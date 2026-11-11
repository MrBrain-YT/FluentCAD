"""Numerical constraint engine for independent 2-D sketch geometry.

This module does not mutate UI objects; callers must explicitly apply a successful result.
"""
from dataclasses import dataclass
import math
import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import CubicSpline
from UI.model.expressions import evaluate_expression


@dataclass(frozen=True)
class Constraint:
    kind: str
    entities: tuple
    value: object = None
    driven: bool = False


@dataclass(frozen=True)
class SolveResult:
    points: dict
    radii: dict
    degrees_of_freedom: int
    status: str
    residual: float
    redundant: bool


def arc_center_radius(start, end, middle):
    """Circumcenter of a three-point arc; reject collinear or coincident points."""
    a, b, m = (np.asarray(v, dtype=float) for v in (start, end, middle))
    u, v = b - a, m - a
    determinant = u[0] * v[1] - u[1] * v[0]
    scale = max(np.linalg.norm(u), np.linalg.norm(v), 1e-12)
    if abs(determinant) < 1e-12 * scale * scale:
        raise ValueError('Arc points must be distinct and non-collinear.')
    rhs_u, rhs_v = np.dot(u, u) / 2, np.dot(v, v) / 2
    center = a + np.array([(rhs_u * v[1] - rhs_v * u[1]) / determinant,
                           (u[0] * rhs_v - v[0] * rhs_u) / determinant])
    return center, float(np.linalg.norm(center - a))


def ellipse_contact(center, major, minor, rotation, parameter):
    """Point, unit tangent and curvature vector of a rotated ellipse (angles in degrees)."""
    if major <= 1e-6 or minor <= 1e-6:
        raise ValueError('Ellipse radii must be positive.')
    angle, t = math.radians(rotation), math.radians(parameter)
    matrix = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    ct, st = math.cos(t), math.sin(t)
    point = np.asarray(center, dtype=float) + matrix @ np.array([major * ct, minor * st])
    first = matrix @ np.array([-major * st, minor * ct])
    second = matrix @ np.array([-major * ct, -minor * st])
    speed = np.linalg.norm(first)
    tangent = first / speed
    curvature = (second - tangent * np.dot(second, tangent)) / speed**2
    return point, tangent, curvature


def spline_endpoint(points, end):
    if len(points) != 4 or end not in (0, 1):
        raise ValueError('Cubic spline requires four control points and an endpoint.')
    p = np.asarray(points, dtype=float)
    if end == 0:
        point, first, second = p[0], 3 * (p[1] - p[0]), 6 * (p[0] - 2*p[1] + p[2])
    else:
        point, first, second = p[3], 3 * (p[3] - p[2]), 6 * (p[3] - 2*p[2] + p[1])
    speed = np.linalg.norm(first)
    if speed < 1e-9: raise ValueError('Spline endpoint tangent must be non-zero.')
    tangent = first / speed
    return point, tangent, (second - tangent * np.dot(second, tangent)) / speed**2


def fit_spline(points):
    """Interpolating cubic spline with uniform parameters and not-a-knot ends."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or len(points) < 3 or not np.isfinite(points).all():
        raise ValueError('Fit spline requires at least three finite points.')
    if any(np.linalg.norm(b - a) < 1e-9 for a, b in zip(points, points[1:])):
        raise ValueError('Adjacent fit points must be distinct.')
    return CubicSpline(np.arange(len(points)), points, axis=0)


def fit_spline_endpoint(points, end):
    curve = fit_spline(points)
    t = 0 if end == 0 else len(points) - 1
    point, first, second = curve(t), curve(t, 1), curve(t, 2)
    speed = np.linalg.norm(first)
    if speed < 1e-9: raise ValueError('Spline endpoint tangent must be non-zero.')
    tangent = first / speed
    return point, tangent, (second - tangent * np.dot(second, tangent)) / speed**2


def slot_contact(points, component, t):
    start, end, width = (np.asarray(v, dtype=float) for v in points)
    axis = end - start
    length = np.linalg.norm(axis)
    if length < 1e-9: raise ValueError('Slot axis must be non-zero.')
    axis /= length
    normal = np.array([-axis[1], axis[0]])
    r = abs(np.dot(width - start, normal))
    if r < 1e-6: raise ValueError('Slot width must be non-zero.')
    if not 0 <= t <= 1: raise ValueError('Slot contact parameter must be between 0 and 1.')
    if component in ('left', 'right'):
        sign = 1 if component == 'left' else -1
        return start + t * (end - start) + sign*r*normal, axis, np.zeros(2)
    angle = math.pi*t
    if component == 'end_cap':
        radial = math.cos(angle)*normal + math.sin(angle)*axis
        tangent = -math.sin(angle)*normal + math.cos(angle)*axis
        return end + r*radial, tangent, -radial/r
    if component == 'start_cap':
        radial = -math.cos(angle)*normal - math.sin(angle)*axis
        tangent = math.sin(angle)*normal - math.cos(angle)*axis
        return start + r*radial, tangent, -radial/r
    raise ValueError('Unknown slot component.')


def composite_slot_contact(points, radius, component, t):
    if not component.startswith('segment:'):
        raise ValueError('Choose a composite slot segment.')
    try:
        _, index, part = component.split(':')
        index = int(index)
        if not 0 <= index < len(points) - 1: raise ValueError('Invalid segment index.')
        start, end = (np.asarray(p, dtype=float) for p in points[index:index + 2])
    except (ValueError, TypeError) as exc:
        raise ValueError('Invalid composite slot segment.') from exc
    direction = end - start
    length = np.linalg.norm(direction)
    if length < 1e-9 or radius <= 1e-6: raise ValueError('Invalid composite slot geometry.')
    normal = np.array([-direction[1], direction[0]]) / length
    return slot_contact([start, end, start + radius * normal], part, t)


def solve_constraints(points, radii, constraints, parameters=None, tolerance=1e-7):
    """Solve point coordinates and circle radii; report conflicts without changing inputs.

    Line entities are pairs of point names; circles are (center point, radius name).
    Angles are in degrees. Fixed values are captured explicitly as (x, y).
    """
    point_names, radius_names = list(points), list(radii)
    if set(point_names) & set(radius_names):
        raise ValueError('Point and radius names must be distinct.')
    coords = [np.asarray(points[name], dtype=float) for name in point_names]
    if any(p.shape != (2,) or not np.isfinite(p).all() for p in coords):
        raise ValueError('Every sketch point must have two finite coordinates.')
    initial_radii = [float(radii[name]) for name in radius_names]
    if any(not math.isfinite(r) or r <= 0 for r in initial_radii):
        raise ValueError('Circle radii must be positive and finite.')
    x0 = np.asarray([v for p in coords for v in p] + initial_radii, dtype=float)
    if not len(x0):
        raise ValueError('The sketch has no variables.')
    pmap = {name: i * 2 for i, name in enumerate(point_names)}
    rmap = {name: 2 * len(point_names) + i for i, name in enumerate(radius_names)}
    parameters = parameters or {}
    active = [c for c in constraints if not c.driven]

    def residual(x):
        result = []
        def p(name):
            if name not in pmap: raise ValueError(f'Unknown point: {name}')
            i = pmap[name]
            return x[i:i + 2]
        def line(entity):
            a, b = entity
            return p(a), p(b)
        def direction(entity):
            a, b = line(entity)
            v = b - a
            norm = np.linalg.norm(v)
            if norm < 1e-12: raise ValueError('A constrained line has zero length.')
            return v / norm
        def circle(entity):
            center, radius = entity
            if radius not in rmap: raise ValueError(f'Unknown radius: {radius}')
            return p(center), x[rmap[radius]]
        def arc(entity):
            if len(entity) != 3: raise ValueError('Arc requires three named points.')
            return arc_center_radius(*(p(name) for name in entity))
        def arc_end(entity, endpoint):
            if endpoint not in entity[:2]: raise ValueError('Choose an arc endpoint.')
            center, radius = arc(entity)
            return p(endpoint), center, radius
        def spline_end(entity, endpoint):
            fitted = bool(entity and entity[0] == '@fit')
            rational = bool(entity and entity[0] == '@nurbs')
            names = entity[4:] if rational else (entity[1:] if fitted else entity)
            if rational:
                from UI.model.nurbs import nurbs_contact
                if endpoint not in (names[0], names[-1]): raise ValueError('Choose a NURBS endpoint.')
                return nurbs_contact([p(name) for name in names], entity[1], entity[2], entity[3],
                                     0 if endpoint == names[0] else 1)
            if (fitted and len(names) < 3) or (not fitted and len(names) != 4) or endpoint not in (names[0], names[-1]):
                raise ValueError('Choose a valid spline endpoint.')
            values = [p(name) for name in names]
            return (fit_spline_endpoint if fitted else spline_endpoint)(values, 0 if endpoint == names[0] else 1)
        def contact_slot(entity, component, parameter, require_curvature=False):
            if entity and entity[0] == '@composite_path_slot':
                from UI.model.composite_slot import composite_contact
                segments = [(kind, [p(name) for name in names], degree, knots, weights)
                            for kind, names, degree, knots, weights in entity[2]]
                result = composite_contact(segments, entity[1], component, parameter)
                if require_curvature and component.startswith('segment:'):
                    _, index, side = component.split(':')
                    index = int(index)
                    segment = segments[index]
                    if segment[0] == 'nurbs' and 0 < parameter < 1:
                        degree, knots = segment[2], np.asarray(segment[3])
                        normalized = (knots - knots[degree]) / (knots[-degree-1] - knots[degree])
                        if np.min(np.abs(normalized - parameter)) < 1e-9:
                            eps = min(1e-8, parameter / 2, (1-parameter) / 2)
                            before = composite_contact(segments, entity[1], component, parameter-eps)
                            after = composite_contact(segments, entity[1], component, parameter+eps)
                            if np.linalg.norm(before[2]-after[2]) > 1e-5 * max(1., np.linalg.norm(before[2]), np.linalg.norm(after[2])):
                                raise ValueError('Curvature is discontinuous at this composite path knot.')
                    other = index - 1 if parameter == 0 else index + 1 if parameter == 1 else -1
                    if 0 <= other < len(segments):
                        other_t = 1 if parameter == 0 else 0
                        neighbor = composite_contact(segments, entity[1], f'segment:{other}:{side}', other_t)
                        if np.linalg.norm(result[2] - neighbor[2]) > 1e-5 * max(1., np.linalg.norm(result[2]), np.linalg.norm(neighbor[2])):
                            raise ValueError('Curvature is discontinuous at this composite path joint.')
                return result
            if entity and entity[0] == '@spline_slot':
                from UI.model.spline_slot import spline_slot_contact
                points = [p(name) for name in entity[6:]]
                result = spline_slot_contact(points, entity[1], component, parameter, entity[2], entity[3], entity[4], entity[5])
                if entity[2] == 'nurbs' and component in ('left', 'right') and 0 < parameter < 1:
                    knots, degree = np.asarray(entity[4]), entity[3]
                    normalized = (knots - knots[degree]) / (knots[-degree-1] - knots[degree])
                    if np.min(np.abs(normalized - parameter)) < 1e-9:
                        eps = min(1e-8, parameter / 2, (1 - parameter) / 2)
                        left = spline_slot_contact(points, entity[1], component, parameter - eps, entity[2], degree, entity[4], entity[5])
                        right = spline_slot_contact(points, entity[1], component, parameter + eps, entity[2], degree, entity[4], entity[5])
                        if np.linalg.norm(left[1] - right[1]) > 1e-5:
                            raise ValueError('Tangent is discontinuous at this slot path knot.')
                        if require_curvature and np.linalg.norm(left[2] - right[2]) > 1e-5 * max(1., np.linalg.norm(left[2]), np.linalg.norm(right[2])):
                            raise ValueError('Curvature is discontinuous at this slot path knot.')
                return result
            if entity and entity[0] == '@arc_slot':
                from UI.model.arc_slot import arc_slot_contact
                return arc_slot_contact([p(name) for name in entity[2:]], entity[1], component, parameter)
            if entity and entity[0] == '@polyline_slot':
                return composite_slot_contact([p(name) for name in entity[2:]], entity[1], component, parameter)
            return slot_contact([p(name) for name in entity], component, parameter)
        def target(c):
            value = evaluate_expression(c.value, parameters) if isinstance(c.value, str) else float(c.value)
            if not math.isfinite(value): raise ValueError('Constraint value must be finite.')
            return value
        def cross(a, b): return a[0] * b[1] - a[1] * b[0]
        def curve_contact(ref, parameter):
            if not 0 <= parameter <= 1: raise ValueError('Curve parameter must be between 0 and 1.')
            kind = ref[0]
            if kind == 'circle':
                center, radius = circle((ref[1], ref[2]))
                return ellipse_contact(center, radius, radius, 0, 360 * parameter)
            if kind == 'ellipse': return ellipse_contact(p(ref[1]), ref[2], ref[3], ref[4], 360 * parameter)
            names, degree, knots, weights, spans = ref[1:]
            points = [p(name) for name in names]
            if kind in ('line', 'arc'):
                from UI.model.composite_slot import path_contact
                return path_contact((kind, points, degree, knots, weights), parameter)
            if kind == 'periodic_nurbs':
                from UI.model.nurbs import periodic_nurbs_contact
                result = periodic_nurbs_contact(points, degree, weights, parameter, spans)
                breakpoints = np.concatenate(([0.], np.cumsum(spans))) / sum(spans)
                evaluator = lambda t: periodic_nurbs_contact(points, degree, weights, t % 1, spans)
            else:
                from UI.model.spline_slot import spline_path_contact
                result = spline_path_contact(points, parameter, kind, degree, knots, weights)
                breakpoints = ((np.asarray(knots)-knots[degree])/(knots[-degree-1]-knots[degree])) if kind == 'nurbs' else []
                evaluator = lambda t: spline_path_contact(points, t, kind, degree, knots, weights)
            if len(breakpoints) and min(abs(np.asarray(breakpoints)-parameter)) < 1e-9 and (kind == 'periodic_nurbs' or 0 < parameter < 1):
                epsilon = 1e-8
                left, right = evaluator(parameter-epsilon), evaluator(parameter+epsilon)
                if np.linalg.norm(left[1]-right[1]) > 1e-5 or np.linalg.norm(left[2]-right[2]) > 1e-5 * max(1., np.linalg.norm(left[2]), np.linalg.norm(right[2])):
                    raise ValueError('Curve does not have continuous curvature at the selected parameter.')
            return result
        for c in active:
            e, kind = c.entities, c.kind
            if kind == 'coincident': result.extend(p(e[0]) - p(e[1]))
            elif kind == 'curvature_curves':
                if not isinstance(c.value, (tuple, list)) or len(c.value) != 2:
                    raise ValueError('Two curve parameters are required.')
                point1, tangent1, curvature1 = curve_contact(e[0], float(c.value[0]))
                point2, tangent2, curvature2 = curve_contact(e[1], float(c.value[1]))
                result.extend(point1-point2)
                result.append(cross(tangent1, tangent2))
                result.extend(curvature1-curvature2)
            elif kind == 'composite_path_joint':
                from UI.model.composite_slot import path_contact
                def geometry(segment):
                    curve_kind, names, degree, knots, weights = segment
                    return (curve_kind, [p(name) for name in names], degree, knots, weights)
                a, tangent_a, _ = path_contact(geometry(e[0]), 1)
                b, tangent_b, _ = path_contact(geometry(e[1]), 0)
                result.extend(a-b)
                result.extend(tangent_a-tangent_b)
            elif kind == 'point_on_line':
                origin, end = line(e[1]); u = direction(e[1]); delta = p(e[0]) - origin
                along = np.dot(delta, u)
                result.extend((cross(u, delta), max(-along, 0.), max(along - np.linalg.norm(end - origin), 0.)))
            elif kind == 'point_on_circle':
                center, radius = circle(e[1]); result.append(np.linalg.norm(p(e[0]) - center) - radius)
            elif kind == 'point_on_slot':
                position, _, _ = contact_slot(e[1], e[2], target(c))
                result.extend(p(e[0]) - position)
            elif kind == 'point_on_spline':
                from UI.model.spline_slot import spline_path_contact
                ref, parameter = e[1], target(c)
                if ref[0] == '@periodic_nurbs':
                    from UI.model.nurbs import periodic_nurbs_contact
                    position, _, _ = periodic_nurbs_contact([p(name) for name in ref[4:]], ref[1], ref[2], parameter, ref[3])
                elif ref[0] == '@nurbs':
                    position, _, _ = spline_path_contact([p(name) for name in ref[4:]], parameter, 'nurbs', ref[1], ref[2], ref[3])
                elif ref[0] == '@fit':
                    position, _, _ = spline_path_contact([p(name) for name in ref[1:]], parameter, 'fit_spline')
                else:
                    position, _, _ = spline_path_contact([p(name) for name in ref], parameter)
                result.extend(p(e[0]) - position)
            elif kind in ('horizontal', 'vertical'):
                a, b = line(e[0]); result.append((b - a)[1 if kind == 'horizontal' else 0])
            elif kind in ('parallel', 'perpendicular'):
                a, b = direction(e[0]), direction(e[1])
                result.append(cross(a, b) if kind == 'parallel' else np.dot(a, b))
            elif kind == 'collinear':
                a, b = line(e[0]); c1, d = line(e[1]); u = direction(e[0])
                result.extend((cross(u, c1 - a), cross(u, d - a)))
            elif kind == 'concentric': result.extend(circle(e[0])[0] - circle(e[1])[0])
            elif kind == 'equal_length':
                a, b = line(e[0]); c1, d = line(e[1]); result.append(np.linalg.norm(b - a) - np.linalg.norm(d - c1))
            elif kind == 'equal_radius': result.append(circle(e[0])[1] - circle(e[1])[1])
            elif kind == 'midpoint':
                a, b = line(e[1]); result.extend(p(e[0]) - (a + b) / 2)
            elif kind == 'fixed':
                fixed = np.asarray(c.value, dtype=float)
                if fixed.shape != (2,) or not np.isfinite(fixed).all(): raise ValueError('Fixed point requires two finite coordinates.')
                result.extend(p(e[0]) - fixed)
            elif kind == 'symmetric':
                a, b = p(e[0]), p(e[1]); origin, _ = line(e[2]); u = direction(e[2]); mid = (a + b) / 2
                result.extend((cross(u, mid - origin), np.dot(b - a, u)))
            elif kind == 'distance': result.append(np.linalg.norm(p(e[0]) - p(e[1])) - target(c))
            elif kind == 'horizontal_distance': result.append(p(e[1])[0] - p(e[0])[0] - target(c))
            elif kind == 'vertical_distance': result.append(p(e[1])[1] - p(e[0])[1] - target(c))
            elif kind in ('radius', 'diameter'):
                radius_name = e[0][1]
                if radius_name not in rmap: raise ValueError(f'Unknown radius: {radius_name}')
                r = x[rmap[radius_name]]
                result.append(r * (2 if kind == 'diameter' else 1) - target(c))
            elif kind == 'angle':
                a, b = direction(e[0]), direction(e[1]); angle = math.radians(target(c))
                result.extend((np.dot(a, b) - math.cos(angle), cross(a, b) - math.sin(angle)))
            elif kind == 'tangent_line_circle':
                origin, _ = line(e[0]); u = direction(e[0]); center, radius = circle(e[1])
                result.append(abs(cross(u, center - origin)) - radius)
            elif kind in ('arc_radius', 'arc_diameter'):
                _, radius = arc(e[0]); result.append(radius * (2 if kind == 'arc_diameter' else 1) - target(c))
            elif kind in ('equal_radius_arc_arc', 'concentric_arc_arc'):
                center1, radius1 = arc(e[0]); center2, radius2 = arc(e[1])
                if kind == 'equal_radius_arc_arc': result.append(radius1 - radius2)
                else: result.extend(center1 - center2)
            elif kind in ('equal_radius_arc_circle', 'concentric_arc_circle'):
                center1, radius1 = arc(e[0]); center2, radius2 = circle(e[1])
                if kind == 'equal_radius_arc_circle': result.append(radius1 - radius2)
                else: result.extend(center1 - center2)
            elif kind == 'tangent_line_arc':
                origin, end = line(e[0]); u = direction(e[0]); point, center, radius = arc_end(e[1], e[2])
                along = np.dot(point - origin, u)
                result.extend((cross(u, point - origin), np.dot(u, (point - center) / radius),
                               max(-along, 0.), max(along - np.linalg.norm(end - origin), 0.)))
            elif kind in ('tangent_arc_arc', 'curvature_arc_arc'):
                point, center, radius = arc_end(e[0], e[1])
                other, center2, radius2 = arc_end(e[2], e[3])
                result.extend(point - other)
                if kind == 'tangent_arc_arc':
                    result.append(cross((point - center) / radius, (other - center2) / radius2))
                else:
                    result.extend((center - point) / radius**2 - (center2 - other) / radius2**2)
            elif kind in ('tangent_arc_circle', 'curvature_arc_circle'):
                point, center, radius = arc_end(e[0], e[1]); center2, radius2 = circle(e[2])
                result.append(np.linalg.norm(point - center2) - radius2)
                if kind == 'tangent_arc_circle':
                    result.append(cross((point - center) / radius, (point - center2) / radius2))
                else:
                    result.extend((center - point) / radius**2 - (center2 - point) / radius2**2)
            elif kind in ('tangent_arc_ellipse', 'curvature_arc_ellipse'):
                point, center, radius = arc_end(e[0], e[1])
                contact, tangent, curvature = ellipse_contact(p(e[2]), *c.value)
                result.extend(point - contact)
                if kind == 'tangent_arc_ellipse':
                    result.append(np.dot((point - center) / radius, tangent))
                else:
                    result.extend((center - point) / radius**2 - curvature)
            elif kind in ('tangent_arc_slot', 'curvature_arc_slot'):
                point, center, radius = arc_end(e[0], e[1])
                other, tangent, curvature = contact_slot(e[2], e[3], target(c), kind.startswith('curvature_'))
                radial = (point - center) / radius
                result.extend(point - other)
                result.append(np.dot(radial, tangent))
                if kind == 'curvature_arc_slot':
                    result.extend(-radial / radius - curvature)
            elif kind in ('tangent_spline_spline', 'curvature_spline_spline',
                          'tangent_spline_arc', 'curvature_spline_arc',
                          'tangent_spline_slot', 'curvature_spline_slot',
                          'tangent_spline_circle', 'curvature_spline_circle',
                          'tangent_spline_ellipse', 'curvature_spline_ellipse',
                          'tangent_spline_periodic_nurbs', 'curvature_spline_periodic_nurbs'):
                point, tangent, curvature = spline_end(e[0], e[1])
                if kind.endswith('_spline'):
                    other, tangent2, curvature2 = spline_end(e[2], e[3])
                elif kind.endswith('_arc'):
                    other, center2, radius2 = arc_end(e[2], e[3])
                    radial = (other - center2) / radius2
                    tangent2 = np.array([-radial[1], radial[0]])
                    curvature2 = -radial / radius2
                elif kind.endswith('_slot'):
                    other, tangent2, curvature2 = contact_slot(e[2], e[3], target(c), kind.startswith('curvature_'))
                elif kind.endswith('_circle'):
                    circle_center, radius2 = circle(e[2])
                    other, tangent2, curvature2 = ellipse_contact(circle_center, radius2, radius2, 0, target(c))
                elif kind.endswith('_periodic_nurbs'):
                    from UI.model.nurbs import periodic_nurbs_contact
                    reference = e[2]
                    if len(reference) < 7 or reference[0] != '@periodic_nurbs':
                        raise ValueError('Choose a periodic NURBS.')
                    controls, degree, weights, spans, parameter = [p(name) for name in reference[4:]], reference[1], reference[2], reference[3], target(c)
                    other, tangent2, curvature2 = periodic_nurbs_contact(controls, degree, weights, parameter, spans)
                    knots = np.concatenate(([0.], np.cumsum(spans))) / sum(spans)
                    if np.min(np.abs(knots - parameter)) < 1e-9:
                        epsilon = 1e-8
                        left = periodic_nurbs_contact(controls, degree, weights, (parameter - epsilon) % 1, spans)
                        right = periodic_nurbs_contact(controls, degree, weights, (parameter + epsilon) % 1, spans)
                        if np.linalg.norm(left[1] - right[1]) > 1e-5:
                            raise ValueError('Tangent is discontinuous at this NURBS knot; choose a different contact point.')
                        if kind.startswith('curvature_') and np.linalg.norm(left[2] - right[2]) > 1e-5 * max(1., np.linalg.norm(left[2]), np.linalg.norm(right[2])):
                            raise ValueError('Curvature is discontinuous at this NURBS knot; choose a different contact point.')
                else:
                    other, tangent2, curvature2 = ellipse_contact(p(e[2]), *c.value)
                result.extend(point - other)
                if kind.startswith('tangent_'):
                    result.append(cross(tangent, tangent2))
                else:
                    result.append(cross(tangent, tangent2))
                    result.extend(curvature - curvature2)
            elif kind in ('tangent_circles_external', 'tangent_circles_internal'):
                a, ra = circle(e[0]); b, rb = circle(e[1]);
                result.append(np.linalg.norm(a - b) - (ra + rb if kind.endswith('external') else abs(ra - rb)))
            else: raise ValueError(f'Unsupported constraint: {kind}')
        return np.asarray(result, dtype=float)

    # Validate constraints before optimization and avoid mutating supplied data.
    initial_residual = residual(x0)
    if not len(initial_residual):
        return SolveResult(dict(zip(point_names, [p.copy() for p in coords])), dict(radii), len(x0), 'underconstrained', 0., False)
    lower = np.full(len(x0), -np.inf)
    lower[2 * len(point_names):] = 1e-9
    solution = least_squares(residual, x0, bounds=(lower, np.full(len(x0), np.inf)),
                             xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=2000)
    if not solution.success or np.max(np.abs(residual(solution.x))) > tolerance:
        # TRF can stop without moving on underdetermined bounded problems.
        alternative = least_squares(residual, x0, bounds=(lower, np.full(len(x0), np.inf)),
                                    method='dogbox', xtol=1e-12, ftol=1e-12,
                                    gtol=1e-12, max_nfev=2000)
        if np.linalg.norm(residual(alternative.x)) < np.linalg.norm(residual(solution.x)):
            solution = alternative
    error = float(np.max(np.abs(residual(solution.x))))
    rank = int(np.linalg.matrix_rank(solution.jac, tol=1e-7))
    dof = max(0, len(x0) - rank)
    status = 'conflict' if not solution.success or error > tolerance else ('underconstrained' if dof else 'fully_constrained')
    values = x0 if status == 'conflict' else solution.x
    solved_points = {name: values[i:i + 2].copy() for name, i in pmap.items()}
    solved_radii = {name: float(values[i]) for name, i in rmap.items()}
    return SolveResult(solved_points, solved_radii, dof, status, error, len(initial_residual) > rank)
