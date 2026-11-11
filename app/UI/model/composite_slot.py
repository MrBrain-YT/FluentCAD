"""Constant-width slots along smoothly connected line, arc and spline pieces."""
import math
from types import SimpleNamespace
import numpy as np


def path_contact(segment, t):
    kind, points, degree, knots, weights = segment
    points = np.asarray(points, dtype=float)
    if not 0 <= t <= 1: raise ValueError('Path parameter must be between 0 and 1.')
    if kind == 'line':
        a, b = points
        direction = b-a
        length = np.linalg.norm(direction)
        if length < 1e-9: raise ValueError('Path line has zero length.')
        return a + t*direction, direction/length, np.zeros_like(a)
    if kind == 'arc':
        from UI.model.constraints import arc_center_radius
        center, radius = arc_center_radius(*points)
        a, b, middle = (math.atan2(*(p-center)[::-1]) for p in points)
        delta, mid = (b-a) % (2*math.pi), (middle-a) % (2*math.pi)
        sweep = delta if mid < delta else delta-2*math.pi
        angle = a + t*sweep
        radial = np.array([math.cos(angle), math.sin(angle)])
        return center + radius*radial, np.sign(sweep)*np.array([-radial[1], radial[0]]), -radial/radius
    from UI.model.spline_slot import spline_path_contact
    return spline_path_contact(points, t, kind, degree, knots, weights)


def composite_contact(segments, radius, component, t):
    if not math.isfinite(radius) or radius <= 1e-6 or not 0 <= t <= 1:
        raise ValueError('Slot radius must be positive and parameter between 0 and 1.')
    if component in ('start_cap', 'end_cap'):
        index = 0 if component == 'start_cap' else len(segments)-1
        parameter = 0 if component == 'start_cap' else 1
    else:
        try:
            _, number, side = component.split(':')
            index = int(number)
            if not component.startswith('segment:') or not 0 <= index < len(segments) or side not in ('left','right'):
                raise ValueError()
        except (ValueError, TypeError) as exc:
            raise ValueError('Invalid composite path segment.') from exc
        parameter = t
    point, tangent, curvature = path_contact(segments[index], parameter)
    normal = np.array([-tangent[1], tangent[0]])
    if component not in ('start_cap','end_cap'):
        sign = 1 if side == 'left' else -1
        factor = 1-sign*radius*np.dot(curvature, normal)
        if factor <= 1e-6: raise ValueError('Slot offset has a cusp; reduce its radius.')
        return point + sign*radius*normal, tangent, curvature/factor
    angle = math.pi*t
    unit = (math.cos(angle)*normal + math.sin(angle)*tangent) if component == 'end_cap' else (-math.cos(angle)*normal - math.sin(angle)*tangent)
    derivative = (-math.sin(angle)*normal + math.cos(angle)*tangent) if component == 'end_cap' else (math.sin(angle)*normal - math.cos(angle)*tangent)
    return point+radius*unit, derivative, -unit/radius


def composite_slot_edges(segments, plane, radius):
    import build123d as cad
    from UI.model.profiles import sketch_edges
    if not 1 <= len(segments) <= 30 or not math.isfinite(radius) or radius <= 1e-6:
        raise ValueError('Composite path requires 1 to 30 segments and a positive radius.')
    for i, segment in enumerate(segments):
        for t in np.linspace(0,1,65):
            for side in ('left','right'):
                composite_contact(segments, radius, f'segment:{i}:{side}', float(t))
        if i:
            a, tangent_a, _ = path_contact(segments[i-1], 1)
            b, tangent_b, _ = path_contact(segment, 0)
            if np.linalg.norm(a-b) > 1e-6 or np.dot(tangent_a,tangent_b) < 1-1e-6:
                raise ValueError('Composite path segments must meet with matching tangent directions.')
    edges = []
    for kind, points, degree, knots, weights in segments:
        world = [np.asarray(tuple(plane.from_local_coords((p[0],p[1],0)))) for p in points]
        if kind == 'line': tool = SimpleNamespace(type='line', points=world)
        elif kind == 'arc': tool = SimpleNamespace(type='curve', points=world)
        else: tool = SimpleNamespace(type=kind, points=world, degree=degree, knots=knots, weights=weights)
        edges.extend(sketch_edges([tool], np.asarray(tuple(plane.z_dir))))
    wires = cad.Wire.combine(edges)
    if len(wires) != 1 or wires[0].is_closed: raise ValueError('Composite path must form one open wire.')
    path = wires[0]
    boundary = path.offset_2d(radius)
    if not isinstance(boundary, cad.Wire) or not boundary.is_closed or not boundary.is_valid:
        raise ValueError('Unable to construct composite slot boundary.')
    face = cad.Face(boundary)
    expected = 2*radius*path.length + math.pi*radius**2
    if not face.is_valid or abs(face.area-expected) > max(1e-5, expected*1e-4):
        raise ValueError('Composite slot overlaps itself or has an invalid offset.')
    return list(boundary.edges())
