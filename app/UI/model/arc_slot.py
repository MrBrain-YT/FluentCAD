"""Exact constant-width slot along a circular arc."""
import math
import numpy as np


def arc_slot_parameters(points, radius):
    from UI.model.constraints import arc_center_radius
    start, end, middle = (np.asarray(p, dtype=float) for p in points)
    center, path_radius = arc_center_radius(start, end, middle)
    if not math.isfinite(radius) or radius <= 1e-6 or radius >= path_radius - 1e-6:
        raise ValueError('Arc slot radius must be positive and smaller than the path radius.')
    def angle(point):
        delta = point - center
        return math.atan2(delta[1], delta[0])
    a = angle(start)
    end_delta, mid_delta = (angle(end) - a) % (2 * math.pi), (angle(middle) - a) % (2 * math.pi)
    sweep = end_delta if mid_delta < end_delta else end_delta - 2 * math.pi
    if abs(sweep) < 1e-9: raise ValueError('Arc slot path must have non-zero sweep.')
    return center, path_radius, a, sweep


def arc_slot_contact(points, radius, component, t):
    if not 0 <= t <= 1: raise ValueError('Slot contact parameter must be between 0 and 1.')
    center, path_radius, start, sweep = arc_slot_parameters(points, radius)
    sign = 1 if sweep > 0 else -1
    angle = start + sweep * (t if component in ('outer', 'inner') else (1 if component == 'end_cap' else 0))
    radial = np.array([math.cos(angle), math.sin(angle)])
    tangent = sign * np.array([-radial[1], radial[0]])
    if component in ('outer', 'inner'):
        distance = path_radius + (radius if component == 'outer' else -radius)
        return center + distance * radial, tangent, -radial / distance
    if component not in ('start_cap', 'end_cap'): raise ValueError('Unknown arc slot component.')
    theta = math.pi * t
    unit = (math.cos(theta) * radial + math.sin(theta) * tangent) if component == 'end_cap' else (-math.cos(theta) * radial - math.sin(theta) * tangent)
    derivative = (-math.sin(theta) * radial + math.cos(theta) * tangent) if component == 'end_cap' else (math.sin(theta) * radial - math.cos(theta) * tangent)
    return center + path_radius * radial + radius * unit, derivative, -unit / radius


def arc_slot_edges(points, normal, radius):
    import build123d as cad
    points = np.asarray(points, dtype=float)
    origin = points[0]
    x = points[1] - origin
    norm = np.linalg.norm(x)
    if norm < 1e-9: raise ValueError('Arc slot endpoints must be distinct.')
    x /= norm
    y = np.cross(np.asarray(normal, dtype=float), x)
    y /= np.linalg.norm(y)
    local = [[np.dot(p - origin, x), np.dot(p - origin, y)] for p in points]
    _, path_radius, _, sweep = arc_slot_parameters(local, radius)
    def world(point): return tuple(origin + point[0] * x + point[1] * y)
    edges = []
    for component in ('outer', 'end_cap', 'inner', 'start_cap'):
        samples = [world(arc_slot_contact(local, radius, component, t)[0]) for t in (0, .5, 1)]
        if component == 'inner': samples.reverse()
        edges.append(cad.Edge.make_three_point_arc(*samples))
    wire = cad.Wire.combine(edges)
    if len(wire) != 1: raise ValueError('Arc slot boundary could not be joined.')
    face = cad.Face(wire[0])
    expected = 2 * radius * path_radius * abs(sweep) + math.pi * radius**2
    if not face.is_valid or abs(face.area - expected) > max(1e-6, expected * 1e-6):
        raise ValueError('Arc slot overlaps itself; reduce its radius or sweep.')
    return edges
