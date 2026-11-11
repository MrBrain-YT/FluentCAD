"""Constant-width slot about an open cubic Bezier path."""
import math
import numpy as np


def bezier_contact(points, t):
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or len(p) != 4 or not np.isfinite(p).all() or not 0 <= t <= 1:
        raise ValueError('Bezier path requires four finite points and a parameter from 0 to 1.')
    q = 1 - t
    point = q**3*p[0] + 3*q*q*t*p[1] + 3*q*t*t*p[2] + t**3*p[3]
    first = 3*(q*q*(p[1]-p[0]) + 2*q*t*(p[2]-p[1]) + t*t*(p[3]-p[2]))
    second = 6*(q*(p[2]-2*p[1]+p[0]) + t*(p[3]-2*p[2]+p[1]))
    speed = np.linalg.norm(first)
    if speed < 1e-9: raise ValueError('Slot path tangent must be non-zero.')
    tangent = first / speed
    curvature = (second - tangent * np.dot(second, tangent)) / speed**2
    return point, tangent, curvature


def spline_path_contact(points, path_t, kind='spline', degree=None, knots=None, weights=None):
    if not math.isfinite(path_t) or not 0 <= path_t <= 1:
        raise ValueError('Spline parameter must be between 0 and 1.')
    if kind == 'spline':
        point, tangent, curvature = bezier_contact(points, path_t)
    elif kind == 'nurbs':
        from UI.model.nurbs import nurbs_contact
        point, tangent, curvature = nurbs_contact(points, degree, knots, weights, path_t)
    elif kind == 'fit_spline':
        from UI.model.constraints import fit_spline
        curve = fit_spline(points)
        parameter = path_t * (len(points) - 1)
        point, first, second = curve(parameter), curve(parameter, 1), curve(parameter, 2)
        speed = np.linalg.norm(first)
        if speed < 1e-9: raise ValueError('Slot path tangent must be non-zero.')
        tangent = first / speed
        curvature = (second - tangent * np.dot(second, tangent)) / speed**2
    else:
        raise ValueError('Unsupported spline slot path.')
    return point, tangent, curvature


def spline_slot_contact(points, radius, component, t, kind='spline', degree=None, knots=None, weights=None):
    if not math.isfinite(radius) or radius <= 1e-6 or not 0 <= t <= 1:
        raise ValueError('Slot radius must be positive and contact parameter between 0 and 1.')
    path_t = t if component in ('left', 'right') else (1 if component == 'end_cap' else 0)
    point, tangent, curvature = spline_path_contact(points, path_t, kind, degree, knots, weights)
    normal = np.array([-tangent[1], tangent[0]])
    if component in ('left', 'right'):
        side = 1 if component == 'left' else -1
        factor = 1 - side * radius * np.dot(curvature, normal)
        if factor <= 1e-6: raise ValueError('Slot offset has a cusp; reduce its radius.')
        return point + side * radius * normal, tangent, curvature / factor
    if component not in ('start_cap', 'end_cap'): raise ValueError('Unknown spline slot component.')
    angle = math.pi * t
    unit = (math.cos(angle)*normal + math.sin(angle)*tangent) if component == 'end_cap' else (-math.cos(angle)*normal - math.sin(angle)*tangent)
    derivative = (-math.sin(angle)*normal + math.cos(angle)*tangent) if component == 'end_cap' else (math.sin(angle)*normal - math.cos(angle)*tangent)
    return point + radius * unit, derivative, -unit / radius


def spline_slot_edges(points, radius, kind='spline', degree=None, knots=None, weights=None):
    import build123d as cad
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not math.isfinite(radius) or radius <= 1e-6:
        raise ValueError('Spline slot requires 3-D control points and a positive radius.')
    if kind == 'spline':
        if len(points) != 4: raise ValueError('Bezier path requires four points.')
        path = cad.Edge.make_bezier(*[tuple(p) for p in points])
    elif kind == 'nurbs':
        from UI.model.nurbs import nurbs_edge
        path = nurbs_edge(points, degree, knots, weights)
    elif kind == 'fit_spline':
        from UI.model.constraints import fit_spline
        curve = fit_spline(points)
        edges = []
        for i in range(len(points)-1):
            a, b = curve(i), curve(i+1)
            edges.append(cad.Edge.make_bezier(tuple(a), tuple(a+curve(i,1)/3), tuple(b-curve(i+1,1)/3), tuple(b)))
        path = cad.Wire(edges)
    else:
        raise ValueError('Unsupported spline slot path.')
    if path.is_closed: raise ValueError('Slot path must be open.')
    wire = path.offset_2d(radius)
    if not isinstance(wire, cad.Wire) or not wire.is_closed or not wire.is_valid:
        raise ValueError('Unable to construct a closed slot boundary.')
    face = cad.Face(wire)
    expected = 2 * radius * path.length + math.pi * radius**2
    if not face.is_valid or abs(face.area - expected) > max(1e-5, expected * 1e-4):
        raise ValueError('Spline slot overlaps itself or its offset is invalid.')
    return list(wire.edges())
