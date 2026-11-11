"""Open clamped NURBS shared by rendering, profile construction and constraints."""
import numpy as np
from scipy.interpolate import BSpline


def nurbs_data(points, degree, knots=None, weights=None):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or len(points) < 3 or len(points) > 100 or not np.isfinite(points).all():
        raise ValueError('NURBS requires 3 to 100 finite control points.')
    if type(degree) is not int or not 2 <= degree <= min(5, len(points) - 1):
        raise ValueError('NURBS degree must be between 2 and 5 and below the point count.')
    n = len(points)
    weights = np.asarray(weights if weights is not None else np.ones(n), dtype=float)
    if weights.shape != (n,) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError('NURBS weights must be positive and match the control points.')
    if knots is None:
        knots = [0.] * (degree + 1) + [i / (n - degree) for i in range(1, n - degree)] + [1.] * (degree + 1)
    knots = np.asarray(knots, dtype=float)
    if knots.shape != (n + degree + 1,) or not np.isfinite(knots).all() or np.any(np.diff(knots) < 0):
        raise ValueError('Invalid NURBS knot vector.')
    unique, counts = np.unique(knots, return_counts=True)
    if len(unique) < 2 or counts[0] != degree + 1 or counts[-1] != degree + 1 or any(c > degree for c in counts[1:-1]):
        raise ValueError('NURBS must be open and clamped, with valid interior knot multiplicity.')
    return points, degree, knots, weights


def nurbs_contact(points, degree, knots, weights, t):
    points, degree, knots, weights = nurbs_data(points, degree, knots, weights)
    if not 0 <= t <= 1: raise ValueError('NURBS parameter must be between 0 and 1.')
    parameter = knots[degree] + t * (knots[-degree-1] - knots[degree])
    curve = BSpline(knots, np.column_stack((points * weights[:, None], weights)), degree)
    h, h1, h2 = curve(parameter), curve(parameter, 1), curve(parameter, 2)
    point = h[:-1] / h[-1]
    first = (h1[:-1] - point * h1[-1]) / h[-1]
    second = (h2[:-1] - 2 * first * h1[-1] - point * h2[-1]) / h[-1]
    speed = np.linalg.norm(first)
    if speed < 1e-9: raise ValueError('NURBS tangent must be non-zero.')
    tangent = first / speed
    return point, tangent, (second - tangent * np.dot(second, tangent)) / speed**2


def _edge_from_data(points, degree, knots, weights):
    import build123d as cad
    from OCP.Geom import Geom_BSplineCurve
    from OCP.gp import gp_Pnt
    from OCP.TColgp import TColgp_Array1OfPnt
    from OCP.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
    unique, counts = np.unique(knots, return_counts=True)
    poles, w = TColgp_Array1OfPnt(1, len(points)), TColStd_Array1OfReal(1, len(points))
    k, m = TColStd_Array1OfReal(1, len(unique)), TColStd_Array1OfInteger(1, len(unique))
    for i, (point, weight) in enumerate(zip(points, weights), 1):
        poles.SetValue(i, gp_Pnt(*map(float, point)))
        w.SetValue(i, float(weight))
    for i, (knot, count) in enumerate(zip(unique, counts), 1):
        k.SetValue(i, float(knot)); m.SetValue(i, int(count))
    curve = Geom_BSplineCurve(poles, w, k, m, degree, False)
    return cad.Edge(BRepBuilderAPI_MakeEdge(curve).Edge())


def nurbs_edge(points, degree, knots, weights):
    return _edge_from_data(*nurbs_data(points, degree, knots, weights))


def periodic_nurbs_data(points, degree, weights, spans=None):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or not 3 <= len(points) <= 100 or not np.isfinite(points).all():
        raise ValueError('Periodic NURBS requires 3 to 100 finite control points.')
    if type(degree) is not int or not 2 <= degree <= min(5, len(points) - 1):
        raise ValueError('Periodic NURBS degree must be between 2 and 5 and below point count.')
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (len(points),) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError('Periodic NURBS weights must be positive and match control points.')
    expanded = np.concatenate((points, points[:degree]))
    weights = np.concatenate((weights, weights[:degree]))
    count = len(points)
    spans = np.asarray(spans if spans is not None else np.ones(count), dtype=float)
    if spans.shape != (count,) or not np.isfinite(spans).all() or np.any(spans < 0) or np.sum(spans) <= 0:
        raise ValueError('Periodic NURBS knot spans must be nonnegative, match control points and contain a positive span.')
    zero_runs = 0
    for span in np.concatenate((spans, spans)):
        zero_runs = zero_runs + 1 if span == 0 else 0
        if zero_runs >= degree:
            raise ValueError('Periodic NURBS knot multiplicity cannot exceed its degree.')
    # Extend the cyclic knot intervals on both sides of one complete period.
    intervals = [spans[(i - degree) % count] for i in range(count + 2 * degree)]
    knots = np.concatenate(([0.], np.cumsum(intervals)))
    return expanded, degree, knots, weights


def periodic_nurbs_contact(points, degree, weights, t, spans=None):
    if not 0 <= t <= 1: raise ValueError('NURBS parameter must be between 0 and 1.')
    points, degree, knots, weights = periodic_nurbs_data(points, degree, weights, spans)
    # The seam is evaluated from its start side, including repeated end knots.
    parameter = knots[degree] + (0 if t == 1 else t) * (knots[-degree-1] - knots[degree])
    curve = BSpline(knots, np.column_stack((points * weights[:, None], weights)), degree)
    h, h1, h2 = curve(parameter), curve(parameter, 1), curve(parameter, 2)
    point = h[:-1] / h[-1]
    first = (h1[:-1] - point * h1[-1]) / h[-1]
    second = (h2[:-1] - 2 * first * h1[-1] - point * h2[-1]) / h[-1]
    speed = np.linalg.norm(first)
    if speed < 1e-9: raise ValueError('Periodic NURBS tangent must be non-zero.')
    tangent = first / speed
    return point, tangent, (second - tangent * np.dot(second, tangent)) / speed**2


def periodic_nurbs_edge(points, degree, weights, spans=None):
    return _edge_from_data(*periodic_nurbs_data(points, degree, weights, spans))
