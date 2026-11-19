"""Native parameter ranges for edits of elliptical and spline sketch curves."""
import build123d as cad
from OCP.BRep import BRep_Tool
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
from OCP.GeomAPI import GeomAPI_ProjectPointOnCurve
from OCP.GeomConvert import GeomConvert_BSplineCurveToBezierCurve
from OCP.TColStd import TColStd_Array1OfReal


CURVE_TYPES = frozenset({'ellipse', 'spline', 'fit_spline', 'nurbs', 'periodic_nurbs'})
PARAMETER_LIMIT = 100


def native_curve(edge):
    adaptor = edge.geom_adaptor()
    first, last = adaptor.FirstParameter(), adaptor.LastParameter()
    curve = BRep_Tool.Curve_s(edge.wrapped, first, last)
    if edge.geom_type == cad.GeomType.BSPLINE and edge.is_closed and not curve.IsPeriodic():
        curve = curve.Copy()
        curve.SetPeriodic()
    return curve, first, last


def curve_fraction_at_point(edge, point):
    """Project to the bounded native parameter range, in edge traversal direction."""
    point = cad.Vector(point)
    for end in (0, 1):
        if (edge.position_at(end) - point).length <= 1e-6:
            return float(end)
    curve, first, last = native_curve(edge)
    projector = GeomAPI_ProjectPointOnCurve(point.to_pnt(), curve, first, last)
    if not projector.NbPoints():
        raise ValueError('Could not locate the point on the curve.')
    fraction = min(1., max(0., (projector.LowerDistanceParameter() - first) / (last - first)))
    return fraction if edge.is_forward else 1. - fraction


def make_curve_edge(curve, first=None, last=None):
    builder = (BRepBuilderAPI_MakeEdge(curve) if first is None
               else BRepBuilderAPI_MakeEdge(curve, first, last))
    if not builder.IsDone():
        raise ValueError('Cannot construct the requested curve range.')
    return cad.Edge(builder.Edge())


def curve_interval(edge, start, end):
    curve, first, last = native_curve(edge)
    if not edge.is_forward:
        start, end = 1. - end, 1. - start
    u, v = first + start * (last - first), first + end * (last - first)
    if edge.geom_type == cad.GeomType.BEZIER and (start < 0 or end > 1):
        curve = curve.Copy()
        curve.Segment(u, v)
        result = make_curve_edge(curve)
    elif edge.geom_type == cad.GeomType.BSPLINE and not curve.IsPeriodic() and (start < 0 or end > 1):
        if not (end <= 0 or start >= 1):
            raise ValueError('Spline continuation must start at the end of its native curve.')
        converter = GeomConvert_BSplineCurveToBezierCurve(curve, first, last, 1e-10)
        count = converter.NbArcs()
        knots = TColStd_Array1OfReal(1, count + 1)
        converter.Knots(knots)
        span = 1 if start < 0 else count
        a, b = list(knots)[span - 1:span + 1]
        tail = converter.Arc(span)
        tail.Segment((u - a) / (b - a), (v - a) / (b - a))
        result = make_curve_edge(tail)
    else:
        result = make_curve_edge(curve, u, v)
    return result if edge.is_forward else result.reversed()


def validate_curve_spans(raw):
    if not isinstance(raw, dict) or len(raw) > 100:
        raise ValueError('Invalid edited sketch curve ranges.')
    parsed, tokens = {}, set()
    for key, spans in raw.items():
        if not isinstance(key, str) or not key.isascii() or not key.isdecimal() or len(key) > 3:
            raise ValueError('Invalid sketch curve segment.')
        segment = int(key)
        if segment in parsed or not isinstance(spans, list) or len(spans) > 100:
            raise ValueError('Invalid edited sketch curve ranges.')
        parsed[segment] = []
        for item in spans:
            if not isinstance(item, list) or len(item) != 3:
                raise ValueError('Invalid edited sketch curve range.')
            token, start, end = item
            if not (type(token) is int and token == segment or isinstance(token, str) and 0 < len(token) <= 64):
                raise ValueError('Invalid edited sketch curve identity.')
            if (token in tokens or type(start) not in (float, int) or type(end) not in (float, int)
                    or not -PARAMETER_LIMIT <= start < end <= PARAMETER_LIMIT):
                raise ValueError('Invalid edited sketch curve range.')
            tokens.add(token)
            parsed[segment].append((token, start, end))
    return parsed
