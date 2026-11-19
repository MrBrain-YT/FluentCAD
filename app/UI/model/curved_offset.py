"""Planar offsets joined with a bevel or a circular arc."""
import numpy as np
import build123d as cad
from OCP.Geom import Geom_OffsetCurve, Geom_TrimmedCurve
from OCP.GeomAbs import GeomAbs_C1
from OCP.GeomConvert import GeomConvert_ApproxCurve
from OCP.gp import gp_Dir, gp_Pnt, gp_Vec

from UI.model.curve_spans import curve_fraction_at_point, curve_interval, make_curve_edge, native_curve
from UI.model.sketch_modify import NATIVE_CURVES, TOLERANCE, offset_edge


def _left_offset(edge, distance):
    if edge.geom_type not in NATIVE_CURVES:
        if edge.geom_type == cad.GeomType.CIRCLE:
            radial = edge.position_at(.5) - edge.arc_center
            normal = cad.Vector(0, 0, 1).cross(edge.tangent_at(.5))
            distance *= 1 if radial.dot(normal) > 0 else -1
        return offset_edge(edge, distance, (0, 0, 1)).preview
    curve, first, last = native_curve(edge)
    reversed_edge = not edge.is_forward
    signed = -distance if reversed_edge else distance
    point, tangent, second = gp_Pnt(), gp_Vec(), gp_Vec()
    for u in np.linspace(first, last, 65):
        curve.D2(float(u), point, tangent, second)
        speed = tangent.Magnitude()
        if speed <= 1e-12 or 1 - signed * tangent.Crossed(second).Z() / speed**3 <= 1e-6:
            raise ValueError('Offset would create a cusp or reverse a curve.')
    parallel = Geom_OffsetCurve(curve, -signed, gp_Dir(0, 0, 1))
    # Convert the exact normal offset within modelling tolerance for B-rep intersections.
    approx = GeomConvert_ApproxCurve(Geom_TrimmedCurve(parallel, first, last), 1e-8, GeomAbs_C1, 100, 12)
    if not approx.IsDone() or not approx.HasResult() or approx.MaxError() > TOLERANCE / 10:
        raise ValueError('Could not construct a regular offset for this curve.')
    offset = make_curve_edge(approx.Curve())
    return offset.reversed() if reversed_edge else offset


def _corner_cut(a, b):
    if (a.position_at(1) - b.position_at(0)).length <= TOLERANCE:
        return 1., 0.
    cuts = []
    for hit in a.intersect(b) or ():
        if isinstance(hit, cad.Edge):
            raise ValueError('Neighbouring offsets overlap.')
        cuts.append((curve_fraction_at_point(a, hit.center()), curve_fraction_at_point(b, hit.center())))
    return min(cuts, key=lambda pair: 1 - pair[0] + pair[1]) if cuts else (1., 0.)


def _round_corner(a, b, center, radius):
    radial_a, radial_b = a - center, b - center
    if any(abs(radial.length - radius) > TOLERANCE for radial in (radial_a, radial_b)):
        raise ValueError('Cannot match the round join to both offset segments.')
    bisector = radial_a + radial_b
    if bisector.length <= TOLERANCE:
        raise ValueError('A round offset cannot turn back along the source chain.')
    return cad.Edge.make_three_point_arc(a, center + bisector.normalized() * radius, b)


def curved_offset_chain(edges, distance, closed, join='bevel'):
    if closed:
        samples = [tuple(edge.position_at(t))[:2] for edge in edges for t in np.linspace(0, 1, 33, endpoint=False)]
        area2 = sum(a[0] * b[1] - a[1] * b[0] for a, b in zip(samples, samples[1:] + samples[:1]))
        if abs(area2) <= TOLERANCE**2:
            raise ValueError('The source offset chain has no enclosed area.')
        side = -distance if area2 > 0 else distance
    else:
        side = distance
    parallels = [_left_offset(edge, side) for edge in edges]
    starts, ends = [0.] * len(edges), [1.] * len(edges)
    for i in range(len(edges) if closed and len(edges) > 1 else len(edges) - 1):
        j = (i + 1) % len(edges)
        ends[i], starts[j] = _corner_cut(parallels[i], parallels[j])
    trimmed = []
    for edge, first, last in zip(parallels, starts, ends):
        if (last - first) * edge.length <= TOLERANCE:
            raise ValueError('The offset consumes a segment at a corner.')
        trimmed.append(curve_interval(edge, first, last) if first > 0 or last < 1 else edge)
    groups = []
    for i, edge in enumerate(trimmed):
        group = [edge]
        if closed or i + 1 < len(trimmed):
            following = (i + 1) % len(trimmed)
            a, b = edge.position_at(1), trimmed[following].position_at(0)
            if (a - b).length > TOLERANCE:
                turn = edges[i].tangent_at(1).cross(edges[following].tangent_at(0)).Z
                if turn * side > TOLERANCE:
                    raise ValueError('The offset consumes a segment at an inside corner.')
                if join == 'round':
                    group.append(_round_corner(a, b, edges[i].position_at(1), abs(distance)))
                else:
                    group.append(cad.Edge.make_line(a, b))
        groups.append(group)
    wire = cad.Wire([edge for group in groups for edge in group])
    if not wire.is_valid or wire.is_closed != closed:
        raise ValueError('The offset must remain one connected chain.')
    if closed:
        source, face = cad.Face(cad.Wire(edges)), cad.Face(wire)
        if not source.is_valid or not face.is_valid or (face.area - source.area) * distance <= 0:
            raise ValueError('The offset intersects itself or collapses.')
    return groups
