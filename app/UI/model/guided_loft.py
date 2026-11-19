"""Build a closed loft from section boundaries and exact guide subcurves."""
import math

import build123d as cad
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_Sewing
from OCP.Geom import Geom_Plane, Geom_TrimmedCurve
from OCP.GeomAbs import GeomAbs_C1
from OCP.GeomAPI import GeomAPI_IntCS
from OCP.GeomConvert import GeomConvert, GeomConvert_CompCurveToBSplineCurve
from OCP.GeomFill import GeomFill_BSplineCurves, GeomFill_CoonsStyle, GeomFill_StretchStyle
from OCP.Standard import Standard_ConstructionError
from OCP.ShapeUpgrade import ShapeUpgrade_ShapeDivideContinuity
from OCP.TopAbs import TopAbs_SHELL
from OCP.TopoDS import TopoDS
from OCP.gp import gp_Pln

from UI.model.curve_spans import curve_fraction_at_point, curve_interval, native_curve


TOLERANCE = 1e-6


def guide_section_contact(guide, profile, perimeter):
    origin, normal = profile.center(), profile.normal_at()
    curve, first, last = native_curve(guide)
    surface = Geom_Plane(gp_Pln(origin.to_pnt(), normal.to_dir()))
    intersection = GeomAPI_IntCS(Geom_TrimmedCurve(curve, first, last), surface)
    if not intersection.IsDone() or intersection.NbSegments():
        raise ValueError('Each guide must cross the plane of each section.')
    points = [cad.Vector(intersection.Point(i).Coord()) for i in range(1, intersection.NbPoints() + 1)]
    points.extend(point for point in (guide.position_at(0), guide.position_at(1))
                  if abs((point - origin).dot(normal)) <= TOLERANCE)
    hits = []
    for point in points:
        if any((point - previous[0]).length <= TOLERANCE for previous in hits):
            continue
        vertex = cad.Vertex(*tuple(point))
        if vertex.distance_to(guide) > TOLERANCE:
            continue
        for index, edge in enumerate(perimeter):
            if vertex.distance_to(edge) <= TOLERANCE:
                mark = (index + curve_fraction_at_point(edge, point)) % len(perimeter)
                hits.append((point, mark))
                break
    if len(hits) != 1:
        raise ValueError('Each guide must meet each section boundary exactly once.')
    point, mark = hits[0]
    return mark, curve_fraction_at_point(guide, point)


def _opposite_mark(perimeter, mark):
    index = int(mark)
    fraction = mark - index
    distance = sum(edge.length for edge in perimeter[:index])
    if fraction > 1e-9:
        distance += curve_interval(perimeter[index], 0., fraction).length
    length = sum(edge.length for edge in perimeter)
    remaining = (distance + length / 2) % length
    for index, edge in enumerate(perimeter):
        if remaining < edge.length - TOLERANCE:
            point = edge.position_at(remaining / edge.length)
            return (index + curve_fraction_at_point(edge, point)) % len(perimeter)
        remaining = max(0., remaining - edge.length)
    return 0.


def _spline(edge):
    curve, first, last = native_curve(edge)
    spline = GeomConvert.CurveToBSplineCurve_s(Geom_TrimmedCurve(curve, first, last))
    return spline if edge.is_forward else spline.Reversed()


def _section_curve(perimeter, start, end):
    if end <= start:
        end += len(perimeter)
    pieces = []
    for i in range(math.floor(start), math.ceil(end)):
        a, b = max(0., start - i), min(1., end - i)
        edge = perimeter[i % len(perimeter)]
        if (b - a) * edge.length > TOLERANCE:
            pieces.append(_spline(curve_interval(edge, a, b) if a > 0 or b < 1 else edge))
    if not pieces:
        raise ValueError('Guides must meet the section at distinct points.')
    composite = GeomConvert_CompCurveToBSplineCurve(pieces[0])
    for piece in pieces[1:]:
        if not composite.Add(piece, TOLERANCE, True):
            raise ValueError('A section boundary is disconnected.')
    return composite.BSplineCurve()


def _point_at(perimeter, mark):
    index = int(mark)
    edge = perimeter[index]
    curve, first, last = native_curve(edge)
    fraction = mark - index
    if not edge.is_forward:
        fraction = 1. - fraction
    return cad.Vector(curve.Value(first + fraction * (last - first)).Coord())


def guided_loft(profiles, guides, ends=None):
    ends = ends or {}
    if ends:
        from UI.model.loft_patch_continuity import PATCH_SUPPORTS, constrain_loft_patch, support_meets_section
        if len(guides) < 2:
            raise ValueError('Use at least two guides for general Loft end conditions.')
        for side, (_, support) in ends.items():
            if support.geom_type not in PATCH_SUPPORTS:
                raise ValueError('Select a planar, cylindrical or extruded support face for general guided Loft.')
            if not support_meets_section(profiles[0 if side == 'start' else -1], support):
                raise ValueError('The support face boundary must coincide with the end section.')
    if any(profile.inner_wires() for profile in profiles):
        raise ValueError('Guided Loft requires sections without holes.')
    if any(guide.is_closed for guide in guides):
        raise ValueError('Select open curves for Loft guides.')
    perimeters = [profile.outer_wire().order_edges() for profile in profiles]
    normal = profiles[0].normal_at()
    for i, profile in enumerate(profiles):
        if normal.dot(profile.normal_at()) < 0:
            perimeters[i] = [edge.reversed() for edge in reversed(perimeters[i])]
    contacts = [[guide_section_contact(guide, profile, perimeter) for guide in guides]
                for profile, perimeter in zip(profiles, perimeters)]
    marks = [[mark for mark, _ in row] for row in contacts]
    order = sorted(range(len(guides)), key=marks[0].__getitem__)
    for row, perimeter in zip(marks, perimeters):
        ring = len(perimeter)
        relative = [(row[i] - row[order[0]]) % ring for i in order]
        if any(b - a <= 1e-9 for a, b in zip(relative, relative[1:])):
            raise ValueError('Guides must meet every section in the same cyclic order at distinct points.')
    for i in range(len(guides)):
        positions = [row[i][1] for row in contacts]
        direction = 1 if positions[-1] > positions[0] else -1
        if any((b - a) * direction <= 1e-9 for a, b in zip(positions, positions[1:])):
            raise ValueError('Guides must pass through the sections in their listed order.')

    if len(guides) == 1:
        order.append(1)
        for row, perimeter in zip(marks, perimeters):
            row.append(_opposite_mark(perimeter, row[0]))

    sections = [[_section_curve(perimeter, row[i], row[j])
                 for i, j in zip(order, order[1:] + order[:1])]
                for perimeter, row in zip(perimeters, marks)]
    sewing = BRepBuilderAPI_Sewing(TOLERANCE)
    sewing.Add(profiles[0].wrapped)
    sewing.Add(profiles[-1].wrapped)
    for level, (lower, upper) in enumerate(zip(sections, sections[1:])):
        rails = [_spline(curve_interval(guide, *sorted((contacts[level][i][1], contacts[level + 1][i][1]))))
                 for i, guide in enumerate(guides)]
        if len(guides) == 1:
            a, b = (_point_at(perimeters[j], marks[j][1]) for j in (level, level + 1))
            rails.append(_spline(cad.Edge.make_line(a, b)))
        for k, (i, j) in enumerate(zip(order, order[1:] + order[:1])):
            boundaries = lower[k], rails[j], upper[k], rails[i]
            try:
                surface = GeomFill_BSplineCurves(*boundaries, GeomFill_CoonsStyle).Surface()
            except Standard_ConstructionError:
                surface = GeomFill_BSplineCurves(*boundaries, GeomFill_StretchStyle).Surface()
            patch_ends = {}
            if level == 0 and 'start' in ends:
                patch_ends['start'] = ends['start']
            if level == len(sections) - 2 and 'end' in ends:
                patch_ends['end'] = ends['end']
            if patch_ends:
                constrain_loft_patch(surface, lower[k], patch_ends)
            sewing.Add(BRepBuilderAPI_MakeFace(surface, TOLERANCE).Face())
    sewing.Perform()
    shell = sewing.SewedShape()
    if shell.IsNull() or shell.ShapeType() != TopAbs_SHELL or sewing.NbFreeEdges() or sewing.NbMultipleEdges():
        raise ValueError('The guided surfaces do not form one closed shell.')
    solid = cad.Solid(cad.Shell(TopoDS.Shell_s(shell)))
    # Composite section curves may put a sharp corner inside a surface patch.
    divider = ShapeUpgrade_ShapeDivideContinuity(solid.wrapped)
    divider.SetSurfaceCriterion(GeomAbs_C1)
    if divider.Perform():
        solid = cad.Solid(TopoDS.Solid_s(divider.Result()))
    return solid
