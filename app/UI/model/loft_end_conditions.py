"""Construct meridian lofts with exact first- or second-order end conditions."""
import math

import build123d as cad
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.Geom import Geom_TrimmedCurve
from OCP.GeomConvert import GeomConvert
from OCP.gp import gp_Pnt, gp_Vec
from scipy.interpolate import CubicSpline

from UI.model.curve_spans import curve_interval, native_curve
from UI.model.geometry_references import resolve_reference
from UI.model.guided_loft import guide_section_contact
from UI.model.sketch_mirror import same_sketch_curve


TOLERANCE = 1e-6


def _axial_coordinate(point, origin, axis):
    offset = point - origin
    height = offset.dot(axis)
    return height, (offset - axis * height).length


def _derivatives(face, section, center, axis, radius):
    surface = BRepAdaptor_Surface(face.wrapped)
    if face.geom_type not in (cad.GeomType.CYLINDER, cad.GeomType.CONE, cad.GeomType.SPHERE):
        raise ValueError('Select a cylindrical, conical or spherical support face.')
    # A support must meet the section along its entire circumference.
    boundary = face.edges()
    for t in (0., .13, .29, .43, .61, .79, .91):
        vertex = cad.Vertex(*tuple(section.position_at(t)))
        if not any(vertex.distance_to(edge) <= TOLERANCE for edge in boundary):
            raise ValueError('The support face boundary must coincide with the end section.')
    if face.geom_type == cad.GeomType.SPHERE:
        sphere = surface.Sphere()
        height, eccentricity = _axial_coordinate(center, cad.Vector(sphere.Location().Coord()), axis)
        if eccentricity > TOLERANCE or abs(math.hypot(radius, height) - sphere.Radius()) > TOLERANCE:
            raise ValueError('The spherical support must be centered on the loft axis.')
        return -height / radius, -sphere.Radius() ** 2 / radius ** 3
    geom = surface.Cylinder() if face.geom_type == cad.GeomType.CYLINDER else surface.Cone()
    direction = cad.Vector(geom.Axis().Direction().Coord())
    height, eccentricity = _axial_coordinate(center, cad.Vector(geom.Location().Coord()), direction)
    alignment = direction.dot(axis)
    if eccentricity > TOLERANCE or abs(alignment) < 1 - TOLERANCE:
        raise ValueError('The support face must share the loft axis.')
    if face.geom_type == cad.GeomType.CYLINDER:
        if abs(geom.Radius() - radius) > TOLERANCE:
            raise ValueError('The cylinder radius must match the end section.')
        return 0., 0.
    slope = math.tan(geom.SemiAngle())
    signed_radius = geom.RefRadius() + height * slope
    if abs(abs(signed_radius) - radius) > TOLERANCE:
        raise ValueError('The cone radius must match the end section.')
    return slope * alignment * math.copysign(1., signed_radius), 0.


def resolve_loft_support(profile, reference, shape, feature_id, body_names):
    try:
        index = resolve_reference(reference, shape, feature_id, body_names=body_names)
        return shape.faces()[index]
    except ValueError as original:
        # An altered cone can move the side face centroid while the rim still
        # coincides with the section. Retain the body and match this rim exactly.
        if reference.get('version') != 3:
            raise
        local = reference.get('local')
        if not isinstance(local, dict) or local.get('kind') != 'face':
            raise
        owners = [i for i, name in enumerate(body_names or ()) if name == reference.get('body_id')]
        if len(owners) != 1 or owners[0] >= len(shape.solids()):
            raise
        from UI.model.loft_patch_continuity import PATCH_SUPPORTS, support_meets_section
        rims = [cad.Face(wire) for wire in (profile.outer_wire(), *profile.inner_wires())]
        matches = []
        for face in shape.solids()[owners[0]].faces():
            if face.geom_type.name != local.get('geometry'):
                continue
            if (face.geom_type not in (*PATCH_SUPPORTS, cad.GeomType.CONE, cad.GeomType.SPHERE)
                    or not any(support_meets_section(rim, face) for rim in rims)):
                continue
            matches.append(face)
        if len(matches) != 1:
            raise original
        return matches[0]


def _guide_meridian(profiles, rings, guides, origin, axis, conditions):
    meridian = radial = None
    for guide in guides:
        if guide.is_closed:
            raise ValueError('A guide with G1/G2 conditions must be open.')
        fractions = [guide_section_contact(guide, profile, [ring])[1] for profile, ring in zip(profiles, rings)]
        direction = 1 if fractions[-1] > fractions[0] else -1
        if any(direction * (b - a) <= 1e-9 for a, b in zip(fractions, fractions[1:])):
            raise ValueError('Guides must pass through the sections in their listed order.')
        edge = curve_interval(guide, *sorted((fractions[0], fractions[-1])))
        if direction < 0:
            edge = edge.reversed()
        start = edge.position_at(0) - origin
        ray = (start - axis * start.dot(axis)).normalized()
        normal = axis.cross(ray)
        curve, first, last = native_curve(edge)
        spline = GeomConvert.CurveToBSplineCurve_s(Geom_TrimmedCurve(curve, first, last))
        poles = [cad.Vector(spline.Pole(i).Coord()) - origin for i in range(1, spline.NbPoles() + 1)]
        if any(abs(pole.dot(normal)) > TOLERANCE or pole.dot(ray) <= TOLERANCE for pole in poles):
            raise ValueError('With G1/G2, each guide must stay on one side of a plane through the loft axis, away from the axis.')
        if not edge.is_forward:
            poles.reverse()
        if any((b - a).dot(axis) < -TOLERANCE for a, b in zip(poles, poles[1:])):
            raise ValueError('The guide control points must progress along the loft axis.')
        for index, (mode, slope, curvature) in conditions.items():
            at_start = index == 0
            parameter = first if at_start == edge.is_forward else last
            derivative, second = gp_Vec(), gp_Vec()
            curve.D2(parameter, gp_Pnt(), derivative, second)
            dz = cad.Vector(derivative.Coord()).dot(axis)
            if abs(dz) <= 1e-9:
                raise ValueError('The guide must advance along the axis at a constrained end.')
            dr = cad.Vector(derivative.Coord()).dot(ray)
            d2z = cad.Vector(second.Coord()).dot(axis)
            d2r = cad.Vector(second.Coord()).dot(ray)
            end = 'first' if at_start else 'last'
            if abs(dr / dz - slope) > TOLERANCE:
                raise ValueError(f'The guide tangent must match the support at the {end} section.')
            if mode == 'curvature' and abs((d2r - slope * d2z) / dz**2 - curvature) > TOLERANCE:
                raise ValueError(f'The guide curvature must match the support at the {end} section.')
        if meridian is None:
            meridian, radial = edge, ray
        else:
            angle = math.degrees(math.atan2(axis.dot(ray.cross(radial)), ray.dot(radial)))
            aligned = edge.rotate(cad.Axis(origin, axis), angle)
            if abs(aligned.length - meridian.length) > TOLERANCE or not same_sketch_curve(aligned, meridian):
                raise ValueError('Guides with G1/G2 must describe the same meridian around the loft axis.')
    return meridian


def constrained_loft(profiles, ends, guides=()):
    rings = []
    for face in profiles:
        boundary = list(face.outer_wire().edges())
        if (face.inner_wires() or len(boundary) != 1 or boundary[0].geom_type != cad.GeomType.CIRCLE
                or not boundary[0].is_closed):
            raise ValueError('Tangent and curvature end conditions require full circular sections without holes.')
        rings.append(boundary[0])
    origin, last = rings[0].arc_center, rings[-1].arc_center
    if (last - origin).length <= TOLERANCE:
        raise ValueError('Loft section centers must be separated along the axis.')
    axis = (last - origin).normalized()
    heights, radii = [], []
    for face, ring in zip(profiles, rings):
        height, eccentricity = _axial_coordinate(ring.arc_center, origin, axis)
        if eccentricity > TOLERANCE or abs(face.normal_at().dot(axis)) < 1 - TOLERANCE:
            raise ValueError('The circular sections must share one axis.')
        heights.append(height)
        radii.append(ring.radius)
    if any(b - a <= TOLERANCE for a, b in zip(heights, heights[1:])):
        raise ValueError('List the circular sections in order along the loft axis.')
    boundary, curvature, conditions = ['natural', 'natural'], {}, {}
    for side, index in (('start', 0), ('end', -1)):
        if side not in ends:
            continue
        mode, face = ends[side]
        slope, second = _derivatives(face, rings[index], rings[index].arc_center, axis, radii[index])
        conditions[index] = mode, slope, second
        boundary[0 if index == 0 else 1] = (1, slope)
        if mode == 'curvature':
            curvature[index] = second
    if guides:
        meridian = _guide_meridian(profiles, rings, guides, origin, axis, conditions)
        edges = [cad.Edge.make_line(origin, meridian.position_at(0)), meridian,
                 cad.Edge.make_line(meridian.position_at(1), last), cad.Edge.make_line(last, origin)]
        return cad.revolve(cad.Face(cad.Wire(edges)), axis=cad.Axis(origin, axis))
    spline = CubicSpline(heights, radii, bc_type=tuple(boundary))
    slopes = spline(heights, 1)
    seconds = spline(heights, 2)
    for index, second in curvature.items():
        seconds[index] = second
    radial = (rings[0].position_at(0) - origin).normalized()

    def point(height, radius):
        return origin + axis * height + radial * radius

    edges = [cad.Edge.make_line(origin, point(heights[0], radii[0]))]
    for i, (a, b) in enumerate(zip(heights, heights[1:])):
        length = b - a
        r0, r5 = radii[i:i + 2]
        r1 = r0 + length * slopes[i] / 5
        r4 = r5 - length * slopes[i + 1] / 5
        r2 = length**2 * seconds[i] / 20 + 2 * r1 - r0
        r3 = length**2 * seconds[i + 1] / 20 + 2 * r4 - r5
        meridian = cad.Edge.make_bezier(*(point(a + j * length / 5, radius)
                                         for j, radius in enumerate((r0, r1, r2, r3, r4, r5))))
        if any((meridian.position_at(j / 32) - origin).dot(radial) <= TOLERANCE for j in range(33)):
            raise ValueError('The end conditions collapse the loft near its axis.')
        edges.append(meridian)
    edges.extend((cad.Edge.make_line(point(heights[-1], radii[-1]), last), cad.Edge.make_line(last, origin)))
    return cad.revolve(cad.Face(cad.Wire(edges)), axis=cad.Axis(origin, axis))
