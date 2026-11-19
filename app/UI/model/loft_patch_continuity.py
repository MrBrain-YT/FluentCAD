"""Apply end continuity to guided spline patches without changing their rails."""
import build123d as cad
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.gp import gp_Pnt, gp_Vec


TOLERANCE = 1e-6
PATCH_SUPPORTS = (cad.GeomType.PLANE, cad.GeomType.CYLINDER, cad.GeomType.EXTRUSION)


def support_meets_section(profile, support):
    boundaries = support.edges()
    return all(
        any(cad.Vertex(*tuple(edge.position_at(t))).distance_to(boundary) <= TOLERANCE
            for boundary in boundaries)
        for edge in profile.outer_wire().edges() for t in (0., .19, .39, .59, .81, 1.)
    )


def _orient_sections(surface, lower):
    a, b = lower.StartPoint(), lower.EndPoint()
    u0, u1, v0, v1 = surface.Bounds()
    for exchange, reverse, params in (
        (False, False, (u0, v0, u1, v0)), (False, True, (u0, v1, u1, v1)),
        (True, False, (u0, v0, u0, v1)), (True, True, (u1, v0, u1, v1)),
    ):
        p, q = surface.Value(*params[:2]), surface.Value(*params[2:])
        if min(max(p.Distance(a), q.Distance(b)), max(p.Distance(b), q.Distance(a))) > TOLERANCE:
            continue
        if exchange:
            surface.ExchangeUV()
        if reverse:
            surface.VReverse()
        return
    raise ValueError('Could not align a Loft patch with its section.')


def _end_rows(surface, side, mode, face):
    count = surface.NbUPoles()
    row = 1 if side == 'start' else surface.NbVPoles()
    step = 1 if side == 'start' else -1
    adaptor = BRepAdaptor_Surface(face.wrapped)
    planar = face.geom_type == cad.GeomType.PLANE
    if planar:
        direction = face.normal_at()
    elif face.geom_type == cad.GeomType.CYLINDER:
        direction = cad.Vector(adaptor.Cylinder().Axis().Direction().Coord())
    else:
        direction = cad.Vector(adaptor.Direction().Coord())
    for depth in range(1, 3 if mode == 'curvature' else 2):
        adjacent = row + depth * step
        shifts = []
        weights = []
        for i in (1, count):
            base = cad.Vector(surface.Pole(i, row).Coord())
            delta = cad.Vector(surface.Pole(i, adjacent).Coord()) - base
            axial = delta.dot(direction)
            gap = abs(axial) if planar else (delta - direction * axial).length
            if gap > TOLERANCE:
                name = 'first' if side == 'start' else 'last'
                raise ValueError(f'A guide is incompatible with {mode} continuity at the {name} support.')
            shifts.append(axial)
            weights.append(surface.Weight(i, adjacent) / surface.Weight(i, row))
        if not planar:
            if abs(weights[0] - weights[1]) > TOLERANCE:
                raise ValueError('The guides have incompatible rational weights at the support.')
            if depth == 1 and (shifts[0] * shifts[1] <= 0 or min(map(abs, shifts)) <= TOLERANCE):
                raise ValueError('The guides must leave the support in the same extrusion direction.')
        for i in range(2, count):
            if planar:
                pos = cad.Vector(surface.Pole(i, adjacent).Coord())
                base = cad.Vector(surface.Pole(i, row).Coord())
                pos -= direction * (pos - base).dot(direction)
                surface.SetPole(i, adjacent, pos.to_pnt())
            else:
                t = (i - 1) / (count - 1)
                base = cad.Vector(surface.Pole(i, row).Coord())
                pos = base + direction * ((1 - t) * shifts[0] + t * shifts[1])
                surface.SetPole(i, adjacent, pos.to_pnt(), surface.Weight(i, row) * weights[0])
    u0, u1, v0, v1 = surface.Bounds()
    for t in (0., .17, .37, .63, .83, 1.):
        du, dv = gp_Vec(), gp_Vec()
        surface.D1(u0 + t * (u1 - u0), v0 if side == 'start' else v1, gp_Pnt(), du, dv)
        if du.Magnitude() * dv.Magnitude() < 1e-18 or du.Crossed(dv).Magnitude() <= 1e-8 * du.Magnitude() * dv.Magnitude():
            raise ValueError('The end condition produces a singular surface at the support.')


def constrain_loft_patch(surface, lower, ends):
    _orient_sections(surface, lower)
    surface.IncreaseDegree(surface.UDegree(), max(surface.VDegree(), 5))
    for side, (mode, face) in ends.items():
        _end_rows(surface, side, mode, face)
