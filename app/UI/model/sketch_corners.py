"""Rebuild fillets and equal setbacks on bounded sketch curves."""
import math

import build123d as cad
from OCP.ChFi2d import ChFi2d_FilletAPI
from OCP.TopoDS import TopoDS_Edge
from OCP.gp import gp_Pln

from UI.model.curve_spans import curve_fraction_at_point, curve_interval
from UI.model.profiles import indexed_sketch_edges
from UI.model.sketch_modify import NATIVE_CURVES, TOLERANCE
from UI.model.sketch_projection_curves import sketch_curve_name


CORNER_CURVES = NATIVE_CURVES | {cad.GeomType.LINE, cad.GeomType.CIRCLE}


def corner_source_name(entity, segment, edge):
    refs = entity.get('points', [entity.get('center')])
    if (not entity.get('construction', False) and edge.geom_type in CORNER_CURVES
            and not edge.is_closed and all(isinstance(name, str) for name in refs)):
        return sketch_curve_name(entity, segment)
    return None


def common_corner(a, b):
    pairs = [(i, j) for i in (0, 1) for j in (0, 1)
             if (a.position_at(i) - b.position_at(j)).length <= TOLERANCE]
    if len(pairs) != 1:
        raise ValueError('Choose two segments with exactly one common endpoint.')
    return pairs[0]


def _fillet(plane, a, b, corner, radius):
    local_a, local_b = plane.to_local_coords(a), plane.to_local_coords(b)
    algorithm = ChFi2d_FilletAPI()
    algorithm.Init(local_a.wrapped, local_b.wrapped, gp_Pln())
    if not algorithm.Perform(radius):
        raise ValueError('The fillet radius does not fit this corner.')
    trimmed_a, trimmed_b = TopoDS_Edge(), TopoDS_Edge()
    raw = algorithm.Result(plane.to_local_coords(corner).to_pnt(), trimmed_a, trimmed_b)
    if raw.IsNull() or trimmed_a.IsNull() or trimmed_b.IsNull():
        raise ValueError('The fillet radius does not fit this corner.')
    return plane.from_local_coords(cad.Edge(raw))


def build_corner(plane, edges, ends, kind, size):
    """Return an exact connector and two bounded native-parameter cuts."""
    if kind not in ('fillet', 'chamfer') or not math.isfinite(size) or size <= TOLERANCE:
        raise ValueError('Enter a positive corner size.')
    a, b = (edge if end == target else edge.reversed()
            for edge, end, target in zip(edges, ends, (1, 0)))
    corner = a.position_at(1)
    if (corner - b.position_at(0)).length > TOLERANCE:
        raise ValueError('The corner source endpoints are no longer joined.')
    if abs(a.tangent_at(1).dot(b.tangent_at(0))) >= 1. - 1e-9:
        raise ValueError('The source segments must form a non-tangent corner.')
    if kind == 'chamfer':
        if size >= min(a.length, b.length) - TOLERANCE:
            raise ValueError('The chamfer setback exceeds the available segment length.')
        connector = cad.Edge.make_line(a.position_at(1 - size / a.length), b.position_at(size / b.length))
    else:
        try:
            connector = _fillet(plane, a, b, corner, size)
        except RuntimeError as exc:
            raise ValueError('The fillet radius does not fit this corner.') from exc
        if a.distance_to(connector.position_at(0)) > TOLERANCE:
            connector = connector.reversed()
    cuts = []
    for i, (edge, end) in enumerate(zip(edges, ends)):
        point = connector.position_at(i)
        if edge.distance_to(point) > TOLERANCE:
            raise ValueError('The corner size exceeds the available segment length.')
        fraction = curve_fraction_at_point(edge, point)
        if min(fraction, 1 - fraction) * edge.length <= TOLERANCE:
            raise ValueError('The corner size leaves no usable segment.')
        interval = (fraction, 1.) if end == 0 else (0., fraction)
        part = curve_interval(edge, *interval)
        if part.length <= TOLERANCE:
            raise ValueError('The corner size leaves no usable segment.')
        if kind == 'fillet':
            tangent = part.tangent_at(end)
            if end != 1 - i:
                tangent = -tangent
            if tangent.dot(connector.tangent_at(i)) < 1. - 1e-6:
                raise ValueError('Could not construct a tangent fillet within both segments.')
        cuts.append(interval)
    return connector, tuple(cuts)


def rebuild_sketch_corners(entries, plane, entities, tools, normal, number):
    if not isinstance(entries, list) or len(entries) > 1000:
        raise ValueError('Invalid sketch corners.')
    sources, ambiguous = {}, set()
    slot_paths = {index for entity in entities if entity['type'] == 'composite_path_slot'
                  for index in entity['curves']}
    slot_paths.update(entity['curve'] for entity in entities if entity['type'] == 'spline_slot')
    for entity_index, (entity, tool) in enumerate(zip(entities, tools)):
        for index, edge in indexed_sketch_edges(tool, normal):
            name = corner_source_name(entity, index, edge)
            if name is not None:
                if name in sources:
                    ambiguous.add(name)
                sources[name] = tool, index, edge, entity_index
    limits, connectors, provenance = {}, [], {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError('Invalid sketch corner.')
        identifier, refs, ends = entry.get('id'), entry.get('sources'), entry.get('ends')
        if not isinstance(identifier, str) or not identifier or ':' in identifier:
            raise ValueError('Invalid sketch corner ID.')
        key = f'corner:{identifier}'
        if key in provenance:
            raise ValueError('Duplicate sketch corner ID.')
        if (not isinstance(refs, list) or len(refs) != 2 or not all(isinstance(ref, str) for ref in refs)
                or refs[0] == refs[1] or not isinstance(ends, list) or len(ends) != 2
                or any(type(end) is not int or end not in (0, 1) for end in ends)):
            raise ValueError('Invalid sketch corner sources.')
        if any(ref not in sources or ref in ambiguous for ref in refs):
            raise ValueError('A sketch corner source is missing or ambiguous.')
        if any(sources[ref][3] in slot_paths for ref in refs):
            raise ValueError('A corner source is used by a slot path.')
        connector, cuts = build_corner(plane, [sources[ref][2] for ref in refs], ends,
                                       entry.get('kind'), number(entry.get('size')))
        for ref, end, interval in zip(refs, ends, cuts):
            trims = limits.setdefault(ref, [0., 1.])
            if trims[end] != float(end):
                raise ValueError('A sketch segment endpoint is used by more than one corner.')
            trims[end] = interval[end]
            if (trims[1] - trims[0]) * sources[ref][2].length <= TOLERANCE:
                raise ValueError('Neighbouring sketch corners consume the same segment.')
        connectors.append((key, connector))
        provenance[key] = tuple(refs)
    for ref, interval in limits.items():
        tool, index, _, _ = sources[ref]
        if not hasattr(tool, 'corner_limits'):
            tool.corner_limits = {}
        tool.corner_limits[index] = interval
    return tuple(connectors), provenance
