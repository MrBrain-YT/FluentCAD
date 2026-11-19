"""Associative, one-sided offsets of ordered planar sketch curves."""
from collections import defaultdict
import math

import build123d as cad
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCP.BRepOffsetAPI import BRepOffsetAPI_MakeOffset
from OCP.GeomAbs import GeomAbs_Intersection
from OCP.TopAbs import TopAbs_WIRE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS
from OCP.gp import gp_Pln

from UI.model.curve_spans import CURVE_TYPES
from UI.model.curved_offset import curved_offset_chain
from UI.model.sketch_modify import NATIVE_CURVES, TOLERANCE, offset_edge
from UI.model.sketch_projection_curves import named_sketch_curves, sketch_curve_name


OFFSET_JOINS = {'auto': 'Automatic', 'bevel': 'Bevel', 'round': 'Round'}


def offset_source_name(entity, segment):
    if entity['type'] in CURVE_TYPES or entity['type'] in ('line', 'arc', 'circle'):
        refs = entity.get('points', [entity.get('center')])
        if all(isinstance(ref, str) for ref in refs):
            return sketch_curve_name(entity, segment)
    return None


def order_offset_chain(edges, seed, automatic=False):
    """Find a connected nonbranching chain, retaining the picked curve's direction."""
    if edges[seed].is_closed:
        if automatic or len(edges) == 1:
            return [[seed, False]]
        raise ValueError('Select a closed curve on its own.')
    vertices, neighbors, endpoints = [], defaultdict(list), {}
    for name, edge in edges.items():
        if edge.is_closed:
            continue
        ends = []
        for t in (0, 1):
            point = edge.position_at(t)
            index = next((i for i, vertex in enumerate(vertices) if (point - vertex).length <= TOLERANCE), None)
            if index is None:
                index = len(vertices)
                vertices.append(point)
            ends.append(index)
            neighbors[index].append(name)
        endpoints[name] = ends
    component, pending = set(), [seed]
    while pending:
        name = pending.pop()
        if name in component:
            continue
        component.add(name)
        for vertex in endpoints[name]:
            if len(neighbors[vertex]) > 2:
                raise ValueError('Offset chain branches. Turn off Chain selection and pick a single path.')
            pending.extend(neighbors[vertex])
    if not automatic and len(component) != len(edges):
        raise ValueError('Select connected segments for the offset.')
    start = endpoints[seed][0]
    start = next((vertex for vertex, names in neighbors.items()
                  if len(names) == 1 and names[0] in component), start)
    cursor, ordered = start, []
    while component:
        name = next(name for name in neighbors[cursor] if name in component)
        a, b = endpoints[name]
        reverse = cursor == b
        ordered.append([name, reverse])
        component.remove(name)
        cursor = a if reverse else b
    if next(reverse for name, reverse in ordered if name == seed):
        ordered = [[name, not reverse] for name, reverse in reversed(ordered)]
    return ordered


def offset_chain_closed(edges, references):
    first, start_reversed = references[0]
    last, end_reversed = references[-1]
    return (edges[first].position_at(int(start_reversed))
            - edges[last].position_at(int(not end_reversed))).length <= TOLERANCE


def _joined_offset(edges, distance, closed):
    wire = cad.Wire(edges)
    builder = BRepOffsetAPI_MakeOffset()
    if closed:
        builder.Init(GeomAbs_Intersection)
    else:
        # Give OCCT the sketch orientation even for straight or nearly straight wires.
        builder.Init(BRepBuilderAPI_MakeFace(gp_Pln()).Face(), GeomAbs_Intersection, True)
    builder.AddWire(wire.wrapped)
    builder.Perform(distance)
    if not builder.IsDone():
        raise ValueError('The offset collapses or cannot be joined at this distance.')
    explorer = TopExp_Explorer(builder.Shape(), TopAbs_WIRE)
    if not explorer.More():
        raise ValueError('The offset did not produce a wire.')
    result = cad.Wire(TopoDS.Wire_s(explorer.Current()))
    explorer.Next()
    if explorer.More() or result.is_closed != closed or not result.is_valid:
        raise ValueError('The offset must remain one connected chain.')
    if closed:
        source_face, face = cad.Face(wire), cad.Face(result)
        if not source_face.is_valid or not face.is_valid or (face.area - source_face.area) * distance <= 0:
            raise ValueError('The closed offset intersects itself or collapses.')
    # Generated edges are tied to the ordered source edges, not OCCT's result order.
    output = result.edges()
    groups = []
    for source in wire.edges():
        generated = list(builder.Generated(source.wrapped))
        pieces = [edge for edge in output if any(edge.wrapped.IsSame(item) for item in generated)]
        pieces.sort(key=lambda edge: (edge.position_at(.5) - source.position_at(0)).length)
        groups.append(pieces)
    if len(groups) != len(edges) or not all(groups) or sum(map(len, groups)) != len(output):
        raise ValueError('Offset changed the chain topology and its segments could not be matched.')
    return groups


def build_offset_chain(plane, edges, references, distance, expected_closed=None, *, join='auto'):
    """Return locally offset curves in world coordinates, grouped by source segment."""
    if not math.isfinite(distance) or abs(distance) <= TOLERANCE:
        raise ValueError('Offset distance must be non-zero.')
    if not isinstance(join, str) or join not in OFFSET_JOINS:
        raise ValueError('Invalid offset corner mode.')
    local = []
    for name, reverse in references:
        if name not in edges:
            raise ValueError('An offset source segment is missing or ambiguous.')
        edge = plane.to_local_coords(edges[name])
        local.append(edge.reversed() if reverse else edge)
    closed = offset_chain_closed(edges, references)
    if expected_closed is not None and closed != expected_closed:
        raise ValueError('The source offset chain changed between open and closed.')
    if len(local) > 1 and (any(edge.is_closed for edge in local) or any(
        (a.position_at(1) - b.position_at(0)).length > TOLERANCE for a, b in zip(local, local[1:])
    )):
        raise ValueError('Offset source segments no longer form a connected chain.')
    if any(edge.geom_type in NATIVE_CURVES for edge in local) or (len(local) > 1 and join != 'auto'):
        try:
            groups = curved_offset_chain(local, distance, closed, 'round' if join == 'round' else 'bevel')
        except (RuntimeError, ValueError) as exc:
            raise ValueError(f'Cannot offset this chain. {exc}') from exc
    elif len(local) == 1:
        edge = local[0]
        if edge.geom_type == cad.GeomType.CIRCLE and not edge.is_closed:
            radial = edge.position_at(.5) - edge.arc_center
            left = cad.Vector(0, 0, 1).cross(edge.tangent_at(.5))
            distance *= 1 if radial.dot(left) > 0 else -1
        groups = [offset_edge(edge, distance, (0, 0, 1)).replacement]
    else:
        try:
            groups = _joined_offset(local, distance, closed)
        except (RuntimeError, ValueError) as exc:
            raise ValueError(f'Cannot offset this chain. {exc}') from exc
    return tuple(tuple(plane.from_local_coords(edge) for edge in group) for group in groups)


def rebuild_sketch_offsets(offsets, plane, entities, tools, normal, number, corner_edges=()):
    if not isinstance(offsets, list) or len(offsets) > 1000:
        raise ValueError('Invalid sketch offsets.')
    native = [(entity, tool) for entity, tool in zip(entities, tools) if offset_source_name(entity, 0)]
    edges, ambiguous = dict(corner_edges), set()
    if native:
        definitions, geometry = zip(*native)
        for name, edge in named_sketch_curves(definitions, geometry, normal):
            if name in edges:
                ambiguous.add(name)
            edges[name] = edge
    for name in ambiguous:
        del edges[name]
    output, identifiers = [], set()
    for entry in offsets:
        if not isinstance(entry, dict):
            raise ValueError('Invalid sketch offset.')
        offset_id, sources, closed = entry.get('id'), entry.get('sources'), entry.get('closed')
        if not isinstance(offset_id, str) or not offset_id or ':' in offset_id or offset_id in identifiers:
            raise ValueError('Invalid or duplicate sketch offset ID.')
        identifiers.add(offset_id)
        if not isinstance(sources, list) or not 1 <= len(sources) <= 1000 or any(
            not isinstance(ref, list) or len(ref) != 2 or not isinstance(ref[0], str) or type(ref[1]) is not bool
            for ref in sources
        ) or len({name for name, _ in sources}) != len(sources):
            raise ValueError('Invalid sketch offset sources.')
        if closed is not None and type(closed) is not bool:
            raise ValueError('Invalid sketch offset closure flag.')
        groups = build_offset_chain(plane, edges, sources, number(entry.get('distance')),
                                    expected_closed=closed, join=entry.get('join', 'auto'))
        output.extend((f'offset:{offset_id}:{i}:{j}', edge)
                      for i, group in enumerate(groups) for j, edge in enumerate(group))
    return tuple(output)
