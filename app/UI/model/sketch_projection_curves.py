"""Identify projectable sketch edges by their source entities and point names."""
import json

import build123d as cad

from UI.model.profiles import indexed_sketch_edges


def sketch_curve_name(entity, segment):
    kind = entity['type']
    refs = entity.get('points', [entity.get('center')])
    identity = [kind, refs[segment:segment + 2]] if kind == 'line' else [kind, refs, segment]
    return json.dumps(identity, separators=(',', ':'), sort_keys=True)


def named_sketch_curves(entities, tools, normal):
    curves = []
    for entity, tool in zip(entities, tools):
        curves.extend((sketch_curve_name(entity, i), edge) for i, edge in indexed_sketch_edges(tool, normal))
    return tuple(curves)


def projectable_sketch_curves(sketch):
    if isinstance(sketch, cad.Face):
        # The four edges of a rectangle_sketch are created in a fixed order.
        return tuple((f'rectangle-{i}', edge) for i, edge in enumerate(sketch.edges()))
    if not hasattr(sketch, 'projectable_curves'):
        return ()
    return (*sketch.projectable_curves,
            *((f'derived:{name}', edge)
              for name, edge in (*sketch.linked_edges, *sketch.detached_edges, *sketch.offset_edges,
                                  *sketch.corner_edges, *sketch.mirror_edges, *sketch.pattern_edges)))
