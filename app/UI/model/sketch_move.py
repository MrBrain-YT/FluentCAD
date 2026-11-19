"""Move or duplicate native sketch geometry and its internal constraints."""
from copy import deepcopy
from uuid import uuid4

import numpy as np

from UI.model.feature_graph import FeatureGraph
from UI.model.parametric_sketch import circle_radius_name
from UI.model.profiles import indexed_sketch_edges
from UI.model.sketch_drag import move_sketch_points, rotate_sketch_ellipse, sketch_transform
from UI.model.sketch_projection_curves import sketch_curve_name


def sketch_selection(params, selection):
    entities = params['entities']
    indices = [ref for kind, ref in selection if kind == 'entity']
    names = dict.fromkeys(ref for kind, ref in selection if kind == 'point')
    for index in indices:
        entity = entities[index]
        for name in entity.get('points', [entity['center']] if 'center' in entity else []):
            if not isinstance(name, str):
                raise ValueError('Select geometry with named sketch points.')
            names[name] = None
        paths = entity.get('curves', [entity['curve']] if 'curve' in entity else [])
        if any(path not in indices for path in paths):
            raise ValueError('Select the slot and its path together.')
    return indices, names


def transform_sketch_selection(graph, feature_id, selection, translation, angle, copy):
    features = graph.features
    feature = next(item for item in features if item.id == feature_id)
    params = feature.parameters
    entities = params['entities']
    indices, names = sketch_selection(params, selection)
    if not copy:
        return move_sketch_points(graph, feature_id, names, translation, angle, entities=indices)

    sketch = graph.results[feature_id]
    state = sketch.constraint_status
    if not names or state is None or any(name not in state.points for name in names):
        raise ValueError('Select entities or points with named sketch coordinates.')
    transform = sketch_transform(translation, angle)
    if all(abs(value) < 1e-9 for value in (*translation, angle % 360)):
        raise ValueError('Enter a displacement or rotation for the copy.')
    prefix = uuid4().hex
    names_map = {name: f'copy:{prefix}:{i}' for i, name in enumerate(names)}
    index_map = {index: len(entities) + i for i, index in enumerate(indices)}
    refs_map = {**names_map, **{circle_radius_name(i): circle_radius_name(index_map[i])
                              for i in indices if entities[i]['type'] == 'circle'}}
    targets = {new: transform(state.points[old]) for old, new in names_map.items()}
    params['points'].update(targets)
    curve_map = {}
    for index in indices:
        entity = deepcopy(entities[index])
        if 'points' in entity:
            entity['points'] = [names_map[name] for name in entity['points']]
        if 'center' in entity:
            entity['center'] = names_map[entity['center']]
        if 'curve' in entity:
            entity['curve'] = index_map[entity['curve']]
        if 'curves' in entity:
            entity['curves'] = [index_map[path] for path in entity['curves']]
        if entity['type'] == 'circle' and not isinstance(entity['radius'], str):
            entity['radius'] = float(state.radii[circle_radius_name(index)])
        rotate_sketch_ellipse(entity, angle)
        if params.get('corners'):
            for segment, _ in indexed_sketch_edges(sketch.tools[index], tuple(sketch.plane.z_dir)):
                curve_map[sketch_curve_name(entities[index], segment)] = sketch_curve_name(entity, segment)
        entities.append(entity)
    for corner in tuple(params.get('corners', [])):
        sources = corner['sources']
        if any(source in curve_map for source in sources):
            if not all(source in curve_map for source in sources):
                raise ValueError('Select both source curves to copy a fillet or chamfer.')
            params['corners'].append({
                **deepcopy(corner), 'id': uuid4().hex, 'sources': [curve_map[source] for source in sources],
            })

    def remap(ref):
        if isinstance(ref, list):
            if len(ref) == 2 and isinstance(ref[0], str) and ref[0].startswith('@') and type(ref[1]) is int:
                return [ref[0], index_map[ref[1]]]
            return [remap(item) for item in ref]
        if isinstance(ref, str):
            if ref in refs_map:
                return refs_map[ref]
            if ref not in state.points and (ref in ('@fit', 'left', 'right', 'inner', 'outer', 'start_cap', 'end_cap')
                                            or ref.startswith('segment:')):
                return ref
            raise KeyError(ref)
        return index_map[ref] if type(ref) is int else ref

    constraints = params.setdefault('constraints', [])
    for entry in tuple(constraints):
        try:
            refs = remap(entry['entities'])
        except KeyError:
            continue  # External references remain attached to the original geometry.
        cloned = {**deepcopy(entry), 'entities': refs}
        if entry['kind'] == 'fixed':
            cloned['value'] = targets[refs[0]]
        if 'placement' in cloned:
            cloned['placement'] = transform(cloned['placement'])
        constraints.append(cloned)
    candidate = FeatureGraph(features, graph.parameters)
    final = candidate.results[feature_id].constraint_status
    if (any(np.linalg.norm(final.points[name] - point) > 1e-6 for name, point in state.points.items())
            or any(abs(final.radii[name] - radius) > 1e-6 for name, radius in state.radii.items())):
        raise ValueError('The copy would change the original sketch geometry.')
    if any(np.linalg.norm(final.points[name] - target) > 1e-6 for name, target in targets.items()):
        raise ValueError('The copied constraints prevent the requested transform.')
    if any(abs(final.radii[circle_radius_name(index_map[index])] - state.radii[circle_radius_name(index)]) > 1e-6
           for index in indices if entities[index]['type'] == 'circle'):
        raise ValueError('The copied constraints prevent keeping the circle radius.')
    return candidate
