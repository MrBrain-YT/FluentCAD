"""Safe point dragging for named-point parametric sketches."""
from copy import deepcopy
import numpy as np
from UI.model.constraints import Constraint
from UI.model.feature_graph import FeatureGraph


def move_sketch_point(graph, feature_id, point_name, position):
    feature = next((item for item in graph.features if item.id == feature_id), None)
    if feature is None or feature.kind != 'planar_sketch' or feature_id not in graph.results:
        raise ValueError('Choose an active parametric sketch.')
    params = deepcopy(feature.parameters)
    points = params.get('points')
    if not isinstance(points, dict) or point_name not in points:
        raise ValueError('The sketch does not contain this named point.')
    position = np.asarray(position, dtype=float)
    if position.shape != (2,) or not np.isfinite(position).all():
        raise ValueError('Point coordinates must be finite.')
    before = graph.results[feature_id].constraint_status
    if before is None:
        raise ValueError('The sketch has no named points.')
    probe = FeatureGraph.from_data(graph.to_data())
    temporary = deepcopy(params)
    temporary.setdefault('constraints', []).append({'kind': 'fixed', 'entities': [point_name], 'value': position.tolist()})
    probe.edit(feature_id, parameters=temporary)
    solved = probe.results[feature_id].constraint_status
    for name, new_position in solved.points.items():
        if np.linalg.norm(new_position - before.points[name]) > 1e-7:
            if any(isinstance(value, str) for value in points[name]):
                raise ValueError(f'Point {name} uses an expression; edit its parameter instead.')
            points[name] = [float(value) for value in new_position]
    result = FeatureGraph.from_data(graph.to_data())
    result.edit(feature_id, parameters=params)
    return result


def set_circle_radius(graph, feature_id, entity_index, radius):
    feature = next((item for item in graph.features if item.id == feature_id), None)
    if feature is None or feature.kind != 'planar_sketch':
        raise ValueError('Choose a parametric sketch.')
    params = deepcopy(feature.parameters)
    entities = params.get('entities', [])
    if type(entity_index) is not int or not 0 <= entity_index < len(entities) or entities[entity_index].get('type') != 'circle':
        raise ValueError('Choose a circle.')
    if isinstance(entities[entity_index].get('radius'), str):
        raise ValueError('The radius uses an expression; edit its parameter instead.')
    if not np.isfinite(radius) or radius <= 1e-6:
        raise ValueError('Radius must be positive.')
    entities[entity_index]['radius'] = float(radius)
    result = FeatureGraph.from_data(graph.to_data())
    result.edit(feature_id, parameters=params)
    return result


def add_sketch_constraint(graph, feature_id, constraint):
    feature = next((item for item in graph.features if item.id == feature_id), None)
    if feature is None or feature.kind != 'planar_sketch':
        raise ValueError('Choose a parametric sketch.')
    params = deepcopy(feature.parameters)
    params.setdefault('constraints', []).append(deepcopy(constraint))
    result = FeatureGraph.from_data(graph.to_data())
    result.edit(feature_id, parameters=params)
    return result
