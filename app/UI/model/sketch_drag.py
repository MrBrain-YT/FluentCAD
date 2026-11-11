"""Safe point dragging for named-point parametric sketches."""
from copy import deepcopy
import numpy as np
from UI.model.feature_graph import FeatureGraph
from UI.model.parametric_sketch import circle_radius_name


def _apply_solution(params, before, solved):
    for name, new_position in solved.points.items():
        if np.linalg.norm(new_position - before.points[name]) > 1e-7:
            if any(isinstance(value, str) for value in params['points'][name]):
                raise ValueError(f'Point {name} uses an expression; edit its parameter instead.')
            params['points'][name] = [float(value) for value in new_position]
    for index, entity in enumerate(params.get('entities', [])):
        if entity.get('type') != 'circle':
            continue
        name = circle_radius_name(index)
        if abs(solved.radii[name] - before.radii[name]) > 1e-7:
            if isinstance(entity['radius'], str):
                raise ValueError('A radius uses an expression; edit its parameter instead.')
            entity['radius'] = float(solved.radii[name])


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
    _apply_solution(params, before, solved)
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
    before = graph.results[feature_id].constraint_status
    if before is None:
        raise ValueError('Mouse editing requires named sketch points.')
    probe = FeatureGraph.from_data(graph.to_data())
    temporary = deepcopy(params)
    entity = entities[entity_index]
    temporary.setdefault('constraints', []).append({'kind': 'radius',
        'entities': [[entity['center'], circle_radius_name(entity_index)]], 'value': float(radius)})
    probe.edit(feature_id, parameters=temporary)
    _apply_solution(params, before, probe.results[feature_id].constraint_status)
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


def remove_sketch_constraint(graph, feature_id, index):
    feature = next((item for item in graph.features if item.id == feature_id), None)
    if feature is None or feature.kind != 'planar_sketch':
        raise ValueError('Choose a parametric sketch.')
    params = deepcopy(feature.parameters)
    constraints = params.get('constraints', [])
    if type(index) is not int or not 0 <= index < len(constraints):
        raise ValueError('Choose an existing constraint.')
    constraints.pop(index)
    result = FeatureGraph.from_data(graph.to_data())
    result.edit(feature_id, parameters=params)
    return result


def edit_nurbs_parameters(graph, feature_id, index, degree, weights, knots_or_spans):
    feature = next((item for item in graph.features if item.id == feature_id), None)
    if feature is None or feature.kind != 'planar_sketch':
        raise ValueError('Choose a parametric sketch.')
    params = deepcopy(feature.parameters)
    entities = params.get('entities', [])
    if type(index) is not int or not 0 <= index < len(entities) or entities[index].get('type') not in ('nurbs', 'periodic_nurbs'):
        raise ValueError('Choose a NURBS.')
    entity = entities[index]
    entity.update(degree=degree, weights=deepcopy(weights))
    entity['knots' if entity['type'] == 'nurbs' else 'spans'] = deepcopy(knots_or_spans)
    result = FeatureGraph.from_data(graph.to_data())
    result.edit(feature_id, parameters=params)
    return result
