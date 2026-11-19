"""Uniform scaling of native sketch geometry about a picked sketch point."""
import numpy as np

from UI.model.parametric_sketch import circle_radius_name
from UI.model.profiles import indexed_sketch_edges
from UI.model.sketch_drag import solve_sketch_transform
from UI.model.sketch_move import sketch_selection
from UI.model.sketch_projection_curves import sketch_curve_name


LENGTH_CONSTRAINTS = frozenset({
    'distance', 'horizontal_distance', 'vertical_distance',
    'radius', 'diameter', 'arc_radius', 'arc_diameter',
})


def _scale_value(value, factor, origin=0.):
    if isinstance(value, str):
        return f'({origin}) + ({factor}) * (({value}) - ({origin}))' if origin else f'({factor}) * ({value})'
    return float(origin + factor * (value - origin))


def _selected_reference(ref, names, indices):
    if isinstance(ref, list):
        return bool(ref) and all(_selected_reference(item, names, indices) for item in ref)
    if isinstance(ref, str) and ref.startswith('@circle:'):
        return any(ref == circle_radius_name(index) for index in indices)
    return isinstance(ref, str) and ref in names


def scale_sketch_selection(graph, feature_id, selection, center, factor):
    center = np.asarray(center, dtype=float)
    if center.shape != (2,) or not np.isfinite(center).all() or not np.isfinite(factor) or factor <= 0:
        raise ValueError('Enter a finite positive scale factor and choose a center.')
    feature = next(item for item in graph.features if item.id == feature_id)
    params = feature.parameters
    indices, names = sketch_selection(params, selection)
    sketch = graph.results[feature_id]
    state = sketch.constraint_status
    if not names or state is None or any(name not in state.points for name in names):
        raise ValueError('Select geometry with named sketch points.')

    def transform(point):
        return (center + factor * (np.asarray(point) - center)).tolist()

    targets = {name: transform(state.points[name]) for name in names}
    for name, target in targets.items():
        params['points'][name] = [
            _scale_value(value, factor, float(origin)) if isinstance(value, str) else coordinate
            for value, origin, coordinate in zip(params['points'][name], center, target)
        ]
    radii, curves = {}, set()
    for index in indices:
        entity = params['entities'][index]
        for field in ('radius', 'radius_x', 'radius_y'):
            if field in entity:
                entity[field] = _scale_value(entity[field], factor)
        if entity['type'] == 'circle':
            radius = state.radii[circle_radius_name(index)] * factor
            radii[index] = radius
            if not isinstance(entity['radius'], str):
                entity['radius'] = float(radius)
        if params.get('corners'):
            curves.update(sketch_curve_name(entity, segment)
                          for segment, _ in indexed_sketch_edges(sketch.tools[index], tuple(sketch.plane.z_dir)))
    for corner in params.get('corners', []):
        if all(source in curves for source in corner['sources']):
            corner['size'] = _scale_value(corner['size'], factor)
    for constraint in params.get('constraints', []):
        if not _selected_reference(constraint['entities'], names, indices):
            continue
        if constraint['kind'] in LENGTH_CONSTRAINTS:
            constraint['value'] = _scale_value(constraint['value'], factor)
        if 'placement' in constraint:
            constraint['placement'] = transform(constraint['placement'])
    return solve_sketch_transform(graph, feature_id, params, targets, radii, reseeded=True)
