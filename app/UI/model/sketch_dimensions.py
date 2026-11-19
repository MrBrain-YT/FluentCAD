"""Driving dimensions for named sketch geometry."""
from dataclasses import replace
import math

import numpy as np

from UI.model.constraints import arc_center_radius
from UI.model.expressions import evaluate_expression
from UI.model.feature_graph import FeatureGraph, resolve_parameters


DIMENSION_NAMES = {
    'distance': 'Distance', 'horizontal_distance': 'Horizontal distance',
    'vertical_distance': 'Vertical distance', 'angle': 'Angle',
    'radius': 'Radius', 'diameter': 'Diameter',
    'arc_radius': 'Radius', 'arc_diameter': 'Diameter',
}
LINEAR = ('distance', 'horizontal_distance', 'vertical_distance')
SCALE = {'radius': 1, 'arc_radius': 1, 'diameter': 2, 'arc_diameter': 2}


def dimension_from_selection(mode, selected, state):
    if not selected:
        return None
    geometry, refs = selected[0]
    if mode == 'auto':
        mode = {'arc': 'radius', 'circle': 'diameter'}.get(geometry, 'distance')
    if mode in LINEAR:
        if geometry == 'point' and len(selected) == 2 and selected[1][0] == 'point':
            entities = [refs[0], selected[1][1][0]]
        elif geometry == 'line' and len(selected) == 1:
            entities = list(refs)
        else:
            return None
        if entities[0] == entities[1]:
            return None
        a, b = (state.points[name] for name in entities)
        delta = b - a
        value = float(np.linalg.norm(delta) if mode == 'distance' else delta[int(mode == 'vertical_distance')])
        if mode == 'distance' and value < 1e-7:
            return None
    elif mode == 'angle' and len(selected) == 2 and all(kind == 'line' for kind, _ in selected):
        if selected[0] == selected[1]:
            return None
        entities = [list(refs) for _, refs in selected]
        u, v = (_line_direction(state, line) for line in entities)
        value = math.degrees(math.atan2(_cross(u, v), np.dot(u, v)))
    elif mode in ('radius', 'diameter') and geometry in ('arc', 'circle'):
        mode = 'arc_' + mode if geometry == 'arc' else mode
        entities = [list(refs)]
        _, radius = _radial_geometry(mode, refs, state)
        value = radius * SCALE[mode]
    else:
        return None
    return {'kind': mode, 'entities': entities, 'value': value}


def _cross(u, v):
    return u[0] * v[1] - u[1] * v[0]


def _line_direction(state, refs):
    direction = state.points[refs[1]] - state.points[refs[0]]
    length = np.linalg.norm(direction)
    if length < 1e-9:
        raise ValueError('Select a nonzero line segment.')
    return direction / length


def _radial_geometry(kind, refs, state):
    if kind.startswith('arc_'):
        return arc_center_radius(*(state.points[name] for name in refs))
    return state.points[refs[0]], state.radii[refs[1]]


def dimension_annotation(constraint, state, variables):
    """Return label anchor, leader polylines and display value in sketch coordinates."""
    kind, refs = constraint['kind'], constraint['entities']
    placement = constraint.get('placement')
    anchor = np.asarray(placement, dtype=float) if placement is not None else None
    if anchor is not None and (anchor.shape != (2,) or not np.isfinite(anchor).all()):
        raise ValueError('Dimension placement must contain two finite coordinates.')
    if kind in LINEAR:
        a, b = (state.points[name] for name in refs)
        direction = _line_direction(state, refs) if kind == 'distance' else (
            np.array([1., 0.]) if kind == 'horizontal_distance' else np.array([0., 1.]))
        perpendicular = np.array([-direction[1], direction[0]])
        mid = (a + b) / 2
        if anchor is None:
            anchor = mid + perpendicular * max(.5, np.linalg.norm(b - a) * .25)
        p, q = (point + perpendicular * np.dot(anchor - point, perpendicular) for point in (a, b))
        anchor = (p + q) / 2
        paths = [(a, p), (p, q), (b, q)]
        prefix, suffix = '', ''
    elif kind in SCALE:
        center, radius = _radial_geometry(kind, refs[0], state)
        if anchor is None:
            anchor = center + radius * np.array([1.35, .75])
        delta = anchor - center
        length = np.linalg.norm(delta)
        direction = delta / length if length > 1e-9 else np.array([1., 0.])
        start = center - radius * direction if SCALE[kind] == 2 else center
        paths = [(start, center + radius * direction, anchor)]
        prefix, suffix = ('⌀' if SCALE[kind] == 2 else 'R'), ''
    else:
        first, second = refs
        a, b = state.points[first[0]], state.points[second[0]]
        u, v = _line_direction(state, first), _line_direction(state, second)
        cross = _cross(u, v)
        turn = math.atan2(cross, np.dot(u, v))
        pivot = a + (_cross(b - a, v) / cross) * u if abs(cross) > 1e-7 else (a + b) / 2
        start = math.atan2(u[1], u[0])
        size = min(np.linalg.norm(state.points[first[1]] - a), np.linalg.norm(state.points[second[1]] - b))
        radius = np.linalg.norm(anchor - pivot) if anchor is not None else size * .65
        radius = max(radius, size * .05)
        angles = start + np.linspace(0, turn, 33)
        arc = pivot + radius * np.column_stack((np.cos(angles), np.sin(angles)))
        anchor = arc[len(arc) // 2]
        paths = [(pivot, arc[0]), arc, (pivot, arc[-1])]
        prefix, suffix = '', '°'
    value = constraint['value']
    if isinstance(value, str):
        value = evaluate_expression(value, variables)
    return anchor, paths, f'{prefix}{value:.7g}{suffix}'


def update_sketch_dimension(graph, feature_id, dimension, text, index=None):
    """Return a rebuilt candidate, leaving the current sketch and solid untouched."""
    text = text.strip()
    value = evaluate_expression(text, resolve_parameters(graph.parameters))
    kind = dimension['kind']
    if kind in SCALE or kind == 'distance':
        if value <= 1e-7:
            raise ValueError('Enter a positive dimension.')
    elif kind == 'angle' and not -180 <= value <= 180:
        raise ValueError('Enter an angle between −180° and 180°.')
    try:
        stored = float(text)
    except ValueError:
        stored = text
    features = graph.features
    feature = next(item for item in features if item.id == feature_id)
    constraints = feature.parameters.setdefault('constraints', [])
    entry = {**dimension, 'value': stored}
    if index is None:
        constraints.append(entry)
    else:
        constraints[index] = entry
    return FeatureGraph((replace(item, parameters=feature.parameters) if item.id == feature_id else item
                         for item in features), graph.parameters)
