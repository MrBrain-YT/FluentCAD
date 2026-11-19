"""Map viewport selections to the parametric sketch solver's references."""
from dataclasses import dataclass


SCENE_CONSTRAINTS = {
    'coincident': 'Coincident', 'horizontal': 'Horizontal', 'vertical': 'Vertical',
    'parallel': 'Parallel', 'perpendicular': 'Perpendicular', 'collinear': 'Collinear',
    'equal': 'Equal', 'concentric': 'Concentric', 'midpoint': 'Midpoint', 'symmetric': 'Symmetry',
    'fixed': 'Fix / Unfix point', 'tangent': 'Tangent', 'tangent_internal': 'Internal circle tangency',
    'curvature': 'Curvature',
}

_POINT = ('point',)
_TWO_POINTS = ('point', 'point')
_POINT_LINE = ('point', 'line')
_LINE_PAIR = ('line', 'line')
_ROUND_PAIRS = (('arc', 'arc'), ('arc', 'circle'), ('circle', 'arc'), ('circle', 'circle'))
_SMOOTH_PAIRS = _ROUND_PAIRS[:-1] + (('spline', 'spline'), ('spline', 'arc'), ('arc', 'spline'),
                                   ('spline', 'circle'), ('circle', 'spline'))
SELECTIONS = {
    'fixed': (_POINT,),
    'coincident': (_TWO_POINTS, _POINT_LINE, ('line', 'point'), ('point', 'circle'), ('circle', 'point'),
                   ('point', 'arc'), ('arc', 'point')),
    'horizontal': (('line',), _TWO_POINTS),
    'vertical': (('line',), _TWO_POINTS),
    'parallel': (_LINE_PAIR,), 'perpendicular': (_LINE_PAIR,), 'collinear': (_LINE_PAIR,),
    'equal': (_LINE_PAIR, *_ROUND_PAIRS),
    'concentric': _ROUND_PAIRS,
    'midpoint': (_POINT_LINE, ('line', 'point')),
    'symmetric': (('point', 'point', 'line'),),
    'tangent': (*_SMOOTH_PAIRS, _ROUND_PAIRS[-1], ('line', 'circle'), ('circle', 'line'),
                ('line', 'arc'), ('arc', 'line')),
    'tangent_internal': (_ROUND_PAIRS[-1],),
    'curvature': _SMOOTH_PAIRS,
}


@dataclass(frozen=True)
class ConstraintTarget:
    kind: str
    refs: tuple
    endpoint: str | None = None
    angle: float | None = None


def constraint_from_targets(mode, targets, points):
    kinds = tuple(target.kind for target in targets)
    if kinds not in SELECTIONS[mode]:
        return None
    kind = mode
    entities = [target.refs[0] if target.kind == 'point' else list(target.refs) for target in targets]
    entry = {}
    if mode == 'fixed':
        entry['value'] = points[entities[0]].tolist()
    elif mode in ('horizontal', 'vertical') and kinds == _TWO_POINTS:
        entities = [entities]
    elif mode in ('coincident', 'midpoint', 'symmetric'):
        if kinds[0] != 'point':
            kinds = kinds[::-1]
            entities.reverse()
        if kinds != _TWO_POINTS:
            point_refs, curve = entities[:-1], entities[-1]
            names = curve[:1] if kinds[-1] == 'circle' else curve
            if any(point in names for point in point_refs):
                raise ValueError('Choose a point distinct from the curve handles.')
            if mode == 'coincident':
                kind = 'point_on_' + kinds[-1]
    elif mode in ('equal', 'concentric'):
        if kinds == _LINE_PAIR:
            kind = 'equal_length'
        else:
            kind = 'equal_radius' if mode == 'equal' else 'concentric'
            if kinds[1] == 'arc' and kinds[0] == 'circle':
                kinds = kinds[::-1]
                entities.reverse()
            if kinds[0] == 'arc':
                kind += '_arc_' + kinds[1]
    elif mode == 'tangent_internal':
        kind = 'tangent_circles_internal'
    elif mode in ('tangent', 'curvature'):
        # Solver references put splines first, then lines, then arcs, then circles.
        order = ('spline', 'line', 'arc', 'circle')
        ordered = sorted(targets, key=lambda target: order.index(target.kind))
        a, b = ordered
        kind = f'{mode}_{a.kind}_{b.kind}' if a.kind != 'circle' else 'tangent_circles_external'
        entities = []
        for target in ordered:
            entities.append(list(target.refs))
            if target.endpoint is not None:
                entities.append(target.endpoint)
        if a.kind == 'spline' and b.kind == 'circle':
            entry['value'] = b.angle
    return dict(entry, kind=kind, entities=entities)
