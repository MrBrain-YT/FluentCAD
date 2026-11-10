"""Conservative geometry references for feature results.

These signatures reject topology changes and ambiguous matches. They are not a
replacement for operation-generated persistent naming across arbitrary edits.
"""
import numpy as np


def _entities(shape, kind):
    if kind == 'face': return list(shape.faces())
    if kind == 'edge': return list(shape.edges())
    raise ValueError('Reference must identify a face or edge.')


def _frame(shape):
    box = shape.bounding_box()
    minimum = np.array([box.min.X, box.min.Y, box.min.Z], dtype=float)
    maximum = np.array([box.max.X, box.max.Y, box.max.Z], dtype=float)
    span = maximum - minimum
    span[span < 1e-9] = 1.0
    return minimum, span


def _normalised(shape, point):
    minimum, span = _frame(shape)
    return (np.asarray(tuple(point), dtype=float) - minimum) / span


def _topology(shape):
    return {'solids': len(shape.solids()), 'faces': len(shape.faces()), 'edges': len(shape.edges())}


def _descriptor(shape, item, kind):
    result = {'geometry': item.geom_type.name, 'center': _normalised(shape, item.center()).tolist()}
    if kind == 'edge':
        vertices = [tuple(_normalised(shape, vertex.center())) for vertex in item.vertices()]
        result['vertices'] = [list(point) for point in sorted(vertices)]
    elif item.geom_type.name == 'PLANE':
        result['normal'] = list(tuple(item.normal_at()))
        result['boundary_edges'] = len(item.edges())
    return result


def make_reference(feature_id, shape, kind, index):
    if not isinstance(feature_id, str) or not feature_id:
        raise ValueError('A source feature ID is required.')
    candidates = _entities(shape, kind)
    if type(index) is not int or not 0 <= index < len(candidates):
        raise ValueError('Invalid geometry index.')
    reference = {'version': 2, 'feature_id': feature_id, 'kind': kind,
                 'topology': _topology(shape), **_descriptor(shape, candidates[index], kind)}
    if resolve_reference(reference, shape, feature_id) != index:
        raise ValueError('Geometry cannot be referenced unambiguously.')
    return reference


def _vector(value, label):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Invalid {label} in geometry reference.') from exc
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f'Invalid {label} in geometry reference.')
    return result


def resolve_reference(reference, shape, feature_id, tolerance=0.05):
    if not isinstance(reference, dict) or reference.get('feature_id') != feature_id:
        raise ValueError('Geometry reference points to a different feature.')
    kind = reference.get('kind')
    entities = _entities(shape, kind)
    version = reference.get('version', 1)
    if version not in (1, 2):
        raise ValueError('Unsupported geometry reference version.')
    if version == 2 and reference.get('topology') != _topology(shape):
        raise ValueError('Source topology changed; reselect the referenced geometry.')
    center = _vector(reference.get('center'), 'center')
    geometry = reference.get('geometry')
    matches = []
    for index, item in enumerate(entities):
        if item.geom_type.name != geometry:
            continue
        descriptor = _descriptor(shape, item, kind)
        if np.linalg.norm(_vector(descriptor['center'], 'center') - center) > tolerance:
            continue
        if version == 2 and kind == 'edge':
            expected, actual = reference.get('vertices'), descriptor['vertices']
            if not isinstance(expected, list) or len(expected) != len(actual):
                continue
            expected = sorted(tuple(_vector(point, 'vertex')) for point in expected)
            if any(np.linalg.norm(np.asarray(a) - np.asarray(b)) > tolerance for a, b in zip(expected, actual)):
                continue
        if version == 2 and kind == 'face' and geometry == 'PLANE':
            if reference.get('boundary_edges') != descriptor['boundary_edges']:
                continue
            if np.linalg.norm(_vector(reference.get('normal'), 'normal') - _vector(descriptor['normal'], 'normal')) > 1e-4:
                continue
        matches.append(index)
    if not matches:
        raise ValueError('Referenced geometry could not be matched after rebuild.')
    if len(matches) != 1:
        raise ValueError('Geometry reference is ambiguous after rebuild.')
    return matches[0]
