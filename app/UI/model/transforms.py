"""Non-parametric solid transforms; each function leaves its input unchanged."""
import numpy as np
import build123d as cad

MAX_PATTERN_BODIES = 100


def _vector(value, label):
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f'{label} must contain three finite coordinates.')
    return vector


def _require_solid(shape):
    if shape is None or not shape.solids() or not shape.is_valid:
        raise ValueError('Create a valid solid body first.')


def _copies(shapes):
    solids = [solid for shape in shapes for solid in shape.solids()]
    if not solids or len(solids) > MAX_PATTERN_BODIES:
        raise ValueError(f'The result must contain 1 to {MAX_PATTERN_BODIES} bodies.')
    result = cad.Part(solids)
    if not result.is_valid:
        raise ValueError('The resulting bodies are invalid.')
    return result


def move_copy(shape, translation=(0, 0, 0), rotation=(0, 0, 0), pivot=(0, 0, 0), copy=False):
    _require_solid(shape)
    translation = _vector(translation, 'Translation')
    rotation = _vector(rotation, 'Rotation')
    pivot = _vector(pivot, 'Pivot')
    transformed = (shape.moved(cad.Pos(*(-pivot))).moved(cad.Rot(*rotation))
                   .moved(cad.Pos(*(pivot + translation))))
    return _copies([shape, transformed]) if copy else transformed


def scale_model(shape, factor, center=(0, 0, 0), copy=False):
    _require_solid(shape)
    center = _vector(center, 'Scale center')
    if not np.isfinite(factor) or factor <= 1e-6:
        raise ValueError('Scale factor must be positive and finite.')
    transformed = shape.moved(cad.Pos(*(-center))).scale(float(factor)).moved(cad.Pos(*center))
    return _copies([shape, transformed]) if copy else transformed


def mirror_model(shape, base, offset=0, copy=True):
    _require_solid(shape)
    normals = {'XY': (0, 0, 1), 'XZ': (0, 1, 0), 'YZ': (1, 0, 0)}
    if base not in normals or not np.isfinite(offset):
        raise ValueError('Choose a valid mirror plane and finite offset.')
    normal = np.asarray(normals[base], dtype=float)
    plane = cad.Plane(origin=tuple(normal * offset), z_dir=tuple(normal))
    transformed = shape.mirror(plane)
    return _copies([shape, transformed]) if copy else transformed


def rectangular_pattern(shape, count_x, count_y, distance_x, distance_y):
    _require_solid(shape)
    if type(count_x) is not int or type(count_y) is not int or min(count_x, count_y) < 1:
        raise ValueError('Pattern counts must be positive whole numbers.')
    if count_x * count_y * len(shape.solids()) > MAX_PATTERN_BODIES:
        raise ValueError(f'Pattern is limited to {MAX_PATTERN_BODIES} bodies.')
    if not np.isfinite([distance_x, distance_y]).all():
        raise ValueError('Pattern distances must be finite.')
    if (count_x > 1 and abs(distance_x) < 1e-9) or (count_y > 1 and abs(distance_y) < 1e-9):
        raise ValueError('Pattern spacing must be non-zero for repeated directions.')
    return _copies([shape.moved(cad.Pos(x * distance_x, y * distance_y, 0))
                    for x in range(count_x) for y in range(count_y)])


def circular_pattern(shape, count, angle, axis='Z', center=(0, 0, 0)):
    _require_solid(shape)
    center = _vector(center, 'Pattern center')
    directions = {'X': (1, 0, 0), 'Y': (0, 1, 0), 'Z': (0, 0, 1)}
    if axis not in directions or type(count) is not int or count < 2:
        raise ValueError('Choose a valid axis and at least two instances.')
    if count * len(shape.solids()) > MAX_PATTERN_BODIES:
        raise ValueError(f'Pattern is limited to {MAX_PATTERN_BODIES} bodies.')
    if not np.isfinite(angle) or abs(angle) < 1e-9 or abs(angle) > 360:
        raise ValueError('Pattern angle must be non-zero and no more than 360 degrees.')
    # A full turn distributes instances without repeating the first one.
    denominator = count if np.isclose(abs(angle), 360) else count - 1
    pattern_axis = cad.Axis(tuple(center), directions[axis])
    return _copies([shape.rotate(pattern_axis, float(angle) * index / denominator)
                    for index in range(count)])


def body_index_for_face(shape, face_index):
    _require_solid(shape)
    faces = list(shape.faces())
    if type(face_index) is not int or not 0 <= face_index < len(faces):
        return None
    face = faces[face_index]
    matches = [index for index, body in enumerate(shape.solids())
               if any(candidate.is_same(face) for candidate in body.faces())]
    # Shared topology is ambiguous; never guess which body to modify.
    return matches[0] if len(matches) == 1 else None


def apply_to_body(shape, body_index, operation):
    _require_solid(shape)
    solids = list(shape.solids())
    if type(body_index) is not int or not 0 <= body_index < len(solids):
        raise ValueError('Select a body by clicking one of its faces.')
    changed = operation(solids[body_index])
    if changed is None or not changed.solids() or not changed.is_valid:
        raise ValueError('The operation did not produce a valid body.')
    result_solids = solids[:body_index] + list(changed.solids()) + solids[body_index + 1:]
    if len(result_solids) > MAX_PATTERN_BODIES:
        raise ValueError(f'The result is limited to {MAX_PATTERN_BODIES} bodies.')
    result = cad.Part(result_solids)
    if not result.is_valid:
        raise ValueError('The resulting model is invalid.')
    return result
