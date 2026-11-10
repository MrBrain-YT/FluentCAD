"""Solid operations independent of the viewport."""
import build123d as cad
import numpy as np


def build_extrusion(face, normal, distance, operation='new_body', current_part=None,
                    offset=0.0, taper=0.0, symmetric=False, second_distance=None, extent="distance"):
    normal = np.asarray(normal, dtype=float)
    length = np.linalg.norm(normal)
    if length < 1e-12:
        raise ValueError('The sketch plane has no valid normal.')
    if extent not in ('distance', 'all'):
        raise ValueError('Unknown extrusion extent.')
    if extent == 'distance' and abs(distance) < 1e-9:
        raise ValueError('Distance must be non-zero.')
    if not -89.0 < taper < 89.0:
        raise ValueError('Taper must be between -89 and 89 degrees.')
    normal /= length
    if extent == 'all':
        if current_part is None or operation not in ('cut', 'intersect'):
            raise ValueError('Through all requires an existing body and Cut or Intersect.')
        bounds = current_part.bounding_box()
        origin = np.asarray(tuple(face.center()), dtype=float)
        corners = np.array([[x, y, z] for x in (bounds.min.X, bounds.max.X)
                            for y in (bounds.min.Y, bounds.max.Y)
                            for z in (bounds.min.Z, bounds.max.Z)])
        projected = (corners - origin) @ normal
        span = float(projected.max() - projected.min())
        margin = max(span * 1e-5, 1e-5)
        start = float(projected.min()) - margin
        shifted_face = face.moved(cad.Location(tuple(normal * start)))
        tool = cad.extrude(shifted_face, amount=span + 2 * margin, dir=tuple(normal))
    elif second_distance is not None:
        shifted_face = face.moved(cad.Location(tuple(normal * offset)))
        if symmetric or distance <= 0 or second_distance <= 0:
            raise ValueError('Two-sided distances must both be positive.')
        first = cad.extrude(shifted_face, amount=distance, dir=tuple(normal), taper=taper)
        second = cad.extrude(shifted_face, amount=-second_distance, dir=tuple(normal), taper=taper)
        tool = first.fuse(second)
    else:
        shifted_face = face.moved(cad.Location(tuple(normal * offset)))
        tool = cad.extrude(shifted_face, amount=distance, dir=tuple(normal),
                           both=symmetric, taper=taper)
    if not tool.is_valid or not tool.solids() or tool.volume <= 1e-9:
        raise ValueError('The extrusion does not produce a valid solid.')
    return combine_solid(tool, operation, current_part)


def combine_solid(tool, operation, current_part):
    if operation == 'new_body':
        result = cad.Part([*current_part.solids(), *tool.solids()]) if current_part is not None else tool
    elif current_part is None:
        raise ValueError('This operation requires an existing body.')
    elif operation == 'join':
        result = current_part.fuse(tool)
        if result is None or len(result.solids()) >= len(current_part.solids()) + len(tool.solids()):
            raise ValueError('The new solid must touch an existing body to join it.')
    elif operation == 'cut':
        result = current_part.cut(tool)
        if result is None or abs(result.volume - current_part.volume) < 1e-9:
            raise ValueError('The new solid does not remove material from the body.')
    elif operation == 'intersect':
        result = current_part.intersect(tool)
    else:
        raise ValueError('Unknown solid operation.')
    if result is None or not result.solids() or not result.is_valid or result.volume <= 1e-9:
        raise ValueError('The operation would produce an empty or invalid solid.')
    return result


def build_revolution(face, origin, direction, angle, operation='new_body', current_part=None):
    direction = np.asarray(direction, dtype=float)
    if np.linalg.norm(direction) < 1e-12:
        raise ValueError('Choose a valid axis.')
    if not 0 < abs(angle) <= 360:
        raise ValueError('The angle must be greater than zero and at most 360 degrees.')
    axis = cad.Axis(tuple(origin), tuple(direction / np.linalg.norm(direction)))
    tool = cad.revolve(face, axis=axis, revolution_arc=angle)
    if not tool.is_valid or not tool.solids() or tool.volume <= 1e-9:
        raise ValueError('The selected profile and axis do not produce a valid solid.')
    return combine_solid(tool, operation, current_part)


def build_primitive(kind, dimensions, position=(0, 0, 0), operation='new_body', current_part=None):
    values = np.asarray(dimensions, dtype=float)
    location = np.asarray(position, dtype=float)
    if location.shape != (3,) or not np.isfinite(location).all():
        raise ValueError('Position must have three finite coordinates.')
    expected = {'box': 3, 'cylinder': 2, 'sphere': 1, 'cone': 3, 'torus': 2}
    if kind not in expected or values.shape != (expected[kind],):
        raise ValueError('Invalid primitive or dimensions.')
    if not np.isfinite(values).all():
        raise ValueError('All dimensions must be finite.')
    if kind == 'cone':
        if np.any(values[:2] < 0) or max(values[:2]) <= 1e-6 or values[2] <= 1e-6:
            raise ValueError('Cone radii must be nonnegative, at least one radius and the height must be positive.')
    elif np.any(values <= 1e-6):
        raise ValueError('All dimensions must be positive and finite.')
    if kind == 'torus' and values[0] <= values[1]:
        raise ValueError('The major radius must exceed the tube radius.')
    if kind == 'box':
        tool = cad.Box(*values)
    elif kind == 'cylinder':
        tool = cad.Cylinder(values[0], values[1])
    elif kind == 'sphere':
        tool = cad.Sphere(values[0])
    elif kind == 'cone':
        tool = cad.Cone(*values)
    else:
        tool = cad.Torus(*values)
    tool = tool.moved(cad.Location(tuple(float(value) for value in location)))
    return combine_solid(tool, operation, current_part)


def import_step_body(path, current_part=None):
    try:
        shape = cad.import_step(path)
    except Exception as exc:
        raise ValueError('Unable to read the STEP file.') from exc
    if shape is None or not shape.solids() or not shape.is_valid:
        raise ValueError('The STEP file does not contain valid solid geometry.')
    return combine_solid(shape, 'new_body', current_part)
