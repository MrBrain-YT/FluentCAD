"""Solid operations independent of the viewport."""
import build123d as cad
import numpy as np


def build_extrusion(face, normal, distance, operation='new_body', current_part=None,
                    offset=0.0, taper=0.0, symmetric=False, second_distance=None, extent="distance",
                    target=None, reverse=False):
    profiles = tuple(face) if isinstance(face, (tuple, list)) else (face,)
    if not profiles:
        raise ValueError('Select at least one closed profile.')
    normal = np.array(normal, dtype=float)
    length = np.linalg.norm(normal)
    if normal.shape != (3,) or not np.isfinite(normal).all() or length < 1e-12:
        raise ValueError('The sketch plane has no valid normal.')
    if extent not in ('distance', 'all', 'to_object'):
        raise ValueError('Unknown extrusion extent.')
    if extent == 'distance' and abs(distance) < 1e-9:
        raise ValueError('Distance must be non-zero.')
    if not -89.0 < taper < 89.0:
        raise ValueError('Taper must be between -89 and 89 degrees.')
    normal /= length
    origin = profiles[0].center()
    for profile in profiles:
        if (profile.geom_type != cad.GeomType.PLANE
                or abs(cad.Vector(normal).dot(profile.normal_at())) < 1 - 1e-8
                or abs(cad.Vector(normal).dot(profile.center() - origin)) > 1e-6):
            raise ValueError('Select profiles in the same plane.')
    if extent == 'to_object':
        if symmetric or second_distance is not None or taper:
            raise ValueError('To Object requires a one-sided extrusion with no taper.')
        if target is None:
            raise ValueError('Select a target face, body or plane.')
        direction = -normal if reverse else normal
        tools = [_extrude_to_object(profile.moved(cad.Location(tuple(normal * offset))), direction, target)
                 for profile in profiles]
    else:
        tools = [_extrusion_tool(profile, normal, distance, operation, current_part,
                                 offset, taper, symmetric, second_distance, extent) for profile in profiles]
    if any(not tool.is_valid or not tool.solids() or tool.volume <= 1e-9 for tool in tools):
        raise ValueError('The extrusion does not produce a valid solid.')
    tool = tools[0]
    if len(tools) > 1:
        tool = _solid_result(tool.fuse(*tools[1:]))
    return combine_solid(tool, operation, current_part)


def _extrusion_tool(face, normal, distance, operation, current_part, offset, taper, symmetric, second_distance, extent):
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
        return cad.extrude(shifted_face, amount=span + 2 * margin, dir=tuple(normal))
    shifted_face = face.moved(cad.Location(tuple(normal * offset)))
    if second_distance is not None:
        if symmetric or distance <= 0 or second_distance <= 0:
            raise ValueError('Two-sided distances must both be positive.')
        first = cad.extrude(shifted_face, amount=distance, dir=tuple(normal), taper=taper)
        second = cad.extrude(shifted_face, amount=-second_distance, dir=tuple(normal), taper=taper)
        return _solid_result(first.fuse(second))
    return cad.extrude(shifted_face, amount=distance, dir=tuple(normal), both=symmetric, taper=taper)


def _solid_result(result):
    return cad.Part([solid for shape in result for solid in shape.solids()]) if isinstance(result, list) else result


def _extrude_to_object(profile, direction, target):
    target_body, curved_face = target if isinstance(target, tuple) else (target, None)
    selected_face = target if isinstance(target, cad.Face) else None
    if isinstance(target, cad.Plane) or selected_face is not None:
        plane = target if selected_face is None else cad.Plane(selected_face)
        alignment = plane.z_dir.dot(cad.Vector(direction))
        if abs(alignment) < 1e-9:
            raise ValueError('The target plane is parallel to the extrusion direction.')
        if alignment < 0:
            plane = -plane
        bounds = plane.to_local_coords(profile).bounding_box()
        if bounds.max.Z >= -1e-7:
            raise ValueError('The target must be ahead of the entire profile. Try Flip direction.')
        distance = -bounds.min.Z / abs(alignment)
        if abs(alignment) >= 1 - 1e-10:
            tool = cad.extrude(profile, amount=distance, dir=tuple(direction))
        else:
            tool = cad.extrude(profile, amount=distance + max(1e-5, distance * 1e-6),
                               dir=tuple(direction)).split(plane, keep=cad.Keep.BOTTOM)
        if selected_face is not None:
            caps = [face for face in tool.faces() if abs(plane.to_local_coords(face).bounding_box().min.Z) < 1e-6]
            _check_target_coverage(caps, selected_face)
        return tool
    probe = profile.moved(cad.Location(tuple(direction * 1e-6)))
    overlap = _intersection_area(probe, target_body)
    tolerance = max(1e-7, profile.area * 1e-6)
    if overlap <= tolerance:
        try:
            tool = cad.Solid.extrude_until(profile, target_body, tuple(direction), until=cad.Until.NEXT)
        except (ValueError, RuntimeError) as exc:
            raise ValueError('The target body must cover the profile in the extrusion direction.') from exc
    elif abs(overlap - profile.area) <= tolerance:
        frame = cad.Plane(profile.center(), z_dir=tuple(direction))
        length = frame.to_local_coords(target_body).bounding_box().max.Z
        prism = cad.Solid.extrude(profile, tuple(direction * (length + max(1e-5, length * 1e-6))))
        clipped = _solid_result(prism.intersect(target_body))
        connected = [solid for solid in clipped.solids() if _intersection_area(profile, solid) >= profile.area - tolerance] if clipped else []
        if len(connected) != 1:
            raise ValueError('Could not find one continuous extrusion from the profile to the target.')
        tool = connected[0]
    else:
        raise ValueError('The profile partly intersects the target body at the start.')
    _check_target_coverage([profile], tool, 'The extrusion must extend continuously from the entire profile.')
    caps = [face for face in tool.faces() if face.normal_at().dot(cad.Vector(direction)) > 1e-7]
    _check_target_coverage(caps, curved_face if curved_face is not None else target_body)
    return tool


def _check_target_coverage(caps, target, message='The target must cover the entire end of every profile.'):
    for cap in caps:
        area = _intersection_area(cap, target)
        if abs(cap.area - area) > max(1e-7, cap.area * 1e-6):
            break
    else:
        if caps:
            return
    raise ValueError(message)


def _intersection_area(face, target):
    common = face.intersect(target)
    return sum(item.area for item in common.faces()) if common is not None else 0


def combine_solid(tool, operation, current_part):
    if operation == 'new_body':
        result = cad.Part([*current_part.solids(), *tool.solids()]) if current_part is not None else tool
    elif current_part is None:
        raise ValueError('This operation requires an existing body.')
    elif operation == 'join':
        result = _solid_result(current_part.fuse(tool))
        if result is None or len(result.solids()) >= len(current_part.solids()) + len(tool.solids()):
            raise ValueError('The new solid must touch an existing body to join it.')
    elif operation == 'cut':
        result = _solid_result(current_part.cut(tool))
        if result is None or abs(result.volume - current_part.volume) < 1e-9:
            raise ValueError('The new solid does not remove material from the body.')
    elif operation == 'intersect':
        result = _solid_result(current_part.intersect(tool))
    else:
        raise ValueError('Unknown solid operation.')
    if result is None or not result.solids() or not result.is_valid or result.volume <= 1e-9:
        raise ValueError('The operation would produce an empty or invalid solid.')
    return result


def build_revolution(face, origin, direction, angle, operation='new_body', current_part=None):
    direction = np.asarray(direction, dtype=float)
    origin = np.asarray(origin, dtype=float)
    if (origin.shape != (3,) or direction.shape != (3,) or not np.isfinite(origin).all()
            or not np.isfinite(direction).all() or np.linalg.norm(direction) < 1e-12):
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
