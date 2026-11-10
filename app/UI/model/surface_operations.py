"""Basic solid modification operations with explicit selection scope."""
import math
import numpy as np
import build123d as cad
from UI.model.solid_operations import build_extrusion


def _body(shape):
    if shape is None or not shape.solids() or not shape.is_valid:
        raise ValueError('Create a valid solid first.')
    return cad.Part(list(shape.solids()))


def _face(shape, face_index):
    faces = list(shape.faces())
    if face_index is None or not 0 <= face_index < len(faces):
        raise ValueError('Select a face first.')
    return faces[face_index]


def _positive(value, label):
    if not math.isfinite(value) or value <= 1e-6:
        raise ValueError(f'{label} must be positive and finite.')


def _validate(result):
    if result is None or not result.is_valid or not result.solids() or result.volume <= 1e-9:
        raise ValueError('The operation did not produce a valid solid.')
    return result


def fillet_all(shape, radius):
    _positive(radius, 'Radius')
    body = _body(shape)
    return _validate(body.fillet(radius, body.edges()))


def chamfer_all(shape, distance):
    _positive(distance, 'Distance')
    body = _body(shape)
    return _validate(body.chamfer(distance, None, body.edges()))


def _selected_edges(shape, indices):
    source_edges = list(shape.edges())
    if not indices or any(type(index) is not int or index < 0 or index >= len(source_edges) for index in indices):
        raise ValueError('Select one or more valid edges first.')
    body = _body(shape)
    body_edges = list(body.edges())
    selected = []
    for index in sorted(set(indices)):
        edge = next((candidate for candidate in body_edges if candidate.is_same(source_edges[index])), None)
        if edge is None:
            raise ValueError('A selected edge no longer belongs to the model.')
        selected.append(edge)
    return body, selected


def fillet_edges(shape, indices, radius):
    _positive(radius, 'Radius')
    body, edges = _selected_edges(shape, indices)
    return _validate(body.fillet(radius, edges))


def chamfer_edges(shape, indices, distance):
    _positive(distance, 'Distance')
    body, edges = _selected_edges(shape, indices)
    return _validate(body.chamfer(distance, None, edges))


def shell_face(shape, face_index, thickness):
    _body(shape)
    if not math.isfinite(thickness) or abs(thickness) <= 1e-6:
        raise ValueError('Thickness must be non-zero and finite.')
    solids = list(shape.solids())
    if len(solids) != 1:
        raise ValueError('Shell currently requires exactly one body.')
    return _validate(solids[0].hollow([_face(shape, face_index)], thickness))


def offset_solid(shape, distance):
    _body(shape)
    if not math.isfinite(distance) or abs(distance) <= 1e-6:
        raise ValueError('Offset must be non-zero and finite.')
    if len(shape.solids()) != 1:
        raise ValueError('Offset currently requires exactly one body.')
    return _validate(cad.offset(shape.solids()[0], amount=distance))


def hole_at_face_center(shape, face_index, diameter, depth, offset_u=0, offset_v=0, through_all=False):
    _body(shape)
    _positive(diameter, 'Diameter')
    if not through_all:
        _positive(depth, 'Depth')
    if not np.isfinite([offset_u, offset_v]).all():
        raise ValueError('Position offsets must be finite.')
    face = _face(shape, face_index)
    if face.geom_type != cad.GeomType.PLANE:
        raise ValueError('Select a planar face for the hole.')
    normal = np.array(tuple(face.normal_at()), dtype=float)
    center = np.array(tuple(face.center()), dtype=float)
    reference = np.array([0., 1., 0.]) if abs(normal[1]) < 0.9 else np.array([1., 0., 0.])
    u = np.cross(normal, reference)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    point = center + u * offset_u + v * offset_v
    if not face.is_inside(tuple(point)):
        raise ValueError('Hole center must lie on the selected face.')
    plane = cad.Plane(origin=tuple(point), z_dir=tuple(normal))
    profile = cad.Face(cad.Wire(cad.Edge.make_circle(diameter / 2, plane)))
    return build_extrusion(profile, normal, -depth, 'cut', shape,
                           extent='all' if through_all else 'distance')
