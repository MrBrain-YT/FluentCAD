"""Resolve the geometry selected for solid operation limits and axes."""
import build123d as cad

from UI.model.geometry_references import resolve_reference
from UI.model.transforms import body_index_for_face


def resolve_operation_geometry(selection, shape, feature_id, body_names):
    if shape is None:
        raise ValueError('The selected body is unavailable.')
    reference = selection.get('reference')
    kind = selection['kind']
    entity_kind = 'edge' if kind == 'edge' else 'face'
    if feature_id is not None and (not isinstance(reference, dict) or reference.get('kind') != entity_kind):
        raise ValueError('The selected geometry needs a valid reference.')
    entities = list(shape.edges() if entity_kind == 'edge' else shape.faces())
    index = (resolve_reference(reference, shape, feature_id, body_names=body_names)
             if reference is not None else selection.get('index'))
    if type(index) is not int or not 0 <= index < len(entities):
        raise ValueError('The selected geometry is unavailable.')
    return index, entities[index]


def extrusion_target(selection, shape, feature_id=None, body_names=None, results=None):
    if not selection:
        raise ValueError('Select a target face, body or plane.')
    kind = selection['kind']
    if kind == 'plane':
        if 'feature_id' in selection:
            plane = (results or {}).get(selection['feature_id'])
            if not isinstance(plane, cad.Plane):
                raise ValueError('The target plane is unavailable.')
            return plane
        return cad.Plane(selection['origin'], z_dir=selection['normal'])
    if kind not in ('face', 'body'):
        raise ValueError('Select a target face, body or plane.')
    index, face = resolve_operation_geometry(selection, shape, feature_id, body_names)
    if kind == 'face' and face.geom_type == cad.GeomType.PLANE:
        return face
    owner = body_index_for_face(shape, index)
    if owner is None:
        raise ValueError('Could not identify the target body.')
    body = list(shape.solids())[owner]
    return body if kind == 'body' else (body, face)


def revolution_axis(selection, sketch=None, shape=None, feature_id=None, body_names=None):
    if not selection:
        raise ValueError('Select a straight sketch line, straight edge or cylindrical face.')
    kind = selection['kind']
    if kind == 'coordinates':
        return tuple(selection['origin']), tuple(selection['direction'])
    if kind == 'sketch_line':
        segments = [segment for reference, segment in sketch.axis_segments if reference == selection] if sketch else []
        if len(segments) != 1:
            raise ValueError('The sketch line selected as the axis is missing or ambiguous.')
        start, end = segments[0]
        return tuple(start), tuple(end - start)
    if kind not in ('edge', 'face'):
        raise ValueError('Select a straight edge or cylindrical face.')
    _, geometry = resolve_operation_geometry(selection, shape, feature_id, body_names)
    if kind == 'edge' and geometry.geom_type == cad.GeomType.LINE:
        start, end = geometry.position_at(0), geometry.position_at(1)
        direction = end - start
    elif kind == 'face' and geometry.geom_type == cad.GeomType.CYLINDER:
        axis = geometry.axis_of_rotation
        if axis is None:
            raise ValueError('Could not determine the cylindrical axis.')
        start, direction = axis.position, axis.direction
    else:
        raise ValueError('Select a straight edge or cylindrical face.')
    if selection.get('forward') is not None and direction.dot(cad.Vector(selection['forward'])) < 0:
        direction = -direction
    return tuple(start), tuple(direction)
