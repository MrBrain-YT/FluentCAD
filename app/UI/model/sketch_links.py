"""Associative projections, sections and spatial curves from earlier features."""
from dataclasses import dataclass
import base64
import io
import math
from uuid import uuid4

import build123d as cad
from OCP.BRep import BRep_Builder, BRep_Tool
from OCP.BRepAlgoAPI import BRepAlgoAPI_Section
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
from OCP.BRepTools import BRepTools
from OCP.Geom import Geom_Plane, Geom_TrimmedCurve
from OCP.GeomProjLib import GeomProjLib
from OCP.TopAbs import TopAbs_EDGE
from OCP.TopoDS import TopoDS, TopoDS_Shape
from OCP.gp import gp_Dir

from UI.model.geometry_references import make_reference, resolve_reference
from UI.model.sketch_projection_curves import projectable_sketch_curves
from UI.model.sketch_silhouette import project_body_silhouette
from UI.model.transforms import body_index_for_face


@dataclass(frozen=True)
class LinkedSketchGeometry:
    edges: tuple
    points: tuple
    spatial: bool = False


MAX_SKETCH_LINK_BYTES = 4 * 1024 * 1024


def freeze_sketch_geometry(geometry, plane):
    """Save exact curves with their identities, using model coordinates for Include."""
    frame = cad.Plane.XY if geometry.spatial else plane
    edges = []
    size = 0
    for name, edge in geometry.edges:
        stream = io.BytesIO()
        if not cad.export_brep(frame.to_local_coords(edge), stream):
            raise ValueError('Could not save the sketch link geometry.')
        size += stream.tell()
        if size > MAX_SKETCH_LINK_BYTES or len(edges) >= 1000:
            raise ValueError('The linked geometry is too large to save.')
        edges.append([name, base64.b64encode(stream.getvalue()).decode('ascii')])
    dimensions = 3 if geometry.spatial else 2
    points = [list(tuple(frame.to_local_coords(point))[:dimensions]) for point in geometry.points]
    snapshot = {'edges': edges, 'points': points}
    if geometry.spatial:
        snapshot['space'] = 'model'
    return snapshot


def restore_sketch_geometry(snapshot, plane, link_id):
    if not isinstance(snapshot, dict):
        raise ValueError('Invalid saved sketch geometry.')
    space = snapshot.get('space', 'sketch')
    if space not in ('sketch', 'model'):
        raise ValueError('Invalid coordinate space for saved sketch geometry.')
    spatial = space == 'model'
    edges, points = snapshot.get('edges'), snapshot.get('points')
    if not isinstance(edges, list) or not isinstance(points, list) or max(len(edges), len(points)) > 1000:
        raise ValueError('Invalid saved sketch geometry.')
    total = 0
    restored = []
    for item in edges:
        if (not isinstance(item, list) or len(item) != 2 or not isinstance(item[0], str)
                or not item[0].startswith(f'{link_id}:') or not isinstance(item[1], str)):
            raise ValueError('Invalid saved sketch curve.')
        name, encoded = item
        total += len(encoded)
        if total > (MAX_SKETCH_LINK_BYTES * 4 // 3) + 4000:
            raise ValueError('The saved sketch geometry is too large.')
        raw = TopoDS_Shape()
        BRepTools.Read_s(raw, io.BytesIO(base64.b64decode(encoded, validate=True)), BRep_Builder())
        if raw.IsNull() or raw.ShapeType() != TopAbs_EDGE:
            raise ValueError('Invalid curve in saved sketch geometry.')
        edge = cad.Edge(TopoDS.Edge_s(raw))
        if not edge.is_valid or edge.length <= 1e-6:
            raise ValueError('Saved sketch curves must be valid.')
        if not spatial:
            bounds = edge.bounding_box()
            if max(abs(bounds.min.Z), abs(bounds.max.Z)) > 1e-5:
                raise ValueError('Saved planar sketch curves must lie on their sketch plane.')
        restored.append((name, edge if spatial else plane.from_local_coords(edge)))
    dimensions = 3 if spatial else 2
    if any(not isinstance(point, list) or len(point) != dimensions
           or any(type(value) not in (int, float) or not math.isfinite(value) for value in point) for point in points):
        raise ValueError('Invalid point in saved sketch geometry.')
    if not restored and not points:
        raise ValueError('The saved sketch geometry is empty.')
    vertices = tuple(cad.Vector(point) if spatial else plane.from_local_coords((*point, 0)) for point in points)
    return LinkedSketchGeometry(tuple(restored), vertices, spatial)


def validate_sketch_links(links):
    if not isinstance(links, list) or len(links) > 1000:
        raise ValueError('Sketch geometry links must be a list with at most 1000 items.')
    identifiers = set()
    for link in links:
        if not isinstance(link, dict):
            raise ValueError('Invalid sketch geometry link.')
        identifier, source = link.get('id'), link.get('source_id')
        operation, kind = link.get('operation'), link.get('kind')
        if (not isinstance(identifier, str) or not identifier or len(identifier) > 128 or identifier in identifiers
                or not isinstance(source, str) or not source or operation not in ('project', 'intersect', 'include')
                or kind not in ('edge', 'face', 'body', 'sketch_curve')
                or (kind == 'sketch_curve' and operation == 'intersect')
                or (operation == 'include' and kind not in ('edge', 'sketch_curve'))):
            raise ValueError('Invalid or duplicate sketch geometry link.')
        anchors = link.get('anchors', [])
        if not isinstance(anchors, list) or len(anchors) > 1000:
            raise ValueError('Invalid linked geometry anchors.')
        identifiers.add(identifier)
    return links


def _body(link, shape, body_names):
    if 'body_id' in link:
        matches = [i for i, name in enumerate(body_names or ()) if name == link['body_id']]
        if not isinstance(link['body_id'], str) or len(matches) != 1:
            raise ValueError('The linked body is missing or ambiguous.')
        return list(shape.solids())[matches[0]]
    reference = link.get('reference')
    if not isinstance(reference, dict) or reference.get('kind') != 'face':
        raise ValueError('The linked body needs a valid face reference.')
    index = resolve_reference(reference, shape, link['source_id'], body_names=body_names)
    owner = body_index_for_face(shape, index)
    if owner is None:
        raise ValueError('The linked body is unavailable.')
    return list(shape.solids())[owner]


def _project_edge(edge, plane):
    normal = plane.z_dir
    if edge.geom_type == cad.GeomType.LINE and edge.tangent_at(0).cross(normal).length < 1e-9:
        local = plane.to_local_coords(edge.position_at(0))
        return plane.from_local_coords((local.X, local.Y, 0))
    adaptor = edge.geom_adaptor()
    first, last = adaptor.FirstParameter(), adaptor.LastParameter()
    curve = Geom_TrimmedCurve(BRep_Tool.Curve_s(edge.wrapped, first, last), first, last)
    projected = GeomProjLib.ProjectOnPlane_s(
        curve, Geom_Plane(plane.wrapped), gp_Dir(*tuple(normal)), True,
    )
    if projected is None:
        raise ValueError('The selected edge cannot be projected to this sketch plane.')
    maker = BRepBuilderAPI_MakeEdge(projected, projected.FirstParameter(), projected.LastParameter())
    if not maker.IsDone():
        raise ValueError('Could not build the projected curve.')
    result = cad.Edge(maker.Edge())
    if result.length <= 1e-6:
        return result.position_at(0)
    return result


def _section(shape, target, plane):
    section = BRepAlgoAPI_Section(target.wrapped, plane.wrapped, False)
    section.Build()
    if not section.IsDone():
        raise ValueError('Could not intersect the geometry with this sketch plane.')
    result = cad.Compound.cast(section.Shape())
    faces = list(shape.faces())
    edges = []
    for edge in result.edges():
        ancestor = TopoDS_Shape()
        index = None
        if section.HasAncestorFaceOn1(edge.wrapped, ancestor):
            index = next((i for i, face in enumerate(faces) if face.wrapped.IsSame(ancestor)), None)
        if edge.length > 1e-6:
            edges.append((index, edge))
    vertices = result.vertices()
    return edges, [vertex.center() for vertex in vertices if not any(vertex.distance_to(edge) < 1e-6 for _, edge in edges)]


def _geometry(link, shape, body_names, plane):
    kind = link['kind']
    if kind == 'sketch_curve':
        key = link.get('curve')
        matches = [edge for name, edge in projectable_sketch_curves(shape) if name == key]
        if 'curve_reference' in link and matches:
            group = cad.Compound(matches)
            index = resolve_reference(link['curve_reference'], group, link['source_id'])
            curve = group.edges()[index]
        elif len(matches) == 1:
            curve = matches[0]
        else:
            raise ValueError('The source sketch curve is missing or ambiguous.')
        if link['operation'] == 'include':
            return [(0, curve)], []
        image = _project_edge(curve, plane)
        return ([(0, image)], []) if isinstance(image, cad.Edge) else ([], [image])
    if not hasattr(shape, 'solids') or not shape.solids():
        raise ValueError('The linked feature has no solid geometry.')
    if kind == 'body':
        target = _body(link, shape, body_names)
    else:
        reference = link.get('reference')
        if not isinstance(reference, dict) or reference.get('kind') != kind:
            raise ValueError('Invalid sketch geometry reference.')
        index = resolve_reference(reference, shape, link['source_id'], body_names=body_names)
        target = (shape.edges() if kind == 'edge' else shape.faces())[index]
    if link['operation'] == 'include':
        return [(0, target)], []
    if link['operation'] == 'intersect':
        return _section(shape, target, plane)
    if kind == 'body':
        return project_body_silhouette(target, plane), []
    source_edges = list(shape.edges()) if kind == 'face' else [target]
    selected_edges = list(target.edges()) if kind == 'face' else [target]
    projected, points = [], []
    for i, edge in enumerate(source_edges):
        if not any(edge.is_same(item) for item in selected_edges):
            continue
        image = _project_edge(edge, plane)
        if isinstance(image, cad.Edge):
            projected.append((i, image))
        else:
            points.append(image)
    return projected, points


def _ancestors(link):
    if link['operation'] == 'project':
        return 'edge' if link['kind'] == 'face' else None
    return 'face' if link['operation'] == 'intersect' and link['kind'] == 'body' else None


def _named_geometry(link, raw_edges, points, anchors):
    if not raw_edges and not points:
        raise ValueError('The selected geometry does not intersect this sketch plane.')
    silhouette = link['operation'] == 'project' and link['kind'] == 'body'
    return LinkedSketchGeometry(
        tuple((f"{link['id']}:{index if silhouette else anchors.get(index, 'link')}", edge)
              for index, edge in raw_edges),
        tuple(points),
        spatial=link['operation'] == 'include',
    )


def make_sketch_link(operation, kind, source_id, shape, index, body_names, plane, link_id=None):
    link = {'id': link_id or uuid4().hex, 'operation': operation, 'kind': kind, 'source_id': source_id}
    validate_sketch_links([link])
    if kind == 'sketch_curve':
        curves = projectable_sketch_curves(shape)
        if type(index) is not int or not 0 <= index < len(curves):
            raise ValueError('Select a curve in the source sketch.')
        key, selected = curves[index]
        link['curve'] = key
        matches = [edge for name, edge in curves if name == key]
        if len(matches) > 1:
            group = cad.Compound(matches)
            member = next(i for i, edge in enumerate(group.edges()) if edge.is_same(selected))
            link['curve_reference'] = make_reference(source_id, group, 'edge', member)
        edges, points = _geometry(link, shape, body_names, plane)
        return link, _named_geometry(link, edges, points, {})
    if kind == 'body':
        owner = body_index_for_face(shape, index)
        name = body_names[owner] if owner is not None and body_names and owner < len(body_names) else None
        if name is not None and body_names.count(name) == 1:
            link['body_id'] = name
    if 'body_id' not in link:
        link['reference'] = make_reference(source_id, shape, 'edge' if kind == 'edge' else 'face',
                                           index, body_names)
    raw_edges, points = _geometry(link, shape, body_names, plane)
    anchors = {}
    anchor_kind = _ancestors(link)
    if anchor_kind is not None:
        for index in sorted({index for index, _ in raw_edges if index is not None}):
            try:
                reference = make_reference(source_id, shape, anchor_kind, index, body_names)
            except ValueError:
                continue
            anchors[index] = len(anchors)
            link.setdefault('anchors', []).append(reference)
    return link, _named_geometry(link, raw_edges, points, anchors)


def rebuild_sketch_link(link, shape, body_names, plane):
    raw_edges, points = _geometry(link, shape, body_names, plane)
    anchors = {}
    for label, reference in enumerate(link.get('anchors', [])):
        if not isinstance(reference, dict) or reference.get('kind') != _ancestors(link):
            raise ValueError('Invalid sketch link boundary reference.')
        try:
            index = resolve_reference(reference, shape, link['source_id'], body_names=body_names)
        except ValueError:
            # The link can still rebuild after a topology change. A dependent profile
            # must reselect its boundary if an original ancestor is no longer present.
            continue
        anchors[index] = label
    geometry = _named_geometry(link, raw_edges, points, anchors)
    previous = link.get('fallback')
    if isinstance(previous, dict):
        old_names = [item[0] for item in previous.get('edges', [])]
        new_names = [name for name, _ in geometry.edges]
        if link['operation'] == 'project' and link['kind'] == 'body':
            # HLR can split or join edges on the same footprint loop between rebuilds.
            old_names, new_names = set(old_names), set(new_names)
        else:
            old_names, new_names = sorted(old_names), sorted(new_names)
        if old_names != new_names or len(previous.get('points', [])) != len(points):
            raise ValueError('The linked geometry changed topology. Choose a replacement source.')
    return geometry
