"""Associative reflection of sketch curves and points about a straight sketch edge."""
from dataclasses import dataclass

import build123d as cad

from UI.model.profiles import indexed_sketch_edges
from UI.model.sketch_projection_curves import sketch_curve_name


TOLERANCE = 1e-6


@dataclass(frozen=True)
class SketchReflection:
    edges: tuple = ()
    points: tuple = ()
    construction: frozenset = frozenset()


def reflection_plane(plane, axis):
    if axis.geom_type != cad.GeomType.LINE or axis.length <= TOLERANCE:
        raise ValueError('Choose a straight sketch segment as the mirror axis.')
    direction = axis.position_at(1) - axis.position_at(0)
    return cad.Plane(axis.position_at(0), x_dir=direction, z_dir=direction.cross(plane.z_dir))


def same_sketch_curve(edge, reflected):
    original, mirror = edge.bounding_box(), reflected.bounding_box()
    if max((original.min - mirror.min).length, (original.max - mirror.max).length) > TOLERANCE:
        return False
    return all(cad.Vertex(*tuple(a.position_at(t))).distance_to(b) <= TOLERANCE
               for a, b in ((edge, reflected), (reflected, edge)) for t in (0., .17, .41, .63, .83, 1.))


def reflect_sketch(plane, axis, edges, points, identifier):
    """Mirror exact CAD edges, preserving identities by position in the saved selection."""
    frame = reflection_plane(plane, axis)
    output, reflected_points, construction = [], [], set()
    for i, (edge, is_construction) in enumerate(edges):
        reflected = edge.mirror(frame)
        if same_sketch_curve(edge, reflected):
            continue
        name = f'mirror:{identifier}:edge:{i}'
        output.append((name, reflected))
        if is_construction:
            construction.add(name)
    for i, value in enumerate(points):
        point = cad.Vector(value)
        distance = (point - frame.origin).dot(frame.z_dir)
        if abs(distance) > TOLERANCE:
            reflected_points.append((f'mirror:{identifier}:point:{i}', point - frame.z_dir * (2 * distance)))
    return SketchReflection(tuple(output), tuple(reflected_points), frozenset(construction))


def rebuild_sketch_mirrors(entries, plane, entities, tools, normal, derived_edges, named_points):
    if not isinstance(entries, list) or len(entries) > 1000:
        raise ValueError('Invalid sketch mirrors.')
    edges, ambiguous = {}, set()
    for entity, tool in zip(entities, tools):
        for segment, edge in indexed_sketch_edges(tool, normal):
            name = sketch_curve_name(entity, segment)
            if name in edges:
                ambiguous.add(name)
            edges[name] = edge, tool.is_construction
    for name, edge in derived_edges:
        if name in edges:
            ambiguous.add(name)
        edges[name] = edge, False
    for name in ambiguous:
        del edges[name]
    points = {name: plane.from_local_coords((*position, 0)) for name, position in named_points.items()}
    identifiers, output, reflected_points, construction = set(), [], [], set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError('Invalid sketch mirror.')
        identifier, axis = entry.get('id'), entry.get('axis')
        if not isinstance(identifier, str) or not identifier or ':' in identifier or identifier in identifiers:
            raise ValueError('Invalid or duplicate sketch mirror ID.')
        identifiers.add(identifier)
        refs, point_refs = entry.get('curves', []), entry.get('points', [])
        for selection in (refs, point_refs):
            if (not isinstance(selection, list) or len(selection) > 1000
                    or not all(isinstance(name, str) for name in selection) or len(set(selection)) != len(selection)):
                raise ValueError('Invalid sketch mirror sources.')
        if not refs and not point_refs:
            raise ValueError('Select at least one sketch curve or point to mirror.')
        if not isinstance(axis, str) or axis not in edges:
            raise ValueError('The sketch mirror axis is missing or ambiguous.')
        if axis in refs or any(name not in edges for name in refs) or any(name not in points for name in point_refs):
            raise ValueError('A sketch mirror source is missing or ambiguous, or also used as the axis.')
        result = reflect_sketch(plane, edges[axis][0], [edges[name] for name in refs],
                                [points[name] for name in point_refs], identifier)
        if any(name in edges for name, _ in result.edges) or any(name in points for name, _ in result.points):
            raise ValueError('A reflected sketch identity conflicts with existing geometry.')
        for name, edge in result.edges:
            edges[name] = edge, name in result.construction
        points.update(result.points)
        output.extend(result.edges)
        reflected_points.extend(result.points)
        construction.update(result.construction)
    return SketchReflection(tuple(output), tuple(reflected_points), frozenset(construction))
