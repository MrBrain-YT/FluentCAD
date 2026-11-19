"""Associative planar sketch patterns with stable identities for each instance."""
import math

import build123d as cad

from UI.model.sketch_mirror import SketchReflection, same_sketch_curve
from UI.model.sketch_projection_curves import named_sketch_curves


def build_sketch_pattern(plane, identifier, kind, edges, points, reference, values):
    if kind == 'rectangular':
        nx, ny, dx, dy = values
        if nx * ny < 2 or nx * ny > 100 or (nx > 1 and dx == 0) or (ny > 1 and dy == 0):
            raise ValueError('Use 2 to 100 instances and nonzero spacing along each repeated direction.')
        if reference.geom_type != cad.GeomType.LINE:
            raise ValueError('Select a straight sketch segment for the pattern direction.')
        u = (reference.position_at(1) - reference.position_at(0)).normalized()
        v = plane.z_dir.cross(u).normalized()
        instances = [(f'{i}:{j}', u * (i * dx) + v * (j * dy))
                     for i in range(nx) for j in range(ny) if i or j]
        move_edge = lambda edge, delta: edge.translate(delta)
        move_point = lambda point, delta: cad.Vector(point) + delta
    elif kind == 'circular':
        count, angle = values
        if count < 2 or not 0 < abs(angle) <= 360:
            raise ValueError('Use at least two instances and a nonzero sweep of at most 360 degrees.')
        axis = cad.Axis(reference, plane.z_dir)
        step = angle / (count if math.isclose(abs(angle), 360, abs_tol=1e-9) else count - 1)
        instances = [(str(i), i * step) for i in range(1, count)]
        move_edge = lambda edge, degrees: edge.rotate(axis, degrees)
        move_point = lambda point, degrees: cad.Vertex(*tuple(point)).rotate(axis, degrees).center()
    else:
        raise ValueError('Invalid sketch pattern type.')
    if len(instances) * (len(edges) + len(points)) > 5000:
        raise ValueError('Reduce the number of selected objects or instances.')
    output, output_points, construction = [], [], set()
    seen_edges = [[edge] for edge, _ in edges]
    seen_points = [[cad.Vector(point)] for point in points]
    for instance, transform in instances:
        prefix = f'pattern:{identifier}:{instance}'
        for i, (edge, is_construction) in enumerate(edges):
            name = f'{prefix}:edge:{i}'
            copy = move_edge(edge, transform)
            if kind == 'circular':
                if any(same_sketch_curve(original, copy) for original in seen_edges[i]):
                    continue
                seen_edges[i].append(copy)
            output.append((name, copy))
            if is_construction:
                construction.add(name)
        for i, point in enumerate(points):
            copy = move_point(point, transform)
            if any((copy - previous).length <= 1e-6 for previous in seen_points[i]):
                continue
            seen_points[i].append(copy)
            output_points.append((f'{prefix}:point:{i}', copy))
    return SketchReflection(tuple(output), tuple(output_points), frozenset(construction))


def rebuild_sketch_patterns(entries, plane, entities, tools, normal, derived, named_points, number):
    if not isinstance(entries, list) or len(entries) > 1000:
        raise ValueError('Invalid sketch patterns.')
    edges, ambiguous = {}, set()
    for entity, tool in zip(entities, tools):
        for name, edge in named_sketch_curves([entity], [tool], normal):
            if name in edges:
                ambiguous.add(name)
            edges[name] = edge, tool.is_construction
    edges.update(derived)
    for name in ambiguous:
        del edges[name]
    points = dict(named_points)
    output, output_points, construction, identifiers = [], [], set(), set()

    def count(value):
        value = number(value)
        if not value.is_integer() or not 1 <= value <= 100:
            raise ValueError('Pattern count must be an integer from 1 to 100.')
        return int(value)

    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError('Invalid sketch pattern.')
        identifier, kind = entry.get('id'), entry.get('kind')
        if not isinstance(identifier, str) or not identifier or ':' in identifier or identifier in identifiers:
            raise ValueError('Invalid or duplicate sketch pattern ID.')
        identifiers.add(identifier)
        curves, handles = entry.get('curves', []), entry.get('points', [])
        for refs in (curves, handles):
            if (not isinstance(refs, list) or not all(isinstance(ref, str) for ref in refs)
                    or len(refs) != len(set(refs))):
                raise ValueError('Invalid sketch pattern sources.')
        if (not curves and not handles) or any(ref not in edges for ref in curves) or any(ref not in points for ref in handles):
            raise ValueError('A sketch pattern source is missing or ambiguous.')
        if kind == 'rectangular':
            direction = entry.get('direction')
            if not isinstance(direction, str) or direction not in edges:
                raise ValueError('The sketch pattern direction is missing or ambiguous.')
            reference = edges[direction][0]
            values = (count(entry.get('count_x')), count(entry.get('count_y')),
                      number(entry.get('spacing_x')), number(entry.get('spacing_y')))
        elif kind == 'circular':
            center = entry.get('center')
            if isinstance(center, str) and center in points:
                reference = points[center]
            elif isinstance(center, list) and len(center) == 2:
                reference = plane.from_local_coords((*map(number, center), 0))
            else:
                raise ValueError('The sketch pattern center is missing.')
            values = count(entry.get('count')), number(entry.get('angle'))
        else:
            raise ValueError('Invalid sketch pattern type.')
        result = build_sketch_pattern(plane, identifier, kind, [edges[ref] for ref in curves],
                                      [points[ref] for ref in handles], reference, values)
        for name, edge in result.edges:
            edges[name] = edge, name in result.construction
        points.update(result.points)
        output.extend(result.edges)
        output_points.extend(result.points)
        construction.update(result.construction)
    return SketchReflection(tuple(output), tuple(output_points), frozenset(construction))
