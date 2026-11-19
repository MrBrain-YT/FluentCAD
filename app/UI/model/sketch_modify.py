"""Geometry and parameter edits for the in-scene sketch modification tools."""
from bisect import bisect_right
from copy import deepcopy
from dataclasses import dataclass
import math
from uuid import uuid4

import build123d as cad

from UI.model.curve_spans import (
    CURVE_TYPES, PARAMETER_LIMIT, curve_fraction_at_point, curve_interval, native_curve,
)
from UI.model.profiles import native_sketch_edges


TOLERANCE = 1e-6
NATIVE_CURVES = frozenset({cad.GeomType.ELLIPSE, cad.GeomType.BEZIER, cad.GeomType.BSPLINE})


@dataclass(frozen=True)
class SketchModification:
    preview: cad.Edge | cad.Wire
    replacement: tuple[cad.Edge, ...]
    ranges: tuple = ()
    extension: bool = False


def _intersection_parameters(edge, boundaries, extending=False):
    parameters = []
    tolerance = TOLERANCE / edge.length
    def position(point):
        return curve_fraction_at_point(edge, point) if edge.geom_type in NATIVE_CURVES else edge.param_at_point(point)

    for boundary in boundaries:
        for hit in edge.intersect(boundary) or ():
            if isinstance(hit, cad.Edge):
                if not extending:
                    raise ValueError('Trim is ambiguous where sketch curves overlap.')
                value = min(position(vertex.center()) for vertex in hit.vertices())
                if value <= tolerance:
                    raise ValueError('The extension would overlap a curve at this endpoint.')
            else:
                value = position(hit.center())
            if edge.is_closed and value >= 1 - tolerance:
                value = 0.0
            if not any(abs(value - existing) <= tolerance for existing in parameters):
                parameters.append(value)
    return sorted(parameters)


def _portion(edge, first, last):
    if edge.geom_type in NATIVE_CURVES:
        return curve_interval(edge, first, last)
    if edge.is_closed and last > 1:
        points = [edge.position_at(t % 1) for t in (first, (first + last) / 2, last)]
        return cad.Edge.make_three_point_arc(*points)
    return edge.trim(float(first), float(last))


def _trim_edit(edge, first, last, keep):
    if edge.is_closed and edge.geom_type == cad.GeomType.BEZIER:
        def split_seam(a, b):
            if b > 1:
                return [(a, 1.), (0., b - 1.)]
            return [(a, b)]

        preview = [_portion(edge, a, b) for a, b in split_seam(first, last)]
        ranges = tuple(part for a, b in keep for part in split_seam(a, b))
        return SketchModification(preview[0] if len(preview) == 1 else cad.Wire(preview),
                                  tuple(_portion(edge, a, b) for a, b in ranges), ranges)
    return SketchModification(_portion(edge, first, last), tuple(_portion(edge, a, b) for a, b in keep), tuple(keep))


def trim_sections(edge, boundaries):
    """Return clickable intervals, their removed edges and their remaining geometry."""
    tolerance = TOLERANCE / edge.length
    hits = _intersection_parameters(edge, boundaries)
    if edge.is_closed:
        if len(hits) < 2:
            return (0.0,), (SketchModification(edge, ()),)
        successors = [*hits[1:], hits[0] + 1]
        edits = []
        for first, last in zip(hits, successors):
            remaining_start = last % 1
            remaining_end = first + (first < remaining_start)
            edits.append(_trim_edit(edge, first, last, [(remaining_start, remaining_end)]))
        return hits, edits
    cuts = [0.0, *(t for t in hits if tolerance < t < 1 - tolerance), 1.0]
    edits = []
    for first, last in zip(cuts, cuts[1:]):
        keep = []
        if first > tolerance:
            keep.append((0, first))
        if last < 1 - tolerance:
            keep.append((last, 1))
        edits.append(_trim_edit(edge, first, last, keep))
    return cuts[:-1], edits


def trim_at(sections, parameter):
    starts, edits = sections
    return edits[bisect_right(starts, min(parameter, 1.0)) - 1]


def source_curve_span(sketch, index, token):
    tool = sketch.tools[index]
    spans = getattr(tool, 'curve_spans', {})
    native = native_sketch_edges([tool], tuple(sketch.plane.z_dir))
    for segment, edge in enumerate(native):
        for name, start, end in spans.get(segment, [(segment, 0., 1.)]):
            if name == token:
                return segment, edge, start, end
    raise ValueError('The source curve segment is missing.')


def extend_curve_span(base, start, end, boundaries, at_start):
    """Continue the conic or the spline's end polynomial up to the first boundary."""
    if not boundaries:
        raise ValueError('There is no boundary in that direction.')
    curve, first, last = native_curve(base)
    tip = start if at_start else end
    direction = -1 if at_start else 1
    committed = []
    if curve.IsPeriodic():
        room = curve.Period() / (last - first) - (end - start) - 1e-8
        limit = min(PARAMETER_LIMIT, max(-PARAMETER_LIMIT, tip + direction * room))
        limits = [(tip, limit, False)] if direction * (limit - tip) > 1e-8 else []
    else:
        nodes = (1., 0.) if at_start else (0., 1.)
        bounds = [tip, *(node for node in nodes if direction * (node - tip) > 1e-8)]
        limits = [(a, b, True) for a, b in zip(bounds, bounds[1:])]
        outside = bounds[-1]
        available = PARAMETER_LIMIT - direction * outside
        for step in (.125, .25, .5, 1., 2., 4., 8., 16., 32., 64., PARAMETER_LIMIT * 2):
            if available <= 1e-8:
                break
            limits.append((outside, outside + direction * min(step, available), False))
            if step >= available:
                break
    for position, limit, bounded in limits:
        a, b = sorted((position, limit))
        try:
            edge = curve_interval(base, a, b)
        except (ValueError, RuntimeError):
            break
        directed = edge.reversed() if at_start else edge
        hits = [t for t in _intersection_parameters(directed, boundaries, extending=True)
                if t > TOLERANCE / directed.length]
        if hits:
            hit = position + (limit - position) * hits[0]
            ranges = sorted([*committed, tuple(sorted((position, hit)))])
            pieces = tuple(curve_interval(base, u, v) for u, v in ranges)
            preview = pieces[0] if len(pieces) == 1 else cad.Wire(pieces)
            return SketchModification(preview, pieces, tuple(ranges), extension=True)
        if bounded:
            committed.append((a, b))
    raise ValueError('No boundary was reached along the curve continuation.')


def extend_edge(edge, boundaries, at_start, normal):
    if edge.geom_type in NATIVE_CURVES and not edge.is_closed:
        return extend_curve_span(edge, 0., 1., boundaries, at_start)
    endpoint = edge.position_at(0 if at_start else 1)
    if not boundaries:
        raise ValueError('There is no boundary in that direction.')
    if edge.geom_type == cad.GeomType.LINE:
        direction = edge.tangent_at(0) * (-1 if at_start else 1)
        boxes = [boundary.bounding_box() for boundary in boundaries]
        length = max((corner - endpoint).length for box in boxes for corner in (box.min, box.max))
        if length <= TOLERANCE:
            raise ValueError('There is no boundary in that direction.')
        extension = cad.Edge.make_line(endpoint, endpoint + direction * length * 2)
    elif edge.geom_type == cad.GeomType.CIRCLE and not edge.is_closed:
        radial = endpoint - edge.arc_center
        outward = edge.tangent_at(0 if at_start else 1) * (-1 if at_start else 1)
        z = cad.Vector(normal)
        if z.cross(radial).dot(outward) < 0:
            z = -z
        sweep = 360 - math.degrees(edge.length / edge.radius)
        extension = cad.Edge.make_circle(edge.radius, cad.Plane(edge.arc_center, x_dir=radial, z_dir=z),
                                         end_angle=sweep)
    else:
        raise ValueError('Extend requires a line, open arc or open spline.')
    tolerance = TOLERANCE / extension.length
    hits = [t for t in _intersection_parameters(extension, boundaries, extending=True) if t > tolerance]
    if not hits:
        raise ValueError('There is no boundary in that direction.')
    hit = hits[0]
    end = extension.position_at(hit)
    first = end if at_start else edge.position_at(0)
    last = edge.position_at(1) if at_start else end
    if edge.geom_type == cad.GeomType.LINE:
        result = cad.Edge.make_line(first, last)
    else:
        if (first - last).length <= TOLERANCE:
            raise ValueError('Extending to this boundary would close the arc.')
        result = cad.Edge.make_three_point_arc(first, edge.position_at(.5), last)
    return SketchModification(_portion(extension, 0, hit), (result,))


def offset_edge(edge, distance, normal):
    if not math.isfinite(distance) or abs(distance) <= TOLERANCE:
        raise ValueError('Offset distance must be non-zero.')
    if edge.geom_type == cad.GeomType.LINE:
        shift = cad.Vector(normal).cross(edge.tangent_at(0)) * distance
        copy = cad.Edge.make_line(edge.position_at(0) + shift, edge.position_at(1) + shift)
    else:
        center = edge.arc_center
        radius = edge.radius + distance
        if radius <= TOLERANCE:
            raise ValueError('The offset radius must remain positive.')
        if edge.is_closed:
            copy = cad.Edge.make_circle(radius, cad.Plane(center, z_dir=normal))
        else:
            points = [center + (edge.position_at(t) - center) * (radius / edge.radius) for t in (0, .5, 1)]
            copy = cad.Edge.make_three_point_arc(*points)
    return SketchModification(copy, (copy,))


def _mentions(value, names, entity_index):
    if isinstance(value, str):
        return value in names or value == f'@circle:{entity_index}'
    if not isinstance(value, (list, tuple)):
        return False
    if len(value) >= 2 and isinstance(value[0], str) and value[0].startswith('@') and value[1] == entity_index:
        return True
    return any(_mentions(item, names, entity_index) for item in value)


def _shift_references(value, removed):
    if isinstance(value, str) and value.startswith('@circle:'):
        index = int(value.split(':')[1])
        return f'@circle:{index - (index > removed)}'
    if not isinstance(value, list):
        return value
    if len(value) >= 2 and isinstance(value[0], str) and value[0].startswith('@') and type(value[1]) is int:
        return [value[0], value[1] - (value[1] > removed), *[_shift_references(item, removed) for item in value[2:]]]
    return [_shift_references(item, removed) for item in value]


def modified_sketch_parameters(parameters, sketch, index, segment, edge, modification, copy_only=False):
    """Keep unaffected entities and refuse edits that would detach constraints or paths."""
    params = deepcopy(parameters)
    entities, points = params['entities'], params.setdefault('points', {})
    entity = entities[index]
    kind = entity['type']
    if not copy_only and segment in getattr(sketch.tools[index], 'corner_limits', {}):
        raise ValueError('This segment is used by a sketch fillet or chamfer.')
    if not copy_only and any(index in item.get('curves', []) or item.get('curve') == index for item in entities):
        raise ValueError('This segment is used by a slot path.')
    if not copy_only and kind in CURVE_TYPES:
        native_index, _, start, end = source_curve_span(sketch, index, segment)
        spans = entity.setdefault('curve_spans', {})
        previous = spans.get(str(native_index), [[native_index, 0., 1.]])
        if modification.extension:
            ranges = [*previous, *[[uuid4().hex, a, b] for a, b in modification.ranges]]
        else:
            ranges = [item for item in previous if item[0] != segment]
            for a, b in modification.ranges:
                identity = segment if len(modification.ranges) == 1 else uuid4().hex
                ranges.append([identity, start + (end - start) * a, start + (end - start) * b])
        spans[str(native_index)] = sorted(ranges, key=lambda item: item[1])
        return params
    names = entity['points'][segment:segment + 2] if kind == 'line' else entity.get('points', [entity.get('center')])
    named = {name for name in names if isinstance(name, str)}
    if not copy_only:
        if any(_mentions(c['entities'], named, index) for c in params.get('constraints', [])):
            raise ValueError('Remove constraints on this segment before using Trim or Extend.')
        point_values = [points[name] for name in named] + [value for value in names if isinstance(value, list)]
        if isinstance(entity.get('radius'), str) or any(isinstance(v, str) for pair in point_values for v in pair):
            raise ValueError('This segment uses an expression. Edit its parameter before changing the geometry.')

    next_point = 1

    def point_at(position):
        nonlocal next_point
        if not copy_only and kind != 'circle':
            for name, t in zip(names[:2], (0, 1)):
                if (position - edge.position_at(t)).length <= TOLERANCE:
                    return name
        while f'Edit.P{next_point}' in points:
            next_point += 1
        name = f'Edit.P{next_point}'
        local = sketch.plane.to_local_coords(position)
        points[name] = [local.X, local.Y]
        next_point += 1
        return name

    def line(vertices):
        return {**entity, 'points': vertices}

    replacements = []
    if not copy_only and kind == 'line' and segment > 0:
        replacements.append(line(entity['points'][:segment + 1]))
    for item in modification.replacement:
        if item.geom_type == cad.GeomType.LINE:
            new_entity = line([point_at(item.position_at(0)), point_at(item.position_at(1))])
        elif item.is_closed:
            new_entity = {'type': 'circle', 'center': point_at(item.arc_center), 'radius': item.radius}
        else:
            new_entity = {'type': 'arc', 'points': [point_at(item.position_at(t)) for t in (0, 1, .5)]}
        new_entity['construction'] = entity.get('construction', False)
        replacements.append(new_entity)
    if not copy_only and kind == 'line' and segment + 2 < len(entity['points']):
        replacements.append(line(entity['points'][segment + 1:]))
    if copy_only:
        entities.extend(replacements)
    elif replacements:
        entities[index] = replacements[0]
        entities.extend(replacements[1:])
    else:
        entities.pop(index)
        for item in entities:
            if 'curve' in item:
                item['curve'] -= item['curve'] > index
            if 'curves' in item:
                item['curves'] = [i - (i > index) for i in item['curves']]
        for constraint in params.get('constraints', []):
            constraint['entities'] = _shift_references(constraint['entities'], index)
    if not copy_only:
        retained = {value for item in entities for value in item.get('points', [item.get('center')]) if isinstance(value, str)}
        for name in named - retained:
            points.pop(name)
    return params
