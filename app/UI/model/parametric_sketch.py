"""Serializable 2-D geometry for parametric feature sketches."""
from dataclasses import dataclass
from types import SimpleNamespace
import math
import numpy as np
from UI.model.profiles import build_profiles
from UI.model.constraints import Constraint, solve_constraints


@dataclass(frozen=True)
class ParametricSketch:
    plane: object
    profiles: tuple
    constraint_status: object = None
    profile_keys: tuple = ()
    tools: tuple = ()


def circle_radius_name(index):
    return f'@circle:{index}'


def build_sketch(plane, entities, number, points=None, constraints=None):
    if not isinstance(entities, list) or len(entities) > 1000:
        raise ValueError('Sketch entities must be a list with at most 1000 items.')
    solved = None
    named_points = {}
    if points or constraints:
        if not isinstance(points, dict) or not points or len(points) > 1000:
            raise ValueError('Constrained sketches require named points.')
        for name, coordinates in points.items():
            if not isinstance(name, str) or not isinstance(coordinates, (list, tuple)) or len(coordinates) != 2:
                raise ValueError('Invalid named point.')
            named_points[name] = [number(value) for value in coordinates]
        if not isinstance(constraints, (list, type(None))) or len(constraints or []) > 1000:
            raise ValueError('Invalid constraint list.')
        def tuples(value):
            return tuple(tuples(item) for item in value) if isinstance(value, (list, tuple)) else value
        def path_descriptors(indices):
            if not isinstance(indices, list) or not 1 <= len(indices) <= 30:
                raise ValueError('Composite slot requires 1 to 30 path segments.')
            result = []
            for index in indices:
                if type(index) is not int or not 0 <= index < len(entities): raise ValueError('Missing path segment.')
                source = entities[index]
                kind, names = source.get('type'), source.get('points')
                if kind not in ('line', 'arc', 'spline', 'fit_spline', 'nurbs') or not isinstance(names, list):
                    raise ValueError('Unsupported composite path segment.')
                if (kind == 'line' and len(names) != 2) or not all(isinstance(name, str) for name in names):
                    raise ValueError('Path segments require named points; lines require exactly two.')
                result.append([kind, names, source.get('degree', 0),
                               [number(x) for x in source.get('knots', [])],
                               [number(x) for x in source.get('weights', [])]])
            return result
        def expand_nurbs(value):
            if isinstance(value, list) and len(value) == 2 and value[0] == '@curve':
                index = value[1]
                if type(index) is not int or not 0 <= index < len(entities): raise ValueError('Missing curve.')
                item = entities[index]
                kind = item.get('type')
                if kind == 'circle': return ['circle', item['center'], circle_radius_name(index)]
                if kind == 'ellipse':
                    if not item.get('is_closed', True): raise ValueError('General curvature constraints on elliptical arcs are not yet supported.')
                    return ['ellipse', item['center'], number(item['radius_x']), number(item['radius_y']), number(item.get('rotation', 0))]
                if kind in ('line','arc','spline','fit_spline','nurbs','periodic_nurbs'):
                    names = item.get('points')
                    if not isinstance(names, list) or not all(isinstance(name, str) for name in names): raise ValueError('Curve requires named points.')
                    if kind == 'line' and len(names) != 2: raise ValueError('Select an individual line segment.')
                    return [kind, names, item.get('degree', 0),
                            [number(x) for x in item.get('knots', [])],
                            [number(x) for x in item.get('weights', [])],
                            [number(x) for x in item.get('spans', [1]*len(names))]]
                raise ValueError('Unsupported curve type for curvature continuity.')
            if isinstance(value, list) and len(value) == 2 and value[0] == '@composite_path_slot':
                index = value[1]
                if type(index) is not int or not 0 <= index < len(entities) or entities[index].get('type') != 'composite_path_slot':
                    raise ValueError('Constraint references a missing composite path slot.')
                item = entities[index]
                return ['@composite_path_slot', number(item['radius']), path_descriptors(item.get('curves'))]
            if isinstance(value, list) and len(value) == 2 and value[0] == '@spline_slot':
                index = value[1]
                if type(index) is not int or not 0 <= index < len(entities) or entities[index].get('type') != 'spline_slot':
                    raise ValueError('Constraint references a missing spline slot.')
                slot = entities[index]
                curve = slot.get('curve')
                if type(curve) is not int or not 0 <= curve < len(entities) or entities[curve].get('type') not in ('spline', 'fit_spline', 'nurbs'):
                    raise ValueError('Spline slot requires a supported open spline path.')
                source = entities[curve]
                names = source['points']
                if not all(isinstance(point, str) for point in names):
                    raise ValueError('Constrained spline slot requires named points.')
                return ['@spline_slot', number(slot['radius']), source['type'], source.get('degree', 0),
                        [number(x) for x in source.get('knots', [])], [number(x) for x in source.get('weights', [])], *names]
            if isinstance(value, list) and len(value) == 2 and value[0] == '@periodic_nurbs':
                index = value[1]
                if type(index) is not int or not 0 <= index < len(entities) or entities[index].get('type') != 'periodic_nurbs':
                    raise ValueError('Constraint references a missing periodic NURBS.')
                item = entities[index]
                return ['@periodic_nurbs', item['degree'], [number(x) for x in item['weights']],
                        [number(x) for x in item.get('spans', [1] * len(item['points']))], *item['points']]
            if isinstance(value, list) and len(value) == 2 and value[0] == '@arc_slot':
                index = value[1]
                if type(index) is not int or not 0 <= index < len(entities) or entities[index].get('type') != 'arc_slot':
                    raise ValueError('Constraint references a missing arc slot.')
                item = entities[index]
                if not all(isinstance(point, str) for point in item['points']):
                    raise ValueError('Constrained arc slot requires named points.')
                return ['@arc_slot', number(item['radius']), *item['points']]
            if isinstance(value, list) and len(value) == 2 and value[0] == '@polyline_slot':
                index = value[1]
                if type(index) is not int or not 0 <= index < len(entities) or entities[index].get('type') != 'polyline_slot':
                    raise ValueError('Constraint references a missing composite slot.')
                item = entities[index]
                if not all(isinstance(point, str) for point in item['points']):
                    raise ValueError('Constrained composite slot requires named points.')
                return ['@polyline_slot', number(item['radius']), *item['points']]
            if isinstance(value, list) and len(value) == 2 and value[0] == '@nurbs':
                index = value[1]
                if type(index) is not int or not 0 <= index < len(entities) or entities[index].get('type') != 'nurbs':
                    raise ValueError('Constraint references a missing NURBS.')
                item = entities[index]
                return ['@nurbs', item['degree'], [number(x) for x in item['knots']],
                        [number(x) for x in item['weights']], *item['points']]
            return [expand_nurbs(item) for item in value] if isinstance(value, list) else value
        radii = {}
        parsed = []
        for index, entity in enumerate(entities):
            if isinstance(entity, dict) and entity.get('type') == 'circle':
                name = circle_radius_name(index)
                radii[name] = number(entity['radius'])
                if isinstance(entity['radius'], str):
                    parsed.append(Constraint('radius', ((entity.get('center'), name),), radii[name]))
            elif isinstance(entity, dict) and entity.get('type') == 'composite_path_slot':
                descriptors = path_descriptors(entity.get('curves'))
                parsed.extend(Constraint('composite_path_joint', tuples([a, b])) for a, b in zip(descriptors, descriptors[1:]))
        for entry in constraints or []:
            if not isinstance(entry, dict) or not isinstance(entry.get('kind'), str) or not isinstance(entry.get('entities'), list):
                raise ValueError('Invalid sketch constraint.')
            if entry.get('driven', False):
                raise ValueError('Driven dimensions are not yet supported in parametric sketches.')
            value = entry.get('value')
            constraint_entities = entry['entities']
            if entry['kind'] in ('tangent_arc_ellipse', 'curvature_arc_ellipse',
                                  'tangent_spline_ellipse', 'curvature_spline_ellipse'):
                if len(constraint_entities) != 3 or type(constraint_entities[2]) is not int:
                    raise ValueError('Arc / ellipse constraint requires an ellipse index.')
                index = constraint_entities[2]
                if not 0 <= index < len(entities) or entities[index].get('type') != 'ellipse':
                    raise ValueError('Arc / ellipse constraint references a missing ellipse.')
                ellipse = entities[index]
                if not ellipse.get('is_closed', True):
                    raise ValueError('This constraint requires a closed ellipse.')
                if not isinstance(ellipse.get('center'), str):
                    raise ValueError('Constrained ellipse requires a named center.')
                constraint_entities = [constraint_entities[0], constraint_entities[1], ellipse['center']]
                value = [ellipse['radius_x'], ellipse['radius_y'], ellipse.get('rotation', 0), value]
            if isinstance(value, list):
                value = tuple(number(item) for item in value)
            elif value is not None:
                value = number(value)
            parsed.append(Constraint(entry['kind'], tuples(expand_nurbs(constraint_entities)), value))
        solved = solve_constraints(named_points, radii, parsed)
        if solved.status == 'conflict':
            raise ValueError(f'Sketch constraints conflict (residual {solved.residual:.3g}).')
        named_points = solved.points
        # Capsule components hidden by the union are not valid boundary targets.
        from UI.model.constraints import composite_slot_contact
        from UI.model.tools.Slot import polyline_slot_edges
        import build123d as cad
        for constraint in parsed:
            entity = constraint.entities[2] if constraint.kind in ('tangent_arc_slot', 'curvature_arc_slot', 'tangent_spline_slot', 'curvature_spline_slot') else (constraint.entities[1] if constraint.kind == 'point_on_slot' else ())
            if entity and entity[0] == '@polyline_slot':
                radius, names = entity[1], entity[2:]
                component = constraint.entities[2] if constraint.kind == 'point_on_slot' else constraint.entities[3]
                contact, _, _ = composite_slot_contact([named_points[name] for name in names], radius, component, constraint.value)
                edges = polyline_slot_edges([(named_points[name][0], named_points[name][1], 0) for name in names], (0, 0, 1), radius)
                vertex = cad.Vertex(float(contact[0]), float(contact[1]), 0)
                if not edges or min(vertex.distance_to(edge) for edge in edges) > 1e-6:
                    raise ValueError('Selected composite slot contact is hidden inside the slot.')

    def point(value):
        if isinstance(value, str):
            if value not in named_points:
                raise ValueError(f'Unknown sketch point: {value}')
            value = named_points[value].tolist() if hasattr(named_points[value], 'tolist') else named_points[value]
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError('Sketch point must contain two coordinates.')
        return np.asarray(tuple(plane.from_local_coords((number(value[0]), number(value[1]), 0))), dtype=float)
    tools = []
    for index, entity in enumerate(entities):
        if not isinstance(entity, dict):
            raise ValueError('Invalid sketch entity.')
        kind = entity.get('type')
        construction = entity.get('construction', False)
        if type(construction) is not bool:
            raise ValueError('Construction flag must be true or false.')
        if kind == 'line':
            values = entity.get('points')
            if not isinstance(values, list) or len(values) < 2:
                raise ValueError('Line requires at least two points.')
            tool = SimpleNamespace(type='line', points=[point(value) for value in values])
        elif kind == 'spline':
            values = entity.get('points')
            if not isinstance(values, list) or len(values) != 4:
                raise ValueError('Cubic spline requires four control points.')
            control = [point(value) for value in values]
            if np.linalg.norm(control[1] - control[0]) < 1e-9 or np.linalg.norm(control[3] - control[2]) < 1e-9:
                raise ValueError('Spline endpoint tangent must be non-zero.')
            tool = SimpleNamespace(type='spline', points=control)
        elif kind == 'fit_spline':
            from UI.model.constraints import fit_spline, fit_spline_endpoint
            values = entity.get('points')
            if not isinstance(values, list) or not 3 <= len(values) <= 100:
                raise ValueError('Fit spline requires between 3 and 100 points.')
            control = [point(value) for value in values]
            fit_spline(control)
            fit_spline_endpoint(control, 0)
            fit_spline_endpoint(control, 1)
            tool = SimpleNamespace(type='fit_spline', points=control)
        elif kind == 'nurbs':
            from UI.model.nurbs import nurbs_data, nurbs_contact
            values = entity.get('points')
            if not isinstance(values, list): raise ValueError('NURBS requires control points.')
            control = [point(value) for value in values]
            knots = [number(x) for x in entity['knots']]
            weights = [number(x) for x in entity['weights']]
            control, degree, knots, weights = nurbs_data(control, entity['degree'], knots, weights)
            nurbs_contact(control, degree, knots, weights, 0)
            nurbs_contact(control, degree, knots, weights, 1)
            tool = SimpleNamespace(type='nurbs', points=control, degree=degree, knots=knots, weights=weights)
        elif kind == 'periodic_nurbs':
            from UI.model.nurbs import periodic_nurbs_data, periodic_nurbs_contact
            values = entity.get('points')
            if not isinstance(values, list): raise ValueError('Periodic NURBS requires control points.')
            control = [point(value) for value in values]
            weights = [number(x) for x in entity['weights']]
            spans = [number(x) for x in entity.get('spans', [1] * len(values))]
            periodic_nurbs_data(control, entity['degree'], weights, spans)
            periodic_nurbs_contact(control, entity['degree'], weights, 0, spans)
            tool = SimpleNamespace(type='periodic_nurbs', points=control, degree=entity['degree'], weights=weights, spans=spans)
        elif kind == 'circle':
            center = point(entity['center'])
            radius = solved.radii[circle_radius_name(index)] if solved is not None else number(entity['radius'])
            if radius <= 1e-6: raise ValueError('Circle radius must be positive.')
            tool = SimpleNamespace(type='circle', points=[], center=center, radius=radius)
        elif kind == 'arc':
            values = entity.get('points')
            if not isinstance(values, list) or len(values) != 3:
                raise ValueError('Arc requires start, end and intermediate points.')
            tool = SimpleNamespace(type='curve', points=[point(value) for value in values])
        elif kind == 'ellipse':
            center = point(entity['center'])
            a, b = number(entity['radius_x']), number(entity['radius_y'])
            if min(a, b) <= 1e-6: raise ValueError('Ellipse radii must be positive.')
            angle = math.radians(number(entity.get('rotation', 0)))
            x, y = np.asarray(tuple(plane.x_dir)), np.asarray(tuple(plane.y_dir))
            axis = np.cos(angle) * x + np.sin(angle) * y
            closed = entity.get('is_closed', True)
            if type(closed) is not bool: raise ValueError('Ellipse closure flag must be boolean.')
            start = math.radians(number(entity.get('start_angle', 0)))
            end = math.radians(number(entity.get('end_angle', 360)))
            if not closed and (end <= start or end-start >= 2*math.pi-1e-9):
                raise ValueError('Elliptical arc sweep must be between 0 and 360 degrees.')
            tool = SimpleNamespace(type='ellipse', points=[] if closed else [center]*5, center=center, major_radius=a,
                                   minor_radius=b, major_axis_vector=axis, is_closed=closed, start_angle=start, end_angle=end)
        elif kind == 'slot':
            values = entity.get('points')
            if not isinstance(values, list) or len(values) != 3:
                raise ValueError('Slot requires two centers and a width point.')
            tool = SimpleNamespace(type='slot', points=[point(value) for value in values])
        elif kind == 'polyline_slot':
            from UI.model.tools.Slot import polyline_slot_edges
            values = entity.get('points')
            if not isinstance(values, list) or not 2 <= len(values) <= 100:
                raise ValueError('Polyline slot requires 2 to 100 path points.')
            radius = number(entity['radius'])
            control = [point(value) for value in values]
            polyline_slot_edges(control, np.asarray(tuple(plane.z_dir)), radius)
            tool = SimpleNamespace(type='polyline_slot', points=control, radius=radius)
        elif kind == 'arc_slot':
            from UI.model.arc_slot import arc_slot_edges
            values = entity.get('points')
            if not isinstance(values, list) or len(values) != 3:
                raise ValueError('Arc slot requires start, end and intermediate path points.')
            radius = number(entity['radius'])
            control = [point(value) for value in values]
            arc_slot_edges(control, np.asarray(tuple(plane.z_dir)), radius)
            tool = SimpleNamespace(type='arc_slot', points=control, radius=radius)
        elif kind == 'spline_slot':
            from UI.model.spline_slot import spline_slot_edges, spline_slot_contact
            index = entity.get('curve')
            if type(index) is not int or not 0 <= index < len(entities) or entities[index].get('type') not in ('spline', 'fit_spline', 'nurbs'):
                raise ValueError('Spline slot requires a supported open spline path.')
            source = entities[index]
            values = source['points']
            radius = number(entity['radius'])
            path_kind, degree = source['type'], source.get('degree')
            knots = [number(x) for x in source.get('knots', [])]
            weights = [number(x) for x in source.get('weights', [])]
            local = [named_points[v] if isinstance(v, str) else [number(q) for q in v] for v in values]
            for t in np.linspace(0, 1, 129):
                spline_slot_contact(local, radius, 'left', float(t), path_kind, degree, knots, weights)
                spline_slot_contact(local, radius, 'right', float(t), path_kind, degree, knots, weights)
            control = [point(value) for value in values]
            spline_slot_edges(control, radius, path_kind, degree, knots, weights)
            tool = SimpleNamespace(type='spline_slot', points=control, radius=radius, path_kind=path_kind, degree=degree, knots=knots, weights=weights)
        elif kind == 'composite_path_slot':
            from UI.model.composite_slot import composite_slot_edges
            descriptors = path_descriptors(entity.get('curves'))
            segments = [(kind, [named_points[name] for name in names], degree, knots, weights)
                        for kind, names, degree, knots, weights in descriptors]
            radius = number(entity['radius'])
            composite_slot_edges(segments, plane, radius)
            tool = SimpleNamespace(type='composite_path_slot', points=[], segments=segments, plane=plane, radius=radius)
        else:
            raise ValueError(f'Unsupported sketch entity: {kind}')
        tool.is_construction = construction
        tools.append(tool)
    profiles = build_profiles(tools, np.asarray(tuple(plane.z_dir)))
    from UI.model.profile_identity import profile_keys
    keys = profile_keys(profiles, entities, tools, np.asarray(tuple(plane.z_dir)))
    return ParametricSketch(plane, tuple(profiles), solved, tuple(keys), tuple(tools))
