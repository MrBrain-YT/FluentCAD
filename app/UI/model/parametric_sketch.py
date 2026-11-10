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


def build_sketch(plane, entities, number, points=None, constraints=None):
    if not isinstance(entities, list) or len(entities) > 1000:
        raise ValueError('Sketch entities must be a list with at most 1000 items.')
    solved = None
    named_points = {}
    if points is not None or constraints:
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
        parsed = []
        for entry in constraints or []:
            if not isinstance(entry, dict) or not isinstance(entry.get('kind'), str) or not isinstance(entry.get('entities'), list):
                raise ValueError('Invalid sketch constraint.')
            if entry.get('driven', False):
                raise ValueError('Driven dimensions are not yet supported in parametric sketches.')
            value = entry.get('value')
            if isinstance(value, list):
                value = tuple(number(item) for item in value)
            elif value is not None:
                value = number(value)
            parsed.append(Constraint(entry['kind'], tuples(entry['entities']), value))
        solved = solve_constraints(named_points, {}, parsed)
        if solved.status == 'conflict':
            raise ValueError(f'Sketch constraints conflict (residual {solved.residual:.3g}).')
        named_points = solved.points

    def point(value):
        if isinstance(value, str):
            if value not in named_points:
                raise ValueError(f'Unknown sketch point: {value}')
            value = named_points[value].tolist() if hasattr(named_points[value], 'tolist') else named_points[value]
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError('Sketch point must contain two coordinates.')
        return np.asarray(tuple(plane.from_local_coords((number(value[0]), number(value[1]), 0))), dtype=float)
    tools = []
    for entity in entities:
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
        elif kind == 'circle':
            center, radius = point(entity['center']), number(entity['radius'])
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
            tool = SimpleNamespace(type='ellipse', points=[], center=center, major_radius=a,
                                   minor_radius=b, major_axis_vector=axis, is_closed=True)
        elif kind == 'slot':
            values = entity.get('points')
            if not isinstance(values, list) or len(values) != 3:
                raise ValueError('Slot requires two centers and a width point.')
            tool = SimpleNamespace(type='slot', points=[point(value) for value in values])
        else:
            raise ValueError(f'Unsupported sketch entity: {kind}')
        tool.is_construction = construction
        tools.append(tool)
    profiles = build_profiles(tools, np.asarray(tuple(plane.z_dir)))
    return ParametricSketch(plane, tuple(profiles), solved)
