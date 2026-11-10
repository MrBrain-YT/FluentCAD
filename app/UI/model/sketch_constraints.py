"""Bridge line endpoints in an active sketch to the numerical solver."""
import numpy as np
from UI.model.constraints import Constraint, solve_constraints


def line_points(manager):
    sketch = manager.get_current_sketch_data()
    result = {}
    for ti, tool in enumerate(sketch['sketch_func']):
        if tool.type != 'line':
            continue
        for pi, point in enumerate(tool.points):
            result[f'L{ti + 1}.P{pi + 1}'] = (tool, pi, manager.get_local_pos_in_sketch(np.asarray(point)))
    return result


def add_automatic_coincidences(sketch, points, tolerance=1e-7):
    known = sketch.setdefault('constraint_known_points', set())
    constraints = sketch.setdefault('constraints', [])
    # Only newly created endpoints may acquire automatic coincidence links.
    for name, (_, _, position) in points.items():
        if name in known:
            continue
        matches = [other for other in known if other in points and
                   np.linalg.norm(position - points[other][2]) <= tolerance]
        if matches:
            constraints.append(Constraint('coincident', (sorted(matches)[0], name)))
        known.add(name)


def apply_constraint(manager, constraint):
    sketch = manager.get_current_sketch_data()
    points = line_points(manager)
    if not points:
        raise ValueError('Create a line or a polygon first.')
    add_automatic_coincidences(sketch, points)
    existing = sketch.setdefault('constraints', [])
    result = solve_constraints({name: item[2] for name, item in points.items()}, {},
                               existing + [constraint])
    if result.status == 'conflict':
        raise ValueError(f'Constraints conflict (residual {result.residual:.3g}); geometry was not changed.')
    # Compute and validate every new point before modifying the sketch.
    updates = [(tool, index, manager.get_global_pos_in_sketch(value))
               for name, value in result.points.items()
               for tool, index, _ in [points[name]]]
    for tool, index, position in updates:
        tool.points[index] = position
    existing.append(constraint)
    return result
