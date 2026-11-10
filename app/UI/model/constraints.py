"""Numerical constraint engine for independent 2-D sketch geometry.

This module does not mutate UI objects; callers must explicitly apply a successful result.
"""
from dataclasses import dataclass
import math
import numpy as np
from scipy.optimize import least_squares
from UI.model.expressions import evaluate_expression


@dataclass(frozen=True)
class Constraint:
    kind: str
    entities: tuple
    value: object = None
    driven: bool = False


@dataclass(frozen=True)
class SolveResult:
    points: dict
    radii: dict
    degrees_of_freedom: int
    status: str
    residual: float
    redundant: bool


def solve_constraints(points, radii, constraints, parameters=None, tolerance=1e-7):
    """Solve point coordinates and circle radii; report conflicts without changing inputs.

    Line entities are pairs of point names; circles are (center point, radius name).
    Angles are in degrees. Fixed values are captured explicitly as (x, y).
    """
    point_names, radius_names = list(points), list(radii)
    if set(point_names) & set(radius_names):
        raise ValueError('Point and radius names must be distinct.')
    coords = [np.asarray(points[name], dtype=float) for name in point_names]
    if any(p.shape != (2,) or not np.isfinite(p).all() for p in coords):
        raise ValueError('Every sketch point must have two finite coordinates.')
    initial_radii = [float(radii[name]) for name in radius_names]
    if any(not math.isfinite(r) or r <= 0 for r in initial_radii):
        raise ValueError('Circle radii must be positive and finite.')
    x0 = np.asarray([v for p in coords for v in p] + initial_radii, dtype=float)
    if not len(x0):
        raise ValueError('The sketch has no variables.')
    pmap = {name: i * 2 for i, name in enumerate(point_names)}
    rmap = {name: 2 * len(point_names) + i for i, name in enumerate(radius_names)}
    parameters = parameters or {}
    active = [c for c in constraints if not c.driven]

    def residual(x):
        result = []
        def p(name):
            if name not in pmap: raise ValueError(f'Unknown point: {name}')
            i = pmap[name]
            return x[i:i + 2]
        def line(entity):
            a, b = entity
            return p(a), p(b)
        def direction(entity):
            a, b = line(entity)
            v = b - a
            norm = np.linalg.norm(v)
            if norm < 1e-12: raise ValueError('A constrained line has zero length.')
            return v / norm
        def circle(entity):
            center, radius = entity
            if radius not in rmap: raise ValueError(f'Unknown radius: {radius}')
            return p(center), x[rmap[radius]]
        def target(c):
            value = evaluate_expression(c.value, parameters) if isinstance(c.value, str) else float(c.value)
            if not math.isfinite(value): raise ValueError('Constraint value must be finite.')
            return value
        def cross(a, b): return a[0] * b[1] - a[1] * b[0]
        for c in active:
            e, kind = c.entities, c.kind
            if kind == 'coincident': result.extend(p(e[0]) - p(e[1]))
            elif kind in ('horizontal', 'vertical'):
                a, b = line(e[0]); result.append((b - a)[1 if kind == 'horizontal' else 0])
            elif kind in ('parallel', 'perpendicular'):
                a, b = direction(e[0]), direction(e[1])
                result.append(cross(a, b) if kind == 'parallel' else np.dot(a, b))
            elif kind == 'collinear':
                a, b = line(e[0]); c1, d = line(e[1]); u = direction(e[0])
                result.extend((cross(u, c1 - a), cross(u, d - a)))
            elif kind == 'concentric': result.extend(circle(e[0])[0] - circle(e[1])[0])
            elif kind == 'equal_length':
                a, b = line(e[0]); c1, d = line(e[1]); result.append(np.linalg.norm(b - a) - np.linalg.norm(d - c1))
            elif kind == 'equal_radius': result.append(circle(e[0])[1] - circle(e[1])[1])
            elif kind == 'midpoint':
                a, b = line(e[1]); result.extend(p(e[0]) - (a + b) / 2)
            elif kind == 'fixed':
                fixed = np.asarray(c.value, dtype=float)
                if fixed.shape != (2,) or not np.isfinite(fixed).all(): raise ValueError('Fixed point requires two finite coordinates.')
                result.extend(p(e[0]) - fixed)
            elif kind == 'symmetric':
                a, b = p(e[0]), p(e[1]); origin, _ = line(e[2]); u = direction(e[2]); mid = (a + b) / 2
                result.extend((cross(u, mid - origin), np.dot(b - a, u)))
            elif kind == 'distance': result.append(np.linalg.norm(p(e[0]) - p(e[1])) - target(c))
            elif kind == 'horizontal_distance': result.append(p(e[1])[0] - p(e[0])[0] - target(c))
            elif kind == 'vertical_distance': result.append(p(e[1])[1] - p(e[0])[1] - target(c))
            elif kind in ('radius', 'diameter'):
                r = circle(e[0])[1]; result.append(r * (2 if kind == 'diameter' else 1) - target(c))
            elif kind == 'angle':
                a, b = direction(e[0]), direction(e[1]); angle = math.radians(target(c))
                result.extend((np.dot(a, b) - math.cos(angle), cross(a, b) - math.sin(angle)))
            elif kind == 'tangent_line_circle':
                origin, _ = line(e[0]); u = direction(e[0]); center, radius = circle(e[1])
                result.append(abs(cross(u, center - origin)) - radius)
            elif kind in ('tangent_circles_external', 'tangent_circles_internal'):
                a, ra = circle(e[0]); b, rb = circle(e[1]);
                result.append(np.linalg.norm(a - b) - (ra + rb if kind.endswith('external') else abs(ra - rb)))
            else: raise ValueError(f'Unsupported constraint: {kind}')
        return np.asarray(result, dtype=float)

    # Validate constraints before optimization and avoid mutating supplied data.
    initial_residual = residual(x0)
    if not len(initial_residual):
        return SolveResult(dict(zip(point_names, [p.copy() for p in coords])), dict(radii), len(x0), 'underconstrained', 0., False)
    lower = np.full(len(x0), -np.inf)
    lower[2 * len(point_names):] = 1e-9
    solution = least_squares(residual, x0, bounds=(lower, np.full(len(x0), np.inf)),
                             xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=2000)
    error = float(np.max(np.abs(residual(solution.x))))
    rank = int(np.linalg.matrix_rank(solution.jac, tol=1e-7))
    dof = max(0, len(x0) - rank)
    status = 'conflict' if not solution.success or error > tolerance else ('underconstrained' if dof else 'fully_constrained')
    values = x0 if status == 'conflict' else solution.x
    solved_points = {name: values[i:i + 2].copy() for name, i in pmap.items()}
    solved_radii = {name: float(values[i]) for name, i in rmap.items()}
    return SolveResult(solved_points, solved_radii, dof, status, error, len(initial_residual) > rank)
