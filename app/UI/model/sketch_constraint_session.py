"""Select sketch geometry, preview a solve and apply it on click."""
import math

import numpy as np
import pyvista as pv

from UI.model.profiles import sketch_edges
from UI.model.scene_constraint_rules import ConstraintTarget, SCENE_CONSTRAINTS, SELECTIONS, constraint_from_targets
from UI.model.sketch_constraints import segment_reference
from UI.model.sketch_drag import add_sketch_constraint, remove_sketch_constraint


INPUT_COLOR = '#c58524'
SOLUTION_COLOR = '#199987'


class SketchConstraintSession:
    command = 'constraint'

    def __init__(self, manager, mode):
        self.manager = manager
        self.mode = mode
        self.title = SCENE_CONSTRAINTS[mode]
        self.patterns = SELECTIONS[mode]
        feature = next(item for item in manager.history_edit_graph.features if item.id == manager.history_edit_id)
        self.entities = feature.parameters.get('entities', [])
        self.selected = []
        self.preview = []
        self.preview_key = None
        self.candidate = None
        self.message = ''

    def _allowed(self):
        prefix = tuple(target.kind for target, _ in self.selected)
        return tuple(dict.fromkeys(pattern[len(prefix)] for pattern in self.patterns
                                   if len(pattern) > len(prefix) and pattern[:len(prefix)] == prefix))

    def prompt(self):
        names = {'point': 'point', 'line': 'line segment', 'arc': 'circular arc',
                 'circle': 'circle', 'spline': 'open spline'}
        options = ' or '.join(names[kind] for kind in self._allowed())
        if self.mode == 'symmetric' and len(self.selected) == 2:
            options = 'line segment for the symmetry axis'
        hint = ' Choose arcs and splines near the desired endpoint.' if self.mode in ('tangent', 'curvature') else ''
        return f'{self.title}: select a {options}.{hint}'

    def clear_preview(self):
        for actor in self.preview:
            self.manager.plotter.remove_actor(actor, render=False)
        if self.preview:
            self.preview.clear()
            self.manager.plotter.render()
        self.preview_key = None
        self.candidate = None

    def close(self):
        self.clear_preview()

    def _pick(self, state):
        manager = self.manager
        allowed = self._allowed()
        excluded = {(target.kind, target.refs) for target, _ in self.selected}
        if 'point' in allowed:
            point = manager._history_handle_at_cursor(
                named_only=True, excluded={refs[0] for kind, refs in excluded if kind == 'point'})
            if point is not None:
                return ConstraintTarget('point', (point,)), None
        if allowed == ('point',):
            return None
        targets = {}
        for actor, (index, segment, edge, _) in manager.history_segment_actors.items():
            reference = segment_reference(self.entities, index, segment, splines='spline' in allowed)
            if reference is not None and reference[0] in allowed and reference not in excluded:
                targets[actor] = reference, index, edge
        picker = manager._pick_actor(targets, tolerance=.006)
        if picker is None:
            return None
        (kind, refs), index, edge = targets[picker.GetActor()]
        endpoint, angle = None, None
        if self.mode in ('tangent', 'curvature'):
            position = manager.get_local_pos_in_sketch(picker.GetPickPosition())
            if kind in ('arc', 'spline'):
                names = self.entities[index]['points']
                ends = (names[0], names[1] if kind == 'arc' else names[-1])
                endpoint = min(ends, key=lambda name: np.linalg.norm(state.points[name] - position))
            elif kind == 'circle' and (not self.selected or self.selected[0][0].kind == 'spline'):
                try:
                    position = manager.get_local_pos_in_sketch(manager._screen_to_sketch_plane())
                except ValueError:
                    pass
                dx, dy = position - state.points[refs[0]]
                angle = round(math.degrees(math.atan2(dy, dx)))
        return ConstraintTarget(kind, refs, endpoint, angle), edge

    def _highlight(self, target, edge, state):
        manager = self.manager
        if edge is not None:
            self.preview.append(manager.draw_sketch_edge(edge, INPUT_COLOR, 4, pickable=False))
        name = target.refs[0] if target.kind == 'point' else target.endpoint
        if name is not None:
            world = manager.get_global_pos_in_sketch(state.points[name])
            self.preview.append(manager.plotter.add_points(world, color=INPUT_COLOR, point_size=15,
                                pickable=False, reset_camera=False, render=False))

    def _draw_solution(self, sketch, initial):
        manager = self.manager
        changed_points = [manager.get_global_pos_in_sketch(point)
                          for name, point in sketch.constraint_status.points.items()
                          if np.linalg.norm(point - initial.points[name]) > 1e-6]
        changed_radii = any(abs(radius - initial.radii[name]) > 1e-6
                            for name, radius in sketch.constraint_status.radii.items())
        if not changed_points and not changed_radii:
            return
        vertices, cells = [], []
        for edge in sketch_edges(sketch.tools, manager.active_normal):
            steps = 2 if edge.geom_type.name == 'LINE' else 65
            cells.extend([steps, *range(len(vertices), len(vertices) + steps)])
            vertices.extend(tuple(edge.position_at(t)) for t in np.linspace(0, 1, steps))
        if vertices:
            mesh = pv.PolyData(np.asarray(vertices))
            mesh.lines = np.asarray(cells, dtype=np.int64)
            self.preview.append(manager.plotter.add_mesh(mesh, color=SOLUTION_COLOR, line_width=3,
                                pickable=False, reset_camera=False, render=False))
        if changed_points:
            self.preview.append(manager.plotter.add_points(np.asarray(changed_points), color=SOLUTION_COLOR,
                                point_size=10, pickable=False, reset_camera=False, render=False))

    def _preview(self, picked, state):
        key = tuple(target for target, _ in self.selected), picked[0] if picked is not None else None
        if key == self.preview_key:
            return
        self.clear_preview()
        self.preview_key = key
        manager = self.manager
        self.message = self.prompt()
        for target, edge in self.selected + ([picked] if picked is not None else []):
            self._highlight(target, edge, state)
        if picked is not None:
            try:
                targets = [target for target, _ in self.selected] + [picked[0]]
                entry = constraint_from_targets(self.mode, targets, state.points)
                if entry is not None:
                    self._solve(entry, state)
            except (ValueError, RuntimeError) as exc:
                self.message = str(exc)
        manager.status_callback(self.message)
        manager.plotter.render()

    def _solve(self, entry, state):
        manager = self.manager
        graph, feature_id = manager.history_edit_graph, manager.history_edit_id
        feature = next(item for item in graph.features if item.id == feature_id)
        existing = feature.parameters.get('constraints', [])
        label = self.title
        if self.mode == 'fixed':
            current = next((i for i, item in enumerate(existing)
                            if item['kind'] == 'fixed' and item['entities'] == entry['entities']), None)
            label = 'Unfix point' if current is not None else 'Fix point'
        else:
            current = None
        if current is not None:
            candidate = remove_sketch_constraint(graph, feature_id, current)
        else:
            if entry in existing:
                raise ValueError('This constraint already exists.')
            candidate = add_sketch_constraint(graph, feature_id, entry)
        self._draw_solution(candidate.results[feature_id], state)
        self.candidate = candidate
        self.message = f'{label} preview. Click to apply.'

    def hover(self):
        manager = self.manager
        state = manager.history_edit_graph.results[manager.history_edit_id].constraint_status
        if state is not None:
            self._preview(self._pick(state), state)

    def apply(self):
        manager = self.manager
        state = manager.history_edit_graph.results[manager.history_edit_id].constraint_status
        picked = self._pick(state) if state is not None else None
        if picked is None:
            return False
        self._preview(picked, state)
        if self.candidate is None:
            kinds = tuple(target.kind for target, _ in self.selected) + (picked[0].kind,)
            if kinds not in self.patterns:
                self.selected.append(picked)
                self._preview(None, state)
            return False
        candidate = self.candidate
        self.clear_preview()
        self.selected.clear()
        manager.history_edit_graph = candidate
        manager.history_selection = None
        manager.render_history_sketch()
        manager.status_callback(f'{self.title} applied. {self.prompt()}')
        return True
