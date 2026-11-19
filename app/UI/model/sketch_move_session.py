"""Select native sketch objects and preview a validated Move / Copy."""
import numpy as np

from UI.model.profiles import sketch_edges
from UI.model.sketch_move import transform_sketch_selection


class SketchMoveSession:
    command = 'sketch_move'
    prompt = 'Move / Copy: click curves and points, then enter the displacement and angle.'

    def __init__(self, manager, settings):
        self.manager = manager
        self.settings = settings
        self.selected = []
        self.preview = []
        self.preview_key = None
        self.candidate = None

    def clear_preview(self):
        for actor in self.preview:
            self.manager.plotter.remove_actor(actor, render=False)
        if self.preview:
            self.manager.plotter.render()
        self.preview.clear()
        self.preview_key = None
        self.candidate = None

    def close(self):
        self.clear_preview()

    def _pick(self):
        manager = self.manager
        name = manager._history_handle_at_cursor(named_only=True)
        if name is not None:
            return 'point', name
        picker = manager._pick_actor(manager.history_entity_actors, tolerance=.006)
        if picker is not None:
            index = manager.history_entity_actors.get(picker.GetActor())
            if index is not None:
                return 'entity', index
        return None

    def refresh_preview(self, hovered=None):
        manager = self.manager
        graph, feature_id = manager.history_edit_graph, manager.history_edit_id
        settings = self.settings()
        key = tuple(self.selected), hovered, settings
        if key == self.preview_key:
            return
        self.clear_preview()
        self.preview_key = key
        sketch = graph.results[feature_id]
        message = self.prompt
        selected = self.selected + ([hovered] if hovered is not None and hovered not in self.selected else [])
        for kind, ref in selected:
            if kind == 'entity':
                self._draw_edges(sketch_edges([sketch.tools[ref]], manager.active_normal), '#c58524', 4)
            else:
                self._draw_points(sketch, [ref], '#c58524')
        if self.selected and self.can_preview(settings):
            try:
                self.candidate, copy = self.transform(graph, feature_id, settings)
                result = self.candidate.results[feature_id]
                if copy:
                    tools = result.tools[len(sketch.tools):]
                    points = set(result.constraint_status.points) - set(sketch.constraint_status.points)
                    edges = sketch_edges(tools, manager.active_normal)
                    edges.extend(edge for _, edge in result.corner_edges[len(sketch.corner_edges):])
                else:
                    points = [name for name, point in result.constraint_status.points.items()
                              if np.linalg.norm(point - sketch.constraint_status.points[name]) > 1e-7]
                    edges = sketch_edges(result.tools, manager.active_normal)
                    edges.extend(edge for _, edge in (*result.corner_edges, *result.offset_edges,
                                                       *result.mirror_edges, *result.pattern_edges))
                self._draw_edges(edges, '#199987', 4)
                self._draw_points(result, points, '#199987')
                message = 'Press Enter or click Apply Transform.'
            except (ValueError, RuntimeError) as exc:
                self.candidate = None
                message = str(exc)
        manager.status_callback(message)
        manager.plotter.render()

    def can_preview(self, settings):
        displacement, angle, _ = settings
        return any(displacement) or angle % 360

    def transform(self, graph, feature_id, settings):
        return transform_sketch_selection(graph, feature_id, self.selected, *settings), settings[2]

    def _draw_edges(self, edges, color, width):
        for edge in edges:
            self.preview.append(self.manager.draw_sketch_edge(edge, color, width, pickable=False))

    def _draw_points(self, sketch, names, color):
        points = [self.manager.get_global_pos_in_sketch(sketch.constraint_status.points[name]) for name in names]
        if points:
            self.preview.append(self.manager.plotter.add_points(
                np.asarray(points), color=color, point_size=14, pickable=False, reset_camera=False, render=False))

    def hover(self):
        self.refresh_preview(None if self.selected else self._pick())

    def apply(self):
        picked = self._pick()
        if picked is None:
            return False
        if picked in self.selected:
            self.selected.remove(picked)
        else:
            self.selected.append(picked)
        self.refresh_preview()
        return True

    def confirm(self):
        self.refresh_preview()
        if self.candidate is None:
            return False
        self.manager.history_edit_graph = self.candidate
        self.clear_preview()
        self.selected.clear()
        self.manager.history_selection = None
        self.manager.render_history_sketch()
        self.manager.status_callback('Transform applied. ' + self.prompt)
        return True
