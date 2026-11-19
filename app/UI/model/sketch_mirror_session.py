"""Viewport selection and preview for an associative sketch mirror."""
from collections import Counter
from uuid import uuid4

import build123d as cad
import numpy as np

from UI.model.feature_graph import FeatureGraph
from UI.model.sketch_mirror import reflect_sketch
from UI.model.sketch_projection_curves import sketch_curve_name


class SketchMirrorSession:
    command = 'sketch_mirror'
    prompt = 'Mirror: select curves and points, then press Enter or click Select Axis.'

    def __init__(self, manager):
        self.manager = manager
        self.selected = []
        self.selecting_axis = False
        self.axis = None
        self.preview = []
        self.preview_key = None
        self._load_sources()

    def _load_sources(self):
        manager = self.manager
        sketch = manager.history_edit_graph.results[manager.history_edit_id]
        self.plane = sketch.plane
        candidates = []
        for actor, (index, segment, edge, _) in manager.history_segment_actors.items():
            if index >= 0:
                name = sketch_curve_name(sketch.entities[index], segment)
                construction = sketch.tools[index].is_construction
            elif isinstance(segment, str):
                name = segment
                if self.command == 'sketch_mirror' and name.startswith('pattern:'):
                    continue
                construction = name in sketch.mirror_construction_edges or name in sketch.pattern_construction_edges
            else:
                continue
            candidates.append((actor, name, edge, construction))
        counts = Counter(name for _, name, _, _ in candidates)
        self.actors = {actor: name for actor, name, _, _ in candidates if counts[name] == 1}
        self.edges = {name: (edge, construction) for actor, name, edge, construction in candidates
                      if actor in self.actors}
        self.points = dict(sketch.mirror_points)
        if self.command != 'sketch_mirror':
            self.points.update(sketch.pattern_points)
        if sketch.constraint_status is not None:
            self.points.update((name, manager.get_global_pos_in_sketch(point))
                               for name, point in sketch.constraint_status.points.items())

    def clear_preview(self):
        for actor in self.preview:
            self.manager.plotter.remove_actor(actor, render=False)
        if self.preview:
            self.preview.clear()
            self.manager.plotter.render()
        self.preview_key = None

    def close(self):
        self.clear_preview()

    def _picked_curve(self):
        actors = self.actors
        if self.selecting_axis:
            selected = {name for kind, name in self.selected if kind == 'curve'}
            actors = {actor: name for actor, name in actors.items()
                      if name not in selected and self.edges[name][0].geom_type == cad.GeomType.LINE}
        picker = self.manager._pick_actor(actors, tolerance=.006)
        return actors.get(picker.GetActor()) if picker is not None else None

    def _picked_source(self):
        manager = self.manager
        name = manager._history_handle_at_cursor(named_only=True)
        if name is None:
            picker = manager._pick_actor(manager.history_mirror_point_actors, tolerance=.006)
            if picker is not None:
                name = manager.history_mirror_point_actors.get(picker.GetActor())
        if name in self.points:
            return 'point', name
        name = self._picked_curve()
        return ('curve', name) if name is not None else None

    def _draw_points(self, points, color):
        vertices = np.asarray([tuple(point) for point in points])
        if len(vertices):
            self.preview.append(self.manager.plotter.add_points(vertices, color=color, point_size=15,
                                pickable=False, reset_camera=False, render=False))

    def _show(self, hovered=None, axis=None):
        key = tuple(self.selected), self.selecting_axis, hovered, axis, self.axis
        if key == self.preview_key:
            return
        self.clear_preview()
        self.preview_key = key
        manager = self.manager
        highlights = [*self.selected, hovered] if hovered is not None and hovered not in self.selected else self.selected
        for kind, name in highlights:
            if kind == 'curve':
                self.preview.append(manager.draw_sketch_edge(self.edges[name][0], '#c58524', 4, pickable=False))
        self._draw_points((self.points[name] for kind, name in highlights if kind == 'point'), '#c58524')
        message = 'Select a straight segment for the mirror axis.' if self.selecting_axis else self.prompt
        if axis is not None:
            line = self.edges[axis][0]
            self.preview.append(manager.draw_sketch_edge(line, '#4499c4', 4, pickable=False))
            try:
                reflected = reflect_sketch(self.plane, line,
                                           [self.edges[name] for kind, name in self.selected if kind == 'curve'],
                                           [self.points[name] for kind, name in self.selected if kind == 'point'], 'preview')
                for _, edge in reflected.edges:
                    self.preview.append(manager.draw_sketch_edge(edge, '#199987', 4, pickable=False))
                self._draw_points((point for _, point in reflected.points), '#199987')
                if reflected.edges or reflected.points:
                    message = ('Press Enter or click Apply Mirror.' if axis == self.axis
                               else 'Click to select this mirror axis.')
                else:
                    message = 'The selected geometry already lies on or is symmetric about this axis.'
            except (ValueError, RuntimeError) as exc:
                message = str(exc)
        manager.status_callback(message)
        manager.plotter.render()

    def hover(self):
        if self.selecting_axis:
            self._show(axis=self._picked_curve() or self.axis)
        else:
            self._show(hovered=self._picked_source())

    def choose_axis(self):
        if not self.selected:
            self.manager.status_callback('Select at least one curve or point to mirror.')
            return False
        self.selecting_axis = True
        self.hover()
        return True

    def apply(self):
        if self.selecting_axis:
            name = self._picked_curve()
            if name is None:
                return False
            self.axis = name
        else:
            source = self._picked_source()
            if source is None:
                return False
            if source in self.selected:
                self.selected.remove(source)
            else:
                self.selected.append(source)
        self.hover()
        return True

    def confirm(self):
        if not self.selecting_axis:
            return self.choose_axis()
        manager = self.manager
        if self.axis is None:
            manager.status_callback('Click a straight sketch segment to select the mirror axis.')
            return False
        graph = manager.history_edit_graph
        features = graph.features
        feature = next(item for item in features if item.id == manager.history_edit_id)
        identifier = uuid4().hex
        feature.parameters.setdefault('mirrors', []).append({
            'id': identifier, 'axis': self.axis,
            'curves': [name for kind, name in self.selected if kind == 'curve'],
            'points': [name for kind, name in self.selected if kind == 'point'],
        })
        try:
            candidate = FeatureGraph(features, graph.parameters)
        except ValueError as exc:
            manager.status_callback(str(exc))
            return False
        sketch = candidate.results[feature.id]
        if not any(name.startswith(f'mirror:{identifier}:') for name, _ in (*sketch.mirror_edges, *sketch.mirror_points)):
            manager.status_callback('The selected geometry already lies on or is symmetric about this axis.')
            return False
        self.clear_preview()
        self.selected.clear()
        self.axis = None
        self.selecting_axis = False
        manager.history_selection = None
        manager.history_edit_graph = candidate
        manager.render_history_sketch()
        self._load_sources()
        manager.status_callback(f'Mirror applied. {self.prompt}')
        return True
