"""Viewport picking and annotations for sketch dimensions."""
import numpy as np
import pyvista as pv

from UI.model.sketch_dimensions import (
    LINEAR, dimension_annotation, dimension_from_selection, update_sketch_dimension,
)
from UI.model.sketch_constraints import segment_reference


COLOR = '#7649a8'
PREVIEW = '#ae6217'


def draw_dimension(manager, annotation, color=COLOR):
    anchor, paths, label = annotation
    vertices, cells = [], []
    for path in paths:
        cells.extend([len(path), *range(len(vertices), len(vertices) + len(path))])
        vertices.extend(manager.get_global_pos_in_sketch(point) for point in path)
    mesh = pv.PolyData(np.asarray(vertices))
    mesh.lines = np.asarray(cells, dtype=np.int64)
    world = manager.get_global_pos_in_sketch(anchor)
    lines = manager.plotter.add_mesh(mesh, color=color, line_width=2, pickable=False,
                                    reset_camera=False, render=False)
    text = manager.plotter.add_point_labels(np.asarray([world]), [label], font_size=13,
        text_color='white', shape_color=color, always_visible=True, show_points=False,
        pickable=False, reset_camera=False, render=False)
    return (lines, text), (world, label)


class SketchDimensionSession:
    command = 'dimension'

    def __init__(self, manager, mode, edit_value):
        self.manager = manager
        self.mode = mode
        self.edit_value = edit_value
        self.first = None
        self.first_edge = None
        self.pending = None
        self.preview = []
        self.preview_key = None
        self.editing = False

    def prompt(self):
        if self.mode == 'angle':
            return 'Select the first line for the angle.' if self.first is None else 'Select the second line for the angle.'
        if self.mode in ('radius', 'diameter'):
            return 'Select a circle or an arc, or click an existing dimension to edit it.'
        if self.first is not None:
            return 'Select the second point.'
        if self.mode in LINEAR:
            return 'Select a line or two points, or click an existing dimension to edit it.'
        return 'Select a line, circle, arc or two points. Click an existing dimension to edit it.'

    def clear_preview(self):
        for actor in self.preview:
            self.manager.plotter.remove_actor(actor, render=False)
        if self.preview:
            self.preview.clear()
            self.manager.plotter.render()
        self.preview_key = None

    def close(self):
        self.clear_preview()

    def _selection(self, entities):
        manager = self.manager
        if self.mode == 'auto' or self.mode in LINEAR:
            point = manager._history_handle_at_cursor(named_only=True)
            if point is not None:
                return ('point', (point,)), None
            if self.first is not None:
                return None, None
        picker = manager._pick_actor(manager.history_segment_actors, tolerance=.006)
        segment = manager.history_segment_actors.get(picker.GetActor()) if picker is not None else None
        if segment is None:
            return None, None
        index, offset, edge, _ = segment
        reference = segment_reference(entities, index, offset)
        if reference is None:
            return None, None
        allowed = {'angle': ('line',), 'radius': ('circle', 'arc'), 'diameter': ('circle', 'arc')}
        if reference[0] not in allowed.get(self.mode, ('line', 'arc', 'circle') if self.mode == 'auto' else ('line',)):
            return None, None
        return reference, edge

    def _label_at_cursor(self):
        manager = self.manager
        x, y = manager.plotter.iren.get_event_position()
        renderer = manager.plotter.renderer
        nearest = None
        best = float('inf')
        for index, (world, label) in manager.history_dimension_labels.items():
            renderer.SetWorldPoint(*world, 1.)
            renderer.WorldToDisplay()
            px, py, pz = renderer.GetDisplayPoint()
            dx, dy = x - px, y - py
            if 0 <= pz <= 1 and -4 <= dx <= 11 * len(label) + 5 and abs(dy) <= 16:
                distance = dx * dx + dy * dy
                if distance < best:
                    nearest, best = index, distance
        return nearest

    def _sketch(self):
        graph, feature_id = self.manager.history_edit_graph, self.manager.history_edit_id
        feature = next(item for item in graph.features if item.id == feature_id)
        return graph.results[feature_id].constraint_status, feature.parameters

    def _placement(self):
        return self.manager.get_local_pos_in_sketch(self.manager._screen_to_sketch_plane())

    def _highlight(self, reference, edge, state):
        manager = self.manager
        if edge is not None:
            self.preview.append(manager.draw_sketch_edge(edge, PREVIEW, 4, pickable=False))
        elif reference is not None:
            world = manager.get_global_pos_in_sketch(state.points[reference[1][0]])
            self.preview.append(manager.plotter.add_points(world, color=PREVIEW, point_size=15,
                pickable=False, reset_camera=False, render=False))

    def hover(self):
        if self.editing:
            return
        manager = self.manager
        state, params = self._sketch()
        try:
            if self.pending is not None:
                placement = self._placement()
                self.clear_preview()
                dimension = {**self.pending, 'placement': list(map(float, placement))}
                actors, _ = draw_dimension(manager, dimension_annotation(dimension, state, {}), PREVIEW)
                self.preview.extend(actors)
                manager.plotter.render()
                return
            label = self._label_at_cursor() if self.first is None else None
            reference, edge = self._selection(params.get('entities', [])) if label is None else (None, None)
            key = label, reference
            if key == self.preview_key:
                return
            self.clear_preview()
            self.preview_key = key
            self._highlight(self.first, self.first_edge, state)
            if reference != self.first:
                self._highlight(reference, edge, state)
            manager.status_callback('Click to edit this dimension.' if label is not None else self.prompt())
            manager.plotter.render()
        except (ValueError, RuntimeError) as exc:
            self.clear_preview()
            manager.status_callback(str(exc))

    def apply(self):
        if self.editing:
            return False
        manager = self.manager
        state, params = self._sketch()
        try:
            if self.pending is not None:
                dimension = {**self.pending, 'placement': list(map(float, self._placement()))}
                return self._edit(dimension)
            index = self._label_at_cursor()
            if index is not None and self.first is None:
                return self._edit(params['constraints'][index], index)
            reference, edge = self._selection(params.get('entities', []))
            if reference is None or reference == self.first:
                return False
            selected = [reference] if self.first is None else [self.first, reference]
            dimension = dimension_from_selection(self.mode, selected, state)
            if dimension is None:
                if self.first is None and (self.mode == 'angle' or reference[0] == 'point'):
                    self.first, self.first_edge = reference, edge
                    self.preview_key = None
                    self.hover()
                else:
                    manager.status_callback('Select distinct geometry for this dimension.')
                return False
            self.pending = dimension
            manager.status_callback('Move the pointer to place the dimension, then click to enter its value.')
            self.hover()
        except (ValueError, RuntimeError) as exc:
            manager.status_callback(str(exc))
        return False

    def _edit(self, dimension, index=None):
        manager = self.manager
        self.clear_preview()
        self.first = self.first_edge = self.pending = None
        self.editing = True
        graph, feature_id = manager.history_edit_graph, manager.history_edit_id
        last = None

        def preview(text):
            nonlocal last
            if last is not None and last[0] == text:
                return last[1]
            candidate = update_sketch_dimension(graph, feature_id, dimension, text, index)
            manager.render_history_sketch(candidate)
            last = text, candidate
            return candidate

        try:
            candidate = self.edit_value(dimension, preview)
            if candidate is not None:
                manager.history_edit_graph = candidate
                manager.history_selection = None
            return candidate is not None
        finally:
            self.editing = False
            manager.render_history_sketch()
            manager.status_callback(self.prompt())
