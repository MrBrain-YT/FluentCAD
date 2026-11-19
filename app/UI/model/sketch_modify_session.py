"""Select, preview and commit segment edits in the active sketch viewport."""
import numpy as np
import pyvista as pv

import build123d as cad
from UI.model.curve_spans import curve_fraction_at_point
from UI.model.feature_graph import FeatureGraph
from UI.model.sketch_modify import (
    NATIVE_CURVES, extend_curve_span, extend_edge, modified_sketch_parameters, offset_edge,
    source_curve_span, trim_at, trim_sections,
)


class SketchModifySession:
    def __init__(self, manager, command, distance):
        self.manager = manager
        self.command = command
        self.distance = distance
        self.actor = None
        self.source = None
        self.sections = None
        self.preview_key = None
        self.parameters = None

    def clear_preview(self):
        if self.actor is not None:
            self.manager.plotter.remove_actor(self.actor, render=False)
            self.actor = None
            self.manager.plotter.render()

    def close(self):
        self.clear_preview()

    def hover(self):
        manager = self.manager
        picker = manager._pick_actor(manager.history_segment_actors, tolerance=.006)
        actor = picker.GetActor() if picker is not None else None
        selected = manager.history_segment_actors.get(actor)
        if selected is None:
            self.clear_preview()
            self.preview_key = self.parameters = None
            return
        index, segment, edge, editable = selected
        native = edge.geom_type in NATIVE_CURVES
        if not editable or (self.command == 'extend' and edge.is_closed) or (self.command == 'offset' and native):
            self.clear_preview()
            self.preview_key = self.parameters = None
            manager.status_callback({
                'extend': 'Choose an open line, arc or spline.',
                'trim': 'Choose a line, conic or spline.',
                'offset': 'Offset copy requires a line, circular arc or circle. Use Offset for splines and ellipses.',
            }[self.command])
            return
        try:
            graph, feature_id = manager.history_edit_graph, manager.history_edit_id
            sketch = graph.results[feature_id]
            if actor != self.source:
                self.source, self.sections = actor, None
            boundaries = [item[2] for other, item in manager.history_segment_actors.items() if other != actor]
            if self.command == 'offset':
                distance = self.distance()
                key = actor, distance
                if key == self.preview_key:
                    return
                edit = offset_edge(edge, distance, tuple(manager.active_normal))
            else:
                _, _, point = cad.Vertex(*picker.GetPickPosition()).distance_to_with_closest_points(edge)
                position = curve_fraction_at_point(edge, point) if native else edge.param_at_point(point)
                if self.command == 'trim':
                    if self.sections is None:
                        self.sections = trim_sections(edge, boundaries)
                    edit = trim_at(self.sections, position)
                    key = actor, id(edit)
                else:
                    key = actor, position < .5
                    if key == self.preview_key:
                        return
                    if native:
                        _, base, start, end = source_curve_span(sketch, index, segment)
                        edit = extend_curve_span(base, start, end, boundaries, position < .5)
                    else:
                        edit = extend_edge(edge, boundaries, position < .5, tuple(manager.active_normal))
            if key == self.preview_key:
                return
            feature = next(item for item in graph.features if item.id == feature_id)
            params = modified_sketch_parameters(feature.parameters, sketch, index, segment,
                                                edge, edit, copy_only=self.command == 'offset')
            points = np.array([tuple(edit.preview.position_at(t)) for t in np.linspace(0, 1, 65)])
            if self.actor is not None:
                manager.plotter.remove_actor(self.actor, render=False)
            self.actor = manager.plotter.add_mesh(
                pv.lines_from_points(points), color='#e25848' if self.command == 'trim' else '#1ca989',
                line_width=5, pickable=False, reset_camera=False, render=False,
            )
            self.preview_key, self.parameters = key, params
            manager.status_callback(f'{self.command.title()} preview. Click to apply or press Enter to finish.')
            manager.plotter.render()
        except (ValueError, RuntimeError) as exc:
            self.clear_preview()
            self.preview_key = self.parameters = None
            manager.status_callback(str(exc))

    def apply(self):
        self.hover()
        if self.parameters is None:
            return False
        manager = self.manager
        try:
            candidate = FeatureGraph.from_data(manager.history_edit_graph.to_data())
            candidate.edit(manager.history_edit_id, parameters=self.parameters)
        except ValueError as exc:
            manager.status_callback(str(exc))
            return False
        self.clear_preview()
        self.source = self.sections = self.preview_key = self.parameters = None
        manager.history_edit_graph = candidate
        manager.history_selection = None
        manager.render_history_sketch()
        manager.status_callback(f'{self.command.title()} applied. Continue selecting segments or press Enter to finish.')
        return True
