"""Select a sketch pattern's sources and reference in the scene."""
from uuid import uuid4

import build123d as cad

from UI.model.feature_graph import FeatureGraph
from UI.model.sketch_mirror_session import SketchMirrorSession
from UI.model.sketch_pattern import build_sketch_pattern


class SketchPatternSession(SketchMirrorSession):
    command = 'sketch_pattern'

    def __init__(self, manager, kind, settings):
        self.kind = kind
        self.settings = settings
        self.reference = None
        label = 'Direction' if kind == 'rectangular' else 'Center'
        self.prompt = f'{kind.title()} Pattern: select curves and points, then click Select {label}.'
        super().__init__(manager)

    def choose_reference(self):
        if not self.selected:
            self.manager.status_callback(self.prompt)
            return False
        self.selecting_axis = True
        self.reference = None
        self.refresh_preview()
        return True

    def _pick_direction(self):
        actors = {actor: name for actor, name in self.actors.items()
                  if self.edges[name][0].geom_type == cad.GeomType.LINE}
        picker = self.manager._pick_actor(actors, tolerance=.006)
        return actors.get(picker.GetActor()) if picker is not None else None

    def _pick_center(self):
        source = self._picked_source()
        if source is not None and source[0] == 'point':
            return source[1]
        manager = self.manager
        _, local = manager.get_snapped_pos(manager._screen_to_sketch_plane())
        return tuple(map(float, local))

    def apply(self):
        if not self.selecting_axis:
            return super().apply()
        try:
            reference = self._pick_direction() if self.kind == 'rectangular' else self._pick_center()
        except (ValueError, RuntimeError) as exc:
            self.manager.status_callback(str(exc))
            return False
        if reference is None:
            return False
        self.reference = reference
        self.refresh_preview()
        return True

    def hover(self):
        if not self.selecting_axis:
            super().hover()
        elif self.kind == 'rectangular':
            self.refresh_preview(self._pick_direction())

    def refresh_preview(self, hovered_direction=None):
        values = self.settings()
        reference = hovered_direction or self.reference
        if not self.selecting_axis:
            self._show()
            return
        key = tuple(self.selected), reference, self.reference, values
        if key == self.preview_key:
            return
        self.clear_preview()
        self.preview_key = key
        manager = self.manager
        edges = [self.edges[name] for kind, name in self.selected if kind == 'curve']
        points = [self.points[name] for kind, name in self.selected if kind == 'point']
        for edge, _ in edges:
            self.preview.append(manager.draw_sketch_edge(edge, '#c58524', 4, pickable=False))
        self._draw_points(points, '#c58524')
        message = ('Click a straight sketch segment for the direction.' if self.kind == 'rectangular'
                   else 'Click a point or a location on the sketch plane for the center.')
        if reference is not None:
            try:
                if self.kind == 'rectangular':
                    geometry = self.edges[reference][0]
                    self.preview.append(manager.draw_sketch_edge(geometry, '#4499c4', 4, pickable=False))
                else:
                    geometry = (self.points[reference] if isinstance(reference, str)
                                else self.plane.from_local_coords((*reference, 0)))
                    self._draw_points([geometry], '#4499c4')
                result = build_sketch_pattern(self.plane, 'preview', self.kind, edges, points, geometry, values)
                for _, edge in result.edges:
                    self.preview.append(manager.draw_sketch_edge(edge, '#199987', 4, pickable=False))
                self._draw_points((point for _, point in result.points), '#199987')
                message = ('Click to select this direction.' if reference != self.reference
                           else 'Press Enter or click Apply Pattern.')
                if not result.edges and not result.points:
                    message = 'All pattern instances coincide with the source.'
            except (ValueError, RuntimeError) as exc:
                message = str(exc)
        manager.status_callback(message)
        manager.plotter.render()

    def confirm(self):
        if not self.selecting_axis:
            return self.choose_reference()
        manager = self.manager
        if self.reference is None:
            self.refresh_preview()
            return False
        graph = manager.history_edit_graph
        features = graph.features
        feature = next(item for item in features if item.id == manager.history_edit_id)
        identifier = uuid4().hex
        entry = {'id': identifier, 'kind': self.kind,
                 'curves': [name for kind, name in self.selected if kind == 'curve'],
                 'points': [name for kind, name in self.selected if kind == 'point']}
        if self.kind == 'rectangular':
            entry['direction'] = self.reference
            fields = ('count_x', 'count_y', 'spacing_x', 'spacing_y')
        else:
            entry['center'] = list(self.reference) if isinstance(self.reference, tuple) else self.reference
            fields = ('count', 'angle')
        entry.update(zip(fields, self.settings()))
        feature.parameters.setdefault('patterns', []).append(entry)
        try:
            candidate = FeatureGraph(features, graph.parameters)
        except ValueError as exc:
            manager.status_callback(str(exc))
            return False
        sketch = candidate.results[feature.id]
        if not any(name.startswith(f'pattern:{identifier}:') for name, _ in (*sketch.pattern_edges, *sketch.pattern_points)):
            manager.status_callback('All pattern instances coincide with the source.')
            return False
        self.clear_preview()
        self.selected.clear()
        self.reference = None
        self.selecting_axis = False
        manager.history_selection = None
        manager.history_edit_graph = candidate
        manager.render_history_sketch()
        self._load_sources()
        manager.status_callback('Pattern applied. ' + self.prompt)
        return True
