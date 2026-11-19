"""Two-segment selection and live corner preview in the sketch viewport."""
from collections import Counter
from uuid import uuid4

from UI.model.curve_spans import curve_interval
from UI.model.feature_graph import FeatureGraph
from UI.model.sketch_corners import build_corner, common_corner, corner_source_name


class SketchCornerSession:
    def __init__(self, manager, command, size):
        self.manager = manager
        self.command = command
        self.size = size
        self.prompt = f'{command.title()}: select the first segment at the corner.'
        self.selected = []
        self.preview = []
        self.preview_key = None
        self._load_sources()

    def _load_sources(self):
        manager = self.manager
        sketch = manager.history_edit_graph.results[manager.history_edit_id]
        sources = [(actor, corner_source_name(sketch.entities[index], segment, edge), edge)
                   for actor, (index, segment, edge, _) in manager.history_segment_actors.items()
                   if index >= 0]
        counts = Counter(name for _, name, _ in sources)
        self.actors = {actor: name for actor, name, _ in sources if name is not None and counts[name] == 1}
        self.edges = {name: edge for actor, name, edge in sources if actor in self.actors}
        self.plane = sketch.plane

    def clear_preview(self):
        for actor in self.preview:
            self.manager.plotter.remove_actor(actor, render=False)
        if self.preview:
            self.preview.clear()
            self.manager.plotter.render()
        self.preview_key = None

    def close(self):
        self.clear_preview()

    def _picked(self):
        picker = self.manager._pick_actor(self.actors, tolerance=.006)
        return self.actors.get(picker.GetActor()) if picker is not None else None

    def _show(self, names):
        size = self.size()
        key = tuple(names), size
        if key == self.preview_key:
            return
        self.clear_preview()
        self.preview_key = key
        manager = self.manager
        edges = [self.edges[name] for name in names]
        for edge in edges:
            self.preview.append(manager.draw_sketch_edge(edge, '#c58524', 4, pickable=False))
        message = self.prompt
        if self.selected:
            message = 'Select the second segment sharing an endpoint.'
        if len(edges) == 2:
            try:
                connector, cuts = build_corner(self.plane, edges, common_corner(*edges), self.command, size)
                for edge, interval in zip(edges, cuts):
                    part = curve_interval(edge, *interval)
                    self.preview.append(manager.draw_sketch_edge(part, '#199987', 5, pickable=False))
                self.preview.append(manager.draw_sketch_edge(connector, '#199987', 5, pickable=False))
            except (ValueError, RuntimeError) as exc:
                message = str(exc)
            else:
                message = 'Press Enter or click Apply Corner.' if len(self.selected) == 2 else 'Click to select this corner.'
        manager.status_callback(message)
        manager.plotter.render()

    def hover(self):
        names = self.selected.copy()
        if len(names) < 2:
            name = self._picked()
            if name is not None and name not in names:
                names.append(name)
        self._show(names)

    def refresh_preview(self):
        self._show(self.selected)

    def apply(self):
        name = self._picked()
        if name is None:
            return False
        if not self.selected or name == self.selected[0]:
            self.selected = [name] if not self.selected else []
        else:
            try:
                common_corner(self.edges[self.selected[0]], self.edges[name])
            except ValueError as exc:
                self.manager.status_callback(str(exc))
                return False
            self.selected = [self.selected[0], name]
        self.preview_key = None
        self.refresh_preview()
        return True

    def confirm(self):
        manager = self.manager
        if len(self.selected) != 2:
            manager.status_callback('Select both segments of the corner first.')
            return False
        graph = manager.history_edit_graph
        features = graph.features
        feature = next(item for item in features if item.id == manager.history_edit_id)
        ends = common_corner(*(self.edges[name] for name in self.selected))
        feature.parameters.setdefault('corners', []).append({
            'id': uuid4().hex, 'kind': self.command, 'size': self.size(),
            'sources': self.selected, 'ends': list(ends),
        })
        try:
            candidate = FeatureGraph(features, graph.parameters)
        except ValueError as exc:
            manager.status_callback(str(exc))
            return False
        self.clear_preview()
        self.selected.clear()
        manager.history_selection = None
        manager.history_edit_graph = candidate
        manager.render_history_sketch()
        self._load_sources()
        manager.status_callback(f'{self.command.title()} applied. {self.prompt}')
        return True
