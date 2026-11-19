"""Pick a chain in the sketch viewport and preview its associative offset."""
from collections import Counter
from uuid import uuid4

import build123d as cad
import numpy as np
import pyvista as pv

from UI.model.feature_graph import FeatureGraph
from UI.model.sketch_offset import build_offset_chain, offset_chain_closed, offset_source_name, order_offset_chain


class SketchOffsetSession:
    command = 'offset_chain'

    def __init__(self, manager, distance, chain_selection, corner_mode=None):
        self.manager = manager
        self.distance = distance
        self.chain_selection = chain_selection
        self.corner_mode = corner_mode or (lambda: 'auto')
        self.prompt = 'Offset: click a chain or select its segments, then press Enter to apply.'
        self.selected = []
        self.seed = None
        self.preview_key = None
        self.preview = []
        self._load_sources()
        if not self.edges:
            raise ValueError('Offset needs a line, conic or spline with named sketch points.')

    def _load_sources(self):
        manager = self.manager
        sketch = manager.history_edit_graph.results[manager.history_edit_id]
        corner_names = {name for name, _ in sketch.corner_edges}
        sources = [(actor, offset_source_name(sketch.entities[index], segment) if index >= 0 else segment, edge)
                   for actor, (index, segment, edge, _) in manager.history_segment_actors.items()
                   if index >= 0 or segment in corner_names]
        counts = Counter(name for _, name, _ in sources)
        self.actors = {actor: name for actor, name, _ in sources if name is not None and counts[name] == 1}
        self.edges = {name: edge for actor, name, edge in sources if actor in self.actors}
        self.chains = {}
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

    def _references(self, name):
        if self.chain_selection():
            if name not in self.chains:
                try:
                    self.chains[name] = order_offset_chain(self.edges, name, automatic=True)
                except ValueError as exc:
                    self.chains[name] = str(exc)
            chain = self.chains[name]
            if isinstance(chain, str):
                raise ValueError(chain)
            return chain
        current = [ref for ref, _ in self.selected]
        if name in current:
            current.remove(name)
        else:
            current.append(name)
        if not current:
            return []
        seed = self.seed if self.seed in current else current[0]
        return order_offset_chain({ref: self.edges[ref] for ref in current}, seed)

    def _draw(self, edges, color, width):
        points, cells = [], []
        for edge in edges:
            count = 2 if edge.geom_type == cad.GeomType.LINE else 97
            cells.extend([count, *range(len(points), len(points) + count)])
            points.extend(tuple(edge.position_at(t)) for t in np.linspace(0, 1, count))
        if points:
            mesh = pv.PolyData(np.asarray(points))
            mesh.lines = np.asarray(cells, dtype=np.int64)
            self.preview.append(self.manager.plotter.add_mesh(mesh, color=color, line_width=width,
                                pickable=False, reset_camera=False, render=False))

    def _show(self, references, message):
        distance, join = self.distance(), self.corner_mode()
        key = tuple(map(tuple, references)), distance, join, message
        if key == self.preview_key:
            return
        self.clear_preview()
        self.preview_key = key
        if references:
            self._draw((self.edges[name] for name, _ in references), '#c58524', 4)
            try:
                groups = build_offset_chain(self.plane, self.edges, references, distance, join=join)
                self._draw((edge for group in groups for edge in group), '#199987', 4)
            except (RuntimeError, ValueError) as exc:
                message = str(exc)
        self.manager.status_callback(message)
        self.manager.plotter.render()

    def hover(self):
        name = self._picked()
        try:
            if name is not None and (not self.selected or (
                not self.chain_selection() and name not in {ref for ref, _ in self.selected}
            )):
                self._show(self._references(name), self.prompt)
                return
        except ValueError as exc:
            self._show(self.selected, str(exc))
            return
        self.refresh_preview()

    def refresh_preview(self):
        self._show(self.selected, self.prompt)

    def reset_selection(self):
        self.selected = []
        self.seed = None
        self.hover()

    def apply(self):
        name = self._picked()
        if name is None:
            return False
        try:
            self.selected = self._references(name)
        except ValueError as exc:
            self._show(self.selected, str(exc))
            return False
        if self.seed not in {ref for ref, _ in self.selected}:
            self.seed = self.selected[0][0] if self.selected else None
        self.refresh_preview()
        return True

    def confirm(self):
        if not self.selected:
            self.manager.status_callback('Offset: select the source chain first.')
            return False
        manager = self.manager
        graph, feature_id = manager.history_edit_graph, manager.history_edit_id
        features = graph.features
        feature = next(item for item in features if item.id == feature_id)
        feature.parameters.setdefault('offsets', []).append({
            'id': uuid4().hex, 'sources': self.selected, 'distance': self.distance(),
            'closed': offset_chain_closed(self.edges, self.selected), 'join': self.corner_mode(),
        })
        try:
            candidate = FeatureGraph(features, graph.parameters)
        except ValueError as exc:
            manager.status_callback(str(exc))
            return False
        self.clear_preview()
        self.selected = []
        self.seed = None
        manager.history_edit_graph = candidate
        manager.history_selection = None
        manager.render_history_sketch()
        self._load_sources()
        manager.status_callback(f'Associative offset applied. {self.prompt}')
        return True
