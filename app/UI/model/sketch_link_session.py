"""Scene selection and preview for associative sketch geometry."""
import numpy as np

from UI.model.edge_selection import face_edge_indices
from UI.model.feature_graph import FeatureGraph
from UI.model.ModelConverter import b123d_to_pro_mesh
from UI.model.sketch_links import make_sketch_link
from UI.model.sketch_projection_curves import projectable_sketch_curves


class SketchLinkSession:
    def __init__(self, manager, command, kind, source_id, replace_id=None):
        self.manager = manager
        self.command = command
        self.kind = kind
        self.source_id = source_id
        self.replace_id = replace_id
        self.shape = manager.history_edit_graph.results[source_id]
        self.source_actor = self.source_outline = None
        self.source_curves = {}
        if self.kind == 'sketch_curve':
            self.source_curves = {manager.draw_sketch_edge(edge, '#4f707c', 3): index
                                  for index, (_, edge) in enumerate(projectable_sketch_curves(self.shape))}
        else:
            self.source_mesh = b123d_to_pro_mesh(self.shape)
            self.source_actor = manager.plotter.add_mesh(
                self.source_mesh, name='sketch_link_source', color='#7d8fa2', opacity=.3,
                pickable=True, reset_camera=False, render=False,
            )
            outlines = self.source_mesh.extract_feature_edges(non_manifold_edges=False, manifold_edges=False)
            if outlines.n_cells:
                self.source_outline = manager.plotter.add_mesh(outlines, color='#586c7f', line_width=1,
                                                               pickable=False, reset_camera=False, render=False)
        self.preview_actors = []
        self.preview_key = self.link = None
        manager.plotter.render()

    def clear_preview(self, render=True):
        for actor in self.preview_actors:
            self.manager.plotter.remove_actor(actor, render=False)
        changed = bool(self.preview_actors)
        self.preview_actors.clear()
        self.preview_key = self.link = None
        if changed and render:
            self.manager.plotter.render()

    def close(self):
        self.clear_preview(render=False)
        if self.source_outline is not None:
            self.manager.plotter.remove_actor(self.source_outline, render=False)
        if self.source_actor is not None:
            self.manager.plotter.remove_actor(self.source_actor)
        else:
            for actor in self.source_curves:
                self.manager.plotter.remove_actor(actor, render=False)
            self.manager.plotter.render()

    def _selection_at_cursor(self):
        if self.kind == 'sketch_curve':
            picker = self.manager._pick_actor(self.source_curves, tolerance=.008)
            return self.source_curves.get(picker.GetActor()) if picker is not None else None
        picker = self.manager._pick_actor([self.source_actor], tolerance=.008)
        if picker is None or not 0 <= picker.GetCellId() < self.source_mesh.n_cells:
            return None
        face_index = int(self.source_mesh.cell_data['face_id'][picker.GetCellId()])
        if self.kind != 'edge':
            return face_index
        edges, indices = face_edge_indices(self.shape, face_index)
        return self.manager._nearest_edge(edges, indices)

    def hover(self):
        manager = self.manager
        index = self._selection_at_cursor()
        if index is None:
            self.clear_preview()
            return
        if index == self.preview_key:
            return
        self.clear_preview(render=False)
        self.preview_key = index
        try:
            graph = manager.history_edit_graph
            sketch = graph.results[manager.history_edit_id]
            self.link, geometry = make_sketch_link(
                self.command, self.kind, self.source_id, self.shape,
                index, graph.body_names.get(self.source_id), sketch.plane, link_id=self.replace_id,
            )
            for _, edge in geometry.edges:
                self.preview_actors.append(manager.draw_sketch_edge(edge, '#cc75ed', 5, pickable=False))
            if geometry.points:
                self.preview_actors.append(manager.plotter.add_points(
                    np.asarray([tuple(point) for point in geometry.points]), color='#cc75ed', point_size=10,
                    pickable=False, reset_camera=False, render=False,
                ))
            manager.status_callback('Replacement preview. Click to relink.' if self.replace_id
                                    else f'{self.command.title()} preview. Click to create the link.')
        except (ValueError, RuntimeError) as exc:
            self.link = None
            manager.status_callback(str(exc))
        manager.plotter.render()

    def apply(self):
        self.hover()
        if self.link is None:
            return False
        manager = self.manager
        graph, feature_id = manager.history_edit_graph, manager.history_edit_id
        params = next(feature.parameters for feature in graph.features if feature.id == feature_id)
        links = params.setdefault('links', [])
        selection_fields = ('kind', 'operation', 'source_id', 'body_id', 'reference', 'curve', 'curve_reference')
        if any(item['id'] != self.replace_id and all(item.get(field) == self.link.get(field) for field in selection_fields)
               for item in links):
            manager.status_callback('This geometry is already linked to the sketch.')
            return False
        if self.replace_id is None:
            links.append(self.link)
        else:
            index = next(i for i, item in enumerate(links) if item['id'] == self.replace_id)
            links[index] = self.link
        try:
            candidate = FeatureGraph.from_data(graph.to_data())
            candidate.edit(feature_id, parameters=params)
        except ValueError as exc:
            manager.status_callback(str(exc))
            return False
        self.clear_preview(render=False)
        manager.history_edit_graph = candidate
        manager.history_selection = None
        manager.render_history_sketch()
        if self.replace_id is not None:
            manager.stop_sketch_modifier()
            manager.status_callback('Sketch geometry relinked. Finish the sketch to apply.')
        else:
            manager.status_callback(f'{self.command.title()} linked. Continue selecting geometry or press Enter to finish.')
        return True
