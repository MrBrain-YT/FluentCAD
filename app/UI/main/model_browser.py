"""Model tree and visibility controls."""
from PySide6 import QtCore, QtWidgets
from UI.model.body_visibility import describe_body, resolve_body_reference


class ModelBrowser(QtWidgets.QTreeWidget):
    def __init__(self, parent, *, overlay=True):
        super().__init__(parent)
        self.setObjectName('modelBrowser')
        self.setHeaderLabel('Model browser')
        self.setAccessibleName('Model browser')
        self.setUniformRowHeights(True)
        self.setIndentation(17)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.manager = None
        self.hidden_bodies = set()
        self.hidden_body_refs = []
        self.hidden_body_ids = []
        self._body_actor = None
        self._body_mesh = None
        self.itemChanged.connect(self._visibility_changed)
        if overlay:
            parent.installEventFilter(self)
            self.reposition()

    def eventFilter(self, watched, event):
        if watched is self.parentWidget() and event.type() in (QtCore.QEvent.Resize, QtCore.QEvent.Show):
            self.reposition()
        return super().eventFilter(watched, event)

    def reposition(self):
        parent = self.parentWidget()
        self.setGeometry(8, 8, min(220, max(1, parent.width() // 3)), max(1, parent.height() - 16))
        self.raise_()

    def refresh(self, manager):
        self.manager = manager
        self.blockSignals(True)
        expanded = {self.topLevelItem(i).text(0): self.topLevelItem(i).isExpanded()
                    for i in range(self.topLevelItemCount())}
        self.clear()
        groups = {name: QtWidgets.QTreeWidgetItem(self, [name]) for name in ('Origin', 'Bodies', 'Sketches', 'Construction planes')}
        for name in ('XY', 'XZ', 'YZ'):
            item = QtWidgets.QTreeWidgetItem(groups['Origin'], [name])
            self._actor_checkbox(item, f'helper_{name}')
        if manager.current_part is not None:
            actor = manager.plotter.actors.get('main_part')
            if actor is not self._body_actor:
                self._body_actor = actor
                self._body_mesh = actor.mapper.dataset if actor is not None else None
                self._resolve_hidden_bodies()
                if self.hidden_bodies:
                    self._apply_body_visibility()
            for index, _ in enumerate(manager.current_part.solids()):
                item = QtWidgets.QTreeWidgetItem(groups['Bodies'], [f'Body {index + 1}'])
                item.setData(0, QtCore.Qt.UserRole, ('body', index))
                item.setCheckState(0, QtCore.Qt.Unchecked if index in self.hidden_bodies else QtCore.Qt.Checked)
        else:
            self._body_actor = self._body_mesh = None
            self.hidden_bodies.clear()
        graph = manager.parametric_graph
        if graph is not None:
            for feature in graph.features:
                if 'sketch' in feature.kind:
                    item = QtWidgets.QTreeWidgetItem(groups['Sketches'], [feature.name])
                    if feature.kind == 'planar_sketch' and feature.id in graph.results:
                        actors = self._sketch_actors(manager, feature.id)
                        item.setData(0, QtCore.Qt.UserRole, ('sketch', feature.id))
                        item.setCheckState(0, QtCore.Qt.Checked if any(actor is not None and actor.GetVisibility() for actor in actors) else QtCore.Qt.Unchecked)
                elif feature.kind == 'datum_plane':
                    QtWidgets.QTreeWidgetItem(groups['Construction planes'], [feature.name])
        else:
            for index, _ in enumerate(manager.sketchs_data, 1):
                QtWidgets.QTreeWidgetItem(groups['Sketches'], [f'Sketch {index}'])
        for plane in manager.construction_planes:
            item = QtWidgets.QTreeWidgetItem(groups['Construction planes'], [plane.get('name', 'Plane')])
            self._actor_checkbox(item, plane.get('name'))
        for name, item in groups.items():
            item.setExpanded(expanded.get(name, name in ('Bodies', 'Sketches')))
        self.blockSignals(False)

    def _actor_checkbox(self, item, name):
        self._actors_checkbox(item, [name] if name else [])

    @staticmethod
    def _sketch_actors(manager, feature_id):
        actors = [manager.plotter.actors.get(profile['name'])
                  for profile in manager.profiles if profile.get('sketch_id') == feature_id]
        if feature_id == getattr(manager, 'spatial_sketch_id', None):
            actors.extend(getattr(manager, 'sketch_spatial_actors', ()))
        return [actor for actor in actors if actor is not None]

    def export_visibility(self):
        manager = self.manager
        planes = [i for i, plane in enumerate(manager.construction_planes)
                  if (actor := manager.plotter.actors.get(plane['name'])) is not None and actor.GetVisibility()]
        sketch = next((p.get('sketch_id') for p in manager.profiles
                       if p.get('sketch_id') and (actor := manager.plotter.actors.get(p['name'])) is not None and actor.GetVisibility()), None)
        spatial_id = getattr(manager, 'spatial_sketch_id', None)
        if sketch is None and spatial_id and any(actor.GetVisibility() for actor in self._sketch_actors(manager, spatial_id)):
            sketch = spatial_id
        return {'hidden_bodies': [], 'hidden_body_refs': list(self.hidden_body_refs), 'hidden_body_ids': list(self.hidden_body_ids),
                'visible_planes': planes, 'visible_sketch': sketch}

    def reset_visibility(self):
        self.hidden_bodies.clear()
        self.hidden_body_refs.clear()
        self.hidden_body_ids.clear()

    def _current_body_ids(self):
        graph = self.manager.parametric_graph
        bodies = list(self.manager.current_part.solids())
        if graph is None:
            return [None] * len(bodies)
        feature = next((item for item in reversed(graph.features) if item.id in graph.results
                        and hasattr(graph.results[item.id], 'solids') and graph.results[item.id].solids()), None)
        if feature is None: return [None] * len(bodies)
        source = list(graph.results[feature.id].solids())
        names = graph.body_names.get(feature.id, ())
        output = []
        for body in bodies:
            matches = [i for i, item in enumerate(source) if item.is_same(body)]
            output.append(names[matches[0]] if len(matches) == 1 and matches[0] < len(names) else None)
        return output

    def _resolve_hidden_bodies(self):
        self.hidden_bodies = set()
        if self.manager.current_part is None:
            return
        bodies = list(self.manager.current_part.solids())
        unresolved = 0
        names = self._current_body_ids()
        for name in self.hidden_body_ids:
            matches = [i for i, candidate in enumerate(names) if candidate == name]
            if len(matches) == 1: self.hidden_bodies.add(matches[0])
            else: unresolved += 1
        for reference in self.hidden_body_refs:
            try:
                self.hidden_bodies.add(resolve_body_reference(reference, bodies))
            except ValueError:
                unresolved += 1
        remaining = []
        for reference in self.hidden_body_refs:
            try:
                index = resolve_body_reference(reference, bodies)
                name = names[index]
                if name is not None and names.count(name) == 1:
                    if name not in self.hidden_body_ids: self.hidden_body_ids.append(name)
                    continue
            except ValueError:
                pass
            remaining.append(reference)
        self.hidden_body_refs = remaining
        if unresolved:
            self.manager.status_callback(f'{unresolved} hidden body reference(s) could not be resolved; visibility was not guessed.')

    def restore_visibility(self, manager, settings):
        self.refresh(manager)
        bodies = list(manager.current_part.solids()) if manager.current_part is not None else []
        self.hidden_body_refs = list(settings.get('hidden_body_refs', []))
        self.hidden_body_ids = list(settings.get('hidden_body_ids', []))
        if not self.hidden_body_refs and not self.hidden_body_ids:
            self.hidden_body_refs = [describe_body(bodies[i]) for i in settings['hidden_bodies'] if i < len(bodies)]
        self._resolve_hidden_bodies()
        self._apply_body_visibility()
        for i, plane in enumerate(manager.construction_planes):
            actor = manager.plotter.actors.get(plane['name'])
            if actor is not None:
                actor.SetVisibility(i in settings['visible_planes'])
        feature_id = settings['visible_sketch']
        if feature_id and manager.parametric_graph is not None and feature_id in manager.parametric_graph.results:
            try:
                manager.show_parametric_profiles(feature_id)
            except ValueError as exc:
                manager.status_callback(str(exc))
        self.refresh(manager)
        manager.plotter.render()

    def _actors_checkbox(self, item, names):
        actors = [self.manager.plotter.actors.get(name) for name in names]
        actors = [actor for actor in actors if actor is not None]
        if actors:
            item.setData(0, QtCore.Qt.UserRole, ('actors', names))
            item.setCheckState(0, QtCore.Qt.Checked if any(actor.GetVisibility() for actor in actors) else QtCore.Qt.Unchecked)

    def _visibility_changed(self, item, column):
        data = item.data(0, QtCore.Qt.UserRole)
        if not data or self.manager is None:
            return
        visible = item.checkState(0) == QtCore.Qt.Checked
        if data[0] == 'actors':
            for name in data[1]:
                actor = self.manager.plotter.actors.get(name)
                if actor is not None: actor.SetVisibility(visible)
        elif data[0] == 'sketch':
            if self.manager.sketch_editing:
                self.refresh(self.manager)
                return
            if visible:
                try:
                    self.manager.show_parametric_profiles(data[1])
                except ValueError as exc:
                    self.manager.status_callback(str(exc))
            else:
                for actor in self._sketch_actors(self.manager, data[1]):
                    actor.SetVisibility(False)
            self.refresh(self.manager)
        elif data[0] == 'body':
            bodies = list(self.manager.current_part.solids())
            names = self._current_body_ids()
            name = names[data[1]]
            if visible:
                self.hidden_bodies.discard(data[1])
                if name is not None:
                    self.hidden_body_ids = [candidate for candidate in self.hidden_body_ids if candidate != name]
                self.hidden_body_refs = [ref for ref in self.hidden_body_refs
                                         if not self._reference_matches_index(ref, bodies, data[1])]
            else:
                if name is not None and names.count(name) == 1:
                    self.hidden_body_ids.append(name)
                    self.hidden_bodies.add(data[1])
                    self._apply_body_visibility()
                    self.manager.plotter.render()
                    return
                reference = describe_body(bodies[data[1]])
                try:
                    resolve_body_reference(reference, bodies)
                except ValueError as exc:
                    self.manager.status_callback(str(exc))
                    self.refresh(self.manager)
                    return
                self.hidden_bodies.add(data[1])
                self.hidden_body_refs.append(reference)
            self._apply_body_visibility()
        self.manager.plotter.render()

    @staticmethod
    def _reference_matches_index(reference, bodies, index):
        try:
            return resolve_body_reference(reference, bodies) == index
        except ValueError:
            return False

    def _apply_body_visibility(self):
        import numpy as np
        if self._body_actor is None or self._body_mesh is None:
            return
        bodies = list(self.manager.current_part.solids())
        faces = list(self.manager.current_part.faces())
        visible_faces = [i for i, face in enumerate(faces)
                         if any(j not in self.hidden_bodies and any(face.is_same(f) for f in body.faces())
                                for j, body in enumerate(bodies))]
        if not visible_faces:
            self._body_actor.SetVisibility(False)
        else:
            mask = np.isin(self._body_mesh.cell_data['face_id'], visible_faces)
            mesh = self._body_mesh if not self.hidden_bodies else self._body_mesh.extract_cells(np.flatnonzero(mask)).extract_surface(algorithm='dataset_surface')
            self._body_actor.mapper.SetInputData(mesh)
            self._body_actor.SetVisibility(True)
        self.manager.selected_face_id = -1
        self.manager.selected_face_ids.clear()
        self.manager._clear_highlight(render=False)
