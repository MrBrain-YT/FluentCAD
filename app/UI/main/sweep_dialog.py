"""Scene selection and a transactional preview for a sketch sweep."""
import numpy as np
from PySide6 import QtWidgets

from UI.model.feature_graph import Feature, FeatureGraph
from UI.model.ModelConverter import b123d_to_pro_mesh
from UI.model.parametric_sketch import ParametricSketch
from UI.model.sketch_sweep import sweep_curves


class SweepDialog(QtWidgets.QDialog):
    def __init__(self, manager, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Sweep')
        self.manager = manager
        self.graph = manager.parametric_graph
        self.profile = None
        self.path = []
        self.choosing_profile = True
        self.actors = {}
        self.hidden = []
        self.preview = None
        self.candidate = None
        self.result_shape = None
        self.curves = dict(sweep_curves(self.graph.results))
        self.sections = [(feature_id, index, face)
                         for feature_id, sketch in self.graph.results.items() if isinstance(sketch, ParametricSketch)
                         for index, face in enumerate(sketch.profiles)]
        if not self.sections or not self.curves:
            raise ValueError('Create a closed sketch profile and sketch curves for the sweep path.')
        layout = QtWidgets.QVBoxLayout(self)
        self.message = QtWidgets.QLabel(self)
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        choose_profile = QtWidgets.QPushButton('Select Profile', self)
        choose_profile.clicked.connect(self.select_profile)
        layout.addWidget(choose_profile)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel, self)
        self.ok = buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.finished.connect(self._cleanup)
        plotter = self.manager.plotter
        for actor in [*(plotter.actors.get(profile['name']) for profile in manager.profiles),
                      *manager.sketch_spatial_actors]:
            if actor is not None and actor.GetVisibility():
                actor.SetVisibility(False)
                self.hidden.append(actor)
        self.select_profile()

    def _clear_actors(self):
        for actor in self.actors:
            self.manager.plotter.remove_actor(actor, render=False)
        self.actors.clear()

    def _clear_preview(self):
        if self.preview is not None:
            self.manager.plotter.remove_actor(self.preview, render=False)
            self.preview = None
        self.candidate = self.result_shape = None
        self.ok.setEnabled(False)

    def _cleanup(self, result):
        self._clear_actors()
        if self.preview is not None:
            self.manager.plotter.remove_actor(self.preview, render=False)
            self.preview = None
        for actor in self.hidden:
            actor.SetVisibility(True)
        self.hidden.clear()
        self.manager.plotter.render()

    def select_profile(self):
        self._clear_actors()
        self._clear_preview()
        self.choosing_profile = True
        self.profile = None
        self.path.clear()
        for feature_id, index, face in self.sections:
            actor = self.manager.plotter.add_mesh(b123d_to_pro_mesh(face), color='#3498db', opacity=.48,
                                                   reset_camera=False, render=False)
            self.actors[actor] = feature_id, index
        self.message.setText('Click a closed sketch profile in the scene.')
        self.manager.plotter.reset_camera_clipping_range()
        self.manager.plotter.render()

    def scene_click(self):
        picker = self.manager._pick_actor(self.actors, tolerance=.006)
        if picker is None:
            return
        actor = picker.GetActor()
        selection = self.actors.get(actor)
        if selection is None:
            return
        if self.choosing_profile:
            self.profile = selection
            self.choosing_profile = False
            self._clear_actors()
            feature_id, index = self.profile
            actor = self.manager.plotter.add_mesh(
                b123d_to_pro_mesh(self.graph.results[feature_id].profiles[index]), color='#3498db',
                opacity=.25, pickable=False, reset_camera=False, render=False)
            self.actors[actor] = None
            for ref, edge in self.curves.items():
                actor = self.manager.draw_sketch_edge(edge, '#258399', 3)
                self.actors[actor] = ref
            self.manager.plotter.reset_camera_clipping_range()
        elif selection in self.path:
            self.path.remove(selection)
            actor.GetProperty().SetColor(.15, .51, .60)
        else:
            self.path.append(selection)
            actor.GetProperty().SetColor(1., .69, .23)
        self._refresh_preview()

    def _refresh_preview(self):
        self._clear_preview()
        if not self.path:
            self.message.setText('Click the segments of a connected path. Click again to remove a segment.')
        else:
            try:
                source_id, index = self.profile
                selection = self.manager.profile_selection_parameters(self.graph.results[source_id], index)
                selection['path'] = [list(ref) for ref in self.path]
                if self.graph.latest_solid_id:
                    selection['base_id'] = self.graph.latest_solid_id
                feature = Feature.create('Sweep', 'sweep', selection, source_id)
                candidate = FeatureGraph([*self.graph.features, feature], self.graph.parameters)
                shape = candidate.results[feature.id]
                self.preview = self.manager.plotter.add_mesh(b123d_to_pro_mesh(shape.solids()[-1]), color='#1ca989', opacity=.42,
                                                              pickable=False, reset_camera=False, render=False)
                self.candidate, self.result_shape = candidate, shape
                self.ok.setEnabled(True)
                self.message.setText(f'{len(self.path)} path segments selected. Click OK to apply the sweep.')
            except (ValueError, RuntimeError) as exc:
                self.message.setText(str(exc))
        self.manager.plotter.render()
