from typing import Union
from pathlib import Path
import json

from PySide6 import QtWidgets, QtCore, QtGui
from pyvistaqt import QtInteractor
import pyvista as pv
import numpy as np
import vtk

from UI.main.icons import tool_icon
from UI.main.parametric_sketch_dialog import ParametricSketchDialog
from UI.main.panel_inputs import PanelInputs
from UI.main.operation_panel import OperationPanel
from UI.main.extrude_dialog import ExtrudeDialog, RevolveDialog, PrimitiveDialog, TransformDialog, SolidFeatureDialog
from UI.model.ActionManager import ActionManager
from UI.model.export import export_body
from UI.model.solid_operations import import_step_body
from UI.model.snapshot import save_snapshot, load_snapshot
from UI.model.feature_graph import Feature, FeatureGraph
from UI.model.geometry_references import make_reference
from UI.model.measurement import measure_face, measure_model, measure_edges
from UI.model.profiles import sketch_edges
from UI.main.ActionManager import ActionManagerUI

class LayoutManager:
    
    def __init__(self, window: QtWidgets.QMainWindow, central_widget: QtWidgets.QWidget):
        self.tool_tabs_widgets = {}
        self.window = window
        self.central_widget = central_widget
        self.main_layout = self.get_main_layout()
        self.plotter: Union[QtInteractor, pv.Plotter] = None 
        self.action_manager = ActionManager(self.plotter)
        self.action_manager.status_callback = lambda message: self.window.statusBar().showMessage(message)
        self.add_splitter()
        self.add_3d_viewport()
        self.panel_inputs = PanelInputs(self)
        self.setup_cad_viewport()
        self.toggle_sidebar()
        self.add_style()
        self.action_manager.update_plotter(self.plotter)
        self.action_manager_ui = ActionManagerUI(self.tool_tabs_widgets)
        self.action_manager.timeline_callback = self.refresh_timeline
        self.refresh_timeline()
        self.shortcuts = []
        def cancel_selection():
            was_selecting_plane = self.action_manager.selecting_sketch_plane
            cancelled = self.action_manager.cancel_pending_selection()
            if cancelled and was_selecting_plane:
                self.tool_tabs.setCurrentWidget(self.tool_tabs_widgets["solid"])
            elif not cancelled and self.action_manager.sketch_editing and self.action_manager.active_tool is not None:
                self.disable_active_tool()
                self.action_manager.status_callback("Sketch tool stopped.")
            elif not cancelled and self.action_manager.history_edit_id is not None:
                self.action_manager.cancel_history_sketch()
                self.tool_tabs.setCurrentWidget(self.tool_tabs_widgets['solid'])
        escape = QtGui.QShortcut(QtGui.QKeySequence("Escape"), self.window)
        escape.activated.connect(cancel_selection)
        self.shortcuts.append(escape)
        def finish_spline():
            tool = self.action_manager.active_tool
            if self.action_manager.sketch_editing and tool is not None and getattr(tool, 'type', '') in ('spline','fit_spline','nurbs','periodic_nurbs'):
                tool.finish_drawing()
        for key in ('Return', 'Enter'):
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self.window)
            shortcut.activated.connect(finish_spline)
            self.shortcuts.append(shortcut)
        for key, callback in [("Ctrl+Z", self.action_manager.undo_solid),
                              ("Ctrl+Y", self.action_manager.redo_solid),
                              ("Ctrl+Shift+Z", self.action_manager.redo_solid)]:
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self.window)
            shortcut.activated.connect(callback)
            self.shortcuts.append(shortcut)
        view_names = ("top", "bottom", "front", "back", "right", "left", "isometric")
        for index, name in enumerate(view_names, 1):
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(f"Ctrl+Alt+{index}"), self.window)
            shortcut.activated.connect(lambda view=name: self.set_standard_view(view))
            self.shortcuts.append(shortcut)

    def set_standard_view(self, name):
        if self.action_manager.sketch_editing:
            self.action_manager.status_callback("Finish the sketch before changing the viewing direction.")
            return
        views = {
            "top": ((0, 0, 1), (0, 1, 0)), "bottom": ((0, 0, -1), (0, 1, 0)),
            "front": ((0, -1, 0), (0, 0, 1)), "back": ((0, 1, 0), (0, 0, 1)),
            "right": ((1, 0, 0), (0, 0, 1)), "left": ((-1, 0, 0), (0, 0, 1)),
        }
        if name == "isometric":
            self.plotter.view_isometric()
        elif name in views:
            vector, up = views[name]
            self.plotter.view_vector(vector, viewup=up)
        else:
            raise ValueError("Unknown standard view.")

    def look_at_selection(self):
        manager = self.action_manager
        if manager.sketch_editing:
            manager.status_callback("Finish the sketch before changing the viewing direction.")
            return False
        face = None
        if manager.selected_profile_index is not None and manager.selected_profile_index < len(manager.profiles):
            face = manager.profiles[manager.selected_profile_index]["face"]
        elif manager.current_part is not None and manager.selected_face_id >= 0:
            faces = list(manager.current_part.faces())
            if manager.selected_face_id < len(faces):
                face = faces[manager.selected_face_id]
        if face is None or face.geom_type.name != "PLANE":
            manager.status_callback("Select a planar face or profile first.")
            return False
        center = np.asarray(tuple(face.center()), dtype=float)
        normal = np.asarray(tuple(face.normal_at()), dtype=float)
        normal /= np.linalg.norm(normal)
        camera = self.plotter.camera
        distance = np.linalg.norm(np.asarray(camera.position) - np.asarray(camera.focal_point))
        if distance < 1e-9:
            distance = 100.0
        up = np.array([0., 0., 1.]) if abs(normal[2]) < 0.9 else np.array([0., 1., 0.])
        up = up - np.dot(up, normal) * normal
        up /= np.linalg.norm(up)
        camera.focal_point = tuple(center)
        camera.position = tuple(center + normal * distance)
        camera.up = tuple(up)
        self.plotter.enable_parallel_projection()
        self.plotter.render()
        return True

    @staticmethod
    def operation_parameters(dialog):
        parameters = {}
        for form in dialog.findChildren(QtWidgets.QFormLayout):
            for row in range(form.rowCount()):
                label_item = form.itemAt(row, QtWidgets.QFormLayout.LabelRole)
                field_item = form.itemAt(row, QtWidgets.QFormLayout.FieldRole)
                if field_item is None:
                    field_item = form.itemAt(row, QtWidgets.QFormLayout.SpanningRole)
                if field_item is None or field_item.widget() is None:
                    continue
                field = field_item.widget()
                label = label_item.widget().text().rstrip(':') if label_item and label_item.widget() else ''
                if isinstance(field, QtWidgets.QCheckBox):
                    label, value = label or field.text(), field.isChecked()
                elif isinstance(field, (QtWidgets.QDoubleSpinBox, QtWidgets.QSpinBox)):
                    value = field.value()
                elif isinstance(field, QtWidgets.QComboBox):
                    value = field.currentText()
                else:
                    continue
                if label:
                    parameters[label] = value
        if hasattr(dialog, 'index'):
            parameters['Profile index'] = dialog.index + 1
        if getattr(dialog, 'selected_body', None) is not None and getattr(dialog, 'scope', None) is not None and dialog.scope.currentData() == 'selected':
            parameters['Selected body index'] = dialog.selected_body + 1
        if getattr(dialog, 'face_index', -1) >= 0:
            parameters['Selected face index'] = dialog.face_index + 1
        if hasattr(dialog, 'edge_indices') and dialog.edge_indices:
            parameters['Selected edges'] = ', '.join(str(i + 1) for i in dialog.edge_indices)
        return parameters

    def refresh_timeline(self):
        timeline = self.action_manager.timeline
        graph = self.action_manager.parametric_graph
        self.timeline_widget.blockSignals(True)
        self.timeline_widget.clear()
        self.timeline_widget.addItem('Start')
        if graph is not None:
            for index, feature in enumerate(graph.features):
                item = QtWidgets.QListWidgetItem(f"{index + 1}. {feature.name} [{feature.kind}]")
                if feature.id in graph.blocked:
                    item.setForeground(QtGui.QColor('#9ba5aa'))
                    item.setToolTip(graph.blocked[feature.id])
                else:
                    item.setToolTip('Right-click to edit this parametric feature')
                self.timeline_widget.addItem(item)
            self.timeline_widget.setCurrentRow(len(graph.features))
            self.timeline_widget.blockSignals(False)
            return
        for index, entry in enumerate(timeline.entries):
            item = QtWidgets.QListWidgetItem(f"{index + 1}. {entry['title']}")
            if index >= timeline.cursor:
                item.setForeground(QtGui.QColor('#9ba5aa'))
            item.setToolTip('Double-click to inspect parameters')
            self.timeline_widget.addItem(item)
        self.timeline_widget.setCurrentRow(timeline.cursor)
        self.timeline_widget.blockSignals(False)

    def navigate_timeline(self, row):
        if row < 0:
            return
        manager = self.action_manager
        if manager.interaction_locked:
            self.refresh_timeline()
            return
        if manager.parametric_graph is not None:
            return
        while manager.timeline.cursor > row:
            if not manager.undo_solid():
                break
        while manager.timeline.cursor < row:
            if not manager.redo_solid():
                break
        self.refresh_timeline()

    def timeline_context_menu(self, position):
        item = self.timeline_widget.itemAt(position)
        if item is None or self.timeline_widget.row(item) <= 0 or self.action_manager.interaction_locked:
            return
        index = self.timeline_widget.row(item) - 1
        if self.action_manager.parametric_graph is not None:
            self.parametric_context_menu(index, position)
            return
        menu = QtWidgets.QMenu(self.timeline_widget)
        rename_action = menu.addAction('Rename operation')
        chosen = menu.exec(self.timeline_widget.mapToGlobal(position))
        if chosen != rename_action:
            return
        old_title = self.action_manager.timeline.entries[index]['title']
        title, accepted = self.panel_inputs.getText(self.window, 'Rename operation', 'Name:', text=old_title)
        if accepted:
            try:
                self.action_manager.timeline.rename(index, title)
                self.refresh_timeline()
            except ValueError as exc:
                QtWidgets.QMessageBox.warning(self.window, 'Rename operation', str(exc))

    def inspect_timeline_entry(self, item):
        index = self.timeline_widget.row(item) - 1
        graph = self.action_manager.parametric_graph
        if graph is not None:
            if 0 <= index < len(graph.features):
                feature = graph.features[index]
                text = json.dumps(feature.parameters, ensure_ascii=False, indent=2)
                self.show_timeline_parameters(feature.name, text)
            return
        entries = self.action_manager.timeline.entries
        if not 0 <= index < len(entries):
            return
        entry = entries[index]
        text = '\n'.join(f'{key}: {value}' for key, value in entry['parameters'].items()) or 'No parameters recorded.'
        self.show_timeline_parameters(entry['title'], text)

    def show_timeline_parameters(self, title, text):
        dialog = QtWidgets.QDialog()
        dialog.setWindowTitle(title)
        layout = QtWidgets.QVBoxLayout(dialog)
        details = QtWidgets.QPlainTextEdit()
        details.setReadOnly(True)
        details.setPlainText(text)
        layout.addWidget(details)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        self.panel_inputs.exec_dialog(dialog)

    def parametric_context_menu(self, index, position):
        graph = self.action_manager.parametric_graph
        feature = graph.features[index]
        menu = QtWidgets.QMenu(self.timeline_widget)
        sketch_action = menu.addAction('Edit sketch in scene') if feature.kind == 'planar_sketch' else None
        fallback_action = menu.addAction('Edit sketch in panel') if feature.kind == 'planar_sketch' else None
        profiles_action = menu.addAction('Show sketch profiles') if feature.kind == 'planar_sketch' else None
        reselect_action = menu.addAction('Reselect extrusion profile') if feature.kind == 'extrude' else None
        edit_action = menu.addAction('Edit parameters (JSON)')
        rename_action = menu.addAction('Rename')
        suppress_action = menu.addAction('Unsuppress' if feature.suppressed else 'Suppress')
        left_action = menu.addAction('Move earlier')
        right_action = menu.addAction('Move later')
        chosen = menu.exec(self.timeline_widget.mapToGlobal(position))
        if chosen is None:
            return
        try:
            if profiles_action is not None and chosen == profiles_action:
                self.action_manager.show_parametric_profiles(feature.id)
                self.action_manager.status_callback('Click a sketch profile, then choose Extrude.')
                return
            if reselect_action is not None and chosen == reselect_action:
                self.reselect_extrusion_profile(feature.id)
                return
            if sketch_action is not None and chosen == sketch_action:
                self.action_manager.start_history_sketch(feature.id)
                self.action_manager_ui.activate_scetch_mode(self.tool_tabs, 'sketch')
                return
            candidate = FeatureGraph.from_data(graph.to_data())
            if chosen == fallback_action and fallback_action is not None:
                dialog = ParametricSketchDialog(candidate, feature.id, self.window, self.panel_inputs)
                if self.panel_inputs.exec_dialog(dialog) != QtWidgets.QDialog.Accepted:
                    return
                candidate = dialog.graph
            elif chosen == edit_action:
                text, ok = self.panel_inputs.getMultiLineText(self.window, 'Edit parameters',
                    'Feature parameters (JSON):', json.dumps(feature.parameters, ensure_ascii=False, indent=2))
                if not ok: return
                value = json.loads(text)
                if not isinstance(value, dict): raise ValueError('Parameters must be a JSON object.')
                candidate.edit(feature.id, parameters=value)
            elif chosen == rename_action:
                value, ok = self.panel_inputs.getText(self.window, 'Rename feature', 'Name:', text=feature.name)
                if not ok: return
                candidate.edit(feature.id, name=value)
            elif chosen == suppress_action:
                candidate.edit(feature.id, suppressed=not feature.suppressed)
            elif chosen in (left_action, right_action):
                candidate.reorder(feature.id, index + (-1 if chosen == left_action else 1))
            self.action_manager.install_parametric_graph(candidate, allow_empty=(chosen == suppress_action))
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self.window, 'Parametric rebuild', str(exc))

    def reselect_extrusion_profile(self, feature_id):
        manager = self.action_manager
        graph = manager.parametric_graph
        feature = next((item for item in graph.features if item.id == feature_id), None)
        if feature is None or feature.kind != 'extrude' or not feature.input_id:
            raise ValueError('Choose an extrusion with a sketch dependency.')
        sketch_id = feature.input_id
        profiles = manager.show_parametric_profiles(sketch_id)
        if not profiles: raise ValueError('The sketch has no closed profiles.')
        def selected(index):
            try:
                current = manager.parametric_graph
                operation = next(item for item in current.features if item.id == feature_id)
                sketch = current.results[sketch_id]
                if not 0 <= index < len(sketch.profiles): raise ValueError('Profile is no longer available.')
                key = sketch.profile_keys[index]
                if key is None and len(sketch.profiles) > 1:
                    raise ValueError('The selected profile cannot be identified unambiguously.')
                params = dict(operation.parameters)
                params.update(profile_index=index, profile_count=len(sketch.profiles))
                params.pop('profile_key', None)
                if key is not None: params['profile_key'] = key
                candidate = FeatureGraph.from_data(current.to_data())
                candidate.edit(feature_id, parameters=params)
                manager.install_parametric_graph(candidate, allow_empty=True)
                if manager.current_part is None: manager.show_parametric_profiles(sketch_id)
                manager.status_callback('Extrusion profile updated.' + (' Unsuppress the operation to rebuild it.' if operation.suppressed else ''))
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self.window, 'Extrusion profile', str(exc))
        manager.request_profile_selection(selected)

    def edit_graph_parameters(self):
        graph = self.action_manager.parametric_graph
        if graph is None:
            QtWidgets.QMessageBox.information(self.window, 'Parameters', 'Open or create a parametric model first.')
            return
        text, ok = self.panel_inputs.getMultiLineText(self.window, 'Parameters',
            'Named parameters as JSON expressions:', json.dumps(graph.parameters, ensure_ascii=False, indent=2))
        if not ok: return
        try:
            values = json.loads(text)
            if not isinstance(values, dict): raise ValueError('Parameters must be a JSON object.')
            candidate = FeatureGraph.from_data(graph.to_data())
            candidate.set_parameters(values)
            self.action_manager.install_parametric_graph(candidate)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self.window, 'Parametric rebuild', str(exc))

    def new_parametric_model(self):
        manager = self.action_manager
        if manager.pending_sketch or manager.sketch_editing or manager.selecting_sketch_plane:
            QtWidgets.QMessageBox.information(self.window, 'Parametric model', 'Finish the current sketch first.')
            return
        if manager.current_part is not None:
            answer = QtWidgets.QMessageBox.question(self.window, 'Parametric model',
                'Replace the current model and its history with a new parametric model?')
            if answer != QtWidgets.QMessageBox.Yes:
                return
        try:
            plane = Feature.create('Plane', 'datum_plane', {'origin': [0, 0, 0], 'normal': [0, 0, 1]})
            sketch = Feature.create('Rectangle', 'rectangle_sketch', {'width': 'width', 'height': 'height'}, plane.id)
            extrude = Feature.create('Extrude', 'extrude', {'distance': 'depth'}, sketch.id)
            graph = FeatureGraph([plane, sketch, extrude], {'width': '10', 'height': '10', 'depth': '10'})
            manager.install_parametric_graph(graph, reset_history=True)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self.window, 'Parametric model', str(exc))

    def add_graph_feature(self):
        graph = self.action_manager.parametric_graph
        if graph is None:
            QtWidgets.QMessageBox.information(self.window, 'Add feature', 'Open or create a parametric model first.')
            return
        kinds = ['primitive', 'datum_plane', 'rectangle_sketch', 'extrude', 'move', 'scale',
                 'mirror', 'rectangular_pattern', 'circular_pattern', 'fillet', 'chamfer', 'shell', 'hole', 'offset',
                 'planar_sketch', 'revolve']
        kind, ok = self.panel_inputs.getItem(self.window, 'Add feature', 'Type:', kinds, 0, False)
        if not ok: return
        templates = {
            'primitive': {'type': 'box', 'dimensions': [10, 10, 10], 'operation': 'new_body'},
            'datum_plane': {'origin': [0, 0, 0], 'normal': [0, 0, 1]},
            'rectangle_sketch': {'width': 10, 'height': 10},
            'extrude': {'distance': 10},
            'revolve': {'angle': 360, 'axis_origin': [0, 0, 0], 'axis_direction': [0, 1, 0]},
            'planar_sketch': {'points': {'a': [0, 0], 'b': [10, 0], 'c': [10, 10], 'd': [0, 10]}, 'entities': [{'type': 'line', 'points': ['a', 'b', 'c', 'd', 'a']}], 'constraints': []},
            'move': {'translation': [0, 0, 0]}, 'scale': {'factor': 1},
            'mirror': {'plane': 'YZ', 'offset': 0, 'copy': True},
            'rectangular_pattern': {'count_x': 2, 'count_y': 1, 'spacing_x': 20, 'spacing_y': 20},
            'circular_pattern': {'count': 4, 'angle': 360, 'axis': 'Z', 'center': [0, 0, 0]},
            'fillet': {'radius': 1}, 'chamfer': {'distance': 1},
            'shell': {'thickness': -1}, 'hole': {'diameter': 2, 'depth': 5}, 'offset': {'distance': 1},
        }
        candidates = [(feature.name, feature.id) for feature in graph.features if feature.id in graph.results]
        needs_input = kind not in ('primitive', 'datum_plane')
        input_id = None
        if kind == 'primitive':
            # Add a primitive to the latest solid result rather than hiding earlier bodies.
            input_id = next((feature.id for feature in reversed(graph.features)
                             if feature.id in graph.results and hasattr(graph.results[feature.id], 'solids')
                             and graph.results[feature.id].solids()), None)
        if needs_input:
            if not candidates:
                QtWidgets.QMessageBox.warning(self.window, 'Add feature', 'No dependency is available.')
                return
            labels = [f'{name} ({fid[:8]})' for name, fid in candidates]
            selected, ok = self.panel_inputs.getItem(self.window, 'Add feature', 'Input feature:', labels, len(labels) - 1, False)
            if not ok: return
            input_id = candidates[labels.index(selected)][1]
        if kind in ('fillet', 'chamfer', 'shell', 'hole'):
            displayed = next((feature.id for feature in reversed(graph.features)
                              if feature.id in graph.results and hasattr(graph.results[feature.id], 'solids')
                              and graph.results[feature.id].solids()), None)
            if input_id != displayed:
                QtWidgets.QMessageBox.warning(self.window, 'Add feature', 'Select the currently displayed solid as the input feature.')
                return
            try:
                if kind in ('fillet', 'chamfer'):
                    indices = sorted(self.action_manager.selected_edge_indices)
                    if not indices: raise ValueError('Select one or more edges first.')
                    templates[kind]['edges'] = [make_reference(input_id, graph.results[input_id], 'edge', i) for i in indices]
                else:
                    face_index = self.action_manager.selected_face_id
                    if face_index < 0: raise ValueError('Select a face first.')
                    templates[kind]['face'] = make_reference(input_id, graph.results[input_id], 'face', face_index)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self.window, 'Add feature', str(exc))
                return
        text, ok = self.panel_inputs.getMultiLineText(self.window, 'Add feature', 'Parameters (JSON):',
                                                           json.dumps(templates[kind], indent=2))
        if not ok: return
        try:
            params = json.loads(text)
            if not isinstance(params, dict): raise ValueError('Parameters must be a JSON object.')
            feature = Feature.create(kind, kind, params, input_id)
            data = graph.to_data()
            data['features'].append({'id': feature.id, 'name': feature.name, 'kind': feature.kind,
                'parameters': feature.parameters, 'input_id': feature.input_id, 'suppressed': False})
            self.action_manager.install_parametric_graph(FeatureGraph.from_data(data))
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self.window, 'Add feature', str(exc))

    def require_snapshot_mode(self):
        if self.action_manager.parametric_graph is not None:
            QtWidgets.QMessageBox.information(self.window, 'Parametric model',
                'This command is not yet supported in parametric mode. Edit the existing features in the Timeline.')
            return False
        return True

    def show_scene_operation(self, dialog, title):
        """Keep the scene interactive while preventing conflicting model edits."""
        manager = self.action_manager
        self.active_operation_dialog = dialog
        manager.interaction_locked = True
        self.tool_tabs.setEnabled(False)
        for shortcut in self.shortcuts:
            shortcut.setEnabled(False)
        def finished(result):
            manager.interaction_locked = False
            self.tool_tabs.setEnabled(True)
            for shortcut in self.shortcuts:
                shortcut.setEnabled(True)
            if result == QtWidgets.QDialog.Accepted and dialog.result_shape is not None:
                try:
                    manager.commit_result(dialog.result_shape, dialog.windowTitle(), self.operation_parameters(dialog))
                except Exception as exc:
                    QtWidgets.QMessageBox.warning(self.window, title, str(exc))
            self.active_operation_dialog = None
            self.operation_panel.hide()
            self.operation_content.layout().removeWidget(dialog)
            dialog.deleteLater()
        dialog.finished.connect(finished)
        self.show_operation_panel(dialog)

    def show_operation_panel(self, dialog):
        self.operation_title.setText(dialog.windowTitle())
        self.operation_title.setChecked(True)
        self.operation_title.setArrowType(QtCore.Qt.DownArrow)
        dialog.setParent(self.operation_content, QtCore.Qt.Widget)
        dialog.setMinimumWidth(0)
        for combo in dialog.findChildren(QtWidgets.QComboBox):
            combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(8)
            combo.setMinimumWidth(0)
            combo.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        for number in dialog.findChildren(QtWidgets.QAbstractSpinBox):
            number.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
            number.setMinimumWidth(0)
            number.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        self.operation_content.layout().addWidget(dialog)
        self.operation_content.show()
        self.operation_scroll.show()
        self.operation_panel.show()
        dialog.show()
        if hasattr(self.operation_panel, 'queue_reposition'):
            self.operation_panel.queue_reposition()

    def get_main_layout(self) -> QtWidgets.QVBoxLayout:
        main_layout = QtWidgets.QVBoxLayout(self.central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        return main_layout
        
    def add_splitter(self):
        self.workspace = QtWidgets.QFrame()
        self.workspace_container = QtWidgets.QVBoxLayout(self.workspace)
        self.workspace_container.setContentsMargins(0, 0, 0, 0)
        self.workspace_container.setSpacing(0)
        
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.projects_bar = QtWidgets.QFrame()
        self.projects_bar.setMaximumWidth(320)
        
        self.tool_tabs = QtWidgets.QTabWidget()
        self.tool_tabs.setFixedHeight(112)
        # add tabs
        solid_tab = self.create_solid_tab()
        self.tool_tabs.addTab(solid_tab, "Solid")
        sketch_tab = self.create_scetch_tab()
        self.tool_tabs.addTab(sketch_tab, "Sketch")
        self.workspace_container.addWidget(self.tool_tabs)

        self.render_container = QtWidgets.QFrame()
        self.workspace_container.addWidget(self.render_container)
        self.timeline_widget = QtWidgets.QListWidget()
        self.timeline_widget.setObjectName("solidTimeline")
        self.timeline_widget.setFlow(QtWidgets.QListView.LeftToRight)
        self.timeline_widget.setWrapping(False)
        self.timeline_widget.setFixedHeight(64)
        self.timeline_widget.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.timeline_widget.currentRowChanged.connect(self.navigate_timeline)
        self.timeline_widget.itemDoubleClicked.connect(self.inspect_timeline_entry)
        self.timeline_widget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.timeline_widget.customContextMenuRequested.connect(self.timeline_context_menu)
        self.workspace_container.addWidget(self.timeline_widget)

        self.splitter.addWidget(self.projects_bar)
        self.splitter.addWidget(self.workspace)
        self.main_layout.addWidget(self.splitter)
        
    def add_3d_viewport(self):
        render_layout = QtWidgets.QHBoxLayout(self.render_container)
        render_layout.setContentsMargins(0,0,0,0)
        render_layout.setSpacing(0)

        self.plotter = QtInteractor(self.render_container)
        render_layout.addWidget(self.plotter, 1)
        self.operation_panel = OperationPanel(self.render_container)
        panel_layout = QtWidgets.QVBoxLayout(self.operation_panel)
        panel_layout.setContentsMargins(1, 1, 1, 1)
        panel_layout.setSpacing(0)
        self.operation_title = QtWidgets.QToolButton()
        self.operation_title.setCheckable(True)
        self.operation_title.setChecked(True)
        self.operation_title.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.operation_title.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.operation_title.setObjectName('operationTitle')
        panel_layout.addWidget(self.operation_title)
        self.operation_content = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(self.operation_content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        self.operation_scroll = QtWidgets.QScrollArea()
        self.operation_scroll.setWidgetResizable(True)
        self.operation_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.operation_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.operation_scroll.setWidget(self.operation_content)
        panel_layout.addWidget(self.operation_scroll)
        self.operation_panel.title = self.operation_title
        self.operation_panel.content = self.operation_content
        self.operation_panel.scroll = self.operation_scroll
        self.operation_title.toggled.connect(self.operation_scroll.setVisible)
        self.operation_title.toggled.connect(lambda expanded: self.operation_title.setArrowType(
            QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow))
        self.operation_title.toggled.connect(self.operation_panel.queue_reposition)
        self.operation_panel.setStyleSheet('''
            QFrame#operationPanel { background: #f3f3f3; border: 1px solid #b6b6b6; }
            QToolButton#operationTitle { background: #e5e5e5; border: 0; border-bottom: 1px solid #c6c6c6;
                color: #303030; text-align: left; padding: 3px 5px; font-weight: bold; }
            QScrollArea, QWidget { background: #f3f3f3; }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit {
                background: white; border: 1px solid #c6c6c6; padding: 2px; }
            QPushButton { background: #eeeeee; border: 1px solid #bcbcbc; padding: 3px 10px; }
            QPushButton:hover { background: #e0edf5; border-color: #73a9c8; }
        ''')
        self.operation_panel.hide()
        
    def setup_cad_viewport(self):
        self.plotter.set_background("#CECECE", top="#CECECE")
        # Intersecting translucent datum planes must be composited per pixel;
        # sorting whole plane actors makes them appear cut at their crossings.
        self.plotter.ren_win.SetAlphaBitPlanes(1)
        self.plotter.ren_win.SetMultiSamples(0)
        self.plotter.renderer.SetUseDepthPeeling(True)
        self.plotter.renderer.SetMaximumNumberOfPeels(8)
        self.plotter.renderer.SetOcclusionRatio(0.0)
        try:
            orientation = self.plotter.add_camera_orientation_widget(n_frames=20)
            def navigation_size():
                representation = orientation.GetRepresentation()
                size, padding = representation.GetSize(), representation.GetPadding()
                ratio = max(1., self.plotter.devicePixelRatioF())
                return tuple(int((size[i] + 2 * padding[i]) / ratio + .999) for i in (0, 1))
            self.operation_panel.navigation_size = navigation_size
            self.operation_panel.queue_reposition()
        except AttributeError:
            self.plotter.add_axes()

        # 3. ОСИ В НАЧАЛЕ КООРДИНАТ (0,0,0)
        # Чтобы всегда понимать, где центр мира
        self.plotter.add_axes_at_origin(labels_off=True, line_width=5)

        self.plotter.enable_lightkit() 
        self.plotter.enable_parallel_projection()
        self.plotter.view_isometric()
        
    def toggle_sidebar(self):
        # Если видима — прячем, если нет — показываем
        self.projects_bar.setVisible(not self.projects_bar.isVisible())
        
    def add_style(self):
        self.window.setStyleSheet("""
            QTabWidget::pane { border: 0; background: #f5f7f8; }
            QTabBar::tab { color: #52616b; background: #edf1f3; padding: 5px 15px; border: 0; }
            QTabBar::tab:selected { color: #08769a; background: #f8fafb; border-bottom: 2px solid #22a1c4; }
            QFrame#ribbon { background: #f8fafb; border-bottom: 1px solid #cbd4d9; }
            QToolButton[ribbon="true"] { border: 1px solid transparent; border-radius: 3px; padding: 3px; background: transparent; }
            QToolButton[ribbon="true"]:hover { border: 1px solid #c9dce5; background: #e5f2f8; }
            QToolButton[groupLabel="true"] { color: #52616b; border: 0; background: transparent; padding: 0 3px; font-size: 10px; }
            QToolButton[groupLabel="true"]:hover { color: #08769a; background: #e5f2f8; }
            QToolButton#finishSketch { color: #286f50; background: #e6f4ec; border: 1px solid #bddfcb; padding: 3px; border-radius: 3px; }
            QToolButton#finishSketch:hover { background: #d3eddd; }
            QMenu { background: white; color: #26343c; border: 1px solid #cbd4d9; padding: 4px; }
            QMenu::item { padding: 6px 24px 6px 10px; }
            QMenu::item:selected { background: #e5f2f8; }
        """)

    def ribbon_group(self, layout, label, commands, visible):
        group = QtWidgets.QFrame()
        group_layout = QtWidgets.QVBoxLayout(group)
        group_layout.setContentsMargins(4, 1, 4, 1)
        group_layout.setSpacing(0)
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(1)
        menu = QtWidgets.QMenu(group)
        for key, title, callback in commands:
            icon = tool_icon(key)
            action = menu.addAction(icon, title)
            action.triggered.connect(callback)
            if key in visible:
                button = QtWidgets.QToolButton()
                button.setProperty("ribbon", True)
                button.setIcon(icon)
                button.setIconSize(QtCore.QSize(40, 40))
                button.setFixedSize(49, 49)
                button.setToolTip(title)
                button.setAccessibleName(title)
                button.clicked.connect(callback)
                row.addWidget(button)
        group_layout.addLayout(row)
        label_button = QtWidgets.QToolButton()
        label_button.setProperty("groupLabel", True)
        label_button.setText(label.upper() + "  ▾")
        label_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        label_button.setMenu(menu)
        group_layout.addWidget(label_button, alignment=QtCore.Qt.AlignHCenter)
        layout.addWidget(group)
        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.VLine)
        separator.setStyleSheet("color: #d8e0e4;")
        layout.addWidget(separator)

    def ribbon_layout(self):
        ribbon = QtWidgets.QFrame()
        ribbon.setObjectName("ribbon")
        layout = QtWidgets.QHBoxLayout(ribbon)
        layout.setContentsMargins(7, 2, 7, 2)
        layout.setSpacing(4)
        return ribbon, layout

    def create_solid_tab(self):
        tool_bar, top_layout = self.ribbon_layout()
        
        def open_sketch():
            self.action_manager.activate_sketch_mode()
            self.action_manager_ui.activate_scetch_mode(self.tool_tabs, "sketch")
            
        def extrude():
            manager = self.action_manager
            if manager.sketch_editing:
                self.disable_active_tool()
                try:
                    manager.exit_sketch_mode()
                except Exception as exc:
                    QtWidgets.QMessageBox.warning(self.window, 'Sketch', str(exc))
                    return
            if not manager.profiles:
                try:
                    if manager.parametric_graph is not None:
                        QtWidgets.QMessageBox.information(self.window, 'Extrude', 'Finish a parametric sketch or use Show sketch profiles in the Timeline.')
                        return
                    if not manager.prepare_selected_face_profile():
                        QtWidgets.QMessageBox.information(self.window, "Extrude", "Select a planar face or finish a sketch containing a closed profile first.")
                        return
                except ValueError as exc:
                    QtWidgets.QMessageBox.warning(self.window, "Extrude", str(exc))
                    return
            def extrude_selected(index):
                dialog = ExtrudeDialog(manager, index, self.window)
                self.active_operation_dialog = dialog
                manager.interaction_locked = True
                self.tool_tabs.setEnabled(False)
                for shortcut in self.shortcuts:
                    shortcut.setEnabled(False)
                def finished(result):
                    manager.interaction_locked = False
                    self.tool_tabs.setEnabled(True)
                    for shortcut in self.shortcuts:
                        shortcut.setEnabled(True)
                    if result == QtWidgets.QDialog.Accepted and dialog.result_shape is not None:
                        try:
                            if manager.parametric_graph is not None:
                                manager.commit_parametric_extrusion(index, dialog.distance.value(), dialog.operation.currentData(),
                                    dialog.offset.value(), dialog.taper.value(), dialog.direction.currentData() == 'symmetric',
                                    dialog.second_distance.value() if dialog.direction.currentData() == 'two' else None,
                                    dialog.extent.currentData())
                            else:
                                manager.commit_result(dialog.result_shape, dialog.windowTitle(), self.operation_parameters(dialog))
                        except Exception as exc:
                            QtWidgets.QMessageBox.warning(self.window, "Extrude", str(exc))
                    manager.clear_temporary_face_profile()
                    self.active_operation_dialog = None
                    self.operation_panel.hide()
                    self.operation_content.layout().removeWidget(dialog)
                    dialog.deleteLater()
                dialog.finished.connect(finished)
                self.show_operation_panel(dialog)

            if manager.selected_profile_index is not None:
                extrude_selected(manager.selected_profile_index)
            else:
                manager.request_profile_selection(extrude_selected)

        def revolve():
            if not self.require_snapshot_mode():
                return
            manager = self.action_manager
            if manager.sketch_editing:
                self.disable_active_tool()
                manager.exit_sketch_mode()
            if not manager.profiles:
                QtWidgets.QMessageBox.information(self.window, "Revolve", "Finish a sketch containing a closed profile first.")
                return
            def revolve_selected(index):
                dialog = RevolveDialog(manager, index, self.window)
                self.show_scene_operation(dialog, "Revolve")
            if manager.selected_profile_index is not None:
                revolve_selected(manager.selected_profile_index)
            else:
                manager.request_profile_selection(revolve_selected)

        def create_primitive(kind):
            if not self.require_snapshot_mode():
                return
            manager = self.action_manager
            if manager.pending_sketch or manager.sketch_editing or manager.selecting_sketch_plane:
                QtWidgets.QMessageBox.information(self.window, "Create solid", "Finish the current sketch and solid operation first.")
                return
            dialog = PrimitiveDialog(manager, kind, self.window)
            self.show_scene_operation(dialog, "Create solid")

        self.ribbon_group(top_layout, "Create", [
            ("sketch", "Create sketch", open_sketch),
            ("extrude", "Extrude", extrude),
            ("revolve", "Revolve", revolve),
            ("box", "Box", lambda: create_primitive("box")),
            ("cylinder", "Cylinder", lambda: create_primitive("cylinder")),
            ("sphere", "Sphere", lambda: create_primitive("sphere")),
            ("cone", "Cone / frustum", lambda: create_primitive("cone")),
            ("torus", "Torus", lambda: create_primitive("torus")),
        ], {"sketch", "extrude", "revolve"})

        def transform_model(kind):
            if not self.require_snapshot_mode():
                return
            manager = self.action_manager
            if manager.pending_sketch or manager.sketch_editing or manager.selecting_sketch_plane:
                QtWidgets.QMessageBox.information(self.window, "Modify", "Finish the current sketch and solid operation first.")
                return
            if manager.current_part is None:
                QtWidgets.QMessageBox.information(self.window, "Modify", "Create or import a solid body first.")
                return
            dialog = TransformDialog(manager, kind, self.window)
            self.show_scene_operation(dialog, "Modify")

        def solid_feature(kind):
            if not self.require_snapshot_mode():
                return
            manager = self.action_manager
            if manager.pending_sketch or manager.sketch_editing or manager.selecting_sketch_plane:
                QtWidgets.QMessageBox.information(self.window, "Modify", "Finish the current sketch and solid operation first.")
                return
            if manager.current_part is None:
                QtWidgets.QMessageBox.information(self.window, "Modify", "Create or import a solid body first.")
                return
            if kind in ("fillet_selected", "chamfer_selected") and not manager.selected_edge_indices:
                QtWidgets.QMessageBox.information(self.window, "Modify", "Select one or more edges first.")
                return
            if kind in ("shell", "hole") and manager.selected_face_id < 0:
                QtWidgets.QMessageBox.information(self.window, "Modify", "Select a face of the body first.")
                return
            dialog = SolidFeatureDialog(manager, kind, self.window)
            self.show_scene_operation(dialog, "Modify")

        def delete_selected_body():
            if not self.require_snapshot_mode():
                return
            manager = self.action_manager
            if manager.pending_sketch or manager.sketch_editing or manager.selecting_sketch_plane:
                QtWidgets.QMessageBox.information(self.window, "Delete body", "Finish the current sketch and solid operation first.")
                return
            if manager.selected_body_index() is None:
                QtWidgets.QMessageBox.information(self.window, "Delete body", "Select a body by clicking one of its faces.")
                return
            answer = QtWidgets.QMessageBox.question(self.window, "Delete body", "Delete the selected body? You can undo this operation.")
            if answer == QtWidgets.QMessageBox.Yes:
                try:
                    manager.delete_selected_body()
                except Exception as exc:
                    QtWidgets.QMessageBox.warning(self.window, "Delete body", str(exc))

        self.ribbon_group(top_layout, "Modify", [
            ("move", "Move / Copy bodies", lambda: transform_model("move")),
            ("scale", "Scale bodies", lambda: transform_model("scale")),
            ("mirror", "Mirror bodies", lambda: transform_model("mirror")),
            ("rectangular_pattern", "Rectangular pattern", lambda: transform_model("rectangular")),
            ("circular_pattern", "Circular pattern", lambda: transform_model("circular")),
            ("fillet_all", "Fillet selected edges", lambda: solid_feature("fillet_selected")),
            ("chamfer_all", "Chamfer selected edges", lambda: solid_feature("chamfer_selected")),
            ("fillet_all", "Fillet all edges", lambda: solid_feature("fillet")),
            ("chamfer_all", "Chamfer all edges", lambda: solid_feature("chamfer")),
            ("shell", "Shell selected face", lambda: solid_feature("shell")),
            ("offset_solid", "Offset solid", lambda: solid_feature("offset")),
            ("hole", "Hole at selected face center", lambda: solid_feature("hole")),
            ("delete_body", "Delete selected body", delete_selected_body),
        ], {"move"})

        def start_edge_selection():
            try:
                self.action_manager.start_edge_selection()
            except ValueError as exc:
                QtWidgets.QMessageBox.information(self.window, "Select edges", str(exc))

        def clear_edge_selection():
            self.action_manager.clear_selected_edges()
            self.action_manager.status_callback("Edge selection cleared.")

        self.ribbon_group(top_layout, "Select", [
            ("select_edge", "Select edges", start_edge_selection),
            ("clear_edges", "Clear selected edges", clear_edge_selection),
        ], {"select_edge"})

        def create_offset_plane():
            manager = self.action_manager
            if manager.sketch_editing or manager.selecting_sketch_plane:
                QtWidgets.QMessageBox.information(self.window, "Offset plane", "Finish the active sketch or plane selection first.")
                return
            bases = ["XY", "XZ", "YZ", "Selected planar face"]
            default = 3 if manager.current_part is not None and manager.selected_face_id >= 0 else 0
            base, ok = self.panel_inputs.getItem(self.window, "Offset plane", "Base plane:", bases, default, False)
            if not ok:
                return
            offset, ok = self.panel_inputs.getDouble(self.window, "Offset plane", "Offset:", 10.0, -1000000, 1000000, 3)
            if ok:
                try:
                    if base == "Selected planar face":
                        manager.create_plane_from_selected_face(offset)
                    else:
                        manager.create_offset_plane(base, offset)
                except ValueError as exc:
                    QtWidgets.QMessageBox.warning(self.window, "Offset plane", str(exc))

        def toggle_construction_planes():
            manager = self.action_manager
            if not manager.construction_planes:
                self.window.statusBar().showMessage("There are no construction planes.", 3000)
                return
            if manager.sketch_editing:
                self.window.statusBar().showMessage("Finish the sketch before showing construction planes.", 3000)
                return
            actors = [manager.plotter.actors.get(plane["name"]) for plane in manager.construction_planes]
            visible = any(actor is not None and actor.GetVisibility() for actor in actors)
            manager.set_construction_planes_visible(not visible)
            manager.plotter.render()

        self.ribbon_group(top_layout, "Construct", [
            ("plane", "Offset plane", create_offset_plane),
            ("plane_visibility", "Show / hide construction planes", toggle_construction_planes),
        ], {"plane"})

        def export_model():
            part = getattr(self.action_manager, "current_part", None)
            if part is None:
                QtWidgets.QMessageBox.information(self.window, "Export", "Create a solid body first.")
                return
            path, selected = QtWidgets.QFileDialog.getSaveFileName(
                self.window, "Export model", "", "STEP model (*.step);;STL mesh (*.stl)")
            if not path:
                return
            suffix = Path(path).suffix.lower()
            format_name = 'stl' if suffix == '.stl' else 'step' if suffix in ('.step', '.stp') else ('stl' if selected.startswith('STL') else 'step')
            if not suffix:
                path += '.' + format_name
            try:
                export_body(part, path, format_name)
                self.window.statusBar().showMessage(f"Exported: {path}", 5000)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self.window, "Export", str(exc))

        def import_step():
            if not self.require_snapshot_mode():
                return
            manager = self.action_manager
            if manager.pending_sketch or manager.sketch_editing or manager.selecting_sketch_plane:
                QtWidgets.QMessageBox.information(self.window, "Import STEP", "Finish the current sketch and solid operation first.")
                return
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self.window, "Import STEP", "", "STEP model (*.step *.stp)")
            if not path:
                return
            try:
                result = import_step_body(path, manager.current_part)
                manager.commit_result(result, "Import STEP", {"File": Path(path).name})
                self.window.statusBar().showMessage(f"Imported: {path}", 5000)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self.window, "Import STEP", str(exc))

        def save_model_snapshot():
            manager = self.action_manager
            if manager.pending_sketch or manager.sketch_editing or manager.selecting_sketch_plane:
                QtWidgets.QMessageBox.information(self.window, "Save model", "Finish the solid operation before saving. Model snapshots do not include sketches.")
                return
            if manager.current_part is None:
                QtWidgets.QMessageBox.information(self.window, "Save model", "Create a solid body first.")
                return
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self.window, "Save model snapshot", "", "FluentCAD model (*.fcad)")
            if not path:
                return
            if not Path(path).suffix:
                path += '.fcad'
            try:
                save_snapshot(path, manager.current_part, manager.undo_stack, manager.redo_stack, manager.construction_planes,
                              {'entries': manager.timeline.entries, 'cursor': manager.timeline.cursor}, manager.parametric_graph,
                              {'undo': manager.graph_undo, 'redo': manager.graph_redo})
                self.window.statusBar().showMessage(f"Model snapshot saved: {path}", 5000)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self.window, "Save model", str(exc))

        def open_model_snapshot():
            manager = self.action_manager
            if manager.pending_sketch or manager.sketch_editing or manager.selecting_sketch_plane:
                QtWidgets.QMessageBox.information(self.window, "Open model", "Finish the current sketch and solid operation first.")
                return
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self.window, "Open model snapshot", "", "FluentCAD model (*.fcad)")
            if not path:
                return
            try:
                current, undo, redo, planes, timeline_data, feature_graph, graph_history = load_snapshot(
                    path, include_planes=True, include_timeline=True, include_feature_graph=True, include_graph_history=True)
                if feature_graph is not None and (undo or redo):
                    raise ValueError('This parametric file contains legacy undo snapshots which cannot yet be restored safely.')
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self.window, "Open model", str(exc))
                return
            if manager.current_part is not None:
                answer = QtWidgets.QMessageBox.question(self.window, "Open model", "Replace the current model and its undo history?")
                if answer != QtWidgets.QMessageBox.Yes:
                    return
            try:
                if feature_graph is not None:
                    manager.install_parametric_graph(feature_graph, reset_history=True)
                    manager.graph_undo, manager.graph_redo = graph_history['undo'], graph_history['redo']
                else:
                    manager.display_result(current)
                    manager.parametric_graph = None
                manager.clear_construction_planes()
                for plane in planes:
                    manager.create_offset_plane(plane["base"], plane["offset"],
                                                plane.get("origin"), plane.get("normal"))
                if feature_graph is None:
                    manager.undo_stack, manager.redo_stack = undo, redo
                    manager.timeline.entries = timeline_data['entries']
                    manager.timeline.cursor = timeline_data['cursor']
                manager.timeline_callback()
                manager.sketchs_data.clear()
                manager.activated_sketch_id = None
                manager.legacy_migration_redo = None
                self.window.statusBar().showMessage(f"Model snapshot opened: {path}", 5000)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self.window, "Open model", str(exc))

        def convert_legacy_model():
            try:
                self.action_manager.convert_legacy_model()
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self.window, 'Convert model', str(exc))
                return
            QtWidgets.QMessageBox.information(self.window, 'Convert model',
                'The existing solid is now a frozen base feature. Earlier operations cannot be reconstructed; Undo restores the previous snapshot history.')

        self.ribbon_group(top_layout, "File", [
            ("parametric_new", "New parametric model", self.new_parametric_model),
            ("parametric_add", "Add parametric feature", self.add_graph_feature),
            ("parametric_parameters", "Edit named parameters", self.edit_graph_parameters),
            ("parametric_convert", "Convert legacy model to parametric", convert_legacy_model),
            ("save", "Save solid model snapshot", save_model_snapshot),
            ("open", "Open solid model snapshot", open_model_snapshot),
            ("export", "Export STEP / STL", export_model),
            ("import", "Import STEP as new body", import_step),
        ], {"save", "open", "export"})
        self.ribbon_group(top_layout, "History", [
            ("undo", "Undo solid operation (Ctrl+Z)", self.action_manager.undo_solid),
            ("redo", "Redo solid operation (Ctrl+Y)", self.action_manager.redo_solid),
        ], {"undo", "redo"})

        def inspect_selection():
            manager = self.action_manager
            face = None
            if manager.selected_profile_index is not None and manager.selected_profile_index < len(manager.profiles):
                face = manager.profiles[manager.selected_profile_index]["face"]
            elif manager.current_part is not None and manager.selected_face_id >= 0:
                faces = list(manager.current_part.faces())
                if manager.selected_face_id < len(faces):
                    face = faces[manager.selected_face_id]
            try:
                data = measure_face(face)
                center = ", ".join(f"{value:.4f}" for value in data["center"])
                QtWidgets.QMessageBox.information(self.window, "Inspect selection",
                    f"Surface: {data['geometry']}\nArea: {data['area']:.4f} square units\n"
                    f"Perimeter: {data['perimeter']:.4f} units\nCenter: ({center})")
            except ValueError as exc:
                QtWidgets.QMessageBox.information(self.window, "Inspect selection", str(exc))

        def inspect_model():
            try:
                data = measure_model(self.action_manager.current_part)
                dimensions = " × ".join(f"{value:.4f}" for value in data["dimensions"])
                QtWidgets.QMessageBox.information(self.window, "Inspect model",
                    f"Bodies: {data['bodies']}\nVolume: {data['volume']:.4f} cubic units\n"
                    f"Surface area: {data['area']:.4f} square units\nBounds: {dimensions} units")
            except ValueError as exc:
                QtWidgets.QMessageBox.information(self.window, "Inspect model", str(exc))

        def inspect_edges():
            try:
                manager = self.action_manager
                data = measure_edges(manager.current_part, manager.selected_edge_indices)
                edge_types = ", ".join(data["types"])
                QtWidgets.QMessageBox.information(self.window, "Inspect edges",
                    f"Edges: {data['count']}\nTotal length: {data['length']:.4f} units\nTypes: {edge_types}")
            except ValueError as exc:
                QtWidgets.QMessageBox.information(self.window, "Inspect edges", str(exc))

        def inspect_selected_body():
            manager = self.action_manager
            index = manager.selected_body_index()
            if index is None:
                QtWidgets.QMessageBox.information(self.window, "Inspect body", "Select a body by clicking one of its faces.")
                return
            data = measure_model(list(manager.current_part.solids())[index])
            dimensions = " × ".join(f"{value:.4f}" for value in data["dimensions"])
            QtWidgets.QMessageBox.information(self.window, "Inspect body",
                f"Volume: {data['volume']:.4f} cubic units\nSurface area: {data['area']:.4f} square units\nBounds: {dimensions} units")

        self.ribbon_group(top_layout, "Inspect", [
            ("inspect", "Measure selected face or profile", inspect_selection),
            ("inspect_model", "Measure model", inspect_model),
            ("inspect_model", "Measure selected edges", inspect_edges),
            ("inspect_model", "Measure selected body", inspect_selected_body),
        ], {"inspect"})

        def fit_view():
            self.plotter.reset_camera()
            self.plotter.render()

        def toggle_projection():
            if self.plotter.camera.GetParallelProjection():
                self.plotter.disable_parallel_projection()
                self.window.statusBar().showMessage("Perspective view", 3000)
            else:
                self.plotter.enable_parallel_projection()
                self.window.statusBar().showMessage("Orthographic view", 3000)
            self.plotter.render()

        def zoom_selection():
            manager = self.action_manager
            face = None
            if manager.selected_profile_index is not None and manager.selected_profile_index < len(manager.profiles):
                face = manager.profiles[manager.selected_profile_index]["face"]
            elif manager.current_part is not None and manager.selected_face_id >= 0:
                faces = list(manager.current_part.faces())
                if manager.selected_face_id < len(faces):
                    face = faces[manager.selected_face_id]
            if face is None:
                self.window.statusBar().showMessage("Select a profile or face first.", 3000)
                return
            box = face.bounding_box()
            bounds = (box.min.X, box.max.X, box.min.Y, box.max.Y, box.min.Z, box.max.Z)
            self.plotter.reset_camera(bounds=bounds)
            self.plotter.render()

        def zoom_selected_body():
            manager = self.action_manager
            index = manager.selected_body_index()
            if index is None:
                self.window.statusBar().showMessage("Select a body by clicking one of its faces.", 3000)
                return
            box = list(manager.current_part.solids())[index].bounding_box()
            bounds = (box.min.X, box.max.X, box.min.Y, box.max.Y, box.min.Z, box.max.Z)
            self.plotter.reset_camera(bounds=bounds)
            self.plotter.render()

        self.ribbon_group(top_layout, "View", [
            ("fit", "Zoom to fit", fit_view),
            ("projection", "Toggle orthographic / perspective", toggle_projection),
            ("view_menu", "Zoom to selection", zoom_selection),
            ("view_menu", "Zoom to selected body", zoom_selected_body),
            ("view_menu", "Look at selected face", self.look_at_selection),
            ("view_menu", "Top (Ctrl+Alt+1)", lambda: self.set_standard_view("top")),
            ("view_menu", "Bottom (Ctrl+Alt+2)", lambda: self.set_standard_view("bottom")),
            ("view_menu", "Front (Ctrl+Alt+3)", lambda: self.set_standard_view("front")),
            ("view_menu", "Back (Ctrl+Alt+4)", lambda: self.set_standard_view("back")),
            ("view_menu", "Right (Ctrl+Alt+5)", lambda: self.set_standard_view("right")),
            ("view_menu", "Left (Ctrl+Alt+6)", lambda: self.set_standard_view("left")),
            ("view_menu", "Isometric (Ctrl+Alt+7)", lambda: self.set_standard_view("isometric")),
        ], {"fit", "projection"})
        top_layout.addStretch()
        self.tool_tabs_widgets["solid"] = tool_bar
        return tool_bar
    
    def create_scetch_tab(self):
        tool_bar, top_layout = self.ribbon_layout()
        
        def start_line():
            if not self.action_manager.sketch_editing:
                QtWidgets.QMessageBox.information(self.window, "Sketch", "Select a sketch plane first.")
                return
            self.disable_active_tool()
            self.action_manager.activate_line_mode()
        def start_circle():
            if not self.action_manager.sketch_editing:
                QtWidgets.QMessageBox.information(self.window, "Sketch", "Select a sketch plane first.")
                return
            self.disable_active_tool()
            self.action_manager.activate_circle_mode()
        def start_curve():
            if not self.action_manager.sketch_editing:
                QtWidgets.QMessageBox.information(self.window, "Sketch", "Select a sketch plane first.")
                return
            self.disable_active_tool()
            self.action_manager.activate_curve_mode()
        def start_ellipse_arc():
            if not self.action_manager.sketch_editing:
                QtWidgets.QMessageBox.information(self.window, "Sketch", "Select a sketch plane first.")
                return
            self.disable_active_tool()
            self.action_manager.activate_ellipse_arc_mode()

        def start_spline(kind):
            if not self.action_manager.sketch_editing:
                QtWidgets.QMessageBox.information(self.window, 'Sketch', 'Select a sketch plane first.')
                return
            self.disable_active_tool()
            self.action_manager.activate_spline_mode(kind)

        def start_parametric_slot(method):
            try:
                getattr(scene_commands(), method)()
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self.window, 'Slot', str(exc))
            
        def exit_sketch():
            self.disable_active_tool()
            try:
                self.action_manager.exit_sketch_mode()
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self.window, 'Sketch', str(exc))
                return
            self.tool_tabs.setCurrentWidget(self.tool_tabs_widgets["solid"])
            if not self.action_manager.profiles:
                QtWidgets.QMessageBox.information(self.window, "Sketch", "No closed profile was found. Connect the endpoints to form a closed contour.")

        def start_shape(kind):
            if not self.action_manager.sketch_editing:
                QtWidgets.QMessageBox.information(self.window, "Sketch", "Select a sketch plane first.")
                return
            sides = 6
            if kind in ("polygon", "circumscribed_polygon", "edge_polygon"):
                sides, ok = self.panel_inputs.getInt(self.window, "Polygon", "Number of sides:", 6, 3, 128)
                if not ok:
                    return
            self.disable_active_tool()
            self.action_manager.activate_straight_shape(kind, sides)

        def start_point_circle(mode):
            if not self.action_manager.sketch_editing:
                QtWidgets.QMessageBox.information(self.window, "Sketch", "Select a sketch plane first.")
                return
            self.disable_active_tool()
            self.action_manager.activate_point_circle(mode)

        def start_point():
            if not self.action_manager.sketch_editing:
                QtWidgets.QMessageBox.information(self.window, "Sketch", "Select a sketch plane first.")
                return
            self.disable_active_tool()
            self.action_manager.activate_sketch_point()

        def start_slot(mode):
            if not self.action_manager.sketch_editing:
                QtWidgets.QMessageBox.information(self.window, "Sketch", "Select a sketch plane first.")
                return
            self.disable_active_tool()
            self.action_manager.activate_slot(mode)

        def start_arc_variant(mode):
            if not self.action_manager.sketch_editing:
                QtWidgets.QMessageBox.information(self.window, "Sketch", "Select a sketch plane first.")
                return
            self.disable_active_tool()
            if mode == "center":
                self.action_manager.activate_center_arc()
            else:
                self.action_manager.activate_elliptical_arc()

        create_commands = [
            ("line", "Line", start_line),
            ("midpoint_line", "Midpoint line", lambda: start_shape("midpoint_line")),
            ("rectangle", "2-point rectangle", lambda: start_shape("rectangle")),
            ("center_rectangle", "Center rectangle", lambda: start_shape("center_rectangle")),
            ("three_point_rectangle", "3-point rectangle", lambda: start_shape("three_point_rectangle")),
            ("circle", "Center circle", start_circle),
            ("circle_2point", "2-point circle", lambda: start_point_circle("diameter")),
            ("circle_3point", "3-point circle", lambda: start_point_circle("three_point")),
            ("curve", "3-point arc", start_curve),
            ("spline", "Bezier spline", lambda: start_spline('spline')),
            ("fit_spline", "Fit-point spline", lambda: start_spline('fit_spline')),
            ("nurbs", "NURBS", lambda: start_spline('nurbs')),
            ("periodic_nurbs", "Periodic NURBS", lambda: start_spline('periodic_nurbs')),
            ("center_arc", "Center-point arc", lambda: start_arc_variant("center")),
            ("ellipse", "Ellipse", start_ellipse_arc),
            ("ellipse_arc", "Elliptical arc", lambda: start_arc_variant("ellipse")),
            ("point", "Point", start_point),
            ("slot", "Center-to-center slot", lambda: start_slot("center_to_center")),
            ("slot_overall", "Overall slot", lambda: start_slot("overall")),
            ("slot_center", "Center-point slot", lambda: start_slot("center_point")),
            ("slot_polyline", "Polyline slot", lambda: start_parametric_slot('create_polyline_slot')),
            ("slot_arc", "Arc slot", lambda: start_parametric_slot('create_arc_slot')),
            ("slot_spline", "Spline slot", lambda: start_parametric_slot('create_spline_slot')),
            ("slot_composite", "Composite curve slot", lambda: start_parametric_slot('create_composite_slot')),
            ("polygon", "Inscribed polygon", lambda: start_shape("polygon")),
            ("circumscribed_polygon", "Circumscribed polygon", lambda: start_shape("circumscribed_polygon")),
            ("edge_polygon", "Edge polygon", lambda: start_shape("edge_polygon")),
        ]
        self.ribbon_group(top_layout, "Create", create_commands,
                          {"line", "rectangle", "circle", "curve", "ellipse", "spline"})
        def scene_commands():
            if not self.action_manager.sketch_editing:
                raise ValueError('Select a sketch plane first.')
            self.disable_active_tool()
            manager = self.action_manager
            if manager.history_edit_id is None:
                manager.promote_current_sketch()
            else:
                manager.flush_history_tools()
            from UI.main.scene_sketch_commands import SceneSketchCommands
            return SceneSketchCommands(manager, self.window, self.panel_inputs)

        def open_constraints():
            try:
                manager = self.action_manager
                if manager.sketch_editing and manager.history_edit_id is None and manager.parametric_graph is None and manager.current_part is not None:
                    from UI.main.constraint_dialog import ConstraintDialog
                    self.disable_active_tool()
                    self.panel_inputs.exec_dialog(ConstraintDialog(manager, self.window))
                    return
                scene_commands().create_constraint()
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self.window, 'Constraints', str(exc))

        constraint_button = QtWidgets.QToolButton()
        constraint_button.setProperty("ribbon", True)
        constraint_button.setText("Constraints")
        constraint_button.setToolTip("Sketch constraints and dimensions")
        constraint_button.clicked.connect(open_constraints)
        top_layout.addWidget(constraint_button)
        remove_constraints = QtWidgets.QToolButton()
        remove_constraints.setProperty('ribbon', True)
        remove_constraints.setText('Remove constraint')
        def remove_constraint():
            try: scene_commands().remove_constraint()
            except Exception as exc: QtWidgets.QMessageBox.warning(self.window, 'Constraints', str(exc))
        remove_constraints.clicked.connect(remove_constraint)
        top_layout.addWidget(remove_constraints)
        nurbs_parameters = QtWidgets.QToolButton()
        nurbs_parameters.setProperty('ribbon', True)
        nurbs_parameters.setText('NURBS parameters')
        def edit_nurbs():
            try: scene_commands().edit_nurbs_parameters()
            except Exception as exc: QtWidgets.QMessageBox.warning(self.window, 'NURBS', str(exc))
        nurbs_parameters.clicked.connect(edit_nurbs)
        top_layout.addWidget(nurbs_parameters)

        grid_button = QtWidgets.QToolButton()
        grid_button.setProperty("ribbon", True)
        grid_button.setText("Grid ▾")
        grid_button.setToolTip("Grid visibility, spacing and snapping")
        grid_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        grid_menu = QtWidgets.QMenu(grid_button)
        visible_action = grid_menu.addAction("Show grid")
        visible_action.setCheckable(True)
        visible_action.setChecked(self.action_manager.grid_visible)
        visible_action.toggled.connect(self.action_manager.set_grid_visible)
        grid_snap_action = grid_menu.addAction("Snap to grid")
        grid_snap_action.setCheckable(True)
        grid_snap_action.setChecked(self.action_manager.snap_to_grid)
        grid_snap_action.toggled.connect(lambda value: setattr(self.action_manager, "snap_to_grid", value))
        point_snap_action = grid_menu.addAction("Snap to points and intersections")
        point_snap_action.setCheckable(True)
        point_snap_action.setChecked(self.action_manager.snap_to_points)
        point_snap_action.toggled.connect(lambda value: setattr(self.action_manager, "snap_to_points", value))
        grid_menu.addSeparator()
        spacing_widget = QtWidgets.QWidget()
        spacing_layout = QtWidgets.QHBoxLayout(spacing_widget)
        spacing_layout.setContentsMargins(8, 2, 8, 2)
        spacing_layout.addWidget(QtWidgets.QLabel("Spacing"))
        spacing = QtWidgets.QDoubleSpinBox()
        spacing.setObjectName("gridSpacing")
        spacing.setDecimals(4)
        spacing.setRange(0.0001, 1000000)
        spacing.setValue(self.action_manager.grid_step)
        spacing.setKeyboardTracking(False)
        spacing.valueChanged.connect(self.action_manager.set_grid_step)
        spacing_layout.addWidget(spacing)
        spacing_action = QtWidgets.QWidgetAction(grid_menu)
        spacing_action.setDefaultWidget(spacing_widget)
        grid_menu.addAction(spacing_action)
        grid_button.setMenu(grid_menu)
        top_layout.addWidget(grid_button)

        construction = QtWidgets.QCheckBox("Construction")
        construction.setToolTip("New sketch geometry is a reference and will not create profiles")
        construction.setChecked(self.action_manager.construction_mode)
        def toggle_construction(checked):
            self.disable_active_tool()
            self.action_manager.construction_mode = checked
        construction.toggled.connect(toggle_construction)
        top_layout.addWidget(construction)
        finish = QtWidgets.QToolButton()
        finish.setObjectName("finishSketch")
        finish.setIcon(tool_icon("finish"))
        finish.setIconSize(QtCore.QSize(40, 40))
        finish.setFixedSize(49, 49)
        finish.setAccessibleName("Finish sketch")
        finish.setToolTip("Finish the current sketch and find its profiles")
        finish.clicked.connect(exit_sketch)
        top_layout.addWidget(finish)

        top_layout.addStretch() 
        self.tool_tabs_widgets["sketch"] = tool_bar
        return tool_bar
    
    def disable_active_tool(self):
        manager = self.action_manager
        tool = manager.active_tool
        if tool is None:
            return
        tool.stop_drawing()
        # Remove unfinished input while retaining completed line segments and shapes.
        complete = bool(tool.points) if tool.type == 'point' else bool(sketch_edges([tool], manager.active_normal))
        if not complete:
            sketch = manager.get_current_sketch_data()
            if sketch:
                for index, candidate in enumerate(sketch["sketch_func"]):
                    if candidate is tool:
                        del sketch["sketch_func"][index]
                        break
            for actor in tool.render_obj["points"]:
                manager.plotter.remove_actor(actor)
        manager.active_tool = None
