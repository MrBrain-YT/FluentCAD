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
from UI.main.navigation_cube import NavigationCube
from UI.main.camera_transition import CameraTransition
from UI.main.mouse_navigation import MouseNavigation
from UI.main.workspace_controls import ViewportNavigation, chrome_button
from UI.main.extrude_dialog import ExtrudeDialog, RevolveDialog, PrimitiveDialog, TransformDialog, SolidFeatureDialog
from UI.main.sweep_dialog import SweepDialog
from UI.main.loft_dialog import LoftDialog
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
        self.command_actions = {}
        self.global_actions = []
        self.document_path = None
        self.window = window
        self.central_widget = central_widget
        self.theme_settings = QtCore.QSettings('FluentCAD', 'FluentCAD')
        self.theme_mode = self.theme_settings.value('appearance/theme', 'system')
        self.main_layout = self.get_main_layout()
        self.plotter: Union[QtInteractor, pv.Plotter] = None 
        self.action_manager = ActionManager(self.plotter)
        self.action_manager.status_callback = lambda message: self.window.statusBar().showMessage(message)
        self.add_workspace()
        self.add_3d_viewport()
        self.panel_inputs = PanelInputs(self)
        self.setup_cad_viewport()
        self.add_style()
        QtGui.QGuiApplication.styleHints().colorSchemeChanged.connect(self.system_theme_changed)
        self.action_manager.update_plotter(self.plotter)
        self.mouse_navigation = MouseNavigation(self.plotter, self.action_manager)
        self.action_manager_ui = ActionManagerUI(self.tool_tabs_widgets)
        self.action_manager.timeline_callback = self.refresh_timeline
        self.refresh_timeline()
        self.shortcuts = []
        def cancel_selection():
            if self.stop_sketch_modifier():
                return
            was_selecting_plane = self.action_manager.selecting_sketch_plane
            cancelled = self.action_manager.cancel_pending_selection()
            if cancelled and was_selecting_plane:
                self.leave_sketch_workspace()
            elif not cancelled and self.action_manager.sketch_editing and self.action_manager.active_tool is not None:
                self.disable_active_tool()
                self.action_manager.status_callback("Sketch tool stopped.")
            elif not cancelled and self.action_manager.history_edit_id is not None:
                self.action_manager.cancel_history_sketch()
                self.leave_sketch_workspace()
        escape = QtGui.QShortcut(QtGui.QKeySequence("Escape"), self.window)
        escape.activated.connect(cancel_selection)
        self.shortcuts.append(escape)
        def finish_spline():
            modifier = self.action_manager.sketch_modifier
            if modifier is not None and modifier.command == 'offset_chain':
                self.sketch_offset_distance.interpretText()
                modifier.confirm()
                return
            if modifier is not None and modifier.command in ('fillet', 'chamfer'):
                self.sketch_corner_size.interpretText()
                modifier.confirm()
                return
            if modifier is not None and modifier.command == 'sketch_pattern':
                modifier.confirm()
                return
            if modifier is not None and modifier.command == 'sketch_scale':
                self.sketch_scale_factor.interpretText()
                modifier.confirm()
                return
            if modifier is not None and modifier.command == 'sketch_move':
                for field in self.sketch_move_fields:
                    field.interpretText()
                modifier.confirm()
                return
            if modifier is not None and modifier.command == 'sketch_mirror':
                modifier.confirm()
                return
            if self.stop_sketch_modifier() or self.action_manager.finish_profile_selection():
                return
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
        for key, callback in (('Ctrl+O', self.command_actions['open'].trigger),
                              ('Ctrl+S', self.command_actions['save'].trigger),
                              ('Ctrl+Alt+B', self.browser_toggle.toggle),
                              ('Ctrl+Alt+N', self.toggle_navigation),
                              ('S', self.show_command_search)):
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self.window)
            shortcut.activated.connect(callback)
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
                if not form.isRowVisible(row):
                    continue
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
        if hasattr(dialog, 'indices'):
            parameters['Profile indices'] = ', '.join(str(index + 1) for index in dialog.indices)
        elif hasattr(dialog, 'index'):
            parameters['Profile index'] = dialog.index + 1
        if getattr(dialog, 'axis_label', None):
            parameters['Axis geometry'] = dialog.axis_label
        if getattr(dialog, 'target_label', None) and dialog.extent.currentData() == 'to_object':
            parameters['Target geometry'] = dialog.target_label
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
        if hasattr(self, 'model_browser'):
            self.model_browser.refresh(self.action_manager)
        self.timeline_widget.blockSignals(True)
        self.timeline_widget.clear()
        start = QtWidgets.QListWidgetItem(tool_icon('timeline_first'), 'Start')
        self.timeline_widget.addItem(start)
        if graph is not None:
            icons = {'planar_sketch': 'sketch', 'rectangle_sketch': 'sketch', 'datum_plane': 'plane',
                     'frozen_body': 'box', 'fillet': 'fillet_all', 'chamfer': 'chamfer_all'}
            for index, feature in enumerate(graph.features):
                key = (feature.parameters.get('type', 'box') if feature.kind == 'primitive'
                       else icons.get(feature.kind, feature.kind))
                item = QtWidgets.QListWidgetItem(tool_icon(key), feature.name)
                if feature.id in graph.blocked:
                    item.setForeground(self.timeline_widget.palette().color(QtGui.QPalette.PlaceholderText))
                    item.setToolTip(graph.blocked[feature.id])
                elif issues := graph.link_issues.get(feature.id):
                    item.setText(item.text() + f' · {len(issues)} lost link(s)')
                    item.setToolTip('\n'.join(issues.values()))
                else:
                    hint = 'Double-click to edit Loft' if feature.kind == 'loft' else 'Right-click for feature actions'
                    item.setToolTip(f'{index + 1}. {feature.name}\n{hint}')
                self.timeline_widget.addItem(item)
            self.timeline_widget.setCurrentRow(len(graph.features))
            self.timeline_widget.blockSignals(False)
            self.update_timeline_controls()
            return
        for index, entry in enumerate(timeline.entries):
            title = entry['title']
            key = title.casefold().split(maxsplit=1)[0] if title.strip() else 'feature'
            item = QtWidgets.QListWidgetItem(tool_icon(key), title)
            if index >= timeline.cursor:
                item.setForeground(self.timeline_widget.palette().color(QtGui.QPalette.PlaceholderText))
            item.setToolTip(f'{index + 1}. {title}\nDouble-click to inspect parameters')
            self.timeline_widget.addItem(item)
        self.timeline_widget.setCurrentRow(timeline.cursor)
        self.timeline_widget.blockSignals(False)
        self.update_timeline_controls()

    def navigate_timeline(self, row):
        if row < 0:
            return
        manager = self.action_manager
        if manager.interaction_locked:
            self.refresh_timeline()
            return
        if manager.parametric_graph is not None:
            self.update_timeline_controls()
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
        if self.action_manager.interaction_locked:
            return
        index = self.timeline_widget.row(item) - 1
        graph = self.action_manager.parametric_graph
        if graph is not None:
            if 0 <= index < len(graph.features):
                feature = graph.features[index]
                if feature.kind == 'loft' and feature.id in graph.results:
                    try:
                        self.edit_loft(feature.id)
                    except ValueError as exc:
                        self.action_manager.status_callback(str(exc))
                    return
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
        loft_action = menu.addAction('Edit Loft') if feature.kind == 'loft' else None
        if loft_action is not None:
            loft_action.setEnabled(feature.id in graph.results)
        relink_action = menu.addAction('Relink sketch geometry…') if feature.parameters.get('links') else None
        fallback_action = menu.addAction('Edit sketch in panel') if feature.kind == 'planar_sketch' else None
        profiles_action = menu.addAction('Show sketch profiles') if feature.kind == 'planar_sketch' else None
        reselect_text = {'extrude': 'Reselect extrusion profile', 'revolve': 'Reselect revolution profile',
                         'sweep': 'Reselect sweep profile'}
        reselect_action = (menu.addAction(reselect_text[feature.kind])
                           if feature.kind in reselect_text and 'profile_faces' not in feature.parameters else None)
        edit_action = menu.addAction('Edit parameters (JSON)')
        rename_action = menu.addAction('Rename')
        suppress_action = menu.addAction('Unsuppress' if feature.suppressed else 'Suppress')
        left_action = menu.addAction('Move earlier')
        right_action = menu.addAction('Move later')
        chosen = menu.exec(self.timeline_widget.mapToGlobal(position))
        if chosen is None:
            return
        try:
            if loft_action is not None and chosen == loft_action:
                self.edit_loft(feature.id)
                return
            if profiles_action is not None and chosen == profiles_action:
                self.action_manager.show_parametric_profiles(feature.id)
                self.action_manager.status_callback('Click a sketch profile, then choose Extrude or Revolve.')
                return
            if reselect_action is not None and chosen == reselect_action:
                self.reselect_feature_profile(feature.id)
                return
            if (sketch_action is not None and chosen == sketch_action) or (relink_action is not None and chosen == relink_action):
                self.action_manager.start_history_sketch(feature.id)
                self.action_manager_ui.activate_scetch_mode(self.tool_tabs, 'sketch')
                if chosen == relink_action:
                    self.manage_sketch_link(relink=True)
                return
            candidate = FeatureGraph.from_data(graph.to_data())
            if chosen == fallback_action and fallback_action is not None:
                dialog = ParametricSketchDialog(candidate, feature.id, self.window, self.panel_inputs)
                if self.panel_inputs.exec_dialog(dialog) != QtWidgets.QDialog.Accepted:
                    return
                candidate = dialog.graph
            elif chosen == edit_action:
                parameters = feature.parameters
                saved_links = {link['id']: link for link in parameters.get('links', [])}
                if saved_links:
                    parameters = {**parameters, 'links': [
                        {key: value for key, value in link.items() if key != 'fallback'} for link in saved_links.values()
                    ]}
                text, ok = self.panel_inputs.getMultiLineText(self.window, 'Edit parameters',
                    'Feature parameters (JSON):', json.dumps(parameters, ensure_ascii=False, indent=2))
                if not ok: return
                value = json.loads(text)
                if not isinstance(value, dict): raise ValueError('Parameters must be a JSON object.')
                for link in value.get('links', []):
                    if not isinstance(link, dict):
                        continue
                    saved = saved_links.get(link.get('id'))
                    if saved and 'fallback' in saved and link == {key: data for key, data in saved.items() if key != 'fallback'}:
                        link['fallback'] = saved['fallback']
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

    def edit_loft(self, feature_id):
        manager = self.action_manager
        if manager.interaction_locked or manager.sketch_editing or manager.pending_sketch or manager.selecting_sketch_plane:
            raise ValueError('Finish or cancel the active scene operation first.')
        self.show_scene_operation(LoftDialog(manager, self.window, feature_id=feature_id), 'Edit Loft')

    def reselect_feature_profile(self, feature_id):
        manager = self.action_manager
        graph = manager.parametric_graph
        feature = next((item for item in graph.features if item.id == feature_id), None)
        if feature is None or feature.kind not in ('extrude', 'revolve', 'sweep') or not feature.input_id:
            raise ValueError('Choose a feature with a sketch dependency.')
        sketch_id = feature.input_id
        profiles = manager.show_parametric_profiles(sketch_id)
        if not profiles: raise ValueError('The sketch has no closed profiles.')
        def selected(index):
            try:
                current = manager.parametric_graph
                operation = next(item for item in current.features if item.id == feature_id)
                sketch = current.results[sketch_id]
                indices = (index,) if isinstance(index, int) else index
                if any(not 0 <= item < len(sketch.profiles) for item in indices):
                    raise ValueError('Profile is no longer available.')
                selections = [ActionManager.profile_selection_parameters(sketch, item) for item in indices]
                params = dict(operation.parameters)
                for name in ('profiles', 'profile_index', 'profile_count', 'profile_key'):
                    params.pop(name, None)
                if len(selections) == 1:
                    params.update(selections[0])
                else:
                    params['profiles'] = selections
                candidate = FeatureGraph.from_data(current.to_data())
                candidate.edit(feature_id, parameters=params)
                manager.install_parametric_graph(candidate, allow_empty=True)
                if manager.current_part is None: manager.show_parametric_profiles(sketch_id)
                manager.status_callback('Sketch profile updated.' + (' Unsuppress the operation to rebuild it.' if operation.suppressed else ''))
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self.window, 'Sketch profile', str(exc))
        if 'profiles' in feature.parameters:
            manager.request_profile_selection(selected, multiple=True)
        else:
            manager.request_profile_selection(selected)

    def edit_graph_parameters(self):
        manager = self.action_manager
        if manager.sketch_editing or manager.pending_sketch or manager.selecting_sketch_plane:
            manager.status_callback('Finish or cancel the current sketch before editing model parameters.')
            return
        graph = manager.parametric_graph
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
        if manager.current_part is not None or manager.parametric_graph is not None:
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
            self.set_document_path(None)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self.window, 'Parametric model', str(exc))

    def add_graph_feature(self):
        manager = self.action_manager
        if manager.sketch_editing or manager.pending_sketch or manager.selecting_sketch_plane:
            manager.status_callback('Finish or cancel the current sketch before adding a timeline feature.')
            return
        graph = manager.parametric_graph
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
                    templates[kind]['edges'] = [make_reference(input_id, graph.results[input_id], 'edge', i, graph.body_names.get(input_id)) for i in indices]
                else:
                    face_index = self.action_manager.selected_face_id
                    if face_index < 0: raise ValueError('Select a face first.')
                    templates[kind]['face'] = make_reference(input_id, graph.results[input_id], 'face', face_index, graph.body_names.get(input_id))
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
        profile_operation = type(dialog) is ExtrudeDialog or isinstance(dialog, (RevolveDialog, SweepDialog, LoftDialog))
        if profile_operation:
            manager.operation_scene_click = dialog.scene_click
        self.tool_tabs.setEnabled(False)
        if hasattr(self, 'model_commands'):
            self.model_commands.setEnabled(False)
        for shortcut in self.shortcuts:
            shortcut.setEnabled(False)
        def finished(result):
            manager.interaction_locked = False
            if profile_operation:
                manager.operation_scene_click = None
            self.tool_tabs.setEnabled(True)
            if hasattr(self, 'model_commands'):
                self.model_commands.setEnabled(True)
            for shortcut in self.shortcuts:
                shortcut.setEnabled(True)
            if result == QtWidgets.QDialog.Accepted and dialog.result_shape is not None:
                try:
                    if isinstance(dialog, (SweepDialog, LoftDialog)):
                        if manager.parametric_graph is not dialog.graph:
                            raise ValueError(f'The model changed. Open {title} again to refresh the selection.')
                        manager.install_parametric_graph(dialog.candidate)
                    elif manager.parametric_graph is not None and type(dialog) is ExtrudeDialog:
                        manager.commit_parametric_extrusion(*dialog.extrusion_arguments())
                    elif manager.parametric_graph is not None and isinstance(dialog, RevolveDialog):
                        manager.commit_parametric_revolution(dialog.index, dialog.current_axis(),
                                                             dialog.angle.value(), dialog.operation.currentData())
                    elif manager.parametric_graph is not None and isinstance(dialog, (TransformDialog, SolidFeatureDialog)):
                        from UI.main.parametric_operations import commit_parametric_operation
                        commit_parametric_operation(manager, dialog)
                    else:
                        manager.commit_result(dialog.result_shape, dialog.windowTitle(), self.operation_parameters(dialog))
                except Exception as exc:
                    QtWidgets.QMessageBox.warning(self.window, title, str(exc))
            if type(dialog) is ExtrudeDialog:
                manager.clear_temporary_face_profile()
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
        
    def add_workspace(self):
        self.add_application_bar()
        self.tool_tabs = QtWidgets.QTabWidget()
        self.tool_tabs.setDocumentMode(True)
        self.tool_tabs.setFixedHeight(116)
        for key, title, ribbon in (('solid', 'Solid', self.create_solid_tab()),
                                   ('sketch', 'Sketch', self.create_scetch_tab())):
            scroll = QtWidgets.QScrollArea()
            scroll.setObjectName('ribbonScroll')
            scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
            scroll.setWidgetResizable(True)
            scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
            scroll.setWidget(ribbon)
            self.tool_tabs_widgets[key] = scroll
            self.tool_tabs.addTab(scroll, title)
        self.leave_sketch_workspace()
        self.main_layout.addWidget(self.tool_tabs)

        self.render_container = QtWidgets.QFrame()
        self.main_layout.addWidget(self.render_container, 1)
        self.add_timeline()

    def leave_sketch_workspace(self):
        self.tool_tabs.setCurrentWidget(self.tool_tabs_widgets['solid'])
        self.tool_tabs.setTabVisible(self.tool_tabs.indexOf(self.tool_tabs_widgets['sketch']), False)

    def add_application_bar(self):
        bar = QtWidgets.QFrame()
        bar.setObjectName('applicationBar')
        layout = QtWidgets.QHBoxLayout(bar)
        layout.setContentsMargins(10, 4, 10, 4)
        layout.setSpacing(7)
        self.browser_toggle = chrome_button('browser', 'Show or hide model browser (Ctrl+Alt+B)')
        self.browser_toggle.setCheckable(True)
        self.browser_toggle.setChecked(True)
        layout.addWidget(self.browser_toggle)

        self.model_commands = QtWidgets.QWidget(bar)
        actions = QtWidgets.QHBoxLayout(self.model_commands)
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(2)
        file_button = chrome_button('open', 'File commands')
        file_button.setText('File')
        file_button.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        file_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.file_menu = QtWidgets.QMenu(file_button)
        file_button.setMenu(self.file_menu)
        actions.addWidget(file_button)
        actions.addWidget(chrome_button('save', 'Save model (Ctrl+S)', lambda: self.command_actions['save'].trigger()))
        for key, title, callback in (
                ('undo', 'Undo solid operation (Ctrl+Z)', self.action_manager.undo_solid),
                ('redo', 'Redo solid operation (Ctrl+Y)', self.action_manager.redo_solid)):
            action = QtGui.QAction(tool_icon(key), title, bar)
            action.triggered.connect(callback)
            self.global_actions.append(action)
            actions.addWidget(chrome_button(key, title, action.trigger))
        search = chrome_button('inspect', 'Find a command (S)', self.show_command_search)
        actions.addWidget(search)
        layout.addWidget(self.model_commands)
        layout.addStretch(1)
        self.document_label = QtWidgets.QLabel()
        self.document_label.setObjectName('documentLabel')
        self.document_label.setTextFormat(QtCore.Qt.PlainText)
        self.document_label.setMaximumWidth(360)
        self.set_document_path(None)
        layout.addWidget(self.document_label)
        layout.addStretch(1)
        workspace = QtWidgets.QLabel('Design')
        workspace.setProperty('role', 'sectionTitle')
        layout.addWidget(workspace)
        self.main_layout.addWidget(bar)

    def set_document_path(self, path):
        self.document_path = path
        name = Path(path).stem if path else 'Untitled model'
        text = QtGui.QFontMetrics(self.document_label.font()).elidedText(name, QtCore.Qt.ElideMiddle, 334)
        self.document_label.setText(text)
        self.document_label.setToolTip(str(path) if path else name)

    def show_command_search(self):
        if self.action_manager.interaction_locked:
            return
        actions = [*self.tool_tabs.currentWidget().findChildren(QtGui.QAction), *self.global_actions]
        commands = [action for action in actions if action.isEnabled() and action.text() and not action.isSeparator()]
        dialog = QtWidgets.QDialog(self.window)
        dialog.setWindowTitle('Find a command')
        dialog.resize(420, 310)
        layout = QtWidgets.QVBoxLayout(dialog)
        field = QtWidgets.QLineEdit()
        field.setPlaceholderText('Search commands')
        field.setAccessibleName('Search commands')
        results = QtWidgets.QListWidget()
        results.setAccessibleName('Matching commands')
        layout.addWidget(field)
        layout.addWidget(results, 1)
        matches = []

        def filter_commands(text):
            words = text.casefold().split()
            matches[:] = [action for action in commands if all(word in action.text().casefold() for word in words)]
            results.clear()
            for action in matches:
                results.addItem(QtWidgets.QListWidgetItem(action.icon(), action.text()))
            if matches:
                results.setCurrentRow(0)

        field.textChanged.connect(filter_commands)
        field.returnPressed.connect(lambda: dialog.accept() if results.currentRow() >= 0 else None)
        results.itemActivated.connect(lambda _item: dialog.accept())
        next_result = QtGui.QShortcut(QtGui.QKeySequence('Down'), field)
        next_result.setContext(QtCore.Qt.WidgetShortcut)
        next_result.activated.connect(lambda: results.setCurrentRow(min(len(matches) - 1, results.currentRow() + 1)))
        previous_result = QtGui.QShortcut(QtGui.QKeySequence('Up'), field)
        previous_result.setContext(QtCore.Qt.WidgetShortcut)
        previous_result.activated.connect(lambda: results.setCurrentRow(max(0, results.currentRow() - 1)) if matches else None)
        filter_commands('')
        field.setFocus()
        if dialog.exec() == QtWidgets.QDialog.Accepted and 0 <= results.currentRow() < len(matches):
            matches[results.currentRow()].trigger()

    def add_timeline(self):
        strip = QtWidgets.QFrame()
        strip.setObjectName('timelineBar')
        layout = QtWidgets.QHBoxLayout(strip)
        layout.setContentsMargins(12, 4, 8, 4)
        layout.setSpacing(5)
        label = QtWidgets.QLabel('Timeline')
        label.setProperty('role', 'sectionTitle')
        layout.addWidget(label)
        self.timeline_controls = []
        for icon, title, step in (
                ('timeline_first', 'First timeline entry', 'first'),
                ('timeline_prev', 'Previous timeline entry', -1),
                ('timeline_next', 'Next timeline entry', 1),
                ('timeline_last', 'Last timeline entry', 'last')):
            button = chrome_button(icon, title, lambda _checked=False, step=step: self.step_timeline(step))
            self.timeline_controls.append(button)
            layout.addWidget(button)
        self.timeline_widget = QtWidgets.QListWidget()
        self.timeline_widget.setObjectName("solidTimeline")
        self.timeline_widget.setAccessibleName('Model timeline')
        self.timeline_widget.setIconSize(QtCore.QSize(20, 20))
        self.timeline_widget.setFlow(QtWidgets.QListView.LeftToRight)
        self.timeline_widget.setWrapping(False)
        self.timeline_widget.setFixedHeight(44)
        self.timeline_widget.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.timeline_widget.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.timeline_widget.currentRowChanged.connect(self.navigate_timeline)
        self.timeline_widget.itemDoubleClicked.connect(self.inspect_timeline_entry)
        self.timeline_widget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.timeline_widget.customContextMenuRequested.connect(self.timeline_context_menu)
        layout.addWidget(self.timeline_widget, 1)
        self.main_layout.addWidget(strip)

    def update_timeline_controls(self):
        if not hasattr(self, 'timeline_controls'):
            return
        row, last = self.timeline_widget.currentRow(), self.timeline_widget.count() - 1
        for index, button in enumerate(self.timeline_controls):
            button.setEnabled(row > 0 if index < 2 else row < last)

    def step_timeline(self, step):
        if self.action_manager.interaction_locked:
            return
        last = self.timeline_widget.count() - 1
        row = 0 if step == 'first' else last if step == 'last' else self.timeline_widget.currentRow() + step
        self.timeline_widget.setCurrentRow(min(last, max(0, row)))
        
    def add_3d_viewport(self):
        render_layout = QtWidgets.QHBoxLayout(self.render_container)
        render_layout.setContentsMargins(0,0,0,0)
        render_layout.setSpacing(0)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self.render_container)
        splitter.setChildrenCollapsible(False)
        render_layout.addWidget(splitter)
        from UI.main.model_browser import ModelBrowser
        self.model_browser = ModelBrowser(splitter, overlay=False)
        self.model_browser.setMinimumWidth(136)
        self.model_browser.setMaximumWidth(360)
        self.browser_toggle.toggled.connect(self.model_browser.setVisible)
        self.viewport = QtWidgets.QFrame(splitter)
        viewport_layout = QtWidgets.QVBoxLayout(self.viewport)
        viewport_layout.setContentsMargins(0, 0, 0, 0)
        self.plotter = QtInteractor(self.viewport)
        viewport_layout.addWidget(self.plotter)
        splitter.addWidget(self.model_browser)
        splitter.addWidget(self.viewport)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([214, 950])
        self.plotter.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.plotter.customContextMenuRequested.connect(self.show_scene_context_menu)
        self.add_viewport_navigation()
        self.operation_panel = OperationPanel(self.viewport)
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
        self.operation_content.setObjectName('operationContent')
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
        self.operation_content.installEventFilter(self.operation_panel)
        self.operation_panel.scroll = self.operation_scroll
        self.operation_title.toggled.connect(self.operation_scroll.setVisible)
        self.operation_title.toggled.connect(lambda expanded: self.operation_title.setArrowType(
            QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow))
        self.operation_title.toggled.connect(self.operation_panel.queue_reposition)
        self.operation_panel.hide()

    def add_viewport_navigation(self):
        navigation = self.navigation = ViewportNavigation(self.viewport)
        layout = navigation.layout()
        layout.addWidget(chrome_button('view_menu', 'Isometric view', lambda: self.set_standard_view('isometric')))
        for key in ('fit', 'projection'):
            action = self.command_actions[key]
            layout.addWidget(chrome_button(key, action.text(), action.trigger))
        views = chrome_button('view_options', 'View and display settings')
        views.setMenu(self.view_menu)
        views.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        layout.addWidget(views)
        navigation.reposition()

    def toggle_navigation(self):
        self.navigation.setVisible(self.navigation.isHidden())
        
    def setup_cad_viewport(self):
        self.plotter.set_background("#CECECE", top="#CECECE")
        # Intersecting translucent datum planes must be composited per pixel;
        # sorting whole plane actors makes them appear cut at their crossings.
        self.plotter.ren_win.SetAlphaBitPlanes(1)
        self.plotter.ren_win.SetMultiSamples(0)
        self.plotter.renderer.SetUseDepthPeeling(True)
        self.plotter.renderer.SetMaximumNumberOfPeels(8)
        self.plotter.renderer.SetOcclusionRatio(0.0)
        self.camera_transition = CameraTransition(self.viewport, self.plotter, self.action_manager)
        self.navigation_cube = NavigationCube(self.viewport, self.plotter.camera, self.camera_transition.home)
        self.navigation_cube.viewRequested.connect(self.camera_transition.orient)
        self.navigation_cube.dragRequested.connect(self.camera_transition.drag)
        self.navigation_cube.gestureStarted.connect(self.camera_transition.stop)
        self.operation_panel.navigation_size = lambda: (self.navigation_cube.width() + 8,
                                                        self.navigation_cube.height() + 8)
        self.operation_panel.queue_reposition()

        # 3. ОСИ В НАЧАЛЕ КООРДИНАТ (0,0,0)
        # Чтобы всегда понимать, где центр мира
        self.plotter.add_axes_at_origin(labels_off=True, line_width=5)

        self.plotter.enable_lightkit() 
        self.plotter.enable_parallel_projection()
        self.plotter.view_isometric()

    def show_scene_context_menu(self, position):
        manager = self.action_manager
        if manager.interaction_locked or getattr(manager, 'sketch_modifier', None) is not None:
            return
        if manager.sketch_editing:
            selection = manager.history_selection if manager.history_edit_id is not None else None
            if selection is None:
                return
            menu = QtWidgets.QMenu(self.plotter)
            if selection[0] == 'point':
                feature = next(item for item in manager.history_edit_graph.features if item.id == manager.history_edit_id)
                fixed = any(c.get('kind') == 'fixed' and c.get('entities') == [selection[1]]
                            for c in feature.parameters.get('constraints', []))
                menu.addAction('Unfix point' if fixed else 'Fix point', lambda: self.edit_sketch_selection('unfix' if fixed else 'fix'))
                menu.addAction('Set point coordinates', lambda: self.edit_sketch_selection('coordinates'))
            elif selection[0] == 'entity':
                menu.addAction('Toggle construction geometry', lambda: self.edit_sketch_selection('construction'))
                feature = next(item for item in manager.history_edit_graph.features if item.id == manager.history_edit_id)
                entity = feature.parameters['entities'][selection[1]]
                if entity.get('type') == 'line' and len(entity.get('points', [])) == 2:
                    menu.addAction('Horizontal', lambda: self.edit_sketch_selection('horizontal'))
                    menu.addAction('Vertical', lambda: self.edit_sketch_selection('vertical'))
                    menu.addAction('Length dimension', lambda: self.edit_sketch_selection('length'))
                if entity.get('type') in ('circle', 'arc'):
                    menu.addAction('Radius dimension', lambda: self.edit_sketch_selection('radius'))
                    menu.addAction('Diameter dimension', lambda: self.edit_sketch_selection('diameter'))
                if entity.get('type') in ('nurbs', 'periodic_nurbs'):
                    menu.addAction('NURBS weights and knots', lambda: self.edit_sketch_selection('nurbs'))
                if entity.get('type') in ('spline', 'fit_spline', 'nurbs'):
                    menu.addAction('Tangent to curve', lambda: self.edit_sketch_selection('spline_tangent'))
                    menu.addAction('Curvature to curve', lambda: self.edit_sketch_selection('spline_curvature'))
                if entity.get('type') in ('ellipse', 'slot', 'polyline_slot', 'arc_slot', 'spline_slot', 'composite_path_slot'):
                    menu.addAction('Edit geometry parameters', lambda: self.edit_sketch_selection('geometry'))
                if entity.get('type') in ('line', 'arc', 'circle', 'ellipse', 'spline', 'fit_spline', 'nurbs', 'periodic_nurbs'):
                    menu.addAction('Curvature between curves', lambda: self.edit_sketch_selection('curve_curvature'))
                menu.addAction('Add constraint…', lambda: self.edit_sketch_selection('constraint'))
                menu.addAction('Delete element', lambda: self.edit_sketch_selection('delete'))
            if not menu.isEmpty(): menu.exec(self.plotter.mapToGlobal(position))
            return
        commands = getattr(self, 'scene_context_commands', {})
        menu = QtWidgets.QMenu(self.plotter)
        if manager.selected_edge_indices:
            menu.addAction('Fillet selected edges', lambda: commands['fillet_selected']())
            menu.addAction('Chamfer selected edges', lambda: commands['chamfer_selected']())
        if manager.selected_face_id >= 0:
            menu.addAction('Extrude selected face', commands['extrude'])
            menu.addAction('Fillet face edges', lambda: commands['fillet_faces']())
            menu.addAction('Chamfer face edges', lambda: commands['chamfer_faces']())
            menu.addAction('Hole', lambda: commands['hole']())
            menu.addAction('Shell', lambda: commands['shell']())
        if manager.current_part is not None:
            menu.addAction('Move / Copy', lambda: commands['move']())
        if not menu.isEmpty():
            menu.exec(self.plotter.mapToGlobal(position))

    def edit_sketch_selection(self, action):
        manager = self.action_manager
        selection = manager.history_selection
        try:
            if action == 'fix' and selection[0] == 'point':
                from UI.model.sketch_drag import add_sketch_constraint
                point = manager.history_edit_graph.results[manager.history_edit_id].constraint_status.points[selection[1]]
                manager.history_edit_graph = add_sketch_constraint(manager.history_edit_graph, manager.history_edit_id,
                    {'kind': 'fixed', 'entities': [selection[1]], 'value': list(map(float, point))})
            elif action == 'unfix' and selection[0] == 'point':
                from UI.model.sketch_drag import remove_sketch_constraint
                feature = next(item for item in manager.history_edit_graph.features if item.id == manager.history_edit_id)
                index = next(i for i, entry in enumerate(feature.parameters.get('constraints', []))
                             if entry.get('kind') == 'fixed' and entry.get('entities') == [selection[1]])
                manager.history_edit_graph = remove_sketch_constraint(manager.history_edit_graph, manager.history_edit_id, index)
            elif action == 'coordinates' and selection[0] == 'point':
                point = manager.history_edit_graph.results[manager.history_edit_id].constraint_status.points[selection[1]]
                x, ok = self.panel_inputs.getDouble(self.window, 'Point', 'X:', float(point[0]), -1000000, 1000000, 4)
                if not ok: return
                y, ok = self.panel_inputs.getDouble(self.window, 'Point', 'Y:', float(point[1]), -1000000, 1000000, 4)
                if not ok: return
                manager.move_history_point(selection[1], (x, y))
                return
            elif action in ('horizontal', 'vertical') and selection[0] == 'entity':
                from UI.model.sketch_drag import add_sketch_constraint
                feature = next(item for item in manager.history_edit_graph.features if item.id == manager.history_edit_id)
                entity = feature.parameters['entities'][selection[1]]
                if entity.get('type') != 'line' or len(entity.get('points', [])) != 2:
                    raise ValueError('Select a single line segment.')
                manager.history_edit_graph = add_sketch_constraint(manager.history_edit_graph, manager.history_edit_id,
                    {'kind': action, 'entities': [entity['points']]})
            elif action in ('radius', 'diameter') and selection[0] == 'entity':
                from UI.model.sketch_drag import add_sketch_constraint
                from UI.model.parametric_sketch import circle_radius_name
                from UI.model.constraints import arc_center_radius
                feature = next(item for item in manager.history_edit_graph.features if item.id == manager.history_edit_id)
                entity = feature.parameters['entities'][selection[1]]
                state = manager.history_edit_graph.results[manager.history_edit_id].constraint_status
                if entity['type'] == 'circle':
                    radius = state.radii[circle_radius_name(selection[1])]
                    entities, kind = [[entity['center'], circle_radius_name(selection[1])]], action
                elif entity['type'] == 'arc':
                    _, radius = arc_center_radius(*(state.points[name] for name in entity['points']))
                    entities, kind = [entity['points']], 'arc_' + action
                else:
                    raise ValueError('Select a circle or circular arc.')
                default = radius * (2 if action == 'diameter' else 1)
                value, ok = self.panel_inputs.getText(self.window, 'Dimension', 'Value or parameter expression:', text=str(default))
                if not ok: return
                manager.history_edit_graph = add_sketch_constraint(manager.history_edit_graph, manager.history_edit_id,
                    {'kind': kind, 'entities': entities, 'value': value})
            elif action == 'nurbs' and selection[0] == 'entity':
                from UI.main.scene_sketch_commands import SceneSketchCommands
                SceneSketchCommands(manager, self.window, self.panel_inputs).edit_nurbs_parameters(selection[1])
                return
            elif action in ('spline_tangent', 'spline_curvature') and selection[0] == 'entity':
                from UI.main.scene_sketch_commands import SceneSketchCommands
                mode = 'tangent_spline' if action == 'spline_tangent' else 'curvature_spline'
                SceneSketchCommands(manager, self.window, self.panel_inputs).create_spline_constraint(mode, selection[1])
                return
            elif action == 'curve_curvature' and selection[0] == 'entity':
                from UI.main.scene_sketch_commands import SceneSketchCommands
                SceneSketchCommands(manager, self.window, self.panel_inputs).create_curve_curvature(selection[1])
                return
            elif action == 'geometry' and selection[0] == 'entity':
                graph = FeatureGraph.from_data(manager.history_edit_graph.to_data())
                feature = next(item for item in graph.features if item.id == manager.history_edit_id)
                params = feature.parameters
                entity = params['entities'][selection[1]]
                kind = entity['type']
                if kind == 'slot':
                    names = entity['points']
                    state = graph.results[feature.id].constraint_status
                    start, end, width = (np.asarray(state.points[name], dtype=float) for name in names)
                    axis = end - start
                    axis /= np.linalg.norm(axis)
                    normal = np.array([-axis[1], axis[0]])
                    signed_radius = float(np.dot(width - start, normal))
                    value, ok = self.panel_inputs.getDouble(self.window, 'Slot', 'Width:', abs(signed_radius) * 2, .00001, 1000000, 4)
                    if not ok: return
                    manager.move_history_point(names[2], start + normal * np.copysign(value / 2, signed_radius))
                    return
                fields = [('radius_x', 'Major radius'), ('radius_y', 'Minor radius'), ('rotation', 'Rotation (degrees)')] if kind == 'ellipse' else [('radius', 'Half-width')]
                if kind == 'ellipse' and not entity.get('is_closed', True):
                    fields += [('start_angle', 'Start angle'), ('end_angle', 'End angle')]
                for key, label in fields:
                    value, ok = self.panel_inputs.getText(self.window, 'Geometry', label + ':', text=str(entity.get(key, 0)))
                    if not ok: return
                    entity[key] = value
                graph.edit(feature.id, parameters=params)
                manager.history_edit_graph = graph
            elif action == 'constraint' and selection[0] == 'entity':
                from UI.main.scene_sketch_commands import SceneSketchCommands
                SceneSketchCommands(manager, self.window, self.panel_inputs).create_constraint()
                return
            elif action == 'length' and selection[0] == 'entity':
                from UI.model.sketch_drag import add_sketch_constraint
                feature = next(item for item in manager.history_edit_graph.features if item.id == manager.history_edit_id)
                points = feature.parameters['entities'][selection[1]]['points']
                state = manager.history_edit_graph.results[manager.history_edit_id].constraint_status
                length = float(np.linalg.norm(state.points[points[1]] - state.points[points[0]]))
                value, ok = self.panel_inputs.getText(self.window, 'Length', 'Value or parameter expression:', text=str(length))
                if not ok: return
                manager.history_edit_graph = add_sketch_constraint(manager.history_edit_graph, manager.history_edit_id,
                    {'kind': 'distance', 'entities': list(points), 'value': value})
            elif action == 'delete' and selection[0] == 'entity':
                graph = FeatureGraph.from_data(manager.history_edit_graph.to_data())
                feature = next(item for item in graph.features if item.id == manager.history_edit_id)
                params, index = feature.parameters, selection[1]
                if params.get('constraints'):
                    raise ValueError('Remove sketch constraints before deleting an element.')
                if any(e.get('type') in ('spline_slot', 'composite_path_slot') for e in params['entities']):
                    raise ValueError('Remove dependent curve slots before deleting an element.')
                params['entities'].pop(index)
                graph.edit(feature.id, parameters=params)
                manager.history_edit_graph = graph
                manager.history_selection = None
            elif action == 'construction' and selection[0] == 'entity':
                graph = FeatureGraph.from_data(manager.history_edit_graph.to_data())
                feature = next(item for item in graph.features if item.id == manager.history_edit_id)
                params = feature.parameters
                entity = params['entities'][selection[1]]
                entity['construction'] = not entity.get('construction', False)
                graph.edit(feature.id, parameters=params)
                manager.history_edit_graph = graph
            manager.render_history_sketch()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self.window, 'Sketch', str(exc))
        
    def add_style(self):
        from UI.main.theme import DARK, LIGHT, apply_theme
        system_dark = QtGui.QGuiApplication.styleHints().colorScheme() == QtCore.Qt.ColorScheme.Dark
        dark = self.theme_mode == 'dark' or self.theme_mode == 'system' and system_dark
        apply_theme(self.window, dark)
        colors = DARK if dark else LIGHT
        self.plotter.set_background(colors['canvas'], top=colors['canvas_top'])
        self.navigation_cube.reposition()

    def set_theme(self, mode):
        self.theme_mode = mode
        self.theme_settings.setValue('appearance/theme', mode)
        self.add_style()
        self.refresh_timeline()

    def system_theme_changed(self, _scheme):
        if self.theme_mode == 'system':
            self.add_style()
            self.refresh_timeline()

    def ribbon_group(self, layout, label, commands, visible):
        group = QtWidgets.QFrame()
        group_layout = QtWidgets.QVBoxLayout(group)
        group_layout.setContentsMargins(5, 2, 5, 2)
        group_layout.setSpacing(0)
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(2)
        menu = QtWidgets.QMenu(group)
        self.add_menu_commands(menu, commands)
        for (key, title, _callback), action in zip(commands, menu.actions()):
            if key not in visible:
                continue
            button = QtWidgets.QToolButton()
            button.setDefaultAction(action)
            button.setProperty("ribbon", True)
            button.setIconSize(QtCore.QSize(26, 26))
            button.setMinimumWidth(52)
            button.setFixedHeight(54)
            button.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
            button.setToolTip(title)
            button.setAccessibleName(title)
            button.setText({'sketch': 'Sketch', 'move': 'Move / Copy', 'select_edge': 'Edges',
                            'inspect': 'Measure', 'plane': 'Plane', 'shell': 'Shell', 'fillet_all': 'Fillet',
                            'chamfer_all': 'Chamfer', 'curve': 'Arc'}.get(key, title))
            row.addWidget(button)
        if row.count():
            group_layout.addLayout(row)
        else:
            group_layout.addStretch(1)
        label_button = QtWidgets.QToolButton()
        label_button.setProperty("groupLabel", True)
        label_button.setText(label + "  ▾")
        label_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        label_button.setMenu(menu)
        group_layout.addWidget(label_button, alignment=QtCore.Qt.AlignHCenter)
        layout.addWidget(group)
        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.VLine)
        separator.setProperty('role', 'ribbonSeparator')
        layout.addWidget(separator)

    def add_menu_commands(self, menu, commands):
        for key, title, callback in commands:
            action = menu.addAction(tool_icon(key), title)
            action.triggered.connect(callback)
            self.command_actions.setdefault(key, action)

    def ribbon_layout(self):
        ribbon = QtWidgets.QFrame()
        ribbon.setObjectName("ribbon")
        layout = QtWidgets.QHBoxLayout(ribbon)
        layout.setContentsMargins(7, 2, 7, 2)
        layout.setSpacing(2)
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
                self.leave_sketch_workspace()
            if not manager.profiles:
                try:
                    if not manager.prepare_selected_face_profile():
                        QtWidgets.QMessageBox.information(self.window, "Extrude",
                            "Select planar faces or show closed profiles via the sketch in the Timeline.")
                        return
                except ValueError as exc:
                    QtWidgets.QMessageBox.warning(self.window, "Extrude", str(exc))
                    return
            def extrude_selected(index):
                indices = manager.extrusion_profile_indices() or (index,)
                self.show_scene_operation(ExtrudeDialog(manager, indices, self.window), 'Extrude')

            if manager.selected_profile_index is not None:
                extrude_selected(manager.selected_profile_index)
            else:
                manager.request_profile_selection(extrude_selected)

        def revolve():
            manager = self.action_manager
            if manager.sketch_editing:
                self.disable_active_tool()
                try:
                    manager.exit_sketch_mode()
                except Exception as exc:
                    QtWidgets.QMessageBox.warning(self.window, 'Sketch', str(exc))
                    return
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

        def sketch_solid(dialog_type, title):
            manager = self.action_manager
            try:
                if manager.sketch_editing:
                    self.disable_active_tool()
                    manager.exit_sketch_mode()
                if manager.parametric_graph is None or manager.selecting_sketch_plane:
                    raise ValueError(f'Finish the parametric sketches before starting {title}.')
                self.show_scene_operation(dialog_type(manager, self.window), title)
            except (ValueError, RuntimeError) as exc:
                QtWidgets.QMessageBox.warning(self.window, title, str(exc))

        self.ribbon_group(top_layout, "Create", [
            ("sketch", "Create sketch", open_sketch),
            ("extrude", "Extrude", extrude),
            ("revolve", "Revolve", revolve),
            ("sweep", "Sweep", lambda: sketch_solid(SweepDialog, 'Sweep')),
            ("loft", "Loft", lambda: sketch_solid(LoftDialog, 'Loft')),
            ("box", "Box", lambda: create_primitive("box")),
            ("cylinder", "Cylinder", lambda: create_primitive("cylinder")),
            ("sphere", "Sphere", lambda: create_primitive("sphere")),
            ("cone", "Cone / frustum", lambda: create_primitive("cone")),
            ("torus", "Torus", lambda: create_primitive("torus")),
        ], {"sketch", "extrude", "revolve", "sweep", "loft"})

        def transform_model(kind):
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

        def modify_face_edges(kind):
            manager = self.action_manager
            faces = list(manager.current_part.faces())
            indices = manager.selected_face_ids or {manager.selected_face_id}
            selected = [edge for i in indices if 0 <= i < len(faces) for edge in faces[i].edges()]
            manager.selected_edge_indices = {i for i, edge in enumerate(manager.current_part.edges())
                                             if any(edge.is_same(candidate) for candidate in selected)}
            solid_feature(kind + '_selected')

        self.scene_context_commands = {
            'extrude': extrude,
            **{name: (lambda key=name: solid_feature(key)) for name in ('fillet_selected', 'chamfer_selected', 'hole', 'shell')},
            'fillet_faces': lambda: modify_face_edges('fillet'),
            'chamfer_faces': lambda: modify_face_edges('chamfer'),
            'move': lambda: transform_model('move'),
        }

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
        ], {"move", "shell"})

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
                QtWidgets.QMessageBox.information(self.window, "Save model", "Finish the current sketch or solid operation first.")
                return
            if manager.current_part is None and manager.parametric_graph is None:
                QtWidgets.QMessageBox.information(self.window, "Save model", "Create a parametric sketch or a solid body first.")
                return
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self.window, "Save model",
                                                          self.document_path or '', "FluentCAD model (*.fcad)")
            if not path:
                return
            if not Path(path).suffix:
                path += '.fcad'
            try:
                save_snapshot(path, manager.current_part, manager.undo_stack, manager.redo_stack, manager.construction_planes,
                              {'entries': manager.timeline.entries, 'cursor': manager.timeline.cursor}, manager.parametric_graph,
                              {'undo': manager.graph_undo, 'redo': manager.graph_redo}, self.model_browser.export_visibility())
                self.set_document_path(path)
                self.window.statusBar().showMessage(f"Model saved: {path}", 5000)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self.window, "Save model", str(exc))

        def open_model_snapshot():
            manager = self.action_manager
            if manager.pending_sketch or manager.sketch_editing or manager.selecting_sketch_plane:
                QtWidgets.QMessageBox.information(self.window, "Open model", "Finish the current sketch and solid operation first.")
                return
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self.window, "Open model", "", "FluentCAD model (*.fcad)")
            if not path:
                return
            try:
                current, undo, redo, planes, timeline_data, feature_graph, graph_history, visibility = load_snapshot(
                    path, include_planes=True, include_timeline=True, include_feature_graph=True, include_graph_history=True, include_visibility=True)
                if feature_graph is not None and (undo or redo):
                    raise ValueError('This parametric file contains legacy undo snapshots which cannot yet be restored safely.')
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self.window, "Open model", str(exc))
                return
            if manager.current_part is not None or manager.parametric_graph is not None:
                answer = QtWidgets.QMessageBox.question(self.window, "Open model", "Replace the current model and its undo history?")
                if answer != QtWidgets.QMessageBox.Yes:
                    return
            try:
                self.model_browser.reset_visibility()
                if feature_graph is not None:
                    manager.install_parametric_graph(feature_graph, reset_history=True, allow_empty=(current is None))
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
                if visibility is not None:
                    self.model_browser.restore_visibility(manager, visibility)
                manager.sketchs_data.clear()
                manager.activated_sketch_id = None
                manager.legacy_migration_redo = None
                self.set_document_path(path)
                self.window.statusBar().showMessage(f"Model opened: {path}", 5000)
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

        self.add_menu_commands(self.file_menu, [
            ("parametric_new", "New parametric model", self.new_parametric_model),
            ("parametric_add", "Add parametric feature", self.add_graph_feature),
            ("parametric_parameters", "Edit named parameters", self.edit_graph_parameters),
            ("parametric_convert", "Convert legacy model to parametric", convert_legacy_model),
            ("save", "Save model", save_model_snapshot),
            ("open", "Open model", open_model_snapshot),
            ("export", "Export STEP / STL", export_model),
            ("import", "Import STEP as new body", import_step),
        ])
        self.global_actions.extend(self.file_menu.actions())

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
            if self.action_manager.sketch_editing:
                self.action_manager.status_callback('Finish the sketch before switching projection.')
                return
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

        self.view_menu = QtWidgets.QMenu(self.window)
        self.add_menu_commands(self.view_menu, [
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
            ("view_menu", "Theme: System", lambda: self.set_theme("system")),
            ("view_menu", "Theme: Light", lambda: self.set_theme("light")),
            ("view_menu", "Theme: Dark", lambda: self.set_theme("dark")),
        ])
        self.global_actions.extend(self.view_menu.actions())
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
            self.leave_sketch_workspace()
            if not self.action_manager.profiles and not self.action_manager.sketch_spatial_actors:
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
        from UI.main.expression_spinbox import ExpressionSpinBox
        self.sketch_offset_distance = ExpressionSpinBox(tool_bar)
        self.sketch_offset_distance.setAccessibleName('Sketch offset distance')
        self.sketch_offset_distance.setPrefix('Offset ')
        self.sketch_offset_distance.setDecimals(4)
        self.sketch_offset_distance.setRange(-1000000, 1000000)
        self.sketch_offset_distance.setValue(1)
        self.sketch_offset_distance.setKeyboardTracking(False)
        self.sketch_offset_distance.hide()
        self.sketch_offset_chain = QtWidgets.QCheckBox('Chain selection', tool_bar)
        self.sketch_offset_chain.setChecked(True)
        self.sketch_offset_chain.hide()
        from UI.model.sketch_offset import OFFSET_JOINS
        self.sketch_offset_join = QtWidgets.QComboBox(tool_bar)
        self.sketch_offset_join.setAccessibleName('Offset corner mode')
        self.sketch_offset_join.setToolTip('Join the outer corners of the offset chain')
        for key, title in OFFSET_JOINS.items():
            self.sketch_offset_join.addItem(title, key)
        self.sketch_offset_join.hide()
        self.sketch_offset_apply = QtWidgets.QPushButton('Apply Offset', tool_bar)
        self.sketch_offset_apply.hide()
        self.sketch_corner_size = ExpressionSpinBox(tool_bar)
        self.sketch_corner_size.setDecimals(4)
        self.sketch_corner_size.setRange(.0001, 1000000)
        self.sketch_corner_size.setValue(1)
        self.sketch_corner_size.hide()
        self.sketch_corner_apply = QtWidgets.QPushButton('Apply Corner', tool_bar)
        self.sketch_corner_apply.hide()
        self.sketch_mirror_axis = QtWidgets.QPushButton('Select Axis', tool_bar)
        self.sketch_mirror_axis.hide()
        self.sketch_mirror_apply = QtWidgets.QPushButton('Apply Mirror', tool_bar)
        self.sketch_mirror_apply.hide()
        self.sketch_move_panel = QtWidgets.QWidget(tool_bar)
        move_layout = QtWidgets.QHBoxLayout(self.sketch_move_panel)
        move_layout.setContentsMargins(0, 0, 0, 0)
        self.sketch_move_fields = []
        for label in ('ΔX', 'ΔY', 'Angle'):
            field = ExpressionSpinBox(self.sketch_move_panel)
            field.setPrefix(label + ' ')
            field.setAccessibleName(label)
            field.setDecimals(6)
            field.setRange(-1000000, 1000000)
            if label == 'Angle':
                field.setSuffix('°')
                field.setToolTip('Rotate in the sketch plane about the sketch origin')
            move_layout.addWidget(field)
            self.sketch_move_fields.append(field)
        sketch_copy = QtWidgets.QCheckBox('Create Copy', self.sketch_move_panel)
        move_apply = QtWidgets.QPushButton('Apply Transform', self.sketch_move_panel)
        move_layout.addWidget(sketch_copy)
        move_layout.addWidget(move_apply)
        self.sketch_move_panel.hide()
        self.sketch_scale_panel = QtWidgets.QWidget(tool_bar)
        scale_layout = QtWidgets.QHBoxLayout(self.sketch_scale_panel)
        scale_layout.setContentsMargins(0, 0, 0, 0)
        self.sketch_scale_factor = ExpressionSpinBox(self.sketch_scale_panel)
        self.sketch_scale_factor.setPrefix('Factor ')
        self.sketch_scale_factor.setAccessibleName('Scale factor')
        self.sketch_scale_factor.setDecimals(6)
        self.sketch_scale_factor.setRange(.000001, 1000000)
        self.sketch_scale_factor.setValue(1)
        center_button = QtWidgets.QPushButton('Select Center', self.sketch_scale_panel)
        scale_apply = QtWidgets.QPushButton('Apply Scale', self.sketch_scale_panel)
        for control in (center_button, self.sketch_scale_factor, scale_apply):
            scale_layout.addWidget(control)
        self.sketch_scale_panel.hide()
        self.sketch_pattern_panels = {}
        pattern_settings = {}

        def pattern_action(action):
            modifier = self.action_manager.sketch_modifier
            if modifier is not None and modifier.command == 'sketch_pattern':
                getattr(modifier, action)()

        for kind, labels, reference in (
            ('rectangular', ('Count X', 'Count Y', 'Spacing X', 'Spacing Y'), 'Direction'),
            ('circular', ('Count', 'Sweep'), 'Center'),
        ):
            panel = QtWidgets.QWidget(tool_bar)
            layout = QtWidgets.QHBoxLayout(panel)
            layout.setContentsMargins(0, 0, 0, 0)
            fields = []
            for label in labels:
                if label.startswith('Count'):
                    field = QtWidgets.QSpinBox(panel)
                    field.setRange(2 if label == 'Count' else 1, 100)
                    field.setValue({'Count X': 2, 'Count Y': 1, 'Count': 4}[label])
                else:
                    field = ExpressionSpinBox(panel)
                    field.setDecimals(6)
                    limit = 360 if label == 'Sweep' else 1000000
                    field.setRange(-limit, limit)
                    field.setValue(360 if label == 'Sweep' else 10)
                    if label == 'Sweep':
                        field.setSuffix('°')
                field.setPrefix(label + ' ')
                field.setAccessibleName(label)
                field.valueChanged.connect(lambda _value: pattern_action('refresh_preview'))
                layout.addWidget(field)
                fields.append(field)
            choose = QtWidgets.QPushButton(f'Select {reference}', panel)
            choose.clicked.connect(lambda: pattern_action('choose_reference'))
            layout.addWidget(choose)
            apply_pattern = QtWidgets.QPushButton('Apply Pattern', panel)
            apply_pattern.clicked.connect(lambda: pattern_action('confirm'))
            layout.addWidget(apply_pattern)
            panel.hide()
            self.sketch_pattern_panels[kind] = panel
            pattern_settings[kind] = lambda fields=tuple(fields): tuple(field.value() for field in fields)

        def start_pattern(kind):
            self.disable_active_tool()
            try:
                self.action_manager.start_sketch_pattern(kind, pattern_settings[kind])
            except (ValueError, RuntimeError) as exc:
                QtWidgets.QMessageBox.warning(self.window, 'Pattern', str(exc))
                return
            self.sketch_pattern_panels[kind].show()

        def start_scale():
            self.disable_active_tool()
            try:
                self.action_manager.start_sketch_scale(self.sketch_scale_factor.value)
            except (ValueError, RuntimeError) as exc:
                QtWidgets.QMessageBox.warning(self.window, 'Scale', str(exc))
                return
            self.sketch_scale_panel.show()

        def scale_action(action):
            modifier = self.action_manager.sketch_modifier
            if modifier is not None and modifier.command == 'sketch_scale':
                if action == 'confirm':
                    self.sketch_scale_factor.interpretText()
                getattr(modifier, action)()

        center_button.clicked.connect(lambda: scale_action('choose_center'))
        scale_apply.clicked.connect(lambda: scale_action('confirm'))
        self.sketch_scale_factor.valueChanged.connect(lambda _value: scale_action('refresh_preview'))

        def move_settings():
            x, y, angle = (field.value() for field in self.sketch_move_fields)
            return (x, y), angle, sketch_copy.isChecked()

        def refresh_move(_value):
            modifier = self.action_manager.sketch_modifier
            if modifier is not None and modifier.command == 'sketch_move':
                modifier.refresh_preview()

        def apply_move():
            modifier = self.action_manager.sketch_modifier
            if modifier is not None and modifier.command == 'sketch_move':
                for field in self.sketch_move_fields:
                    field.interpretText()
                modifier.confirm()

        def start_move():
            self.disable_active_tool()
            try:
                self.action_manager.start_sketch_move(move_settings)
            except (ValueError, RuntimeError) as exc:
                QtWidgets.QMessageBox.warning(self.window, 'Move / Copy', str(exc))
                return
            self.sketch_move_panel.show()

        for field in self.sketch_move_fields:
            field.valueChanged.connect(refresh_move)
        sketch_copy.toggled.connect(refresh_move)
        move_apply.clicked.connect(apply_move)

        def preview_offset(_value):
            modifier = self.action_manager.sketch_modifier
            if modifier is not None and modifier.command == 'offset':
                modifier.hover()
            elif modifier is not None and modifier.command == 'offset_chain':
                modifier.refresh_preview()

        self.sketch_offset_distance.valueChanged.connect(preview_offset)
        self.sketch_offset_join.currentIndexChanged.connect(preview_offset)

        def reset_offset_selection(_checked):
            modifier = self.action_manager.sketch_modifier
            if modifier is not None and modifier.command == 'offset_chain':
                modifier.reset_selection()

        self.sketch_offset_chain.toggled.connect(reset_offset_selection)

        def apply_offset():
            modifier = self.action_manager.sketch_modifier
            if modifier is not None and modifier.command == 'offset_chain':
                self.sketch_offset_distance.interpretText()
                modifier.confirm()

        self.sketch_offset_apply.clicked.connect(apply_offset)

        def preview_corner(_value):
            modifier = self.action_manager.sketch_modifier
            if modifier is not None and modifier.command in ('fillet', 'chamfer'):
                modifier.refresh_preview()

        def apply_corner():
            modifier = self.action_manager.sketch_modifier
            if modifier is not None and modifier.command in ('fillet', 'chamfer'):
                self.sketch_corner_size.interpretText()
                modifier.confirm()

        self.sketch_corner_size.valueChanged.connect(preview_corner)
        self.sketch_corner_apply.clicked.connect(apply_corner)

        def start_corner(command):
            self.disable_active_tool()
            try:
                self.action_manager.start_sketch_corner(command, self.sketch_corner_size.value)
            except (ValueError, RuntimeError) as exc:
                QtWidgets.QMessageBox.warning(self.window, command.title(), str(exc))
                return
            field = 'Radius' if command == 'fillet' else 'Setback'
            self.sketch_corner_size.setPrefix(field + ' ')
            self.sketch_corner_size.setAccessibleName(field)
            self.sketch_corner_size.show()
            self.sketch_corner_apply.show()

        def start_mirror():
            self.disable_active_tool()
            try:
                self.action_manager.start_sketch_mirror()
            except (ValueError, RuntimeError) as exc:
                QtWidgets.QMessageBox.warning(self.window, 'Mirror', str(exc))
                return
            self.sketch_mirror_axis.show()
            self.sketch_mirror_apply.show()

        def choose_mirror_axis():
            modifier = self.action_manager.sketch_modifier
            if modifier is not None and modifier.command == 'sketch_mirror':
                modifier.choose_axis()

        def apply_mirror():
            modifier = self.action_manager.sketch_modifier
            if modifier is not None and modifier.command == 'sketch_mirror':
                modifier.confirm()

        self.sketch_mirror_axis.clicked.connect(choose_mirror_axis)
        self.sketch_mirror_apply.clicked.connect(apply_mirror)

        def start_offset():
            self.disable_active_tool()
            try:
                self.action_manager.start_sketch_offset(self.sketch_offset_distance.value,
                                                       self.sketch_offset_chain.isChecked,
                                                       self.sketch_offset_join.currentData)
            except (ValueError, RuntimeError) as exc:
                QtWidgets.QMessageBox.warning(self.window, 'Offset', str(exc))
                return
            self.sketch_offset_distance.show()
            self.sketch_offset_chain.show()
            self.sketch_offset_join.show()
            self.sketch_offset_apply.show()

        def start_modify(command):
            self.disable_active_tool()
            try:
                self.action_manager.start_sketch_modifier(command, self.sketch_offset_distance.value)
            except (ValueError, RuntimeError) as exc:
                QtWidgets.QMessageBox.warning(self.window, command.title(), str(exc))
                return
            self.sketch_offset_distance.setVisible(command == 'offset')

        self.ribbon_group(top_layout, 'Modify', [
            ('trim', 'Trim', lambda: start_modify('trim')),
            ('extend', 'Extend', lambda: start_modify('extend')),
            ('offset', 'Offset', start_offset),
            ('offset', 'Offset copy', lambda: start_modify('offset')),
            ('fillet_all', 'Fillet', lambda: start_corner('fillet')),
            ('chamfer_all', 'Chamfer', lambda: start_corner('chamfer')),
            ('mirror', 'Mirror', start_mirror),
            ('move', 'Move / Copy', start_move),
            ('scale', 'Scale', start_scale),
            ('rectangular_pattern', 'Rectangular Pattern', lambda: start_pattern('rectangular')),
            ('circular_pattern', 'Circular Pattern', lambda: start_pattern('circular')),
        ], {'trim', 'extend', 'offset'})
        top_layout.addWidget(self.sketch_offset_distance)
        top_layout.addWidget(self.sketch_offset_chain)
        top_layout.addWidget(self.sketch_offset_join)
        top_layout.addWidget(self.sketch_offset_apply)
        top_layout.addWidget(self.sketch_corner_size)
        top_layout.addWidget(self.sketch_corner_apply)
        top_layout.addWidget(self.sketch_mirror_axis)
        top_layout.addWidget(self.sketch_mirror_apply)
        top_layout.addWidget(self.sketch_move_panel)
        top_layout.addWidget(self.sketch_scale_panel)
        for panel in self.sketch_pattern_panels.values():
            top_layout.addWidget(panel)

        def start_dimension(mode):
            from UI.main.sketch_dimension_dialog import edit_sketch_dimension
            self.disable_active_tool()
            try:
                self.action_manager.start_sketch_dimension(
                    mode, lambda dimension, preview: edit_sketch_dimension(self.window, dimension, preview))
            except (ValueError, RuntimeError) as exc:
                QtWidgets.QMessageBox.warning(self.window, 'Sketch Dimension', str(exc))

        self.ribbon_group(top_layout, 'Dimension', [
            ('dimension', 'Sketch Dimension', lambda: start_dimension('auto')),
            ('dimension_distance', 'Distance', lambda: start_dimension('distance')),
            ('dimension_horizontal', 'Horizontal distance', lambda: start_dimension('horizontal_distance')),
            ('dimension_vertical', 'Vertical distance', lambda: start_dimension('vertical_distance')),
            ('dimension_angle', 'Angle', lambda: start_dimension('angle')),
            ('dimension_radius', 'Radius', lambda: start_dimension('radius')),
            ('dimension_diameter', 'Diameter', lambda: start_dimension('diameter')),
        ], {'dimension'})

        def start_link(command, kind):
            self.disable_active_tool()
            try:
                if not self.action_manager.sketch_editing:
                    raise ValueError('Select a sketch plane first.')
                source_id = None
                if command == 'include' or command == 'project' and kind in ('body', 'sketch_curve'):
                    title = 'Include 3D Geometry' if command == 'include' else 'Project'
                    source_id = self.choose_sketch_link_source(kind, title)
                    if source_id is None:
                        return
                self.action_manager.start_sketch_link(command, kind, source_id)
            except (ValueError, RuntimeError) as exc:
                QtWidgets.QMessageBox.warning(self.window, command.title(), str(exc))

        self.ribbon_group(top_layout, 'Project / Include', [
            ('project', 'Project edges', lambda: start_link('project', 'edge')),
            ('project', 'Project faces', lambda: start_link('project', 'face')),
            ('project', 'Project body silhouette', lambda: start_link('project', 'body')),
            ('project', 'Project curves from another sketch', lambda: start_link('project', 'sketch_curve')),
            ('project', 'Include 3D Geometry — Edges', lambda: start_link('include', 'edge')),
            ('project', 'Include 3D Geometry — Sketch curves', lambda: start_link('include', 'sketch_curve')),
            ('intersect', 'Intersect bodies', lambda: start_link('intersect', 'body')),
            ('intersect', 'Intersect faces', lambda: start_link('intersect', 'face')),
            ('intersect', 'Intersect edges', lambda: start_link('intersect', 'edge')),
            ('project', 'Relink…', lambda: self.manage_sketch_link(relink=True)),
            ('project', 'Break Link…', lambda: self.manage_sketch_link(relink=False)),
        ], set())

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

        def start_constraint(mode):
            self.disable_active_tool()
            try:
                self.action_manager.start_sketch_constraint(mode)
            except (ValueError, RuntimeError) as exc:
                QtWidgets.QMessageBox.warning(self.window, 'Constraints', str(exc))

        def remove_constraint():
            try: scene_commands().remove_constraint()
            except Exception as exc: QtWidgets.QMessageBox.warning(self.window, 'Constraints', str(exc))

        from UI.model.scene_constraint_rules import SCENE_CONSTRAINTS
        self.ribbon_group(top_layout, 'Constraints', [
            ('constraint', title, lambda mode=mode: start_constraint(mode))
            for mode, title in SCENE_CONSTRAINTS.items()
        ] + [
            ('constraint', 'More constraints…', open_constraints),
            ('constraint', 'Remove constraint…', remove_constraint),
        ], set())
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
        finish.setIconSize(QtCore.QSize(22, 22))
        finish.setText("Finish sketch")
        finish.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        finish.setAccessibleName("Finish sketch")
        finish.setToolTip("Finish the current sketch and find its profiles")
        finish.clicked.connect(exit_sketch)
        top_layout.addWidget(finish)

        top_layout.addStretch() 
        self.tool_tabs_widgets["sketch"] = tool_bar
        return tool_bar
    
    def manage_sketch_link(self, relink):
        title = 'Relink' if relink else 'Break Link'
        manager = self.action_manager
        try:
            if not manager.sketch_editing or manager.history_edit_id is None:
                raise ValueError('Edit the linked sketch in the scene first.')
            self.disable_active_tool()
            graph, feature_id = manager.history_edit_graph, manager.history_edit_id
            features = {feature.id: feature for feature in graph.features}
            links = features[feature_id].parameters.get('links', [])
            if not links:
                raise ValueError('This sketch has no geometry links.')
            issues = graph.link_issues.get(feature_id, {})
            labels = []
            for i, link in enumerate(links, 1):
                source = features.get(link['source_id'])
                source_name = source.name if source else 'Missing source'
                status = 'Lost · ' if link['id'] in issues else ''
                kind = {'sketch_curve': 'sketch curve'}.get(link['kind'], link['kind'])
                labels.append(f"{i}. {status}{link['operation'].title()} {kind} · {source_name}")
            initial = next((i for i, link in enumerate(links) if link['id'] in issues), 0)
            choice, ok = self.panel_inputs.getItem(self.window, title, 'Sketch link', labels, initial, False)
            if not ok:
                return
            link = links[labels.index(choice)]
            if not relink:
                manager.break_sketch_link(link['id'])
                return
            source_id = self.choose_sketch_link_source(link['kind'], title)
            if source_id is not None:
                manager.start_sketch_link(link['operation'], link['kind'], source_id, replace_id=link['id'])
        except (ValueError, RuntimeError) as exc:
            QtWidgets.QMessageBox.warning(self.window, title, str(exc))

    def choose_sketch_link_source(self, kind, title):
        sources = self.action_manager.sketch_link_sources(kind)
        label = 'Source sketch' if kind == 'sketch_curve' else 'Source solid feature'
        if not sources:
            raise ValueError(f'{label} must be available before this sketch in the Timeline.')
        if len(sources) == 1:
            return sources[0].id
        labels = [f'{i}. {feature.name}' for i, feature in enumerate(sources, 1)]
        choice, ok = self.panel_inputs.getItem(self.window, title, label, labels, 0, False)
        return sources[labels.index(choice)].id if ok else None

    def stop_sketch_modifier(self):
        if not self.action_manager.stop_sketch_modifier():
            return False
        self.sketch_offset_distance.hide()
        self.sketch_offset_chain.hide()
        self.sketch_offset_join.hide()
        self.sketch_offset_apply.hide()
        self.sketch_corner_size.hide()
        self.sketch_corner_apply.hide()
        self.sketch_mirror_axis.hide()
        self.sketch_mirror_apply.hide()
        self.sketch_move_panel.hide()
        self.sketch_scale_panel.hide()
        for panel in self.sketch_pattern_panels.values():
            panel.hide()
        return True

    def disable_active_tool(self):
        manager = self.action_manager
        if getattr(manager, 'sketch_modifier', None) is not None:
            self.stop_sketch_modifier()
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
