"""Create or edit a Loft with selections made in the viewport."""
from dataclasses import replace

from PySide6 import QtCore, QtWidgets

from UI.model.ModelConverter import b123d_to_pro_mesh
from UI.model.feature_graph import Feature, FeatureGraph, RebuildError, sketch_profile_index
from UI.model.geometry_references import make_reference
from UI.model.loft_end_conditions import resolve_loft_support
from UI.model.loft_patch_continuity import support_meets_section
from UI.model.loft_rims import section_rims
from UI.model.parametric_sketch import ParametricSketch
from UI.model.sketch_sweep import sweep_curves


class LoftDialog(QtWidgets.QDialog):
    def __init__(self, manager, parent=None, *, feature_id=None):
        super().__init__(parent)
        self.setWindowTitle('Loft')
        self.manager = manager
        self.graph = manager.parametric_graph
        features = self.graph.features
        self.edited_feature = None
        self.following_features = []
        self.source_features = features
        if feature_id is not None:
            position = next((i for i, feature in enumerate(features) if feature.id == feature_id), None)
            if position is None or features[position].kind != 'loft' or feature_id not in self.graph.results:
                raise ValueError('Choose an active Loft in the timeline.')
            self.edited_feature = features[position]
            self.source_features = features[:position]
            self.following_features = features[position + 1:]
            self.setWindowTitle('Edit Loft')
        self.source_results = {feature.id: self.graph.results[feature.id] for feature in self.source_features
                               if feature.id in self.graph.results}
        self.support_source = None
        self.source_actor = None
        self.sections = []
        self.guides = []
        self.hole_matches = {}
        self.pending_hole = None
        self.hole_actors = {}
        self.section_pairs = []
        self.section_rim_cache = {}
        self.support_side = None
        self.support_faces = {}
        self.rim_modes = {}
        self.rim_choices = {}
        self.end_sections = {}
        self.end_rims = {}
        self.rim_highlight = []
        self.support_buttons = {}
        self.end_modes = {}
        self.guide_actors = {}
        self.actors = {}
        self.hidden = []
        self.preview = self.candidate = self.result_shape = None
        profiles = [(feature.id, index, face, f'{position}. {feature.name} · Profile {index + 1}')
                    for position, feature in enumerate(self.source_features, 1)
                    if isinstance(sketch := self.source_results.get(feature.id), ParametricSketch)
                    for index, face in enumerate(sketch.profiles)]
        if len(profiles) < 2:
            raise ValueError('Create at least two closed sketch profiles before starting Loft.')
        if self.edited_feature is not None:
            self._load_feature()

        layout = QtWidgets.QVBoxLayout(self)
        self.message = QtWidgets.QLabel(self)
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        self.order = QtWidgets.QListWidget(self)
        self.order.setAccessibleName('Loft section order')
        layout.addWidget(self.order)
        controls = QtWidgets.QHBoxLayout()
        self.up = QtWidgets.QPushButton('Move Up', self)
        self.down = QtWidgets.QPushButton('Move Down', self)
        remove = QtWidgets.QPushButton('Remove', self)
        self.up.clicked.connect(lambda: self.move_section(-1))
        self.down.clicked.connect(lambda: self.move_section(1))
        remove.clicked.connect(lambda: self.remove_section(self.order.currentRow()))
        self.order.currentRowChanged.connect(lambda row: self._update_controls(row, remove))
        for button in (self.up, self.down, remove):
            button.setEnabled(False)
            controls.addWidget(button)
        layout.addLayout(controls)
        self.choose_guides = QtWidgets.QCheckBox('Select Guides', self)
        self.choose_guides.setToolTip('Click open guide curves in the scene. Click again to remove a guide.')
        self.choose_guides.toggled.connect(self._toggle_guides)
        layout.addWidget(self.choose_guides)
        self.match_holes = QtWidgets.QCheckBox('Match Holes', self)
        self.match_holes.setEnabled(False)
        self.match_holes.toggled.connect(self._toggle_hole_matching)
        layout.addWidget(self.match_holes)
        self.match_panel = QtWidgets.QWidget(self)
        matches = QtWidgets.QVBoxLayout(self.match_panel)
        matches.setContentsMargins(0, 0, 0, 0)
        self.match_interval = QtWidgets.QComboBox(self)
        self.match_interval.setAccessibleName('Adjacent sections for hole matching')
        self.match_interval.currentIndexChanged.connect(self._change_match_interval)
        matches.addWidget(self.match_interval)
        self.match_list = QtWidgets.QListWidget(self)
        self.match_list.setAccessibleName('Manual hole matches')
        self.match_list.setMaximumHeight(85)
        matches.addWidget(self.match_list)
        self.remove_match = QtWidgets.QPushButton('Remove Match', self)
        self.remove_match.setEnabled(False)
        self.match_list.currentRowChanged.connect(self._highlight_match)
        self.remove_match.clicked.connect(self._remove_match)
        matches.addWidget(self.remove_match)
        layout.addWidget(self.match_panel)
        self.match_panel.hide()
        ends = QtWidgets.QFormLayout()
        for side, name in (('start', 'First section'), ('end', 'Last section')):
            rim = QtWidgets.QComboBox(self)
            rim.setAccessibleName(name + ' contour')
            rim.currentIndexChanged.connect(lambda _index, side=side: self._change_rim(side))
            mode = QtWidgets.QComboBox(self)
            mode.setAccessibleName(name + ' continuity')
            for label, value in (('Position', None), ('Tangent (G1)', 'tangent'), ('Curvature (G2)', 'curvature')):
                mode.addItem(label, value)
            pick = QtWidgets.QPushButton('Pick Face', self)
            pick.setAccessibleName('Pick support face for ' + name.lower())
            pick.setEnabled(False)
            pick.clicked.connect(lambda _checked=False, side=side: self.choose_support(side))
            mode.currentIndexChanged.connect(lambda _index, side=side: self._change_end(side))
            row = QtWidgets.QHBoxLayout()
            row.addWidget(rim)
            row.addWidget(mode)
            row.addWidget(pick)
            ends.addRow(name, row)
            self.rim_choices[side] = rim
            self.end_modes[side] = mode
            self.support_buttons[side] = pick
        layout.addLayout(ends)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel, self)
        self.ok = buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.finished.connect(self._cleanup)

        self.labels = {}
        self.profile_actors = {}
        try:
            self._open_scene(profiles)
        except Exception:
            self._cleanup(None)
            raise

    def _open_scene(self, profiles):
        plotter = self.manager.plotter
        hidden = [*(plotter.actors.get(profile['name']) for profile in self.manager.profiles),
                  *self.manager.sketch_spatial_actors]
        if self.edited_feature is not None:
            hidden.extend((plotter.actors.get('main_part'), self.manager.highlight_actor,
                           *self.manager.edge_selection_actors))
            self.support_source = next((feature.id for feature in reversed(self.source_features)
                                        if hasattr(shape := self.source_results.get(feature.id), 'solids')
                                        and shape.solids()), None)
            if self.support_source is not None:
                self.source_actor = plotter.add_mesh(b123d_to_pro_mesh(self.source_results[self.support_source]),
                                                     color='#b0b0b0', reset_camera=False, render=False)
        for actor in hidden:
            if actor is not None and actor.GetVisibility():
                actor.SetVisibility(False)
                self.hidden.append(actor)
        for feature_id, index, face, label in profiles:
            selection = feature_id, index
            selected = selection in self.sections
            actor = plotter.add_mesh(b123d_to_pro_mesh(face), color='#ffb03b' if selected else '#3498db',
                                     opacity=.48, reset_camera=False, render=False)
            self.actors[actor] = selection
            self.profile_actors[selection] = actor
            self.labels[selection] = label
        for section in self.sections:
            self.order.addItem(self.labels[section])
        if self.sections:
            self.order.setCurrentRow(0)
        if self.guides:
            self._show_guides(False)
        plotter.reset_camera_clipping_range()
        self._refresh_preview()

    def _load_feature(self):
        params = self.edited_feature.parameters
        for selection in params['sections']:
            sketch_id = selection['sketch_id']
            self.sections.append((sketch_id, sketch_profile_index(self.source_results[sketch_id], selection)))
        self.guides = [tuple(reference) for reference in params.get('guides', [])]
        pairs = list(zip(self.sections, self.sections[1:]))
        for level, left, right in params.get('hole_matches', []):
            self.hole_matches.setdefault(pairs[level], []).append((left, right))

        for side, level in (('start', 0), ('end', -1)):
            section = self.sections[level]
            conditions = params.get('rim_ends', {}).get(side, {})
            legacy = params.get('ends', {}).get(side)
            if legacy is not None:
                reference = legacy['face']
                source_id = reference['feature_id']
                sketch = self.source_results[section[0]]
                face = resolve_loft_support(sketch.profiles[section[1]], reference, self.source_results[source_id],
                                            source_id, self.graph.body_names.get(source_id))
                keys = [key for key, rim in self._section_rims(section) if support_meets_section(rim, face)]
                if not keys or None in keys:
                    raise ValueError('The legacy Loft support cannot be assigned to an unambiguous contour.')
                conditions = dict.fromkeys(keys, legacy)
            for rim, condition in conditions.items():
                key = side, section, rim
                self.rim_modes[key] = condition['mode']
                self.support_faces[key] = condition['face']

    def _update_controls(self, row, remove):
        self.up.setEnabled(row > 0)
        self.down.setEnabled(0 <= row < len(self.sections) - 1)
        remove.setEnabled(row >= 0)

    def _remove_preview(self):
        if self.preview is not None:
            self.manager.plotter.remove_actor(self.preview, render=False)
            self.preview = None

    def _cleanup(self, _result):
        self._remove_preview()
        self._clear_rim_highlight()
        self._clear_hole_actors()
        for actor in (*self.actors, *self.guide_actors):
            self.manager.plotter.remove_actor(actor, render=False)
        if self.source_actor is not None:
            self.manager.plotter.remove_actor(self.source_actor, render=False)
            self.source_actor = None
        self.actors.clear()
        self.guide_actors.clear()
        for actor in self.hidden:
            actor.SetVisibility(True)
        self.hidden.clear()
        self.manager.plotter.reset_camera_clipping_range()
        self.manager.plotter.render()

    def scene_click(self):
        if self.support_side is not None:
            try:
                if self.edited_feature is not None and self.source_actor is None:
                    raise ValueError('There is no solid support before this Loft in the timeline.')
                index, _ = (self.manager._get_face_under_mouse() if self.edited_feature is None
                            else self.manager._get_face_under_mouse(self.source_actor))
                if index < 0:
                    raise ValueError('Click a visible solid face to set the support.')
                if self.edited_feature is None:
                    reference = self.manager.operation_geometry_selection('face', index)['reference']
                else:
                    source_id = self.support_source
                    reference = make_reference(source_id, self.source_results[source_id], 'face', index,
                                               self.graph.body_names.get(source_id))
            except (ValueError, RuntimeError) as exc:
                self.message.setText(str(exc))
                return
            side = self.support_side
            self.support_faces[self._rim_setting_key(side)] = reference
            self.support_side = None
            self._clear_rim_highlight()
            self._show_rim_settings(side)
            self._refresh_preview()
            return
        if self.match_holes.isChecked():
            self._pick_hole()
            return
        guides_mode = self.choose_guides.isChecked()
        actors = self.guide_actors if guides_mode else self.actors
        picker = self.manager._pick_actor(actors, tolerance=.006)
        if picker is None:
            return
        actor = picker.GetActor()
        selection = actors.get(actor)
        if selection is None:
            return
        if guides_mode:
            if selection in self.guides:
                self.guides.remove(selection)
                actor.GetProperty().SetColor(.15, .51, .60)
            else:
                self.guides.append(selection)
                actor.GetProperty().SetColor(1., .69, .23)
            self._refresh_preview()
            return
        if selection in self.sections:
            self.remove_section(self.sections.index(selection))
            return
        self.sections.append(selection)
        self.profile_actors[selection].GetProperty().SetColor(1., .69, .23)
        self.order.addItem(self.labels[selection])
        self.order.setCurrentRow(len(self.sections) - 1)
        self._refresh_preview()

    def _toggle_guides(self, active):
        if active:
            self.match_holes.setChecked(False)
            self.support_side = None
            self._clear_rim_highlight()
        self._show_guides(active)
        self.manager.plotter.reset_camera_clipping_range()
        self._refresh_preview()

    def _show_guides(self, active):
        if not self.guide_actors:
            for reference, edge in sweep_curves(self.source_results):
                if not edge.is_closed:
                    actor = self.manager.draw_sketch_edge(edge, '#258399', 3)
                    self.guide_actors[actor] = reference
        for actor, reference in self.guide_actors.items():
            actor.GetProperty().SetColor(*((1., .69, .23) if reference in self.guides else (.15, .51, .60)))
            actor.SetVisibility(active or reference in self.guides)

    def _section_rims(self, section):
        if section not in self.section_rim_cache:
            feature_id, index = section
            sketch = self.source_results[feature_id]
            self.section_rim_cache[section] = section_rims(sketch, sketch.profiles[index])
        return self.section_rim_cache[section]

    def _active_pair(self):
        index = self.match_interval.currentIndex()
        return self.section_pairs[index] if 0 <= index < len(self.section_pairs) else None

    def _sync_section_pairs(self):
        pairs = list(zip(self.sections, self.sections[1:]))
        if pairs == self.section_pairs:
            return 0
        self.match_holes.setEnabled(any(
            all(any(key is not None for key, _ in self._section_rims(section)[1:]) for section in pair)
            for pair in pairs))
        if not self.match_holes.isEnabled():
            with QtCore.QSignalBlocker(self.match_holes):
                self.match_holes.setChecked(False)
            self.match_panel.hide()
        current = self._active_pair()
        if current is not None and current not in pairs:
            current = current[::-1]
        old_matches = self.hole_matches
        self.hole_matches = {}
        for pair in pairs:
            if pair in old_matches:
                self.hole_matches[pair] = old_matches[pair]
            elif pair[::-1] in old_matches:
                self.hole_matches[pair] = [(right, left) for left, right in old_matches[pair[::-1]]]
        dropped = sum(map(len, old_matches.values())) - sum(map(len, self.hole_matches.values()))
        self.section_pairs = pairs
        with QtCore.QSignalBlocker(self.match_interval):
            self.match_interval.clear()
            for level, (left, right) in enumerate(pairs, 1):
                self.match_interval.addItem(f'{level} → {level + 1} · {self.labels[left]} → {self.labels[right]}')
            if current in pairs:
                self.match_interval.setCurrentIndex(pairs.index(current))
        self.pending_hole = None
        self._show_hole_matches()
        return dropped

    def _clear_hole_actors(self):
        for actor in self.hole_actors:
            self.manager.plotter.remove_actor(actor, render=False)
        self.hole_actors.clear()

    def _show_hole_matches(self):
        self._clear_hole_actors()
        self.match_list.clear()
        pair = self._active_pair()
        if pair is None or not self.match_holes.isChecked():
            return
        matches = self.hole_matches.get(pair, [])
        labels = []
        for side, section in enumerate(pair):
            labels.append({})
            for number, (key, face) in enumerate(self._section_rims(section)[1:], 1):
                if key is None:
                    continue
                labels[side][key] = f'Hole {number}'
                for edge in face.outer_wire().edges():
                    actor = self.manager.draw_sketch_edge(edge, '#5dc5fa', 5)
                    self.hole_actors[actor] = side, key
        for left, right in matches:
            self.match_list.addItem(f'{labels[0][left]} → {labels[1][right]}')
        self._highlight_match(-1)

    def _highlight_match(self, row):
        self.remove_match.setEnabled(row >= 0)
        matches = self.hole_matches.get(self._active_pair(), [])
        selected = matches[row] if 0 <= row < len(matches) else None
        for actor, (side, key) in self.hole_actors.items():
            if (side == 0 and key == self.pending_hole) or (selected is not None and key == selected[side]):
                color = (1., .8, .28)
            elif any(match[side] == key for match in matches):
                color = (.11, .66, .54)
            else:
                color = (.36, .77, .98)
            actor.GetProperty().SetColor(*color)
        if row >= 0:
            self.manager.plotter.render()

    def _toggle_hole_matching(self, active):
        if active:
            self.choose_guides.setChecked(False)
            self.support_side = None
            self._clear_rim_highlight()
        self.match_panel.setVisible(active)
        self._change_match_interval()

    def _change_match_interval(self, _index=None):
        self.pending_hole = None
        self._show_hole_matches()
        self._refresh_preview()

    def _pick_hole(self):
        pair = self._active_pair()
        if pair is None:
            return
        side = int(self.pending_hole is not None)
        actors = [actor for actor, selection in self.hole_actors.items() if selection[0] == side]
        picker = self.manager._pick_actor(actors, tolerance=.012)
        if picker is None:
            return
        _, key = self.hole_actors[picker.GetActor()]
        matches = self.hole_matches.get(pair, [])
        if any(match[side] == key for match in matches):
            self.message.setText('This hole is already matched. Remove its match to choose another.')
            return
        if side == 0:
            self.pending_hole = key
        else:
            self.hole_matches.setdefault(pair, []).append((self.pending_hole, key))
            self.pending_hole = None
        self._show_hole_matches()
        self._refresh_preview()

    def _remove_match(self):
        pair = self._active_pair()
        row = self.match_list.currentRow()
        if pair is None or row < 0:
            return
        matches = self.hole_matches[pair]
        matches.pop(row)
        if not matches:
            del self.hole_matches[pair]
        self.pending_hole = None
        self._show_hole_matches()
        self._refresh_preview()

    def _rim_setting_key(self, side):
        return side, self.end_sections.get(side), self.rim_choices[side].currentData()

    def _show_rim_settings(self, side):
        key = self._rim_setting_key(side)
        selected = key[2] is not None
        mode = self.end_modes[side]
        value = self.rim_modes.get(key)
        with QtCore.QSignalBlocker(mode):
            mode.setCurrentIndex(0 if value is None else mode.findData(value))
        mode.setEnabled(selected)
        pick = self.support_buttons[side]
        pick.setEnabled(selected and value is not None)
        pick.setText('Change Face' if value is not None and key in self.support_faces else 'Pick Face')

    def _sync_rims(self):
        for side, level in (('start', 0), ('end', -1)):
            section = self.sections[level] if len(self.sections) >= 2 else None
            if side in self.end_sections and self.end_sections[side] == section:
                continue
            self.end_sections[side] = section
            self._clear_rim_highlight()
            if self.support_side == side:
                self.support_side = None
            self.end_rims[side] = {}
            choice = self.rim_choices[side]
            with QtCore.QSignalBlocker(choice):
                choice.clear()
                if section is not None:
                    for number, (key, face) in enumerate(self._section_rims(section)):
                        label = 'Outer contour' if number == 0 else f'Hole {number}'
                        choice.addItem(label, key)
                        if key is None:
                            item = choice.model().item(choice.count() - 1)
                            item.setEnabled(False)
                            item.setToolTip('This hole has an ambiguous sketch reference.')
                        else:
                            self.end_rims[side][key] = label, face
            choice.setEnabled(section is not None)
            self._show_rim_settings(side)

    def _clear_rim_highlight(self):
        for actor in self.rim_highlight:
            self.manager.plotter.remove_actor(actor, render=False)
        self.rim_highlight.clear()

    def _highlight_rim(self, side):
        self._clear_rim_highlight()
        rim = self.end_rims[side].get(self.rim_choices[side].currentData())
        if rim is not None:
            for edge in rim[1].outer_wire().edges():
                actor = self.manager.draw_sketch_edge(edge, '#f5ce58', 4, pickable=False)
                self.rim_highlight.append(actor)

    def _change_rim(self, side):
        self.support_side = None
        self._show_rim_settings(side)
        self._highlight_rim(side)
        self._refresh_preview()

    def _change_end(self, side):
        mode = self.end_modes[side].currentData()
        key = self._rim_setting_key(side)
        if key[2] is None:
            return
        if mode is None:
            self.rim_modes.pop(key, None)
        else:
            self.rim_modes[key] = mode
        if mode is None and self.support_side == side:
            self.support_side = None
        self._show_rim_settings(side)
        self._refresh_preview()

    def choose_support(self, side):
        self.match_holes.setChecked(False)
        self.choose_guides.setChecked(False)
        self.support_side = side
        self.ok.setEnabled(False)
        self._highlight_rim(side)
        self.message.setText(f'Click a support face for {self.rim_choices[side].currentText().lower()} '
                             f'on the {"first" if side == "start" else "last"} section.')
        self.manager.plotter.render()

    def remove_section(self, row):
        if not 0 <= row < len(self.sections):
            return
        selection = self.sections.pop(row)
        self.profile_actors[selection].GetProperty().SetColor(.20, .60, .86)
        self.order.takeItem(row)
        self.order.setCurrentRow(min(row, len(self.sections) - 1))
        self._refresh_preview()

    def move_section(self, step):
        row = self.order.currentRow()
        target = row + step
        if not 0 <= row < len(self.sections) or not 0 <= target < len(self.sections):
            return
        self.sections.insert(target, self.sections.pop(row))
        self.order.insertItem(target, self.order.takeItem(row))
        self.order.setCurrentRow(target)
        self._refresh_preview()

    def _rebuild_preview(self, params):
        if self.edited_feature is None:
            if self.graph.latest_solid_id:
                params['base_id'] = self.graph.latest_solid_id
            feature = Feature.create('Loft', 'loft', params)
        else:
            saved = {key: value for key, value in self.edited_feature.parameters.items()
                     if key not in ('sections', 'guides', 'hole_matches', 'ends', 'rim_ends')}
            feature = replace(self.edited_feature, parameters={**saved, **params})
        prefix = [*self.source_features, feature]
        try:
            candidate = FeatureGraph([*prefix, *self.following_features], self.graph.parameters)
        except RebuildError as exc:
            if not any(item.id == exc.feature_id for item in self.following_features):
                raise
            preview = FeatureGraph(prefix, self.graph.parameters)
            return preview.results[feature.id], None, str(exc)
        return candidate.results[feature.id], candidate, None

    def _refresh_preview(self):
        dropped = self._sync_section_pairs()
        self._sync_rims()
        self._remove_preview()
        self.candidate = self.result_shape = None
        self.ok.setEnabled(False)
        hint = ('Click open curves that meet each section boundary.' if self.choose_guides.isChecked()
                else 'Click at least two closed profiles in the scene.')
        if len(self.sections) < 2:
            self.message.setText(hint)
        else:
            try:
                sections = [{'sketch_id': feature_id,
                             **self.manager.profile_selection_parameters(self.source_results[feature_id], index)}
                            for feature_id, index in self.sections]
                params = {'sections': sections}
                if self.guides:
                    params['guides'] = [list(ref) for ref in self.guides]
                matches = [[level, left, right] for level, pair in enumerate(self.section_pairs)
                           for left, right in self.hole_matches.get(pair, [])]
                if matches:
                    params['hole_matches'] = matches
                for key, mode in self.rim_modes.items():
                    side, section, rim = key
                    if self.end_sections[side] != section:
                        continue
                    if key not in self.support_faces:
                        label = self.end_rims[side][rim][0].lower()
                        name = 'first' if side == 'start' else 'last'
                        raise ValueError(f'Pick a support face for {label} on the {name} section.')
                    params.setdefault('rim_ends', {}).setdefault(side, {})[rim] = {
                        'mode': mode, 'face': self.support_faces[key]}
                shape, candidate, rebuild_error = self._rebuild_preview(params)
                self.preview = self.manager.plotter.add_mesh(
                    b123d_to_pro_mesh(shape.solids()[-1]), color='#1ca989', opacity=.42,
                    pickable=False, reset_camera=False, render=False)
                if rebuild_error is not None:
                    self.message.setText(f'A later feature could not rebuild. {rebuild_error}')
                else:
                    self.candidate, self.result_shape = candidate, shape
                    self.ok.setEnabled(self.support_side is None and self.pending_hole is None)
                if rebuild_error is None and self.choose_guides.isChecked():
                    self.message.setText(f'{len(self.guides)} guides selected. {hint}')
                elif rebuild_error is None:
                    self.message.setText('Loft follows the section order shown above.')
            except (ValueError, RuntimeError) as exc:
                self.message.setText(str(exc))
        if self.match_holes.isChecked() and (pair := self._active_pair()) is not None:
            side = int(self.pending_hole is not None)
            section = self.match_interval.currentIndex() + side + 1
            instruction = (f'Click a hole contour in section {section}.'
                           if any(key is not None for key, _ in self._section_rims(pair[side])[1:])
                           else f'Section {section} has no unambiguous hole contours.')
            self.message.setText(f'{instruction}\n{self.message.text()}')
        if dropped:
            self.message.setText(f'Removed {dropped} hole match(es) between sections that are no longer adjacent.\n'
                                 f'{self.message.text()}')
        self.manager.plotter.render()
