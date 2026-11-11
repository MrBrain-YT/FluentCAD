"""Extrusion options and a reversible viewport preview."""
from PySide6 import QtCore, QtWidgets
from UI.model.ModelConverter import b123d_to_pro_mesh
from UI.main.expression_spinbox import ExpressionSpinBox
from UI.main.extrude_handle import ExtrudeHandle, MoveAxisHandles, ParameterHandles, CompositeHandles, create_operation_handles


class ExtrudeDialog(QtWidgets.QDialog):
    def __init__(self, manager, index, parent=None):
        super().__init__(parent)
        self.manager = manager
        self.index = index
        self.result_shape = None
        self.setWindowTitle('Extrude')
        self.setMinimumWidth(290)
        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        self.distance = self.number(10., -1000000., 1000000.)
        self.second_distance = self.number(10., 0., 1000000.)
        self.offset = self.number(0., -1000000., 1000000.)
        self.taper = self.number(0., -88.9, 88.9)
        self.taper.setSuffix('°')
        self.extent = QtWidgets.QComboBox()
        self.extent.addItem('Distance', 'distance')
        self.extent.addItem('Through all (Cut / Intersect)', 'all')
        self.direction = QtWidgets.QComboBox()
        self.direction.addItem('One side', 'one')
        self.direction.addItem('Symmetric (distance per side)', 'symmetric')
        self.direction.addItem('Two sides', 'two')
        self.operation = QtWidgets.QComboBox()
        self.operation.addItem('New body', 'new_body')
        self.operation.addItem('Join', 'join')
        self.operation.addItem('Cut', 'cut')
        self.operation.addItem('Intersect', 'intersect')
        if getattr(manager, 'current_part', None) is not None:
            self.operation.setCurrentIndex(1)
        form.addRow('Extent', self.extent)
        form.addRow('Distance', self.distance)
        form.addRow('Opposite distance', self.second_distance)
        form.addRow('Start offset', self.offset)
        form.addRow('Taper angle', self.taper)
        form.addRow('Direction', self.direction)
        form.addRow('Operation', self.operation)
        layout.addLayout(form)
        self.message = QtWidgets.QLabel()
        self.message.setWordWrap(True)
        self.message.setStyleSheet('color: #a23b32;')
        layout.addWidget(self.message)
        self.buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        for control in (self.distance, self.second_distance, self.offset, self.taper):
            control.valueChanged.connect(self.request_preview)
        self.extent.currentIndexChanged.connect(self.request_preview)
        self.direction.currentIndexChanged.connect(self.request_preview)
        self.operation.currentIndexChanged.connect(self.request_preview)
        self.handle = None
        self.finished.connect(self.clear_preview)
        self.update_preview()
        try:
            self.handle = ExtrudeHandle(self)
        except (AttributeError, RuntimeError, ValueError):
            self.handle = None

    @staticmethod
    def number(value, minimum, maximum):
        control = ExpressionSpinBox()
        control.setDecimals(3)
        control.setRange(minimum, maximum)
        control.setValue(value)
        control.setKeyboardTracking(False)
        return control

    def request_preview(self, *args):
        """Coalesce rapid field and drag events without committing an old result."""
        timer = getattr(self, '_preview_timer', None)
        if timer is None:
            timer = self._preview_timer = QtCore.QTimer(self)
            timer.setSingleShot(True)
            timer.setInterval(40)
            timer.timeout.connect(self.update_preview)
        self.result_shape = None
        self.buttons.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(False)
        if not timer.isActive():
            timer.start()

    def accept(self):
        timer = getattr(self, '_preview_timer', None)
        if timer is not None and timer.isActive():
            timer.stop()
            self.update_preview()
        if self.result_shape is not None:
            super().accept()

    def update_preview(self, *args):
        is_distance = self.extent.currentData() == 'distance'
        self.distance.setEnabled(is_distance)
        self.offset.setEnabled(is_distance)
        self.taper.setEnabled(is_distance)
        self.direction.setEnabled(is_distance)
        self.second_distance.setEnabled(is_distance and self.direction.currentData() == 'two')
        self._update_common_preview()
        if self.handle is not None:
            self.handle.sync()

    def _update_common_preview(self):
        self.manager.plotter.remove_actor('extrude_preview', render=False)
        self.result_shape = None
        body = self.manager.plotter.actors.get('main_part')
        if body is not None:
            body.SetVisibility(True)
        try:
            shape = self.build_shape()
            mesh = b123d_to_pro_mesh(shape, 0.01, 0.1)
            self.manager.plotter.add_mesh(mesh, name='extrude_preview', color='#5aade0',
                                          opacity=0.55, pickable=False, reset_camera=False, render=False)
            if body is not None:
                body.SetVisibility(False)
            self.result_shape = shape
            if hasattr(self.manager, 'refresh_camera_clipping'): self.manager.refresh_camera_clipping()
            self.message.clear()
        except Exception as exc:
            self.message.setText(str(exc))
        self.buttons.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(self.result_shape is not None)
        self.manager.plotter.render()

    def build_shape(self):
        return self.manager.build_extrusion(
            self.index, self.distance.value(), self.operation.currentData(),
            self.offset.value(), self.taper.value(), self.direction.currentData() == 'symmetric',
            self.second_distance.value() if self.direction.currentData() == 'two' else None,
            self.extent.currentData())

    def clear_preview(self, *args):
        timer = getattr(self, '_preview_timer', None)
        if timer is not None:
            timer.stop()
        handle = getattr(self, 'handle', None)
        if handle is not None:
            handle.close()
            self.handle = None
        self.manager.plotter.remove_actor('extrude_preview', render=False)
        body = self.manager.plotter.actors.get('main_part')
        if body is not None:
            body.SetVisibility(True)
        self.manager.plotter.render()


class RevolveDialog(ExtrudeDialog):
    def __init__(self, manager, index, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.manager = manager
        self.index = index
        self.result_shape = None
        self.handle = None
        self.setWindowTitle('Revolve')
        self.setMinimumWidth(290)
        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        self.angle = self.number(360., -360., 360.)
        self.angle.setSuffix('°')
        self.axis = QtWidgets.QComboBox()
        self.axis.addItem('Sketch horizontal axis', 'horizontal')
        self.axis.addItem('Sketch vertical axis', 'vertical')
        self.operation = QtWidgets.QComboBox()
        for label, key in [('New body', 'new_body'), ('Join', 'join'), ('Cut', 'cut'), ('Intersect', 'intersect')]:
            self.operation.addItem(label, key)
        form.addRow('Axis', self.axis)
        form.addRow('Angle', self.angle)
        form.addRow('Operation', self.operation)
        layout.addLayout(form)
        self.message = QtWidgets.QLabel()
        self.message.setWordWrap(True)
        self.message.setStyleSheet('color: #a23b32;')
        layout.addWidget(self.message)
        self.buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.angle.valueChanged.connect(self.request_preview)
        self.axis.currentIndexChanged.connect(self.request_preview)
        self.operation.currentIndexChanged.connect(self.request_preview)
        self.finished.connect(self.clear_preview)
        self.update_preview()
        try:
            self.handle = create_operation_handles(self, [(self.angle, 'revolve', True)])
        except (AttributeError, RuntimeError, ValueError):
            self.handle = None

    def build_shape(self):
        return self.manager.build_revolution(self.index, self.axis.currentData(),
                                             self.angle.value(), self.operation.currentData())

    def update_preview(self, *args):
        self._update_common_preview()
        if self.handle is not None:
            self.handle.sync()

class PrimitiveDialog(ExtrudeDialog):
    """A centered solid primitive with a live preview."""
    def __init__(self, manager, kind, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.manager = manager
        self.kind = kind
        self.result_shape = None
        self.handle = None
        self.setWindowTitle(kind.capitalize())
        self.setMinimumWidth(300)
        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        labels = {'box': ('Length', 'Width', 'Height'),
                  'cylinder': ('Radius', 'Height'), 'sphere': ('Radius',),
                  'cone': ('Bottom radius', 'Top radius', 'Height'),
                  'torus': ('Major radius', 'Tube radius')}[kind]
        self.dimensions = []
        self.position = []
        for index, label in enumerate(labels):
            default = 0. if kind == 'cone' and index == 1 else 3. if kind == 'torus' and index == 1 else 10.
            control = self.number(default, 0., 1000000.)
            self.dimensions.append(control)
            form.addRow(label, control)
        for label in ('Center X', 'Center Y', 'Center Z'):
            control = self.number(0., -1000000., 1000000.)
            self.position.append(control)
            form.addRow(label, control)
        self.operation = QtWidgets.QComboBox()
        for label, key in [('New body', 'new_body'), ('Join', 'join'), ('Cut', 'cut'), ('Intersect', 'intersect')]:
            self.operation.addItem(label, key)
        form.addRow('Operation', self.operation)
        layout.addLayout(form)
        self.message = QtWidgets.QLabel()
        self.message.setWordWrap(True)
        self.message.setStyleSheet('color: #a23b32;')
        layout.addWidget(self.message)
        self.buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        for control in self.dimensions + self.position:
            control.valueChanged.connect(self.request_preview)
        self.operation.currentIndexChanged.connect(self.request_preview)
        self.finished.connect(self.clear_preview)
        self.update_preview()
        axes = {'box': ('X', 'Y', 'Z'), 'cylinder': ('X', 'Z'), 'sphere': ('X',),
                'cone': ('X', 'Y', 'Z'), 'torus': ('X', 'Y')}[kind]
        try:
            self.handle = create_operation_handles(self, [(field, axis, False) for field, axis in zip(self.dimensions, axes)])
        except (AttributeError, RuntimeError, ValueError):
            self.handle = None

    def build_shape(self):
        return self.manager.build_primitive(self.kind, [control.value() for control in self.dimensions],
                                            [control.value() for control in self.position],
                                            self.operation.currentData())

    def update_preview(self, *args):
        self._update_common_preview()
        if self.handle is not None:
            self.handle.sync()


class TransformDialog(ExtrudeDialog):
    """Transform all bodies or the body identified by the selected face."""
    def __init__(self, manager, kind, parent=None):
        from UI.model import transforms
        QtWidgets.QDialog.__init__(self, parent)
        self.manager, self.kind, self.transforms = manager, kind, transforms
        self.result_shape = None
        self.handle = None
        self.setWindowTitle({'move': 'Move / Copy', 'scale': 'Scale', 'mirror': 'Mirror',
                             'rectangular': 'Rectangular pattern', 'circular': 'Circular pattern'}[kind])
        self.setMinimumWidth(300)
        layout = QtWidgets.QVBoxLayout(self)
        description = QtWidgets.QLabel('Select a face before opening this dialog to target its body.')
        description.setWordWrap(True)
        layout.addWidget(description)
        form = QtWidgets.QFormLayout()
        self.controls = {}
        self.selected_body = manager.selected_body_index()
        self.scope = QtWidgets.QComboBox()
        self.scope.addItem('All bodies', 'all')
        if self.selected_body is not None:
            self.scope.addItem('Selected body', 'selected')
            self.scope.setCurrentIndex(1)
        self.scope.currentIndexChanged.connect(self.request_preview)
        form.addRow('Target', self.scope)
        def number(key, label, value=0., minimum=-1000000., maximum=1000000.):
            widget = self.number(value, minimum, maximum)
            widget.valueChanged.connect(self.request_preview)
            self.controls[key] = widget
            form.addRow(label, widget)
        def integer(key, label, value):
            widget = QtWidgets.QSpinBox()
            widget.setRange(1, 100)
            widget.setValue(value)
            widget.valueChanged.connect(self.request_preview)
            self.controls[key] = widget
            form.addRow(label, widget)
        def choice(key, label, choices):
            widget = QtWidgets.QComboBox()
            widget.addItems(choices)
            widget.currentIndexChanged.connect(self.request_preview)
            self.controls[key] = widget
            form.addRow(label, widget)
        def copy_option(default=False):
            widget = QtWidgets.QCheckBox('Keep original bodies')
            widget.setChecked(default)
            widget.toggled.connect(self.request_preview)
            self.controls['copy'] = widget
            form.addRow(widget)
        if kind == 'move':
            for axis in 'XYZ': number('d' + axis, 'Translation ' + axis)
            for axis in 'XYZ': number('r' + axis, 'Rotation ' + axis + ' (°)', 0, -360, 360)
            for axis in 'XYZ': number('p' + axis, 'Pivot ' + axis)
            copy_option()
        elif kind == 'scale':
            number('factor', 'Scale factor', 1., 0., 1000000.)
            for axis in 'XYZ': number('p' + axis, 'Center ' + axis)
            copy_option()
        elif kind == 'mirror':
            choice('plane', 'Mirror plane', ['XY', 'XZ', 'YZ'])
            number('offset', 'Plane offset')
            copy_option(True)
        elif kind == 'rectangular':
            integer('nx', 'Count X', 2)
            number('dx', 'Spacing X', 20.)
            integer('ny', 'Count Y', 1)
            number('dy', 'Spacing Y', 20.)
        else:
            integer('count', 'Count', 4)
            number('angle', 'Total angle (°)', 360., -360., 360.)
            choice('axis', 'Axis', ['Z', 'X', 'Y'])
            for axis in 'XYZ': number('p' + axis, 'Center ' + axis)
        layout.addLayout(form)
        self.message = QtWidgets.QLabel()
        self.message.setWordWrap(True)
        self.message.setStyleSheet('color: #a23b32;')
        layout.addWidget(self.message)
        self.buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.finished.connect(self.clear_preview)
        self.update_preview()
        try:
            if self.kind == 'move':
                move = MoveAxisHandles(self)
                try:
                    self.handle = CompositeHandles(move, ParameterHandles(self, 'rotate'))
                except Exception:
                    move.close()
                    raise
            elif self.kind == 'scale':
                self.handle = ParameterHandles(self, 'scale')
            elif self.kind == 'mirror':
                self.handle = create_operation_handles(self, [(self.controls['offset'], 'mirror', False)])
            elif self.kind == 'rectangular':
                self.handle = create_operation_handles(self, [(self.controls['dx'], 'X', False), (self.controls['dy'], 'Y', False)])
            else:
                self.handle = create_operation_handles(self, [(self.controls['angle'], 'pattern', True)])
        except (AttributeError, RuntimeError, ValueError):
            self.handle = None

    def build_shape(self):
        c = self.controls
        value = lambda name: c[name].value()
        point = lambda prefix: tuple(value(prefix + axis) for axis in 'XYZ')
        shape, functions = self.manager.current_part, self.transforms
        def transform(target):
            if self.kind == 'move':
                return functions.move_copy(target, point('d'), point('r'), point('p'), c['copy'].isChecked())
            if self.kind == 'scale':
                return functions.scale_model(target, value('factor'), point('p'), c['copy'].isChecked())
            if self.kind == 'mirror':
                return functions.mirror_model(target, c['plane'].currentText(), value('offset'), c['copy'].isChecked())
            if self.kind == 'rectangular':
                return functions.rectangular_pattern(target, value('nx'), value('ny'), value('dx'), value('dy'))
            return functions.circular_pattern(target, value('count'), value('angle'), c['axis'].currentText(), point('p'))
        if self.scope.currentData() == 'selected':
            return functions.apply_to_body(shape, self.selected_body, transform)
        return transform(shape)

    def update_preview(self, *args):
        self._update_common_preview()
        if self.handle is not None:
            self.handle.sync()


class SolidFeatureDialog(ExtrudeDialog):
    """Basic solid feature preview with an explicit selection scope."""
    def __init__(self, manager, kind, parent=None):
        from UI.model import surface_operations
        QtWidgets.QDialog.__init__(self, parent)
        self.manager, self.kind, self.operations = manager, kind, surface_operations
        self.result_shape = None
        self.handle = None
        self.face_index = manager.selected_face_id
        self.selected_body = manager.selected_body_index()
        self.edge_indices = sorted(manager.selected_edge_indices)
        self.setWindowTitle({'fillet': 'Fillet all edges', 'chamfer': 'Chamfer all edges',
                             'fillet_selected': 'Fillet selected edges', 'chamfer_selected': 'Chamfer selected edges',
                             'shell': 'Shell selected face', 'offset': 'Offset solid',
                             'hole': 'Hole at face center'}[kind])
        self.setMinimumWidth(300)
        layout = QtWidgets.QVBoxLayout(self)
        if kind in ('fillet', 'chamfer'):
            note = 'This operation applies to every edge of the model.'
        elif kind in ('fillet_selected', 'chamfer_selected'):
            note = f'Applies to {len(self.edge_indices)} selected edge(s).'
        elif kind in ('shell', 'offset'):
            note = 'Select a face to target one body in a multi-body model.'
        else:
            note = 'The hole is positioned from the selected face center.'
        label = QtWidgets.QLabel(note)
        label.setWordWrap(True)
        layout.addWidget(label)
        form = QtWidgets.QFormLayout()
        self.controls = {}
        self.scope = None
        if kind in ('shell', 'offset', 'hole'):
            self.scope = QtWidgets.QComboBox()
            self.scope.addItem('All bodies', 'all')
            if self.selected_body is not None:
                self.scope.addItem('Selected body', 'selected')
                self.scope.setCurrentIndex(1)
            self.scope.currentIndexChanged.connect(self.request_preview)
            form.addRow('Target', self.scope)
        def number(key, title, default, minimum=-1000000., maximum=1000000.):
            field = self.number(default, minimum, maximum)
            self.controls[key] = field
            field.valueChanged.connect(self.request_preview)
            form.addRow(title, field)
        if kind in ('fillet', 'fillet_selected'):
            number('value', 'Radius', 1., 0.)
        elif kind in ('chamfer', 'chamfer_selected'):
            number('value', 'Distance', 1., 0.)
        elif kind == 'shell':
            number('value', 'Thickness (negative inward)', -1.)
        elif kind == 'offset':
            number('value', 'Offset', 1.)
        else:
            number('diameter', 'Diameter', 2., 0.)
            number('depth', 'Depth', 5., 0.)
            number('offset_u', 'Position U', 0.)
            number('offset_v', 'Position V', 0.)
            through = QtWidgets.QCheckBox('Through all')
            through.toggled.connect(self.request_preview)
            self.controls['through'] = through
            form.addRow(through)
        layout.addLayout(form)
        self.message = QtWidgets.QLabel()
        self.message.setWordWrap(True)
        self.message.setStyleSheet('color: #a23b32;')
        layout.addWidget(self.message)
        self.buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.finished.connect(self.clear_preview)
        self.update_preview()
        if kind == 'hole':
            specs = [(self.controls['diameter'], 'radial', False), (self.controls['depth'], 'Z', False)]
        else:
            specs = [(self.controls['value'], 'Z', False)]
        try:
            self.handle = create_operation_handles(self, specs)
        except (AttributeError, RuntimeError, ValueError):
            self.handle = None

    def build_shape(self):
        c, op, shape = self.controls, self.operations, self.manager.current_part
        if self.kind == 'fillet':
            return op.fillet_all(shape, c['value'].value())
        if self.kind == 'chamfer':
            return op.chamfer_all(shape, c['value'].value())
        if self.kind == 'fillet_selected':
            return op.fillet_edges(shape, self.edge_indices, c['value'].value())
        if self.kind == 'chamfer_selected':
            return op.chamfer_edges(shape, self.edge_indices, c['value'].value())
        def apply_feature(target, face_index):
            if self.kind == 'shell':
                return op.shell_face(target, face_index, c['value'].value())
            if self.kind == 'offset':
                return op.offset_solid(target, c['value'].value())
            return op.hole_at_face_center(target, face_index, c['diameter'].value(),
                c['depth'].value(), c['offset_u'].value(), c['offset_v'].value(), c['through'].isChecked())
        if self.scope is not None and self.scope.currentData() == 'selected':
            from UI.model.transforms import apply_to_body
            selected_face = None
            if self.kind in ('shell', 'hole'):
                faces = list(shape.faces())
                if not 0 <= self.face_index < len(faces):
                    raise ValueError('Select a face first.')
                selected_face = faces[self.face_index]
            def apply_selected(target):
                local_index = -1
                if selected_face is not None:
                    local_index = next((i for i, face in enumerate(target.faces())
                                        if face.is_same(selected_face)), -1)
                return apply_feature(target, local_index)
            return apply_to_body(shape, self.selected_body, apply_selected)
        return apply_feature(shape, self.face_index)

    def update_preview(self, *args):
        if self.kind == 'hole':
            self.controls['depth'].setEnabled(not self.controls['through'].isChecked())
        self._update_common_preview()
        if self.handle is not None:
            self.handle.sync()
