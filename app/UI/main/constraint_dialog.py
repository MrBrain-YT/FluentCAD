"""Minimal constraint and dimension editor for line endpoints."""
from PySide6 import QtWidgets
from UI.model.constraints import Constraint
from UI.model.expressions import evaluate_expression
from UI.model.sketch_constraints import line_points, apply_constraint, history_line_points


class ConstraintDialog(QtWidgets.QDialog):
    def __init__(self, manager, parent=None):
        super().__init__(parent)
        self.manager = manager
        self.setWindowTitle('Line constraints and dimensions')
        self.setMinimumWidth(340)
        layout = QtWidgets.QVBoxLayout(self)
        info = QtWidgets.QLabel('Choose line endpoints. These commands currently apply to lines and polygon edges.')
        info.setWordWrap(True)
        layout.addWidget(info)
        form = QtWidgets.QFormLayout()
        self.kind = QtWidgets.QComboBox()
        for label, key in [('Coincident', 'coincident'), ('Horizontal', 'horizontal'),
                           ('Vertical', 'vertical'), ('Fix point', 'fixed'),
                           ('Distance', 'distance'), ('Horizontal distance', 'horizontal_distance'),
                           ('Vertical distance', 'vertical_distance')]:
            self.kind.addItem(label, key)
        self.first, self.second = QtWidgets.QComboBox(), QtWidgets.QComboBox()
        names = list(history_line_points(manager) if manager.history_edit_id is not None else line_points(manager))
        self.first.addItems(names)
        self.second.addItems(names)
        if len(names) > 1:
            self.second.setCurrentIndex(1)
        self.value = QtWidgets.QLineEdit('10')
        self.value.setToolTip('Numeric expression, for example 20 / 2')
        form.addRow('Constraint', self.kind)
        form.addRow('First point', self.first)
        form.addRow('Second point', self.second)
        form.addRow('Dimension', self.value)
        layout.addLayout(form)
        self.message = QtWidgets.QLabel()
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Apply | QtWidgets.QDialogButtonBox.Close)
        buttons.button(QtWidgets.QDialogButtonBox.Apply).clicked.connect(self.apply)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.kind.currentIndexChanged.connect(self.update_fields)
        self.update_fields()

    def update_fields(self):
        kind = self.kind.currentData()
        self.second.setEnabled(kind != 'fixed')
        self.value.setEnabled(kind in ('distance', 'horizontal_distance', 'vertical_distance'))

    def apply(self):
        try:
            kind = self.kind.currentData()
            first, second = self.first.currentText(), self.second.currentText()
            if not first or (kind != 'fixed' and (not second or first == second)):
                raise ValueError('Choose distinct endpoints.')
            if self.manager.history_edit_id is not None:
                from UI.model.sketch_drag import add_sketch_constraint
                points = history_line_points(self.manager)
                entry = {'kind':kind}
                if kind == 'fixed':
                    entry.update(entities=[first], value=list(map(float, points[first])))
                else:
                    entry['entities'] = [[first, second]] if kind in ('horizontal','vertical') else [first, second]
                    if kind in ('distance','horizontal_distance','vertical_distance'):
                        entry['value'] = self.value.text()
                candidate = add_sketch_constraint(self.manager.history_edit_graph, self.manager.history_edit_id, entry)
                self.manager.history_edit_graph = candidate
                self.manager.render_history_sketch()
                state = candidate.results[self.manager.history_edit_id].constraint_status
                self.message.setText(f'{state.status}; remaining degrees of freedom: {state.degrees_of_freedom}')
                return
            points = line_points(self.manager)
            if kind == 'fixed':
                constraint = Constraint(kind, (first,), tuple(points[first][2]))
            elif kind in ('horizontal', 'vertical'):
                constraint = Constraint(kind, ((first, second),))
            elif kind == 'coincident':
                constraint = Constraint(kind, (first, second))
            else:
                value = evaluate_expression(self.value.text())
                if kind == 'distance' and value <= 0:
                    raise ValueError('Distance must be positive.')
                constraint = Constraint(kind, (first, second), value)
            result = apply_constraint(self.manager, constraint)
            for tool in self.manager.get_current_sketch_data()['sketch_func']:
                if tool.type == 'line' and tool.render_obj['points']:
                    for actor in tool.render_obj['points']:
                        self.manager.plotter.remove_actor(actor)
                    tool.render_obj['points'] = [self.manager.plotter.add_points(point, point_size=10.0)
                                                  for point in tool.points]
            self.manager.redraw_sketch_geometry()
            self.message.setText(f'{result.status}; remaining degrees of freedom: {result.degrees_of_freedom}')
        except (ValueError, KeyError) as exc:
            self.message.setText(str(exc))
