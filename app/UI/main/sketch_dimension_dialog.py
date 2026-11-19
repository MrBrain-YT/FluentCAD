"""Edit a driving dimension while previewing the rebuilt sketch."""
from PySide6 import QtCore, QtWidgets

from UI.model.sketch_dimensions import DIMENSION_NAMES


def edit_sketch_dimension(parent, dimension, preview):
    dialog = QtWidgets.QDialog(parent)
    dialog.setWindowTitle(DIMENSION_NAMES[dimension['kind']])
    layout = QtWidgets.QVBoxLayout(dialog)
    field = QtWidgets.QLineEdit(str(dimension['value']), dialog)
    field.setMaxLength(256)
    label = QtWidgets.QLabel('Value in degrees' if dimension['kind'] == 'angle' else 'Value in mm', dialog)
    label.setBuddy(field)
    error = QtWidgets.QLabel(dialog)
    error.setWordWrap(True)
    error.hide()
    buttons = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel, dialog)
    ok = buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
    layout.addWidget(label)
    layout.addWidget(field)
    layout.addWidget(error)
    layout.addWidget(buttons)
    dialog.setMinimumWidth(270)
    timer = QtCore.QTimer(dialog)
    timer.setSingleShot(True)
    timer.setInterval(220)
    candidate = None

    def update():
        nonlocal candidate
        timer.stop()
        try:
            candidate = preview(field.text())
        except (ValueError, RuntimeError) as exc:
            candidate = None
            error.setText(str(exc))
        else:
            error.clear()
        error.setVisible(candidate is None)
        ok.setEnabled(candidate is not None)

    def accept():
        update()
        if candidate is not None:
            dialog.accept()

    timer.timeout.connect(update)
    field.textChanged.connect(lambda: timer.start())
    buttons.accepted.connect(accept)
    buttons.rejected.connect(dialog.reject)
    field.selectAll()
    field.setFocus()
    update()
    return candidate if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted else None
