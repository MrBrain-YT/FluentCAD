"""Movable in-scene numeric input linked to an operation parameter."""
from PySide6 import QtCore, QtWidgets
from UI.main.expression_spinbox import ExpressionSpinBox


class DragGrip(QtWidgets.QLabel):
    def __init__(self, owner):
        super().__init__('::', owner)
        self.setObjectName('sceneValueGrip')
        self.setAccessibleName('Move numeric input')
        self.setToolTip('Drag to move the numeric input')
        self.owner = owner
        self.setCursor(QtCore.Qt.SizeAllCursor)
        self._start = None

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self._start = event.globalPosition().toPoint()
            event.accept()

    def mouseMoveEvent(self, event):
        if self._start is not None:
            current = event.globalPosition().toPoint()
            self.owner.offset += current - self._start
            self._start = current
            self.owner.reposition()
            event.accept()

    def mouseReleaseEvent(self, event):
        self._start = None
        event.accept()


class SceneValueEditor(QtWidgets.QFrame):
    def __init__(self, plotter, field):
        super().__init__(plotter)
        self.setObjectName('sceneValueEditor')
        self.plotter, self.field = plotter, field
        self.offset = QtCore.QPoint(18, -18)
        self.point = None
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(2, 1, 2, 1)
        layout.setSpacing(2)
        self.editor = ExpressionSpinBox()
        self.editor.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.editor.setDecimals(field.decimals())
        self.editor.setRange(field.minimum(), field.maximum())
        self.editor.setSuffix(field.suffix())
        self.editor.setKeyboardTracking(False)
        self.editor.setFixedWidth(90)
        layout.addWidget(self.editor)
        layout.addWidget(DragGrip(self))
        self.editor.valueChanged.connect(field.setValue)
        field.valueChanged.connect(self.set_value)
        self.set_value(field.value())
        self.adjustSize()

    def set_value(self, value):
        blocker = QtCore.QSignalBlocker(self.editor)
        self.editor.setValue(value)
        del blocker

    def set_point(self, point):
        self.point = point
        self.reposition()

    def reposition(self):
        if self.point is None:
            return
        renderer = self.plotter.renderer
        renderer.SetWorldPoint(*map(float, self.point), 1.)
        renderer.WorldToDisplay()
        x, y, depth = renderer.GetDisplayPoint()
        if not 0 <= depth <= 1:
            self.hide()
            return
        ratio = max(float(self.plotter.devicePixelRatioF()), 1.)
        x = int(x / ratio) + self.offset.x()
        y = self.plotter.height() - int(y / ratio) + self.offset.y()
        self.move(max(0, min(x, self.plotter.width() - self.width())),
                  max(0, min(y, self.plotter.height() - self.height())))
        self.show()
        self.raise_()

    def dispose(self):
        try:
            self.field.valueChanged.disconnect(self.set_value)
            self.editor.valueChanged.disconnect(self.field.setValue)
        except (RuntimeError, TypeError):
            pass
        self.hide()
        self.setParent(None)
        self.field = None
        self.plotter = None
        self.deleteLater()
