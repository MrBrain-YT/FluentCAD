"""Quick actions and viewport controls for the modeling workspace."""
from PySide6 import QtCore, QtWidgets

from UI.main.icons import tool_icon


def chrome_button(icon, title, callback=None):
    button = QtWidgets.QToolButton()
    button.setProperty('chrome', True)
    button.setIcon(tool_icon(icon))
    button.setIconSize(QtCore.QSize(20, 20))
    button.setToolTip(title)
    button.setAccessibleName(title)
    if callback is not None:
        button.clicked.connect(callback)
    return button


class ViewportNavigation(QtWidgets.QFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.setObjectName('viewportNavigation')
        self.setAccessibleName('Viewport navigation')
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(4, 3, 4, 3)
        layout.setSpacing(2)
        parent.installEventFilter(self)

    def eventFilter(self, watched, event):
        if watched is self.parentWidget() and event.type() in (QtCore.QEvent.Resize, QtCore.QEvent.Show):
            self.reposition()
        return super().eventFilter(watched, event)

    def reposition(self):
        self.adjustSize()
        viewport = self.parentWidget()
        self.move(max(0, (viewport.width() - self.width()) // 2), max(0, viewport.height() - self.height() - 10))
        self.raise_()
