"""A compact, right-anchored overlay that does not resize the viewport."""
from PySide6 import QtCore, QtWidgets


class OperationPanel(QtWidgets.QFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.setObjectName('operationPanel')
        self.title = None
        self.content = None
        self.scroll = None
        self.navigation_size = lambda: (0, 0)
        parent.installEventFilter(self)

    def eventFilter(self, watched, event):
        if watched is self.parentWidget() and event.type() in (QtCore.QEvent.Resize, QtCore.QEvent.Show):
            self.reposition()
        return super().eventFilter(watched, event)

    def reposition(self):
        if not self.title or not self.content or not self.scroll:
            return
        parent = self.parentWidget()
        margin = 12
        nav_width, nav_height = self.navigation_size()
        top = margin + max(0, int(nav_height))
        left_of_navigation = top + 48 > parent.height() - margin
        available_width = max(1, parent.width() - 2 * margin - (int(nav_width) + margin if left_of_navigation else 0))
        if left_of_navigation:
            top = margin
        available_height = max(1, parent.height() - top - margin)
        width = min(320, available_width)
        header = max(24, self.title.sizeHint().height())
        content_height = self.content.sizeHint().height() if self.title.isChecked() else 0
        height = min(available_height, header + content_height + (24 if self.title.isChecked() else 2))
        right = parent.width() - margin - (int(nav_width) + margin if left_of_navigation else 0)
        self.setGeometry(max(0, right - width), top, width, max(1, height))
        self.raise_()

    def queue_reposition(self):
        self.reposition()
        QtCore.QTimer.singleShot(0, self.reposition)
