from PySide6 import QtWidgets, QtCore

class ClickableFrame(QtWidgets.QFrame):
    # Создаем собственный сигнал "клик"
    clicked = QtCore.Signal()

    def mousePressEvent(self, event):
        # Если нажата левая кнопка мыши
        if event.button() == QtCore.Qt.LeftButton:
            self.clicked.emit() # Испускаем сигнал
        super().mousePressEvent(event)