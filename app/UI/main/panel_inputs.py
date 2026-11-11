"""Input prompts hosted in the viewport's fixed operation panel."""
from PySide6 import QtCore, QtWidgets


class PanelInputs:
    def __init__(self, owner):
        self.owner = owner

    def exec_dialog(self, dialog):
        owner = self.owner
        layout = owner.operation_content.layout()
        previous = [layout.itemAt(i).widget() for i in range(layout.count())]
        previous = [widget for widget in previous if widget is not None and not widget.isHidden()]
        old_title = owner.operation_title.text()
        old_expanded = owner.operation_title.isChecked()
        was_open = not owner.operation_panel.isHidden()
        tabs = getattr(owner, 'tool_tabs', None)
        tabs_enabled = tabs.isEnabled() if tabs is not None else False
        if tabs is not None:
            tabs.setEnabled(False)
        for widget in previous:
            widget.hide()
        loop = QtCore.QEventLoop()
        dialog.finished.connect(loop.quit)
        owner.show_operation_panel(dialog)
        loop.exec()
        result = dialog.result()
        layout.removeWidget(dialog)
        dialog.hide()
        dialog.setParent(None)
        if was_open:
            owner.operation_title.setText(old_title)
            for widget in previous:
                widget.show()
            owner.operation_title.setChecked(old_expanded)
            if hasattr(owner.operation_panel, 'queue_reposition'):
                owner.operation_panel.queue_reposition()
        else:
            owner.operation_panel.hide()
        if tabs is not None:
            tabs.setEnabled(tabs_enabled)
        return result

    def _input(self, title, label, mode):
        dialog = QtWidgets.QInputDialog()
        dialog.setWindowTitle(title)
        dialog.setLabelText(label)
        dialog.setInputMode(mode)
        return dialog

    def getItem(self, parent, title, label, items, current=0, editable=True, *args, **kwargs):
        dialog = self._input(title, label, QtWidgets.QInputDialog.TextInput)
        dialog.setComboBoxItems(items)
        dialog.setComboBoxEditable(editable)
        if items:
            dialog.setTextValue(items[current])
        accepted = self.exec_dialog(dialog) == QtWidgets.QDialog.Accepted
        return dialog.textValue(), accepted

    def getText(self, parent, title, label, *args, text='', **kwargs):
        dialog = self._input(title, label, QtWidgets.QInputDialog.TextInput)
        dialog.setTextValue(text)
        accepted = self.exec_dialog(dialog) == QtWidgets.QDialog.Accepted
        return dialog.textValue(), accepted

    def getMultiLineText(self, parent, title, label, text='', *args, **kwargs):
        dialog = self._input(title, label, QtWidgets.QInputDialog.TextInput)
        dialog.setOption(QtWidgets.QInputDialog.UsePlainTextEditForTextInput)
        dialog.setTextValue(text)
        accepted = self.exec_dialog(dialog) == QtWidgets.QDialog.Accepted
        return dialog.textValue(), accepted

    def getInt(self, parent, title, label, value=0, minValue=-2147483647, maxValue=2147483647, step=1, *args, **kwargs):
        dialog = self._input(title, label, QtWidgets.QInputDialog.IntInput)
        dialog.setIntRange(minValue, maxValue)
        dialog.setIntStep(step)
        dialog.setIntValue(value)
        accepted = self.exec_dialog(dialog) == QtWidgets.QDialog.Accepted
        return dialog.intValue(), accepted

    def getDouble(self, parent, title, label, value=0., minValue=-2147483647., maxValue=2147483647., decimals=1, *args, **kwargs):
        dialog = self._input(title, label, QtWidgets.QInputDialog.DoubleInput)
        dialog.setDoubleRange(minValue, maxValue)
        dialog.setDoubleDecimals(decimals)
        dialog.setDoubleValue(value)
        accepted = self.exec_dialog(dialog) == QtWidgets.QDialog.Accepted
        return dialog.doubleValue(), accepted
