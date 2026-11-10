"""Numeric input that accepts simple arithmetic expressions."""
from PySide6 import QtGui, QtWidgets
from UI.model.expressions import evaluate_expression


class ExpressionSpinBox(QtWidgets.QDoubleSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setToolTip('Enter a number or arithmetic expression, e.g. 10 / 2 + 3')

    def _expression(self, text):
        text = text.strip()
        if self.prefix() and text.startswith(self.prefix()):
            text = text[len(self.prefix()):]
        if self.suffix() and text.endswith(self.suffix()):
            text = text[:-len(self.suffix())]
        decimal = self.locale().decimalPoint()
        if decimal != '.':
            text = text.replace(decimal, '.')
        return text.strip()

    def validate(self, text, position):
        expression = self._expression(text)
        try:
            value = evaluate_expression(expression)
            if self.minimum() <= value <= self.maximum():
                return (QtGui.QValidator.Acceptable, text, position)
        except ValueError:
            pass
        allowed = set('0123456789+-*/(). eE')
        if all(char in allowed for char in expression) and len(expression) <= 256:
            return (QtGui.QValidator.Intermediate, text, position)
        return (QtGui.QValidator.Invalid, text, position)

    def valueFromText(self, text):
        try:
            return evaluate_expression(self._expression(text))
        except ValueError:
            return self.value()
