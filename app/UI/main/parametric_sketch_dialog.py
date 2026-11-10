"""Mouse editor for named points, arcs, and circles in parametric sketches."""
import math
import numpy as np
import build123d as cad
from PySide6 import QtWidgets, QtCore, QtGui
from UI.model.feature_graph import resolve_parameters
from UI.model.expressions import evaluate_expression
from UI.model.sketch_drag import move_sketch_point, set_circle_radius, add_sketch_constraint


class PointHandle(QtWidgets.QGraphicsEllipseItem):
    def __init__(self, name, x, y, owner, circle_index=None):
        super().__init__(-5, -5, 10, 10)
        self.name, self.owner, self.circle_index = name, owner, circle_index
        self.setBrush(QtGui.QColor('#e39832' if circle_index is not None else '#1785ae'))
        self.setPen(QtGui.QPen(QtGui.QColor('white'), 1))
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable)
        self.setFlag(QtWidgets.QGraphicsItem.ItemSendsGeometryChanges)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations)
        self.setToolTip('Circle radius' if circle_index is not None else name)
        self.setZValue(2)
        self.setPos(x, y)

    def itemChange(self, change, value):
        result = super().itemChange(change, value)
        if change == QtWidgets.QGraphicsItem.ItemPositionHasChanged:
            self.owner.update_lines()
        return result

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if self.circle_index is None:
            self.owner.finish_drag(self.name, (self.pos().x(), self.pos().y()))
        else:
            self.owner.finish_radius(self.circle_index, (self.pos().x(), self.pos().y()))


class ParametricSketchDialog(QtWidgets.QDialog):
    def __init__(self, graph, feature_id, parent=None):
        super().__init__(parent)
        self.graph, self.feature_id = graph, feature_id
        feature = self.feature()
        if feature.kind != 'planar_sketch' or feature.id not in graph.results:
            raise ValueError('Select an active parametric sketch.')
        entities = feature.parameters.get('entities', [])
        for item in entities:
            kind = item.get('type')
            refs = [item.get('center')] if kind == 'circle' else item.get('points', [])
            if kind not in ('line', 'arc', 'circle') or not refs or not all(isinstance(p, str) for p in refs):
                raise ValueError('Mouse editing requires named points for lines, arcs and circles.')
        state = graph.results[feature_id].constraint_status
        if state is None:
            raise ValueError('Mouse editing requires named sketch points.')
        self.setWindowTitle('Edit parametric sketch')
        self.resize(700, 550)
        layout = QtWidgets.QVBoxLayout(self)
        label = QtWidgets.QLabel('Drag named points or orange circle handles. Rebuild runs on release.')
        label.setWordWrap(True)
        layout.addWidget(label)
        self.scene = QtWidgets.QGraphicsScene(self)
        self.view = QtWidgets.QGraphicsView(self.scene)
        self.view.setRenderHint(QtGui.QPainter.Antialiasing)
        self.view.scale(1, -1)
        layout.addWidget(self.view)
        self.message = QtWidgets.QLabel()
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        add_constraint = QtWidgets.QPushButton('Add point constraint / dimension')
        add_constraint.clicked.connect(self.create_constraint)
        layout.addWidget(add_constraint)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.handles, self.lines, self.arcs, self.circles = {}, [], [], []
        self.ready = False
        for name, pos in state.points.items():
            handle = PointHandle(name, float(pos[0]), float(pos[1]), self)
            self.scene.addItem(handle)
            self.handles[name] = handle
        pen = QtGui.QPen(QtGui.QColor('#216caa'), 0)
        for index, entity in enumerate(entities):
            kind = entity['type']
            if kind == 'line':
                for a, b in zip(entity['points'], entity['points'][1:]):
                    self.lines.append((self.scene.addLine(0, 0, 0, 0, pen), a, b))
            elif kind == 'arc':
                self.arcs.append((self.scene.addPath(QtGui.QPainterPath(), pen), entity['points']))
            else:
                circle = self.scene.addEllipse(0, 0, 0, 0, pen)
                handle = PointHandle('', 0, 0, self, index)
                self.scene.addItem(handle)
                self.circles.append((circle, entity['center'], handle, index))
        self.ready = True
        self.refresh_geometry()
        rect = self.scene.itemsBoundingRect().adjusted(-2, -2, 2, 2)
        self.view.fitInView(rect, QtCore.Qt.KeepAspectRatio)

    def feature(self):
        feature = next((item for item in self.graph.features if item.id == self.feature_id), None)
        if feature is None: raise ValueError('Sketch no longer exists.')
        return feature

    def radius(self, index):
        value = self.feature().parameters['entities'][index]['radius']
        return evaluate_expression(value, resolve_parameters(self.graph.parameters)) if isinstance(value, str) else float(value)

    def update_lines(self):
        if not getattr(self, 'ready', False): return
        for line, a, b in self.lines:
            line.setLine(QtCore.QLineF(self.handles[a].pos(), self.handles[b].pos()))
        for item, names in self.arcs:
            try:
                points = [self.handles[name].pos() for name in names]
                edge = cad.Edge.make_three_point_arc((points[0].x(), points[0].y(), 0),
                                                    (points[2].x(), points[2].y(), 0),
                                                    (points[1].x(), points[1].y(), 0))
                path = QtGui.QPainterPath()
                for i, t in enumerate(np.linspace(0, 1, 65)):
                    p = edge.position_at(float(t))
                    if i == 0: path.moveTo(p.X, p.Y)
                    else: path.lineTo(p.X, p.Y)
                item.setPath(path)
            except (ValueError, RuntimeError):
                item.setPath(QtGui.QPainterPath())
        for circle, center_name, radius_handle, index in self.circles:
            center = self.handles[center_name].pos()
            radius = math.hypot(radius_handle.pos().x() - center.x(), radius_handle.pos().y() - center.y())
            circle.setRect(center.x() - radius, center.y() - radius, 2 * radius, 2 * radius)

    def refresh_geometry(self):
        self.ready = False
        state = self.graph.results[self.feature_id].constraint_status
        for key, handle in self.handles.items():
            handle.setPos(*map(float, state.points[key]))
        for _, center_name, radius_handle, index in self.circles:
            center = self.handles[center_name].pos()
            radius_handle.setPos(center.x() + self.radius(index), center.y())
        self.ready = True
        self.update_lines()

    def finish_drag(self, name, position):
        try:
            self.graph = move_sketch_point(self.graph, self.feature_id, name, position)
            self.message.setText('Rebuild succeeded. Press OK to apply.')
        except Exception as exc:
            self.message.setText(str(exc))
        self.refresh_geometry()

    def finish_radius(self, index, position):
        try:
            entity = self.feature().parameters['entities'][index]
            center = self.handles[entity['center']].pos()
            radius = math.hypot(position[0] - center.x(), position[1] - center.y())
            self.graph = set_circle_radius(self.graph, self.feature_id, index, radius)
            self.message.setText('Rebuild succeeded. Press OK to apply.')
        except Exception as exc:
            self.message.setText(str(exc))
        self.refresh_geometry()

    def create_constraint(self):
        names = list(self.handles)
        if not names: return
        labels = ['Coincident', 'Horizontal', 'Vertical', 'Fix point', 'Distance', 'Horizontal distance', 'Vertical distance']
        keys = ['coincident', 'horizontal', 'vertical', 'fixed', 'distance', 'horizontal_distance', 'vertical_distance']
        choice, ok = QtWidgets.QInputDialog.getItem(self, 'Constraint', 'Type:', labels, 0, False)
        if not ok: return
        kind = keys[labels.index(choice)]
        first, ok = QtWidgets.QInputDialog.getItem(self, 'Constraint', 'First point:', names, 0, False)
        if not ok: return
        entry = {'kind': kind}
        if kind == 'fixed':
            p = self.handles[first].pos()
            entry.update(entities=[first], value=[p.x(), p.y()])
        else:
            second, ok = QtWidgets.QInputDialog.getItem(self, 'Constraint', 'Second point:', names, min(1, len(names)-1), False)
            if not ok: return
            if first == second:
                self.message.setText('Choose distinct points.')
                return
            entry['entities'] = [[first, second]] if kind in ('horizontal', 'vertical') else [first, second]
            if kind in ('distance', 'horizontal_distance', 'vertical_distance'):
                value, ok = QtWidgets.QInputDialog.getText(self, 'Dimension', 'Value or named-parameter expression:', text='10')
                if not ok: return
                entry['value'] = value
        try:
            self.graph = add_sketch_constraint(self.graph, self.feature_id, entry)
            state = self.graph.results[self.feature_id].constraint_status
            self.message.setText(f'{state.status}; degrees of freedom: {state.degrees_of_freedom}')
        except Exception as exc:
            self.message.setText(str(exc))
        self.refresh_geometry()
