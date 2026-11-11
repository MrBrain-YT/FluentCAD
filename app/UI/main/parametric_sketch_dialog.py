"""Mouse editor for named points, arcs, and circles in parametric sketches."""
import math
import numpy as np
import build123d as cad
from PySide6 import QtWidgets, QtCore, QtGui
from UI.model.parametric_sketch import circle_radius_name
from UI.model.constraints import arc_center_radius, ellipse_contact, fit_spline
from UI.model.feature_graph import resolve_parameters
from UI.model.expressions import evaluate_expression
from UI.model.sketch_drag import move_sketch_point, set_circle_radius, add_sketch_constraint, remove_sketch_constraint, edit_nurbs_parameters


class SketchView(QtWidgets.QGraphicsView):
    def __init__(self, scene, owner):
        super().__init__(scene)
        self.owner = owner
        self.setMouseTracking(True)

    def mousePressEvent(self, event):
        if self.owner.draft_kind and event.button() == QtCore.Qt.LeftButton:
            self.owner.draft_click(event.position().toPoint())
            event.accept()
            return
        if self.owner.draft_kind and event.button() == QtCore.Qt.RightButton:
            self.owner.finish_draft()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.owner.draft_kind:
            self.owner.draw_draft(self.mapToScene(event.position().toPoint()))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def keyPressEvent(self, event):
        if self.owner.draft_kind and event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
            self.owner.finish_draft(); event.accept(); return
        if self.owner.draft_kind and event.key() == QtCore.Qt.Key_Escape:
            self.owner.cancel_draft(); event.accept(); return
        super().keyPressEvent(event)


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
    def __init__(self, graph, feature_id, parent=None, panel_inputs=None):
        super().__init__(parent)
        self.panel_inputs = panel_inputs or QtWidgets.QInputDialog
        self.graph, self.feature_id = graph, feature_id
        self.draft_kind = None
        self.draft_points = []
        feature = self.feature()
        if feature.kind != 'planar_sketch' or feature.id not in graph.results:
            raise ValueError('Select an active parametric sketch.')
        entities = feature.parameters.get('entities', [])
        for item in entities:
            kind = item.get('type')
            refs = ([p for index in item['curves'] for p in entities[index]['points']] if kind == 'composite_path_slot' else
                    entities[item['curve']]['points'] if kind == 'spline_slot' else ([item.get('center')] if kind in ('circle', 'ellipse') else item.get('points', [])))
            if kind not in ('line', 'arc', 'circle', 'ellipse', 'spline', 'fit_spline', 'nurbs', 'periodic_nurbs', 'slot', 'polyline_slot', 'arc_slot', 'spline_slot', 'composite_path_slot') or not refs or not all(isinstance(p, str) for p in refs):
                raise ValueError('Mouse editing requires named points for all sketch entities.')
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
        self.view = SketchView(self.scene, self)
        self.view.setRenderHint(QtGui.QPainter.Antialiasing)
        self.view.scale(1, -1)
        layout.addWidget(self.view)
        self.message = QtWidgets.QLabel()
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        drawing = QtWidgets.QHBoxLayout()
        for title, kind in [('Draw Bezier', 'spline'), ('Draw fit spline', 'fit_spline'), ('Draw NURBS', 'nurbs'), ('Draw periodic NURBS', 'periodic_nurbs')]:
            button = QtWidgets.QPushButton(title)
            button.clicked.connect(lambda checked=False, value=kind: self.start_draft(value))
            drawing.addWidget(button)
        self.draft_construction = QtWidgets.QCheckBox('Construction')
        drawing.addWidget(self.draft_construction)
        layout.addLayout(drawing)
        add_constraint = QtWidgets.QPushButton('Add constraint / dimension')
        add_constraint.clicked.connect(self.create_constraint)
        layout.addWidget(add_constraint)
        add_spline = QtWidgets.QPushButton('Add cubic spline from four named points')
        add_spline.clicked.connect(self.create_spline)
        layout.addWidget(add_spline)
        add_fit_spline = QtWidgets.QPushButton('Add fit-point spline')
        add_fit_spline.clicked.connect(self.create_fit_spline)
        layout.addWidget(add_fit_spline)
        add_nurbs = QtWidgets.QPushButton('Add NURBS from named control points')
        add_nurbs.clicked.connect(self.create_nurbs)
        layout.addWidget(add_nurbs)
        add_periodic = QtWidgets.QPushButton('Add periodic NURBS')
        add_periodic.clicked.connect(lambda: self.create_nurbs(True))
        layout.addWidget(add_periodic)
        edit_nurbs = QtWidgets.QPushButton('Edit NURBS weights and knots')
        edit_nurbs.clicked.connect(self.edit_nurbs_parameters)
        layout.addWidget(edit_nurbs)
        add_polyline_slot = QtWidgets.QPushButton('Add polyline slot')
        add_polyline_slot.clicked.connect(self.create_polyline_slot)
        layout.addWidget(add_polyline_slot)
        add_arc_slot = QtWidgets.QPushButton('Add arc slot')
        add_arc_slot.clicked.connect(self.create_arc_slot)
        layout.addWidget(add_arc_slot)
        add_spline_slot = QtWidgets.QPushButton('Add spline slot')
        add_spline_slot.clicked.connect(self.create_spline_slot)
        layout.addWidget(add_spline_slot)
        add_composite_slot = QtWidgets.QPushButton('Add composite curve slot')
        add_composite_slot.clicked.connect(self.create_composite_slot)
        layout.addWidget(add_composite_slot)
        remove_constraint = QtWidgets.QPushButton('Remove constraint / dimension')
        remove_constraint.clicked.connect(self.remove_constraint)
        layout.addWidget(remove_constraint)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.handles, self.lines, self.arcs, self.circles = {}, [], [], []
        self.arc_entities = []
        self.ellipses = []
        self.splines = []
        self.periodic_splines = []
        self.spline_polygons = {}
        self.slots = []
        self.polyline_slots = []
        self.arc_slots = []
        self.spline_slots = []
        self.composite_slots = []
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
                self.arc_entities.append((index, entity['points']))
            elif kind == 'ellipse':
                self.ellipses.append((self.scene.addPath(QtGui.QPainterPath(), pen), entity['center'], index))
            elif kind in ('spline', 'fit_spline', 'nurbs', 'periodic_nurbs', 'slot', 'polyline_slot', 'arc_slot', 'spline_slot', 'composite_path_slot'):
                target = self.composite_slots if kind == 'composite_path_slot' else (self.spline_slots if kind == 'spline_slot' else (self.periodic_splines if kind == 'periodic_nurbs' else (self.arc_slots if kind == 'arc_slot' else (self.polyline_slots if kind == 'polyline_slot' else (self.slots if kind == 'slot' else self.splines)))))
                points = entity['curves'] if kind == 'composite_path_slot' else (entities[entity['curve']]['points'] if kind == 'spline_slot' else entity['points'])
                target.append((self.scene.addPath(QtGui.QPainterPath(), pen), points, index))
            else:
                circle = self.scene.addEllipse(0, 0, 0, 0, pen)
                handle = PointHandle('', 0, 0, self, index)
                self.scene.addItem(handle)
                self.circles.append((circle, entity['center'], handle, index))
        self.ready = True
        self.refresh_geometry()
        self.draft_item = self.scene.addPath(QtGui.QPainterPath(), QtGui.QPen(QtGui.QColor('#e39832'), 0, QtCore.Qt.DashLine))
        rect = self.scene.itemsBoundingRect().adjusted(-2, -2, 2, 2)
        self.view.fitInView(rect, QtCore.Qt.KeepAspectRatio)

    def feature(self):
        feature = next((item for item in self.graph.features if item.id == self.feature_id), None)
        if feature is None: raise ValueError('Sketch no longer exists.')
        return feature

    def start_draft(self, kind):
        self.draft_kind, self.draft_points = kind, []
        self.draft_item.setPath(QtGui.QPainterPath())
        self.view.setFocus()
        self.message.setText('Click points; Enter or right-click finishes, Esc cancels. Existing points snap within 8 pixels.')

    def draft_click(self, viewport_position):
        pos = self.view.mapToScene(viewport_position)
        nearest = next((name for name, handle in self.handles.items()
                        if (self.view.mapFromScene(handle.pos()) - viewport_position).manhattanLength() <= 8), None)
        value = nearest if nearest is not None else (float(pos.x()), float(pos.y()))
        self.draft_points.append(value)
        self.draw_draft(pos)
        if self.draft_kind == 'spline' and len(self.draft_points) == 4:
            self.finish_draft()

    def draw_draft(self, cursor):
        path = QtGui.QPainterPath()
        for i, value in enumerate(self.draft_points):
            point = self.handles[value].pos() if isinstance(value, str) else QtCore.QPointF(*value)
            if i == 0: path.moveTo(point)
            else: path.lineTo(point)
        if self.draft_points: path.lineTo(cursor)
        self.draft_item.setPath(path)

    def cancel_draft(self):
        self.draft_kind, self.draft_points = None, []
        self.draft_item.setPath(QtGui.QPainterPath())
        self.message.setText('Drawing cancelled.')

    def edit_nurbs_parameters(self):
        entities = self.feature().parameters['entities']
        choices = [(i, item) for i, item in enumerate(entities) if item.get('type') in ('nurbs', 'periodic_nurbs')]
        if not choices:
            self.message.setText('The sketch has no NURBS.')
            return
        labels = [f"{item['type']} {i+1}" for i, item in choices]
        selected, ok = self.panel_inputs.getItem(self, 'NURBS', 'Curve:', labels, 0, False)
        if not ok: return
        index, entity = choices[labels.index(selected)]
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle('NURBS parameters')
        form = QtWidgets.QFormLayout(dialog)
        degree = QtWidgets.QSpinBox(); degree.setRange(2, min(5, len(entity['points'])-1)); degree.setValue(entity['degree'])
        weights = QtWidgets.QLineEdit(', '.join(map(str, entity['weights'])))
        field = 'knots' if entity['type'] == 'nurbs' else 'spans'
        knots = QtWidgets.QLineEdit(', '.join(map(str, entity.get(field, [1]*len(entity['points'])))))
        form.addRow('Degree', degree)
        form.addRow('Weights (comma separated)', weights)
        form.addRow('Knots' if field == 'knots' else 'Periodic knot spans', knots)
        note = QtWidgets.QLabel('Values may be numbers or named-parameter expressions.'); note.setWordWrap(True)
        form.addRow(note)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept); buttons.rejected.connect(dialog.reject); form.addRow(buttons)
        result = self.panel_inputs.exec_dialog(dialog) if hasattr(self.panel_inputs, 'exec_dialog') else dialog.exec()
        if result != QtWidgets.QDialog.Accepted: return
        def parse(text):
            entries = [part.strip() for part in text.split(',')]
            if any(not entry for entry in entries): raise ValueError('Values cannot be empty.')
            result = []
            for entry in entries:
                try: result.append(float(entry))
                except ValueError: result.append(entry)
            return result
        try:
            self.graph = edit_nurbs_parameters(self.graph, self.feature_id, index, degree.value(), parse(weights.text()), parse(knots.text()))
            self.message.setText('NURBS updated. Press OK to apply.')
        except Exception as exc:
            self.message.setText(str(exc))
        self.refresh_geometry()

    def finish_draft(self):
        kind, values = self.draft_kind, list(self.draft_points)
        if not kind: return
        if len(values) < (4 if kind == 'spline' else 3):
            self.message.setText('Not enough points; continue drawing or press Esc.')
            return
        try:
            from copy import deepcopy
            from UI.model.feature_graph import FeatureGraph
            params = deepcopy(self.feature().parameters)
            names = []
            for value in values:
                if isinstance(value, str):
                    names.append(value)
                else:
                    count = 1
                    while f'P{count}' in params['points']: count += 1
                    name = f'P{count}'
                    params['points'][name] = list(value)
                    names.append(name)
            if len(set(names)) != len(names): raise ValueError('Choose distinct points.')
            entity = {'type': kind, 'points': names, 'construction': self.draft_construction.isChecked()}
            if kind in ('nurbs', 'periodic_nurbs'):
                degree = min(3, len(names)-1)
                count = len(names)
                entity.update(degree=degree, weights=[1.] * count)
                if kind == 'nurbs':
                    entity['knots'] = [0.]*(degree+1) + [i/(count-degree) for i in range(1,count-degree)] + [1.]*(degree+1)
                else:
                    entity['spans'] = [1.] * count
            params['entities'].append(entity)
            candidate = FeatureGraph.from_data(self.graph.to_data())
            candidate.edit(self.feature_id, parameters=params)
            self.graph = candidate
            self.ready = False
            for name in names:
                if name not in self.handles:
                    pos = candidate.results[self.feature_id].constraint_status.points[name]
                    handle = PointHandle(name, float(pos[0]), float(pos[1]), self)
                    self.scene.addItem(handle); self.handles[name] = handle
            self.ready = True
            index = len(params['entities'])-1
            target = self.periodic_splines if kind == 'periodic_nurbs' else self.splines
            target.append((self.scene.addPath(QtGui.QPainterPath(), QtGui.QPen(QtGui.QColor('#216caa'), 0)), names, index))
            self.draft_kind, self.draft_points = None, []
            self.draft_item.setPath(QtGui.QPainterPath())
            self.message.setText('Spline added; drag its points. Press OK to apply.')
            self.refresh_geometry()
        except Exception as exc:
            self.message.setText(str(exc))

    def radius(self, index):
        return self.graph.results[self.feature_id].constraint_status.radii[circle_radius_name(index)]

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
        for item, names, index in self.splines:
            points = [self.handles[name].pos() for name in names]
            self.update_spline_polygon(index, points)
            path = QtGui.QPainterPath(points[0])
            entity = self.feature().parameters['entities'][index]
            if entity['type'] == 'nurbs':
                from UI.model.nurbs import nurbs_contact
                variables = resolve_parameters(self.graph.parameters)
                def number(value):
                    return evaluate_expression(value, variables) if isinstance(value, str) else float(value)
                try:
                    values = [(p.x(), p.y()) for p in points]
                    knots = [number(x) for x in entity['knots']]
                    weights = [number(x) for x in entity['weights']]
                    for t in np.linspace(0, 1, 100)[1:]:
                        position, _, _ = nurbs_contact(values, entity['degree'], knots, weights, float(t))
                        path.lineTo(*position)
                except ValueError:
                    path = QtGui.QPainterPath()
            elif entity['type'] == 'fit_spline':
                try:
                    curve = fit_spline([(p.x(), p.y()) for p in points])
                    for t in np.linspace(0, len(points) - 1, max(65, 24 * len(points)))[1:]:
                        position = curve(float(t))
                        path.lineTo(*position)
                except ValueError:
                    path = QtGui.QPainterPath()
            else:
                path.cubicTo(points[1], points[2], points[3])
            item.setPath(path)
        from UI.model.nurbs import periodic_nurbs_contact
        values = resolve_parameters(self.graph.parameters)
        for item, names, index in self.periodic_splines:
            entity = self.feature().parameters['entities'][index]
            points = [(self.handles[name].pos().x(), self.handles[name].pos().y()) for name in names]
            self.update_spline_polygon(index, [QtCore.QPointF(*point) for point in points], closed=True)
            weights = [evaluate_expression(w, values) if isinstance(w, str) else float(w) for w in entity['weights']]
            spans = [evaluate_expression(w, values) if isinstance(w, str) else float(w) for w in entity.get('spans', [1] * len(names))]
            path = QtGui.QPainterPath()
            try:
                for i, t in enumerate(np.linspace(0, 1, 129)):
                    point, _, _ = periodic_nurbs_contact(points, entity['degree'], weights, float(t), spans)
                    if i == 0: path.moveTo(*point)
                    else: path.lineTo(*point)
                path.closeSubpath()
            except ValueError:
                pass
            item.setPath(path)
        from UI.model.tools.Slot import slot_edges
        for item, names, _ in self.slots:
            points = [self.handles[name].pos() for name in names]
            edges = slot_edges([(p.x(), p.y(), 0) for p in points], (0, 0, 1))
            path = QtGui.QPainterPath()
            for edge in edges:
                for i, t in enumerate(np.linspace(0, 1, 25)):
                    p = edge.position_at(float(t))
                    if path.isEmpty() and i == 0: path.moveTo(p.X, p.Y)
                    else: path.lineTo(p.X, p.Y)
            item.setPath(path)
        from UI.model.tools.Slot import polyline_slot_edges
        variables = resolve_parameters(self.graph.parameters)
        from UI.model.arc_slot import arc_slot_edges
        from UI.model.spline_slot import spline_slot_edges
        for item, names, index in self.polyline_slots + self.arc_slots + self.spline_slots:
            points = [self.handles[name].pos() for name in names]
            radius = self.feature().parameters['entities'][index]['radius']
            radius = evaluate_expression(radius, variables) if isinstance(radius, str) else float(radius)
            path = QtGui.QPainterPath()
            try:
                kind = self.feature().parameters['entities'][index]['type']
                coords = [(p.x(), p.y(), 0) for p in points]
                if kind == 'spline_slot':
                    source = self.feature().parameters['entities'][self.feature().parameters['entities'][index]['curve']]
                    number = lambda value: evaluate_expression(value, variables) if isinstance(value, str) else float(value)
                    edges = spline_slot_edges(coords, radius, source['type'], source.get('degree'),
                                              [number(x) for x in source.get('knots', [])],
                                              [number(x) for x in source.get('weights', [])])
                else:
                    edges = arc_slot_edges(coords, (0, 0, 1), radius) if kind == 'arc_slot' else polyline_slot_edges(coords, (0, 0, 1), radius)
                for edge in edges:
                    for i, t in enumerate(np.linspace(0, 1, 25)):
                        p = edge.position_at(float(t))
                        if i == 0: path.moveTo(p.X, p.Y)
                        else: path.lineTo(p.X, p.Y)
            except (ValueError, RuntimeError):
                pass
            item.setPath(path)
        from UI.model.composite_slot import composite_slot_edges
        for item, indices, index in self.composite_slots:
            path = QtGui.QPainterPath()
            try:
                entities = self.feature().parameters['entities']
                def number(value): return evaluate_expression(value, variables) if isinstance(value, str) else float(value)
                segments = []
                for source_index in indices:
                    source = entities[source_index]
                    points = [(self.handles[n].pos().x(), self.handles[n].pos().y()) for n in source['points']]
                    segments.append((source['type'], points, source.get('degree', 0), [number(x) for x in source.get('knots', [])], [number(x) for x in source.get('weights', [])]))
                edges = composite_slot_edges(segments, cad.Plane.XY, number(entities[index]['radius']))
                for edge in edges:
                    for i, t in enumerate(np.linspace(0,1,25)):
                        p = edge.position_at(float(t))
                        if i == 0: path.moveTo(p.X,p.Y)
                        else: path.lineTo(p.X,p.Y)
            except (ValueError, RuntimeError):
                pass
            item.setPath(path)
        for circle, center_name, radius_handle, index in self.circles:
            center = self.handles[center_name].pos()
            radius = math.hypot(radius_handle.pos().x() - center.x(), radius_handle.pos().y() - center.y())
            circle.setRect(center.x() - radius, center.y() - radius, 2 * radius, 2 * radius)
        values = resolve_parameters(self.graph.parameters)
        for item, center_name, index in self.ellipses:
            entity = self.feature().parameters['entities'][index]
            def number(value):
                return evaluate_expression(value, values) if isinstance(value, str) else float(value)
            a, b, angle = number(entity['radius_x']), number(entity['radius_y']), number(entity.get('rotation', 0))
            center = self.handles[center_name].pos()
            path = QtGui.QPainterPath()
            closed = entity.get('is_closed', True)
            start = 0 if closed else number(entity['start_angle'])
            end = 360 if closed else number(entity['end_angle'])
            for i, t in enumerate(np.linspace(start, end, 129)):
                point, _, _ = ellipse_contact((center.x(), center.y()), a, b, angle, float(t))
                if i == 0: path.moveTo(*point)
                else: path.lineTo(*point)
            if closed: path.closeSubpath()
            item.setPath(path)

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

    def update_spline_polygon(self, index, points, closed=False):
        if index not in self.spline_polygons:
            pen = QtGui.QPen(QtGui.QColor('#95a5af'), 0, QtCore.Qt.DashLine)
            self.spline_polygons[index] = self.scene.addPath(QtGui.QPainterPath(), pen)
            self.spline_polygons[index].setZValue(-1)
        path = QtGui.QPainterPath(points[0])
        for point in points[1:]: path.lineTo(point)
        if closed: path.closeSubpath()
        self.spline_polygons[index].setPath(path)

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

    def create_spline(self):
        names = list(self.handles)
        if len(names) < 4:
            self.message.setText('Create at least four named points before adding a spline.')
            return
        points = []
        for index in range(4):
            name, ok = self.panel_inputs.getItem(self, 'Spline', f'Control point {index + 1}:', names, index, False)
            if not ok: return
            points.append(name)
        if len(set(points)) != 4:
            self.message.setText('Choose four distinct control points.')
            return
        try:
            from copy import deepcopy
            from UI.model.feature_graph import FeatureGraph
            params = deepcopy(self.feature().parameters)
            params['entities'].append({'type': 'spline', 'points': points})
            candidate = FeatureGraph.from_data(self.graph.to_data())
            candidate.edit(self.feature_id, parameters=params)
            self.graph = candidate
            index = len(params['entities']) - 1
            pen = QtGui.QPen(QtGui.QColor('#216caa'), 0)
            self.splines.append((self.scene.addPath(QtGui.QPainterPath(), pen), points, index))
            self.message.setText('Spline added. Press OK to apply.')
        except Exception as exc:
            self.message.setText(str(exc))
        self.refresh_geometry()

    def create_fit_spline(self):
        names = list(self.handles)
        if len(names) < 3:
            self.message.setText('Create at least three named points first.')
            return
        count, ok = self.panel_inputs.getInt(self, 'Fit spline', 'Number of fit points:',
                                                  min(4, len(names)), 3, min(100, len(names)))
        if not ok: return
        points = []
        for index in range(count):
            name, ok = self.panel_inputs.getItem(self, 'Fit spline', f'Fit point {index + 1}:', names, index, False)
            if not ok: return
            points.append(name)
        if len(set(points)) != len(points):
            self.message.setText('Choose distinct fit points.')
            return
        try:
            from copy import deepcopy
            from UI.model.feature_graph import FeatureGraph
            params = deepcopy(self.feature().parameters)
            params['entities'].append({'type': 'fit_spline', 'points': points})
            candidate = FeatureGraph.from_data(self.graph.to_data())
            candidate.edit(self.feature_id, parameters=params)
            self.graph = candidate
            index = len(params['entities']) - 1
            self.splines.append((self.scene.addPath(QtGui.QPainterPath(), QtGui.QPen(QtGui.QColor('#216caa'), 0)), points, index))
            self.message.setText('Fit spline added. Press OK to apply.')
        except Exception as exc:
            self.message.setText(str(exc))
        self.refresh_geometry()

    def create_nurbs(self, periodic=False):
        names = list(self.handles)
        if len(names) < 3:
            self.message.setText('Create at least three named control points first.')
            return
        count, ok = self.panel_inputs.getInt(self, 'NURBS', 'Control point count:', min(4, len(names)), 3, min(100, len(names)))
        if not ok: return
        degree, ok = self.panel_inputs.getInt(self, 'NURBS', 'Degree:', min(3, count - 1), 2, min(5, count - 1))
        if not ok: return
        points = []
        for index in range(count):
            name, ok = self.panel_inputs.getItem(self, 'NURBS', f'Control point {index + 1}:', names, index, False)
            if not ok: return
            points.append(name)
        if len(set(points)) != len(points):
            self.message.setText('Choose distinct control points.')
            return
        knots = [0.] * (degree + 1) + [i / (count - degree) for i in range(1, count - degree)] + [1.] * (degree + 1)
        try:
            from copy import deepcopy
            from UI.model.feature_graph import FeatureGraph
            params = deepcopy(self.feature().parameters)
            entity = {'type': 'periodic_nurbs' if periodic else 'nurbs', 'points': points, 'degree': degree, 'weights': [1.] * count}
            if not periodic: entity['knots'] = knots
            else: entity['spans'] = [1.] * count
            params['entities'].append(entity)
            candidate = FeatureGraph.from_data(self.graph.to_data())
            candidate.edit(self.feature_id, parameters=params)
            self.graph = candidate
            index = len(params['entities']) - 1
            (self.periodic_splines if periodic else self.splines).append((self.scene.addPath(QtGui.QPainterPath(), QtGui.QPen(QtGui.QColor('#216caa'), 0)), points, index))
            self.message.setText('NURBS added. Weights and knots can be edited in the feature parameters.')
        except Exception as exc:
            self.message.setText(str(exc))
        self.refresh_geometry()

    def create_polyline_slot(self):
        names = list(self.handles)
        if len(names) < 2:
            self.message.setText('Create at least two named path points first.')
            return
        count, ok = self.panel_inputs.getInt(self, 'Polyline slot', 'Path point count:', min(3, len(names)), 2, min(100, len(names)))
        if not ok: return
        points = []
        for index in range(count):
            name, ok = self.panel_inputs.getItem(self, 'Polyline slot', f'Path point {index + 1}:', names, index, False)
            if not ok: return
            points.append(name)
        if any(a == b for a, b in zip(points, points[1:])):
            self.message.setText('Adjacent path points must be distinct.')
            return
        radius, ok = self.panel_inputs.getText(self, 'Polyline slot', 'Radius or parameter expression:', text='1')
        if not ok: return
        try:
            from copy import deepcopy
            from UI.model.feature_graph import FeatureGraph
            params = deepcopy(self.feature().parameters)
            params['entities'].append({'type': 'polyline_slot', 'points': points, 'radius': radius})
            candidate = FeatureGraph.from_data(self.graph.to_data())
            candidate.edit(self.feature_id, parameters=params)
            self.graph = candidate
            index = len(params['entities']) - 1
            self.polyline_slots.append((self.scene.addPath(QtGui.QPainterPath(), QtGui.QPen(QtGui.QColor('#216caa'), 0)), points, index))
            self.message.setText('Polyline slot added. Press OK to apply.')
        except Exception as exc:
            self.message.setText(str(exc))
        self.refresh_geometry()

    def create_arc_slot(self):
        names = list(self.handles)
        if len(names) < 3:
            self.message.setText('Create three named path points first.')
            return
        points = []
        for index, label in enumerate(('Start', 'End', 'Intermediate')):
            name, ok = self.panel_inputs.getItem(self, 'Arc slot', f'{label} point:', names, index, False)
            if not ok: return
            points.append(name)
        if len(set(points)) != 3:
            self.message.setText('Choose three different points.')
            return
        radius, ok = self.panel_inputs.getText(self, 'Arc slot', 'Radius or parameter expression:', text='1')
        if not ok: return
        try:
            from copy import deepcopy
            from UI.model.feature_graph import FeatureGraph
            params = deepcopy(self.feature().parameters)
            params['entities'].append({'type': 'arc_slot', 'points': points, 'radius': radius})
            candidate = FeatureGraph.from_data(self.graph.to_data())
            candidate.edit(self.feature_id, parameters=params)
            self.graph = candidate
            index = len(params['entities']) - 1
            self.arc_slots.append((self.scene.addPath(QtGui.QPainterPath(), QtGui.QPen(QtGui.QColor('#216caa'), 0)), points, index))
            self.message.setText('Arc slot added. Press OK to apply.')
        except Exception as exc:
            self.message.setText(str(exc))
        self.refresh_geometry()

    def create_spline_slot(self):
        candidates = [(index, points) for _, points, index in self.splines
                      if self.feature().parameters['entities'][index]['type'] in ('spline', 'fit_spline', 'nurbs')]
        if not candidates:
            self.message.setText('Add an open spline path first.')
            return
        labels = [f'Spline {index + 1}: {points[0]} → {points[-1]}' for index, points in candidates]
        selected, ok = self.panel_inputs.getItem(self, 'Spline slot', 'Path:', labels, 0, False)
        if not ok: return
        radius, ok = self.panel_inputs.getText(self, 'Spline slot', 'Radius or parameter expression:', text='1')
        if not ok: return
        curve_index, points = candidates[labels.index(selected)]
        try:
            from copy import deepcopy
            from UI.model.feature_graph import FeatureGraph
            params = deepcopy(self.feature().parameters)
            params['entities'].append({'type': 'spline_slot', 'curve': curve_index, 'radius': radius})
            candidate = FeatureGraph.from_data(self.graph.to_data())
            candidate.edit(self.feature_id, parameters=params)
            self.graph = candidate
            index = len(params['entities']) - 1
            self.spline_slots.append((self.scene.addPath(QtGui.QPainterPath(), QtGui.QPen(QtGui.QColor('#216caa'), 0)), points, index))
            self.message.setText('Spline slot added. Use construction geometry for its path.')
        except Exception as exc:
            self.message.setText(str(exc))
        self.refresh_geometry()

    def create_composite_slot(self):
        entities = self.feature().parameters['entities']
        candidates = [(index, entity) for index, entity in enumerate(entities)
                      if entity.get('type') in ('line','arc','spline','fit_spline','nurbs')
                      and (entity['type'] != 'line' or len(entity.get('points', [])) == 2)]
        if not candidates:
            self.message.setText('Add line, arc or open spline path segments first.')
            return
        labels = [f"{entity['type']} {index+1}: {entity['points'][0]} → {entity['points'][1] if entity['type'] == 'arc' else entity['points'][-1]}" for index, entity in candidates]
        count, ok = self.panel_inputs.getInt(self, 'Composite slot', 'Number of ordered segments:', min(2,len(labels)), 1, min(30,len(labels)))
        if not ok: return
        indices = []
        for i in range(count):
            selected, ok = self.panel_inputs.getItem(self, 'Composite slot', f'Segment {i+1}:', labels, min(i,len(labels)-1), False)
            if not ok: return
            indices.append(candidates[labels.index(selected)][0])
        if len(set(indices)) != len(indices):
            self.message.setText('Choose distinct path segments.')
            return
        radius, ok = self.panel_inputs.getText(self, 'Composite slot', 'Radius or parameter expression:', text='1')
        if not ok: return
        try:
            from copy import deepcopy
            from UI.model.feature_graph import FeatureGraph
            params = deepcopy(self.feature().parameters)
            params['entities'].append({'type':'composite_path_slot','curves':indices,'radius':radius})
            candidate = FeatureGraph.from_data(self.graph.to_data())
            candidate.edit(self.feature_id, parameters=params)
            self.graph = candidate
            index = len(params['entities']) - 1
            self.composite_slots.append((self.scene.addPath(QtGui.QPainterPath(), QtGui.QPen(QtGui.QColor('#216caa'), 0)), indices, index))
            self.message.setText('Composite slot added. Use construction geometry for its path.')
        except Exception as exc:
            self.message.setText(str(exc))
        self.refresh_geometry()

    def create_constraint(self):
        names = list(self.handles)
        if not names: return
        labels = ['Coincident', 'Horizontal', 'Vertical', 'Fix point', 'Distance', 'Horizontal distance', 'Vertical distance',
                  'Parallel', 'Perpendicular', 'Collinear', 'Equal length', 'Angle', 'Midpoint', 'Symmetry',
                  'Radius', 'Diameter', 'Concentric', 'Equal radius', 'Tangent line / circle',
                  'Tangent circles (external)', 'Tangent circles (internal)', 'Arc radius', 'Arc diameter',
                  'Tangent line / arc', 'Tangent arcs at endpoints', 'Tangent arc / circle at endpoint',
                  'Curvature arcs at endpoints', 'Curvature arc / circle at endpoint',
                  'Tangent arc / ellipse at endpoint', 'Curvature arc / ellipse at endpoint',
                  'Equal radii of two arcs', 'Concentric arcs', 'Equal radius arc / circle', 'Concentric arc / circle',
                  'Tangent spline / curve', 'Curvature spline / curve',
                  'Tangent arc / slot', 'Curvature arc / slot', 'Point on line segment', 'Point on circle', 'Point at spline parameter', 'Point at slot parameter', 'Curvature between curves']
        keys = ['coincident', 'horizontal', 'vertical', 'fixed', 'distance', 'horizontal_distance', 'vertical_distance',
                'parallel', 'perpendicular', 'collinear', 'equal_length', 'angle', 'midpoint', 'symmetric',
                'radius', 'diameter', 'concentric', 'equal_radius', 'tangent_line_circle',
                'tangent_circles_external', 'tangent_circles_internal', 'arc_radius', 'arc_diameter',
                'tangent_line_arc', 'tangent_arc_arc', 'tangent_arc_circle',
                'curvature_arc_arc', 'curvature_arc_circle', 'tangent_arc_ellipse', 'curvature_arc_ellipse',
                'equal_radius_arc_arc', 'concentric_arc_arc', 'equal_radius_arc_circle', 'concentric_arc_circle',
                'tangent_spline', 'curvature_spline', 'tangent_arc_slot', 'curvature_arc_slot',
                'point_on_line', 'point_on_circle', 'point_on_spline', 'point_on_slot', 'curvature_curves']
        choice, ok = self.panel_inputs.getItem(self, 'Constraint', 'Type:', labels, 0, False)
        if not ok: return
        kind = keys[labels.index(choice)]
        if kind == 'curvature_curves':
            self.create_curve_curvature()
            return
        if kind == 'point_on_slot':
            self.create_point_on_slot()
            return
        if kind == 'point_on_spline':
            self.create_point_on_spline()
            return
        if kind in ('tangent_spline', 'curvature_spline'):
            self.create_spline_constraint(kind)
            return
        if kind in ('arc_radius', 'arc_diameter', 'tangent_line_arc', 'tangent_arc_arc',
                    'tangent_arc_circle', 'curvature_arc_arc', 'curvature_arc_circle',
                    'tangent_arc_ellipse', 'curvature_arc_ellipse', 'equal_radius_arc_arc',
                    'concentric_arc_arc', 'equal_radius_arc_circle', 'concentric_arc_circle',
                    'tangent_arc_slot', 'curvature_arc_slot'):
            self.create_arc_constraint(kind)
            return
        if kind in ('parallel', 'perpendicular', 'collinear', 'equal_length', 'angle', 'midpoint', 'symmetric', 'point_on_line'):
            self.create_line_constraint(kind, names)
            return
        if kind in ('radius', 'diameter', 'concentric', 'equal_radius', 'tangent_line_circle', 'point_on_circle',
                    'tangent_circles_external', 'tangent_circles_internal'):
            self.create_circle_constraint(kind)
            return
        first, ok = self.panel_inputs.getItem(self, 'Constraint', 'First point:', names, 0, False)
        if not ok: return
        entry = {'kind': kind}
        if kind == 'fixed':
            p = self.handles[first].pos()
            entry.update(entities=[first], value=[p.x(), p.y()])
        else:
            second, ok = self.panel_inputs.getItem(self, 'Constraint', 'Second point:', names, min(1, len(names)-1), False)
            if not ok: return
            if first == second:
                self.message.setText('Choose distinct points.')
                return
            entry['entities'] = [[first, second]] if kind in ('horizontal', 'vertical') else [first, second]
            if kind in ('distance', 'horizontal_distance', 'vertical_distance'):
                value, ok = self.panel_inputs.getText(self, 'Dimension', 'Value or named-parameter expression:', text='10')
                if not ok: return
                entry['value'] = value
        try:
            self.graph = add_sketch_constraint(self.graph, self.feature_id, entry)
            state = self.graph.results[self.feature_id].constraint_status
            self.message.setText(f'{state.status}; degrees of freedom: {state.degrees_of_freedom}')
        except Exception as exc:
            self.message.setText(str(exc))
        self.refresh_geometry()

    def create_line_constraint(self, kind, names):
        lines = [(a, b) for _, a, b in self.lines if a != b]
        if not lines:
            self.message.setText('The sketch has no line segments.')
            return
        labels = [f'{a} → {b} (segment {i + 1})' for i, (a, b) in enumerate(lines)]
        entry = {'kind': kind}
        if kind in ('midpoint', 'symmetric', 'point_on_line'):
            point, ok = self.panel_inputs.getItem(self, 'Constraint', 'Point:', names, 0, False)
            if not ok: return
            if kind == 'symmetric':
                other, ok = self.panel_inputs.getItem(self, 'Constraint', 'Second point:', names, min(1, len(names)-1), False)
                if not ok: return
                if other == point:
                    self.message.setText('Choose distinct points.')
                    return
            selected, ok = self.panel_inputs.getItem(self, 'Constraint', 'Line:', labels, 0, False)
            if not ok: return
            line = lines[labels.index(selected)]
            if point in line or (kind == 'symmetric' and other in line):
                self.message.setText('Choose a point distinct from the line endpoints.')
                return
            entry['entities'] = [point, list(line)] if kind in ('midpoint', 'point_on_line') else [point, other, list(line)]
        else:
            if len(lines) < 2:
                self.message.setText('Choose two different line segments.')
                return
            first, ok = self.panel_inputs.getItem(self, 'Constraint', 'First line:', labels, 0, False)
            if not ok: return
            second, ok = self.panel_inputs.getItem(self, 'Constraint', 'Second line:', labels, 1, False)
            if not ok: return
            if first == second:
                self.message.setText('Choose two different line segments.')
                return
            entry['entities'] = [list(lines[labels.index(first)]), list(lines[labels.index(second)])]
            if kind == 'angle':
                value, ok = self.panel_inputs.getText(self, 'Angle', 'Angle in degrees or parameter expression:', text='90')
                if not ok: return
                entry['value'] = value
        try:
            candidate = add_sketch_constraint(self.graph, self.feature_id, entry)
            self.graph = candidate
            state = candidate.results[self.feature_id].constraint_status
            self.message.setText(f'{state.status}; degrees of freedom: {state.degrees_of_freedom}')
        except Exception as exc:
            self.message.setText(str(exc))
        self.refresh_geometry()

    def create_circle_constraint(self, kind):
        circles = [(index, center) for _, center, _, index in self.circles]
        if not circles:
            self.message.setText('The sketch has no circles.')
            return
        labels = [f'Circle {index + 1} (center {center})' for index, center in circles]
        first, ok = self.panel_inputs.getItem(self, 'Constraint', 'Circle:', labels, 0, False)
        if not ok: return
        def circle_ref(label):
            index, center = circles[labels.index(label)]
            return [center, circle_radius_name(index)]
        entry = {'kind': kind}
        if kind == 'point_on_circle':
            point, ok = self.panel_inputs.getItem(self, 'Constraint', 'Point:', list(self.handles), 0, False)
            if not ok: return
            if point == circle_ref(first)[0]:
                self.message.setText('Choose a point distinct from the circle center.')
                return
            entry['entities'] = [point, circle_ref(first)]
        elif kind in ('radius', 'diameter'):
            entry['entities'] = [circle_ref(first)]
            index, _ = circles[labels.index(first)]
            default = self.radius(index) * (2 if kind == 'diameter' else 1)
            value, ok = self.panel_inputs.getText(self, 'Dimension', 'Value or parameter expression:', text=str(default))
            if not ok: return
            entry['value'] = value
        elif kind == 'tangent_line_circle':
            lines = [(a, b) for _, a, b in self.lines if a != b]
            if not lines:
                self.message.setText('The sketch has no line segments.')
                return
            line_labels = [f'{a} → {b} (segment {i + 1})' for i, (a, b) in enumerate(lines)]
            line, ok = self.panel_inputs.getItem(self, 'Constraint', 'Line:', line_labels, 0, False)
            if not ok: return
            entry['entities'] = [list(lines[line_labels.index(line)]), circle_ref(first)]
        else:
            if len(circles) < 2:
                self.message.setText('Choose two different circles.')
                return
            second, ok = self.panel_inputs.getItem(self, 'Constraint', 'Second circle:', labels, 1, False)
            if not ok: return
            if first == second:
                self.message.setText('Choose two different circles.')
                return
            entry['entities'] = [circle_ref(first), circle_ref(second)]
        try:
            self.graph = add_sketch_constraint(self.graph, self.feature_id, entry)
            state = self.graph.results[self.feature_id].constraint_status
            self.message.setText(f'{state.status}; degrees of freedom: {state.degrees_of_freedom}')
        except Exception as exc:
            self.message.setText(str(exc))
        self.refresh_geometry()

    def create_arc_constraint(self, kind):
        if not self.arc_entities:
            self.message.setText('The sketch has no arcs.')
            return
        labels = [f'Arc {index + 1}: {points[0]} → {points[1]}' for index, points in self.arc_entities]
        selected, ok = self.panel_inputs.getItem(self, 'Constraint', 'Arc:', labels, 0, False)
        if not ok: return
        first_index = labels.index(selected)
        first = self.arc_entities[first_index][1]
        entry = {'kind': kind}
        if kind in ('arc_radius', 'arc_diameter'):
            positions = [self.handles[name].pos() for name in first]
            _, radius = arc_center_radius(*[(p.x(), p.y()) for p in positions])
            value, ok = self.panel_inputs.getText(self, 'Dimension', 'Value or parameter expression:',
                                                       text=str(radius * (2 if kind == 'arc_diameter' else 1)))
            if not ok: return
            entry.update(entities=[list(first)], value=value)
        elif kind in ('equal_radius_arc_arc', 'concentric_arc_arc', 'equal_radius_arc_circle', 'concentric_arc_circle'):
            if kind.endswith('_arc'):
                choices = [label for label in labels if label != selected]
                if not choices:
                    self.message.setText('Choose two different arcs.')
                    return
                other, ok = self.panel_inputs.getItem(self, 'Constraint', 'Second arc:', choices, 0, False)
                if not ok: return
                entry['entities'] = [list(first), list(self.arc_entities[labels.index(other)][1])]
            else:
                if not self.circles:
                    self.message.setText('The sketch has no circles.')
                    return
                circles = [(index, center) for _, center, _, index in self.circles]
                choices = [f'Circle {index + 1} (center {center})' for index, center in circles]
                other, ok = self.panel_inputs.getItem(self, 'Constraint', 'Circle:', choices, 0, False)
                if not ok: return
                index, center = circles[choices.index(other)]
                entry['entities'] = [list(first), [center, circle_radius_name(index)]]
        else:
            endpoint, ok = self.panel_inputs.getItem(self, 'Constraint', 'Arc endpoint:', first[:2], 0, False)
            if not ok: return
            if kind == 'tangent_line_arc':
                lines = [(a, b) for _, a, b in self.lines if a != b]
                if not lines:
                    self.message.setText('The sketch has no line segments.')
                    return
                line_labels = [f'{a} → {b} (segment {i + 1})' for i, (a, b) in enumerate(lines)]
                line, ok = self.panel_inputs.getItem(self, 'Constraint', 'Line:', line_labels, 0, False)
                if not ok: return
                entry['entities'] = [list(lines[line_labels.index(line)]), list(first), endpoint]
            elif kind.endswith('_arc'):
                if len(labels) < 2:
                    self.message.setText('Choose two different arcs.')
                    return
                second, ok = self.panel_inputs.getItem(self, 'Constraint', 'Second arc:', labels,
                                                             1 if first_index == 0 else 0, False)
                if not ok: return
                if second == selected:
                    self.message.setText('Choose two different arcs.')
                    return
                other = self.arc_entities[labels.index(second)][1]
                other_end, ok = self.panel_inputs.getItem(self, 'Constraint', 'Second arc endpoint:', other[:2], 0, False)
                if not ok: return
                entry['entities'] = [list(first), endpoint, list(other), other_end]
            elif kind.endswith('_slot'):
                targets = [('slot', index, points) for _, points, index in self.slots] + [('polyline_slot', index, points) for _, points, index in self.polyline_slots] + [('arc_slot', index, points) for _, points, index in self.arc_slots] + [('spline_slot', index, points) for _, points, index in self.spline_slots] + [('composite_path_slot', index, points) for _, points, index in self.composite_slots]
                if not targets:
                    self.message.setText('The sketch has no slots.')
                    return
                choices = [f'{slot_kind} {index + 1}' for slot_kind, index, _ in targets]
                selected_slot, ok = self.panel_inputs.getItem(self, 'Constraint', 'Slot:', choices, 0, False)
                if not ok: return
                slot_kind, index, points = targets[choices.index(selected_slot)]
                parts = ['left', 'right', 'start_cap', 'end_cap']
                if slot_kind == 'polyline_slot':
                    parts = [f'segment:{i}:{part}' for i in range(len(points) - 1) for part in parts]
                elif slot_kind == 'arc_slot':
                    parts = ['outer', 'inner', 'start_cap', 'end_cap']
                elif slot_kind == 'composite_path_slot':
                    parts = ['start_cap', 'end_cap'] + [f'segment:{i}:{part}' for i in range(len(points)) for part in ('left','right')]
                part, ok = self.panel_inputs.getItem(self, 'Constraint', 'Slot component:', parts, 0, False)
                if not ok: return
                value, ok = self.panel_inputs.getText(self, 'Contact', 'Parameter from 0 to 1 or expression:', text='0.5')
                if not ok: return
                reference = [f'@{slot_kind}', index] if slot_kind in ('polyline_slot', 'arc_slot', 'spline_slot', 'composite_path_slot') else list(points)
                entry.update(entities=[list(first), endpoint, reference, part], value=value)
            elif kind.endswith('_ellipse'):
                if not self.ellipses:
                    self.message.setText('The sketch has no ellipses.')
                    return
                ellipse_labels = [f'Ellipse {index + 1} (center {center})' for _, center, index in self.ellipses]
                ellipse, ok = self.panel_inputs.getItem(self, 'Constraint', 'Ellipse:', ellipse_labels, 0, False)
                if not ok: return
                index = self.ellipses[ellipse_labels.index(ellipse)][2]
                angle, ok = self.panel_inputs.getText(self, 'Contact', 'Ellipse parameter angle in degrees or expression:', text='0')
                if not ok: return
                entry.update(entities=[list(first), endpoint, index], value=angle)
            else:
                if not self.circles:
                    self.message.setText('The sketch has no circles.')
                    return
                circles = [(index, center) for _, center, _, index in self.circles]
                circle_labels = [f'Circle {index + 1} (center {center})' for index, center in circles]
                circle, ok = self.panel_inputs.getItem(self, 'Constraint', 'Circle:', circle_labels, 0, False)
                if not ok: return
                index, center = circles[circle_labels.index(circle)]
                entry['entities'] = [list(first), endpoint, [center, circle_radius_name(index)]]
        try:
            self.graph = add_sketch_constraint(self.graph, self.feature_id, entry)
            state = self.graph.results[self.feature_id].constraint_status
            self.message.setText(f'{state.status}; degrees of freedom: {state.degrees_of_freedom}')
        except Exception as exc:
            self.message.setText(str(exc))
        self.refresh_geometry()

    def create_spline_constraint(self, mode):
        if not self.splines:
            self.message.setText('The sketch has no cubic splines.')
            return
        spline_labels = [f'Spline {i + 1}: {pts[0]} → {pts[-1]}' for _, pts, i in self.splines]
        chosen, ok = self.panel_inputs.getItem(self, 'Constraint', 'Spline:', spline_labels, 0, False)
        if not ok: return
        first_index = spline_labels.index(chosen)
        first = self.splines[first_index][1]
        endpoint, ok = self.panel_inputs.getItem(self, 'Constraint', 'Spline endpoint:', [first[0], first[-1]], 0, False)
        if not ok: return
        targets = ([('spline', label, pts) for j, (_, pts, _) in enumerate(self.splines)
                    if j != first_index for label in [spline_labels[j]]] +
                   [('arc', f'Arc {index + 1}', pts) for index, pts in self.arc_entities] +
                   [('circle', f'Circle {index + 1}', (index, center)) for _, center, _, index in self.circles] +
                   [('ellipse', f'Ellipse {index + 1}', index) for _, _, index in self.ellipses] +
                   [('slot', f'Slot {index + 1}', pts) for _, pts, index in self.slots] +
                   [('polyline_slot', f'Polyline slot {index + 1}', (index, pts)) for _, pts, index in self.polyline_slots] +
                   [('arc_slot', f'Arc slot {index + 1}', (index, pts)) for _, pts, index in self.arc_slots] +
                   [('periodic_nurbs', f'Periodic NURBS {index + 1}', index) for _, _, index in self.periodic_splines] +
                   [('spline_slot', f'Spline slot {index + 1}', (index, pts)) for _, pts, index in self.spline_slots] +
                   [('composite_path_slot', f'Composite slot {index + 1}', (index, indices)) for _, indices, index in self.composite_slots])
        if not targets:
            self.message.setText('Add another curve or a slot first.')
            return
        labels = [item[1] for item in targets]
        target, ok = self.panel_inputs.getItem(self, 'Constraint', 'Target curve:', labels, 0, False)
        if not ok: return
        kind, _, data = targets[labels.index(target)]
        def spline_ref(names, index):
            kind = self.feature().parameters['entities'][index]['type']
            if kind == 'nurbs': return ['@nurbs', index]
            return (['@fit'] if kind == 'fit_spline' else []) + list(names)
        entry = {'kind': f"{mode}_{'slot' if kind in ('polyline_slot', 'arc_slot', 'spline_slot', 'composite_path_slot') else kind}", 'entities': [spline_ref(first, self.splines[first_index][2]), endpoint]}
        if kind in ('spline', 'arc'):
            ends = [data[0], data[-1] if kind == 'spline' else data[1]]
            other_end, ok = self.panel_inputs.getItem(self, 'Constraint', 'Target endpoint:', ends, 0, False)
            if not ok: return
            if kind == 'spline':
                target_index = next(index for _, pts, index in self.splines if pts is data)
                entry['entities'] += [spline_ref(data, target_index), other_end]
            else:
                entry['entities'] += [list(data), other_end]
        elif kind in ('slot', 'polyline_slot', 'arc_slot', 'spline_slot', 'composite_path_slot'):
            parts = ['left', 'right', 'start_cap', 'end_cap']
            if kind == 'polyline_slot':
                index, points = data
                parts = [f'segment:{i}:{part}' for i in range(len(points) - 1) for part in parts]
            elif kind == 'arc_slot':
                index, points = data
                parts = ['outer', 'inner', 'start_cap', 'end_cap']
            elif kind == 'spline_slot':
                index, points = data
            elif kind == 'composite_path_slot':
                index, indices = data
                parts = ['start_cap', 'end_cap'] + [f'segment:{i}:{part}' for i in range(len(indices)) for part in ('left','right')]
            part, ok = self.panel_inputs.getItem(self, 'Constraint', 'Slot component:', parts, 0, False)
            if not ok: return
            value, ok = self.panel_inputs.getText(self, 'Contact', 'Parameter from 0 to 1 or expression:', text='0.5')
            if not ok: return
            entry['entities'] += [[f'@{kind}', index] if kind in ('polyline_slot', 'arc_slot', 'spline_slot', 'composite_path_slot') else list(data), part]
            entry['value'] = value
        elif kind == 'periodic_nurbs':
            value, ok = self.panel_inputs.getText(self, 'Contact', 'Parameter from 0 to 1 or expression:', text='0')
            if not ok: return
            entry['entities'].append(['@periodic_nurbs', data])
            entry['value'] = value
        else:
            value, ok = self.panel_inputs.getText(self, 'Contact', 'Parameter angle in degrees or expression:', text='0')
            if not ok: return
            entry['entities'].append([data[1], circle_radius_name(data[0])] if kind == 'circle' else data)
            entry['value'] = value
        try:
            self.graph = add_sketch_constraint(self.graph, self.feature_id, entry)
            state = self.graph.results[self.feature_id].constraint_status
            self.message.setText(f'{state.status}; degrees of freedom: {state.degrees_of_freedom}')
        except Exception as exc:
            self.message.setText(str(exc))
        self.refresh_geometry()

    def create_point_on_spline(self):
        candidates = [('open', index, points) for _, points, index in self.splines] + [('periodic', index, points) for _, points, index in self.periodic_splines]
        if not candidates:
            self.message.setText('The sketch has no splines.')
            return
        names = list(self.handles)
        point, ok = self.panel_inputs.getItem(self, 'Constraint', 'Point:', names, 0, False)
        if not ok: return
        labels = [f'{kind} spline {index + 1}' for kind, index, _ in candidates]
        selected, ok = self.panel_inputs.getItem(self, 'Constraint', 'Spline:', labels, 0, False)
        if not ok: return
        _, index, control = candidates[labels.index(selected)]
        kind = self.feature().parameters['entities'][index]['type']
        reference = (['@periodic_nurbs', index] if kind == 'periodic_nurbs' else
                     ['@nurbs', index] if kind == 'nurbs' else
                     ['@fit', *control] if kind == 'fit_spline' else list(control))
        value, ok = self.panel_inputs.getText(self, 'Constraint', 'Spline parameter from 0 to 1 or expression:', text='0.5')
        if not ok: return
        try:
            self.graph = add_sketch_constraint(self.graph, self.feature_id, {'kind':'point_on_spline', 'entities':[point, reference], 'value':value})
            state = self.graph.results[self.feature_id].constraint_status
            self.message.setText(f'{state.status}; degrees of freedom: {state.degrees_of_freedom}')
        except Exception as exc:
            self.message.setText(str(exc))
        self.refresh_geometry()

    def create_point_on_slot(self):
        targets = ([('slot', index, points) for _, points, index in self.slots] +
                   [('polyline_slot', index, points) for _, points, index in self.polyline_slots] +
                   [('arc_slot', index, points) for _, points, index in self.arc_slots] +
                   [('spline_slot', index, points) for _, points, index in self.spline_slots] +
                   [('composite_path_slot', index, indices) for _, indices, index in self.composite_slots])
        if not targets:
            self.message.setText('The sketch has no slots.')
            return
        point, ok = self.panel_inputs.getItem(self, 'Constraint', 'Point:', list(self.handles), 0, False)
        if not ok: return
        labels = [f'{kind} {index+1}' for kind, index, _ in targets]
        selected, ok = self.panel_inputs.getItem(self, 'Constraint', 'Slot:', labels, 0, False)
        if not ok: return
        kind, index, points = targets[labels.index(selected)]
        if kind == 'polyline_slot':
            parts = [f'segment:{i}:{side}' for i in range(len(points)-1) for side in ('left','right','start_cap','end_cap')]
        elif kind == 'composite_path_slot':
            parts = ['start_cap','end_cap'] + [f'segment:{i}:{side}' for i in range(len(points)) for side in ('left','right')]
        elif kind == 'arc_slot':
            parts = ['outer','inner','start_cap','end_cap']
        else:
            parts = ['left','right','start_cap','end_cap']
        component, ok = self.panel_inputs.getItem(self, 'Constraint', 'Slot component:', parts, 0, False)
        if not ok: return
        value, ok = self.panel_inputs.getText(self, 'Constraint', 'Parameter from 0 to 1 or expression:', text='0.5')
        if not ok: return
        reference = list(points) if kind == 'slot' else [f'@{kind}', index]
        try:
            self.graph = add_sketch_constraint(self.graph, self.feature_id, {'kind':'point_on_slot',
                'entities':[point, reference, component], 'value':value})
            state = self.graph.results[self.feature_id].constraint_status
            self.message.setText(f'{state.status}; degrees of freedom: {state.degrees_of_freedom}')
        except Exception as exc:
            self.message.setText(str(exc))
        self.refresh_geometry()

    def create_curve_curvature(self):
        entries = [(i, item) for i, item in enumerate(self.feature().parameters['entities'])
                   if item.get('type') in ('line','arc','circle','ellipse','spline','fit_spline','nurbs','periodic_nurbs')
                   and (item['type'] != 'line' or len(item.get('points', [])) == 2)]
        if len(entries) < 2:
            self.message.setText('Add two supported curves first.')
            return
        labels = [f"{item['type']} {index+1}" for index, item in entries]
        first, ok = self.panel_inputs.getItem(self, 'Curvature', 'First curve:', labels, 0, False)
        if not ok: return
        second, ok = self.panel_inputs.getItem(self, 'Curvature', 'Second curve:', labels, 1, False)
        if not ok: return
        if first == second:
            self.message.setText('Choose two distinct curves.'); return
        values = []
        for label in (first, second):
            value, ok = self.panel_inputs.getText(self, 'Curvature', f'{label}: parameter from 0 to 1 or expression:', text='0')
            if not ok: return
            values.append(value)
        entry = {'kind':'curvature_curves', 'entities':[['@curve', entries[labels.index(first)][0]],
                                                         ['@curve', entries[labels.index(second)][0]]], 'value':values}
        try:
            self.graph = add_sketch_constraint(self.graph, self.feature_id, entry)
            state = self.graph.results[self.feature_id].constraint_status
            self.message.setText(f'{state.status}; degrees of freedom: {state.degrees_of_freedom}')
        except Exception as exc:
            self.message.setText(str(exc))
        self.refresh_geometry()

    def remove_constraint(self):
        constraints = self.feature().parameters.get('constraints', [])
        if not constraints:
            self.message.setText('The sketch has no constraints.')
            return
        labels = [f"{i + 1}: {item.get('kind')} — {item.get('entities')}" +
                  (f" = {item['value']}" if 'value' in item else '')
                  for i, item in enumerate(constraints)]
        selected, ok = self.panel_inputs.getItem(self, 'Remove constraint', 'Constraint:', labels, 0, False)
        if not ok: return
        try:
            self.graph = remove_sketch_constraint(self.graph, self.feature_id, labels.index(selected))
            state = self.graph.results[self.feature_id].constraint_status
            self.message.setText(f'{state.status}; degrees of freedom: {state.degrees_of_freedom}')
        except Exception as exc:
            self.message.setText(str(exc))
        self.refresh_geometry()
