"""Small vector icons for the sketch ribbon."""
from PySide6 import QtCore, QtGui


def tool_icon(kind):
    pixmap = QtGui.QPixmap(48, 48)
    pixmap.fill(QtCore.Qt.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.Antialiasing)
    painter.scale(1.5, 1.5)
    pen = QtGui.QPen(QtGui.QColor('#347ca4'), 2)
    pen.setCapStyle(QtCore.Qt.RoundCap)
    pen.setJoinStyle(QtCore.Qt.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(QtCore.Qt.NoBrush)
    if kind == 'sketch':
        painter.setBrush(QtGui.QColor('#e6f3f7'))
        painter.drawRect(5, 5, 18, 20)
        painter.drawLine(4, 28, 25, 7)
        painter.setPen(QtGui.QPen(QtGui.QColor('#3f9b73'), 2))
        painter.drawLine(23, 22, 30, 22)
        painter.drawLine(26, 19, 26, 26)
    elif kind in ('box', 'cylinder', 'sphere'):
        if kind == 'box':
            painter.drawRect(6, 9, 16, 16)
            painter.drawLine(6, 9, 11, 5)
            painter.drawLine(22, 9, 27, 5)
            painter.drawLine(11, 5, 27, 5)
            painter.drawLine(27, 5, 27, 21)
            painter.drawLine(22, 25, 27, 21)
        elif kind == 'cylinder':
            painter.drawEllipse(6, 5, 20, 8)
            painter.drawLine(6, 9, 6, 23)
            painter.drawLine(26, 9, 26, 23)
            painter.drawArc(6, 19, 20, 8, 180 * 16, 180 * 16)
        else:
            painter.drawEllipse(5, 5, 22, 22)
            painter.drawEllipse(10, 5, 12, 22)
    elif kind == 'cone':
        painter.drawLine(16, 4, 5, 24)
        painter.drawLine(16, 4, 27, 24)
        painter.drawArc(5, 19, 22, 9, 180 * 16, 180 * 16)
    elif kind == 'torus':
        painter.drawEllipse(4, 9, 24, 14)
        painter.drawEllipse(11, 13, 10, 6)
    elif kind == 'extrude':
        painter.setBrush(QtGui.QColor('#a8d5e9'))
        painter.drawPolygon(QtGui.QPolygon([QtCore.QPoint(6, 11), QtCore.QPoint(19, 11), QtCore.QPoint(25, 6), QtCore.QPoint(12, 6)]))
        painter.drawPolygon(QtGui.QPolygon([QtCore.QPoint(6, 11), QtCore.QPoint(19, 11), QtCore.QPoint(19, 25), QtCore.QPoint(6, 25)]))
        painter.drawPolygon(QtGui.QPolygon([QtCore.QPoint(19, 11), QtCore.QPoint(25, 6), QtCore.QPoint(25, 20), QtCore.QPoint(19, 25)]))
    elif kind in ('undo', 'redo'):
        if kind == 'redo':
            painter.translate(32, 0)
            painter.scale(-1, 1)
        painter.drawArc(6, 8, 20, 16, 10 * 16, 155 * 16)
        painter.drawLine(6, 15, 6, 8)
        painter.drawLine(6, 15, 13, 15)
    elif kind == 'fit':
        painter.drawRect(8, 8, 16, 16)
        painter.drawLine(4, 11, 4, 4)
        painter.drawLine(4, 4, 11, 4)
        painter.drawLine(28, 21, 28, 28)
        painter.drawLine(28, 28, 21, 28)
    elif kind in ('inspect', 'inspect_model'):
        painter.drawEllipse(5, 5, 15, 15)
        painter.drawLine(19, 19, 27, 27)
        painter.drawLine(9, 13, 16, 13)
        painter.drawLine(12, 10, 12, 16)
    elif kind == 'projection':
        painter.drawRect(6, 9, 14, 14)
        painter.drawLine(6, 9, 13, 4)
        painter.drawLine(20, 9, 27, 4)
        painter.drawLine(20, 23, 27, 18)
        painter.drawLine(27, 4, 27, 18)
        painter.drawLine(13, 4, 27, 4)
    elif kind == 'delete_body':
        painter.drawRect(7, 9, 18, 18)
        painter.setPen(QtGui.QPen(QtGui.QColor('#bd4c4c'), 2))
        painter.drawLine(10, 12, 22, 24)
        painter.drawLine(22, 12, 10, 24)
    elif kind in ('select_edge', 'clear_edges'):
        painter.drawRect(5, 6, 21, 20)
        painter.setPen(QtGui.QPen(QtGui.QColor('#ee8e20'), 3))
        painter.drawLine(5, 26, 26, 26)
    elif kind in ('fillet_all', 'chamfer_all', 'shell', 'offset_solid', 'hole'):
        painter.drawRect(6, 6, 20, 20)
        if kind == 'hole': painter.drawEllipse(12, 12, 8, 8)
        elif kind == 'shell': painter.drawRect(10, 10, 12, 12)
        elif kind == 'fillet_all': painter.drawArc(5, 5, 10, 10, 90 * 16, 90 * 16)
        else: painter.drawLine(6, 16, 16, 6)
    elif kind in ('move', 'scale', 'mirror', 'rectangular_pattern', 'circular_pattern'):
        if kind == 'move':
            painter.drawLine(16, 4, 16, 28)
            painter.drawLine(4, 16, 28, 16)
            painter.drawLine(12, 8, 16, 4)
            painter.drawLine(20, 8, 16, 4)
            painter.drawLine(24, 12, 28, 16)
            painter.drawLine(24, 20, 28, 16)
        elif kind == 'scale':
            painter.drawRect(5, 13, 10, 10)
            painter.drawRect(11, 6, 16, 16)
        elif kind == 'mirror':
            painter.drawLine(16, 4, 16, 28)
            painter.drawPolygon(QtGui.QPolygon([QtCore.QPoint(5, 7), QtCore.QPoint(12, 12), QtCore.QPoint(12, 24), QtCore.QPoint(5, 24)]))
            painter.drawPolygon(QtGui.QPolygon([QtCore.QPoint(27, 7), QtCore.QPoint(20, 12), QtCore.QPoint(20, 24), QtCore.QPoint(27, 24)]))
        elif kind == 'rectangular_pattern':
            for x in (5, 18):
                for y in (5, 18): painter.drawRect(x, y, 8, 8)
        else:
            painter.drawEllipse(6, 6, 20, 20)
            for x, y in ((14, 3), (25, 14), (14, 25), (3, 14)): painter.drawEllipse(x, y, 4, 4)
    elif kind in ('plane', 'plane_visibility'):
        painter.setBrush(QtGui.QColor('#f8e3b9'))
        painter.drawPolygon(QtGui.QPolygon([QtCore.QPoint(4, 10), QtCore.QPoint(22, 5), QtCore.QPoint(28, 21), QtCore.QPoint(10, 26)]))
        painter.drawLine(10, 17, 23, 13)
    elif kind in ('parametric_new', 'parametric_add', 'parametric_parameters'):
        painter.drawRect(5, 5, 22, 22)
        painter.drawLine(10, 16, 22, 16)
        painter.drawLine(16, 10, 16, 22)
    elif kind == 'save':
        painter.drawRect(5, 4, 22, 24)
        painter.drawRect(9, 5, 13, 7)
        painter.drawRect(9, 17, 14, 10)
    elif kind == 'open':
        painter.drawRect(4, 10, 24, 16)
        painter.drawLine(5, 9, 5, 6)
        painter.drawLine(5, 6, 14, 6)
        painter.drawLine(14, 6, 17, 10)
    elif kind == 'import':
        painter.drawRect(6, 4, 19, 24)
        painter.drawLine(16, 21, 16, 8)
        painter.drawLine(12, 12, 16, 8)
        painter.drawLine(20, 12, 16, 8)
    elif kind == 'export':
        painter.drawRect(6, 4, 19, 24)
        painter.drawLine(16, 8, 16, 21)
        painter.drawLine(12, 17, 16, 21)
        painter.drawLine(20, 17, 16, 21)
    elif kind == 'revolve':
        painter.drawEllipse(7, 7, 18, 18)
        painter.drawLine(16, 4, 16, 28)
        painter.drawArc(3, 11, 26, 10, 200 * 16, 220 * 16)
    elif kind in ('line', 'midpoint_line'):
        painter.drawLine(5, 25, 26, 7)
        painter.setBrush(QtGui.QColor('#ffffff'))
        painter.drawEllipse(QtCore.QPointF(5, 25), 2, 2)
        painter.drawEllipse(QtCore.QPointF(26, 7), 2, 2)
    elif kind in ('rectangle', 'center_rectangle', 'three_point_rectangle'):
        painter.drawRect(5, 8, 22, 16)
    elif kind in ('circle', 'circle_2point', 'circle_3point'):
        painter.drawEllipse(6, 6, 20, 20)
        painter.drawPoint(16, 16)
    elif kind in ('curve', 'center_arc'):
        painter.drawArc(5, 7, 22, 22, 25 * 16, 130 * 16)
    elif kind in ('ellipse', 'ellipse_arc'):
        painter.drawEllipse(4, 10, 24, 12)
    elif kind in ('polygon', 'circumscribed_polygon', 'edge_polygon'):
        painter.drawPolygon(QtGui.QPolygon([QtCore.QPoint(10, 5), QtCore.QPoint(22, 5), QtCore.QPoint(28, 16), QtCore.QPoint(22, 27), QtCore.QPoint(10, 27), QtCore.QPoint(4, 16)]))
    elif kind in ('slot', 'slot_overall', 'slot_center'):
        painter.drawRoundedRect(4, 10, 24, 12, 6, 6)
    elif kind == 'point':
        painter.setBrush(QtGui.QColor('#347ca4'))
        painter.drawEllipse(12, 12, 8, 8)
    elif kind == 'finish':
        painter.setPen(QtGui.QPen(QtGui.QColor('#399a69'), 3))
        painter.drawLine(5, 17, 13, 24)
        painter.drawLine(13, 24, 27, 8)
    painter.end()
    return QtGui.QIcon(pixmap)
