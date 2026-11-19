"""Handle clicks on the projected arrow shaft as well as its head."""
import numpy as np
from PySide6 import QtCore


def segment_distance(point, start, end):
    point, start, end = (np.asarray(value, dtype=float)[:2] for value in (point, start, end))
    axis = end - start
    length2 = float(np.dot(axis, axis))
    t = max(0., min(1., float(np.dot(point - start, axis) / length2))) if length2 > 1e-12 else 0.
    return float(np.linalg.norm(point - (start + t * axis)))


class ArrowHitFilter(QtCore.QObject):
    def __init__(self, handle):
        super().__init__(handle.plotter)
        self.handle, self.plotter = handle, handle.plotter
        self.segments = {}
        self.active = None
        self.plotter.installEventFilter(self)

    def cursor(self, event):
        ratio = self.plotter.devicePixelRatioF()
        pos = event.position()
        return (pos.x() * ratio, (self.plotter.height() - pos.y()) * ratio)

    def project(self, point):
        renderer = self.plotter.renderer
        renderer.SetWorldPoint(*map(float, point), 1.)
        renderer.WorldToDisplay()
        return renderer.GetDisplayPoint()[:2]

    def eventFilter(self, watched, event):
        kind = event.type()
        if kind == QtCore.QEvent.MouseButtonPress and event.button() == QtCore.Qt.LeftButton:
            pos = self.cursor(event)
            candidates = [(segment_distance(pos, self.project(start), self.project(end)), key)
                          for key, (start, end) in self.segments.items()]
            if candidates:
                distance, key = min(candidates)
                if distance <= 9 * self.plotter.devicePixelRatioF():
                    self.active = key
                    self.handle._pointer_position = pos
                    self.handle.start_drag(key)
                    return True
        if self.active is not None and kind == QtCore.QEvent.MouseMove:
            self.handle._pointer_position = self.cursor(event)
            self.handle.drag(self.active)
            return True
        if self.active is not None and kind == QtCore.QEvent.MouseButtonRelease and event.button() == QtCore.Qt.LeftButton:
            self.handle._pointer_position = self.cursor(event)
            self.handle.drag(self.active)
            self.active = None
            self.handle.end_drag()
            self.handle._pointer_position = None
            return True
        return False

    def dispose(self):
        self.plotter.removeEventFilter(self)
        self.setParent(None)
        self.handle = None
        self.plotter = None
        self.deleteLater()
