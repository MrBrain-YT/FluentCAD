"""Mouse gestures for the CAD viewport, including cursor-anchored zoom."""
import numpy as np
from PySide6 import QtCore
import vtk


class MouseNavigation(QtCore.QObject):
    def __init__(self, viewport, manager):
        super().__init__(viewport)
        self.viewport = viewport
        self.manager = manager
        self.previous = None
        self.picker = vtk.vtkCellPicker()
        self.picker.PickFromListOn()
        viewport.installEventFilter(self)

    def eventFilter(self, watched, event):
        if watched is not self.viewport:
            return False
        kind = event.type()
        if kind in (QtCore.QEvent.MouseButtonPress, QtCore.QEvent.MouseButtonDblClick):
            if event.button() == QtCore.Qt.MiddleButton:
                self.previous = event.position()
                return True
        elif kind == QtCore.QEvent.MouseMove and self.previous is not None:
            if not event.buttons() & QtCore.Qt.MiddleButton:
                self.previous = None
                return False
            current, previous = event.position(), self.previous
            self.previous = current
            if event.modifiers() & QtCore.Qt.ShiftModifier and not self.manager.sketch_editing:
                self.orbit(current - previous)
            else:
                self.pan(previous, current)
            return True
        elif kind == QtCore.QEvent.MouseButtonRelease and event.button() == QtCore.Qt.MiddleButton:
            self.previous = None
            return True
        elif kind in (QtCore.QEvent.UngrabMouse, QtCore.QEvent.Hide, QtCore.QEvent.WindowDeactivate):
            self.previous = None
        elif kind == QtCore.QEvent.Wheel:
            turns = event.angleDelta().y() / 120 or event.pixelDelta().y() / 100
            if turns:
                self.zoom(event.position(), 1.2 ** np.clip(turns, -8, 8))
            return True
        return False

    def _pixel(self, point):
        width, height = self.viewport.ren_win.GetSize()
        return (point.x() * width / max(1, self.viewport.width()),
                height - 1 - point.y() * height / max(1, self.viewport.height()))

    def _focal_depth(self):
        renderer = self.viewport.renderer
        renderer.SetWorldPoint(*self.viewport.camera.GetFocalPoint(), 1.)
        renderer.WorldToDisplay()
        return renderer.GetDisplayPoint()[2]

    def _unproject(self, pixel, depth):
        renderer = self.viewport.renderer
        renderer.SetDisplayPoint(*pixel, depth)
        renderer.DisplayToWorld()
        *point, weight = renderer.GetWorldPoint()
        return np.asarray(point) / weight

    def _translate(self, offset):
        camera = self.viewport.camera
        camera.SetPosition(*(np.asarray(camera.GetPosition()) + offset))
        camera.SetFocalPoint(*(np.asarray(camera.GetFocalPoint()) + offset))

    def _refresh(self):
        if self.manager.sketch_editing:
            self.manager.update_grid_lod()
        self.viewport.render()

    def pan(self, previous, current):
        depth = self._focal_depth()
        offset = self._unproject(self._pixel(previous), depth) - self._unproject(self._pixel(current), depth)
        self._translate(offset)
        self._refresh()

    def orbit(self, delta):
        camera = self.viewport.camera
        camera.Azimuth(-delta.x() * .35)
        camera.Elevation(delta.y() * .35)
        camera.OrthogonalizeViewUp()
        self._refresh()

    def zoom(self, point, factor):
        camera = self.viewport.camera
        pixel = self._pixel(point)
        anchor = self._unproject(pixel, self._focal_depth())
        if camera.GetParallelProjection():
            camera.SetParallelScale(np.clip(camera.GetParallelScale() / factor, 1e-6, 1e9))
            self._translate(anchor - self._unproject(pixel, self._focal_depth()))
        else:
            actor = self.viewport.actors.get('main_part')
            if actor is not None and actor.GetVisibility():
                self.picker.InitializePickList()
                self.picker.AddPickList(actor)
                if self.picker.Pick(*pixel, 0, self.viewport.renderer):
                    anchor = np.asarray(self.picker.GetPickPosition())
            distance = camera.GetDistance()
            ratio = np.clip(distance / factor, 1e-6, 1e9) / distance
            position = anchor + (np.asarray(camera.GetPosition()) - anchor) * ratio
            focal = anchor + (np.asarray(camera.GetFocalPoint()) - anchor) * ratio
            camera.SetPosition(*position)
            camera.SetFocalPoint(*focal)
        self._refresh()
