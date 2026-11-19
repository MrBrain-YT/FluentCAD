"""Interpolate the CAD camera and rotate it with the navigation trackball."""
from dataclasses import dataclass
import weakref

import numpy as np
from PySide6 import QtCore
from scipy.spatial.transform import Rotation, Slerp
import vtk


def _orientation(outward, up):
    outward = np.asarray(outward, dtype=float)
    outward /= np.linalg.norm(outward)
    right = np.cross(up, outward)
    right /= np.linalg.norm(right)
    return Rotation.from_matrix(np.column_stack((right, np.cross(outward, right), outward)))


@dataclass
class _Pose:
    orientation: Rotation
    center: np.ndarray
    distance: float
    scale: float

    @classmethod
    def read(cls, camera):
        return cls(_orientation(-np.asarray(camera.GetDirectionOfProjection()), camera.GetViewUp()),
                   np.asarray(camera.GetFocalPoint()), camera.GetDistance(), camera.GetParallelScale())

    def apply(self, camera):
        _, up, outward = self.orientation.as_matrix().T
        camera.SetFocalPoint(*self.center)
        camera.SetPosition(*(self.center + self.distance * outward))
        camera.SetViewUp(*up)
        camera.SetParallelScale(self.scale)


class CameraTransition(QtCore.QObject):
    def __init__(self, parent, plotter, manager):
        super().__init__(parent)
        self.plotter, self.manager = plotter, manager
        self.camera = plotter.camera
        self._applying = False
        self.animation = QtCore.QVariantAnimation(self)
        self.animation.setDuration(280)
        self.animation.setStartValue(0.)
        self.animation.setEndValue(1.)
        self.animation.setEasingCurve(QtCore.QEasingCurve.InOutCubic)
        self.animation.valueChanged.connect(self._advance)

        transition_ref = weakref.ref(self)

        def camera_changed(_camera, _event):
            transition = transition_ref()
            if transition is not None and not transition._applying:
                transition.stop()

        camera = self.camera
        observer = camera.AddObserver('ModifiedEvent', camera_changed)
        self.destroyed.connect(lambda: camera.RemoveObserver(observer))

    def stop(self):
        self.animation.stop()

    def _can_rotate(self):
        if self.manager.sketch_editing:
            self.manager.status_callback('Finish the sketch before changing the viewing direction.')
            return False
        return True

    def orient(self, direction):
        if not self._can_rotate():
            return
        start = _Pose.read(self.camera)
        up = (0, 0, 1) if direction[0] or direction[1] else (0, 1, 0)
        target = _Pose(_orientation(direction, up), start.center, start.distance, start.scale)
        self._start(start, target)

    def home(self):
        if not self._can_rotate():
            return
        self.stop()
        start = _Pose.read(self.camera)
        saved = vtk.vtkCamera()
        saved.DeepCopy(self.camera)
        camera_set = self.plotter.renderer.camera_set
        self._applying = True
        try:
            self.plotter.view_isometric(render=False)
            target = _Pose.read(self.camera)
        finally:
            self.camera.DeepCopy(saved)
            self.plotter.renderer.camera_set = camera_set
            self._applying = False
        self._start(start, target)

    def _start(self, start, target):
        self.stop()
        self.start_pose, self.target_pose = start, target
        self.interpolate = Slerp([0., 1.], Rotation.concatenate((start.orientation, target.orientation)))
        self.animation.start()

    def _advance(self, amount):
        if self.manager.sketch_editing:
            self.stop()
            return
        start, target = self.start_pose, self.target_pose
        pose = _Pose(self.interpolate(amount), start.center + amount * (target.center - start.center),
                     start.distance + amount * (target.distance - start.distance),
                     start.scale + amount * (target.scale - start.scale))
        self._apply(pose)

    def drag(self, before, after):
        if not self._can_rotate():
            return
        self.stop()
        axis = np.cross(before, after)
        length = np.linalg.norm(axis)
        if length < 1e-10:
            return
        angle = np.arctan2(length, np.dot(before, after))
        pose = _Pose.read(self.camera)
        pose.orientation *= Rotation.from_rotvec(-axis * angle / length)
        self._apply(pose)

    def _apply(self, pose):
        self._applying = True
        try:
            pose.apply(self.camera)
            self.manager.refresh_camera_clipping()
            self.plotter.render()
        finally:
            self._applying = False
