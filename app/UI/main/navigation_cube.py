"""Camera-aligned navigation cube with separate face, edge and corner targets."""
from itertools import product
import weakref

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import vtk
from vtk.util.numpy_support import vtk_to_numpy

AXIS_NAMES = (
    ('Left', 'Right'),
    ('Front', 'Back'),
    ('Bottom', 'Top'),
)


def _regions():
    inner = .73
    free_positions = {
        0: [()],
        1: [(-inner,), (inner,)],
        2: [(-inner, -inner), (inner, -inner), (inner, inner), (-inner, inner)],
    }
    for direction in product((-1, 0, 1), repeat=3):
        active = [axis for axis, sign in enumerate(direction) if sign]
        if not active:
            continue
        free = [axis for axis, sign in enumerate(direction) if not sign]
        polygon = []
        for index, axis in enumerate(active):
            positions = free_positions[len(free)]
            for position in (positions if index % 2 == 0 else reversed(positions)):
                point = np.asarray(direction, dtype=float) * inner
                point[axis] = direction[axis]
                point[free] = position
                polygon.append(point)
        names = [AXIS_NAMES[axis][int(direction[axis] > 0)] for axis in reversed(active)]
        yield direction, np.asarray(polygon), ' / '.join(names)


REGIONS = tuple(_regions())


class HomeButton(QtWidgets.QToolButton):
    def __init__(self, parent, callback):
        super().__init__(parent)
        self.setFixedSize(26, 26)
        self.setAccessibleName('Home · Isometric view')
        self.setToolTip(self.accessibleName())
        self.clicked.connect(callback)
        self.outline = QtGui.QPolygon([QtCore.QPoint(x, y) for x, y in
                                      ((2, 12), (13, 3), (24, 12), (21, 12), (21, 23),
                                       (5, 23), (5, 12))])

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        palette = self.palette()
        active = self.underMouse() or self.hasFocus() or self.isDown()
        color = palette.color(QtGui.QPalette.Link if active else QtGui.QPalette.WindowText)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(color)
        painter.drawPolygon(self.outline)
        painter.setBrush(palette.color(QtGui.QPalette.Window))
        painter.drawRect(11, 15, 4, 8)


class CubeScene:
    """A separate VTK scene with a bevelled solid and cell-based picking."""
    def __init__(self):
        points = vtk.vtkPoints()
        cells = vtk.vtkCellArray()
        self.colors = vtk.vtkUnsignedCharArray()
        self.colors.SetNumberOfComponents(3)
        self.labels = []
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackgroundAlpha(0)
        point_ids = {}
        for direction, vertices, name in REGIONS:
            first, second, third = vertices[:3]
            if np.dot(np.cross(second - first, third - first), direction) < 0:
                vertices = vertices[::-1]
            cells.InsertNextCell(len(vertices))
            for vertex in vertices:
                key = tuple(vertex)
                if key not in point_ids:
                    point_ids[key] = points.InsertNextPoint(*vertex)
                cells.InsertCellPoint(point_ids[key])
            self.colors.InsertNextTuple3(150, 170, 190)
            if direction.count(0) == 2:
                self._add_label(direction, name)

        mesh = vtk.vtkPolyData()
        mesh.SetPoints(points)
        mesh.SetPolys(cells)
        mesh.GetCellData().SetScalars(self.colors)
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(mesh)
        normals.ComputePointNormalsOff()
        normals.ComputeCellNormalsOn()
        normals.ConsistencyOff()
        normals.SplittingOff()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(normals.GetOutputPort())
        mapper.SetColorModeToDirectScalars()
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(mapper)
        prop = self.actor.GetProperty()
        prop.SetInterpolationToFlat()
        prop.SetAmbient(.38)
        prop.SetDiffuse(.65)
        prop.SetSpecular(.25)
        prop.SetSpecularPower(24)
        prop.EdgeVisibilityOn()
        self.renderer.AddActor(self.actor)

        light = vtk.vtkLight()
        light.SetLightTypeToCameraLight()
        light.SetPosition(-3, 4, 6)
        light.SetFocalPoint(0, 0, 0)
        light.SetIntensity(.9)
        self.renderer.AddLight(light)
        self.window = vtk.vtkRenderWindow()
        self.window.SetOffScreenRendering(1)
        self.window.SetAlphaBitPlanes(1)
        self.window.SetMultiSamples(4)
        self.window.AddRenderer(self.renderer)
        self.camera = self.renderer.GetActiveCamera()
        self.camera.ParallelProjectionOn()
        self.camera.SetParallelScale(1.55)
        self.camera.SetClippingRange(2., 12.)

        self.capture = vtk.vtkWindowToImageFilter()
        self.capture.SetInput(self.window)
        self.capture.SetInputBufferTypeToRGBA()
        self.capture.ReadFrontBufferOff()
        self.capture.ShouldRerenderOff()
        self.picker = vtk.vtkCellPicker()
        self.picker.SetTolerance(0.)
        self.picker.PickFromListOn()
        self.picker.AddPickList(self.actor)
        self.image = QtGui.QImage()
        self.state = None
        self.theme = None

    def _add_label(self, direction, name):
        text = vtk.vtkVectorText()
        text.SetText(name.upper())
        text.Update()
        xmin, xmax, ymin, ymax, _, _ = text.GetOutput().GetBounds()
        normal = np.asarray(direction)
        up = np.array((0, 0, 1) if direction[0] or direction[1] else (0, 1, 0))
        right = np.cross(up, normal)
        scale = min(.24, 1.08 / (xmax - xmin))
        center = ((xmin + xmax) / 2 * right + (ymin + ymax) / 2 * up) * scale
        matrices = []
        for sign in (1, -1):
            axes = (right * scale * sign, up * scale * sign, normal, normal * 1.009 - center * sign)
            matrix = vtk.vtkMatrix4x4()
            for row, values in enumerate(np.column_stack(axes)):
                for column, value in enumerate(values):
                    matrix.SetElement(row, column, value)
            matrices.append(matrix)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(text.GetOutputPort())
        label = vtk.vtkActor()
        label.SetMapper(mapper)
        label.SetUserMatrix(matrices[0])
        label.GetProperty().LightingOff()
        label.GetProperty().BackfaceCullingOn()
        label.PickableOff()
        self.renderer.AddActor(label)
        self.labels.append((label, up, matrices))

    def _style(self, dark, hovered):
        for index, (direction, _, _) in enumerate(REGIONS):
            if direction == hovered:
                color = (240, 164, 91) if dark else (209, 116, 44)
            elif direction.count(0) == 2:
                color = (113, 149, 181) if dark else (176, 202, 221)
            else:
                color = (77, 106, 137) if dark else (128, 162, 187)
            self.colors.SetTuple3(index, *color)
        self.colors.Modified()
        self.actor.GetProperty().SetEdgeColor(*((.24, .34, .44) if dark else (.33, .45, .55)))
        for label, _, _ in self.labels:
            label.GetProperty().SetColor(*((.97, .98, 1.) if dark else (.12, .20, .28)))

    def render(self, size, direction, up, dark, hovered):
        state = size, direction, up, dark, hovered
        if state == self.state:
            return self.image
        if (dark, hovered) != self.theme:
            self._style(dark, hovered)
            self.theme = dark, hovered
        self.window.SetSize(*size)
        self.camera.SetPosition(*(-6 * np.asarray(direction)))
        self.camera.SetViewUp(*up)
        self.camera.OrthogonalizeViewUp()
        screen_up = self.camera.GetViewUp()
        for label, label_up, matrices in self.labels:
            label.SetUserMatrix(matrices[int(np.dot(screen_up, label_up) < -1e-6)])
        self.window.Render()
        self.capture.Modified()
        self.capture.Update()
        width, height = size
        pixels = vtk_to_numpy(self.capture.GetOutput().GetPointData().GetScalars())
        pixels = np.ascontiguousarray(pixels.reshape(height, width, 4)[::-1])
        self.image = QtGui.QImage(pixels.data, width, height, width * 4, QtGui.QImage.Format_RGBA8888).copy()
        self.state = state
        return self.image

    def pick(self, x, y):
        if self.picker.Pick(x, y, 0, self.renderer):
            index = self.picker.GetCellId()
            if 0 <= index < len(REGIONS):
                direction, _, name = REGIONS[index]
                return direction, name
        return None, ''


class NavigationCube(QtWidgets.QFrame):
    viewRequested = QtCore.Signal(tuple)
    dragRequested = QtCore.Signal(tuple, tuple)
    gestureStarted = QtCore.Signal()

    def __init__(self, parent, camera, go_home):
        super().__init__(parent)
        self.setObjectName('navigationCube')
        self.setAccessibleName('Navigation cube')
        # The VTK viewport is a native child window. Keep its sibling opaque to
        # the window system and cut away the background with the silhouette.
        self.setAttribute(QtCore.Qt.WA_NativeWindow)
        self.setFixedSize(142, 148)
        self.setMouseTracking(True)
        self.home = HomeButton(self, go_home)
        self.hovered = self.pressed = None
        self.press_point = self.previous_point = None
        self.dragging = False
        self._camera = camera
        self.scene = CubeScene()
        self.canvas = QtCore.QRect(6, 24, 130, 122)
        self.mask_state = None
        self.destroyed.connect(self.scene.window.Finalize)
        self._watch_camera()
        parent.installEventFilter(self)
        self.reposition()

    def _watch_camera(self):
        widget_ref = weakref.ref(self)
        camera = self._camera

        def refresh(_camera, _event):
            widget = widget_ref()
            if widget is not None:
                widget.update()

        observer = camera.AddObserver('ModifiedEvent', refresh)
        self.destroyed.connect(lambda: camera.RemoveObserver(observer))

    def eventFilter(self, watched, event):
        if watched is self.parentWidget() and event.type() in (QtCore.QEvent.Resize, QtCore.QEvent.Show):
            self.reposition()
        return super().eventFilter(watched, event)

    def reposition(self):
        self.home.adjustSize()
        self.home.move(self.width() - self.home.width() - 5, 5)
        self.move(max(0, self.parentWidget().width() - self.width() - 12), 12)
        self.raise_()
        self.update()

    def _image(self):
        ratio = self.devicePixelRatioF()
        size = round(self.canvas.width() * ratio), round(self.canvas.height() * ratio)
        dark = self.palette().color(QtGui.QPalette.Window).lightness() < 125
        direction = tuple(np.round(self._camera.GetDirectionOfProjection(), 12))
        up = tuple(np.round(self._camera.GetViewUp(), 12))
        return self.scene.render(size, direction, up, dark, self.hovered)

    def _target_at(self, point):
        if not QtCore.QRectF(self.canvas).contains(point):
            return None, ''
        image = self._image()
        x = (point.x() - self.canvas.x()) * image.width() / self.canvas.width()
        y = image.height() - 1 - (point.y() - self.canvas.y()) * image.height() / self.canvas.height()
        return self.scene.pick(x, y)

    def paintEvent(self, event):
        super().paintEvent(event)
        image = self._image()
        state = image.cacheKey(), self.home.geometry()
        if state != self.mask_state:
            alpha = image.scaled(self.canvas.size(), QtCore.Qt.IgnoreAspectRatio,
                                 QtCore.Qt.SmoothTransformation).createAlphaMask(QtCore.Qt.ThresholdDither)
            silhouette = QtGui.QRegion(QtGui.QBitmap.fromImage(alpha)).translated(self.canvas.topLeft())
            home = QtGui.QRegion(self.home.outline).translated(state[1].topLeft())
            mask = silhouette.united(home)
            if self.mask() != mask:
                self.setMask(mask)
            self.mask_state = state
        painter = QtGui.QPainter(self)
        painter.drawImage(self.canvas, image)

    def _trackball(self, point):
        offset = point - QtCore.QRectF(self.canvas).center()
        x, y = offset.x() / 52, -offset.y() / 52
        radius = x*x + y*y
        z = np.sqrt(1 - radius) if radius <= .5 else .5 / np.sqrt(radius)
        vector = np.array((x, y, z))
        return tuple(vector / np.linalg.norm(vector))

    def mouseMoveEvent(self, event):
        point = event.position()
        if self.pressed is not None:
            if not event.buttons() & QtCore.Qt.LeftButton:
                self._end_drag()
            elif self.dragging or (point - self.press_point).manhattanLength() >= QtWidgets.QApplication.startDragDistance():
                if not self.dragging:
                    self.dragging = True
                    self._hover(None, '')
                    self.setCursor(QtCore.Qt.ClosedHandCursor)
                self.dragRequested.emit(self._trackball(self.previous_point), self._trackball(point))
                self.previous_point = point
                return
            else:
                return
        hit, name = self._target_at(event.position())
        if hit != self.hovered:
            self._hover(hit, name)

    def _hover(self, hit, name):
        self.hovered = hit
        self.setToolTip(name)
        if hit is None:
            self.unsetCursor()
        else:
            self.setCursor(QtCore.Qt.OpenHandCursor)
        self.update()

    def leaveEvent(self, event):
        if not self.dragging:
            self._hover(None, '')
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.pressed, _ = self._target_at(event.position())
            if self.pressed is not None:
                self.gestureStarted.emit()
                self.press_point = self.previous_point = event.position()

    def mouseReleaseEvent(self, event):
        if event.button() != QtCore.Qt.LeftButton:
            return
        if self.pressed is not None and not self.dragging:
            hit, _ = self._target_at(event.position())
            if hit == self.pressed:
                self.viewRequested.emit(hit)
        self._end_drag()

    def _end_drag(self):
        self.pressed = None
        self.press_point = self.previous_point = None
        self.dragging = False
        self._hover(None, '')
