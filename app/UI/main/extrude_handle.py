"""Interactive 3-D handle for the distance of a one-sided extrusion."""
import math
import numpy as np
import pyvista as pv
import vtk
from PySide6 import QtCore


def world_per_pixel(camera, height):
    height = max(int(height), 1)
    if camera.GetParallelProjection():
        return 2.0 * camera.GetParallelScale() / height
    distance = np.linalg.norm(np.asarray(camera.position) - np.asarray(camera.focal_point))
    return 2.0 * distance * math.tan(math.radians(camera.GetViewAngle()) / 2) / height


def projected_axis_delta(cursor_start, cursor_now, screen_start, screen_end, world_length):
    axis = np.asarray(screen_end, dtype=float)[:2] - np.asarray(screen_start, dtype=float)[:2]
    length2 = float(np.dot(axis, axis))
    if length2 < 9.: return 0.0
    return float(np.dot(np.asarray(cursor_now, dtype=float)[:2] - np.asarray(cursor_start, dtype=float)[:2], axis) / length2 * world_length)


def axis_drag_projection(plotter, anchor, direction):
    renderer = plotter.renderer
    length = max(world_per_pixel(plotter.camera, plotter.ren_win.GetSize()[1]) * 80, 1e-8)
    def project(point):
        renderer.SetWorldPoint(*map(float, point), 1.)
        renderer.WorldToDisplay()
        return tuple(renderer.GetDisplayPoint())
    return project(anchor), project(anchor + np.asarray(direction)*length), length


def add_outline_arrow(plotter, name, start, tip, direction, size, color='#258bcc'):
    """Draw a fine guide and a blue outlined, camera-facing arrowhead."""
    tip, direction = np.asarray(tip, dtype=float), np.asarray(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    view = np.asarray(plotter.camera.position, dtype=float) - tip
    if np.linalg.norm(view) < 1e-9:
        view = np.array([0., 0., 1.])
    view /= np.linalg.norm(view)
    side = np.cross(direction, view)
    if np.linalg.norm(side) < 1e-6:
        side = np.cross(direction, np.array([0., 1., 0.]) if abs(direction[1]) < .9 else np.array([1., 0., 0.]))
    side /= np.linalg.norm(side)
    def polygon_points(coords, offset):
        return np.asarray([tip + size * (along * direction + across * side) + offset * view
                           for along, across in coords])
    outer = [(0, 0), (-.52, .36), (-.52, .14), (-1., .14),
             (-1., -.14), (-.52, -.14), (-.52, -.36)]
    inner = [(-.12, 0), (-.47, .23), (-.47, .08), (-.91, .08),
             (-.91, -.08), (-.47, -.08), (-.47, -.23)]
    points = (np.asarray([start, tip]), polygon_points(outer, 0), polygon_points(inner, size * .002))
    cache = getattr(plotter, '_fluent_arrow_meshes', None)
    if not isinstance(cache, dict):
        cache = plotter._fluent_arrow_meshes = {}
    if name in cache:
        for mesh, vertices in zip(cache[name], points):
            mesh.points = vertices
        return
    meshes = (pv.Line(start, tip),
              pv.PolyData(points[1], np.array([len(outer), *range(len(outer))])),
              pv.PolyData(points[2], np.array([len(inner), *range(len(inner))])))
    for mesh, suffix, shade in zip(meshes, ('_guide', '', '_inset'), (color, color, 'white')):
        plotter.add_mesh(mesh, name=name + suffix, color=shade, line_width=2,
                         lighting=False, pickable=False, reset_camera=False, render=False)
    cache[name] = meshes


def remove_outline_arrow(plotter, name):
    cache = getattr(plotter, '_fluent_arrow_meshes', None)
    if isinstance(cache, dict):
        cache.pop(name, None)
    for suffix in ('', '_guide', '_inset'):
        plotter.remove_actor(name + suffix, render=False)


def observe_arrow_camera(handle):
    def camera_state():
        camera = handle.plotter.camera
        return (tuple(camera.GetPosition()), tuple(camera.GetFocalPoint()), tuple(camera.GetViewUp()),
                camera.GetParallelProjection(), camera.GetParallelScale(), camera.GetViewAngle(),
                tuple(handle.plotter.ren_win.GetSize()))
    handle._camera_state = camera_state()
    handle._camera_timer = QtCore.QTimer()
    handle._camera_timer.setSingleShot(True)
    handle._camera_timer.setInterval(16)
    handle._camera_timer.timeout.connect(handle.sync)
    def refresh(*_args):
        state = camera_state()
        if state == handle._camera_state:
            return
        handle._camera_state = state
        if not handle._camera_timer.isActive():
            handle._camera_timer.start()
    handle._camera_observer = handle.plotter.camera.AddObserver('ModifiedEvent', refresh)


def schedule_handle_sync(handle):
    if QtCore.QCoreApplication.instance() is None or not hasattr(getattr(handle, 'dialog', None), 'request_preview'):
        handle.sync()
        return
    timer = getattr(handle, '_drag_sync_timer', None)
    if timer is None:
        timer = handle._drag_sync_timer = QtCore.QTimer()
        timer.setSingleShot(True)
        timer.setInterval(16)
        timer.timeout.connect(handle.sync)
    if not timer.isActive():
        timer.start()


def stop_handle_sync(handle):
    timer = getattr(handle, '_drag_sync_timer', None)
    if timer is not None:
        timer.stop()


def projected_ring_angle(cursor, center, first, second):
    """Angle in a projected ring basis; None when the ring is edge-on."""
    matrix = np.column_stack((np.asarray(first)[:2] - np.asarray(center)[:2],
                              np.asarray(second)[:2] - np.asarray(center)[:2]))
    if abs(np.linalg.det(matrix)) < 1e-6 or np.linalg.cond(matrix) > 1000:
        return None
    coordinate = np.linalg.solve(matrix, np.asarray(cursor)[:2] - np.asarray(center)[:2])
    if np.linalg.norm(coordinate) < 1e-6:
        return None
    return math.degrees(math.atan2(coordinate[1], coordinate[0]))


class ExtrudeHandle:
    def __init__(self, dialog):
        self.dialog = dialog
        self.plotter = dialog.manager.plotter
        self.anchor = np.asarray(tuple(dialog.manager.profiles[dialog.index]['face'].center()), dtype=float)
        self.normal = np.asarray(dialog.manager.active_normal, dtype=float)
        self.normal /= np.linalg.norm(self.normal)
        self.drag_start = None
        self.updating = False
        self.widgets, self.representations = {}, {}
        for key in ('primary', 'opposite'):
            rep = vtk.vtkPointHandleRepresentation3D()
            rep.SetHandleSize(11)
            rep.GetProperty().SetColor(.22, .57, .82)
            rep.GetSelectedProperty().SetColor(.65, .85, 1.)
            rep.GetProperty().SetOpacity(.01)
            rep.GetSelectedProperty().SetOpacity(.01)
            widget = vtk.vtkHandleWidget()
            widget.SetInteractor(self.plotter.iren.interactor)
            widget.SetRepresentation(rep)
            widget.AddObserver('StartInteractionEvent', lambda *args, k=key: self.start_drag(k))
            widget.AddObserver('InteractionEvent', lambda *args, k=key: self.drag(k))
            widget.AddObserver('EndInteractionEvent', lambda *args: self.end_drag())
            self.widgets[key], self.representations[key] = widget, rep
            widget.EnabledOn()
        self.sync()
        observe_arrow_camera(self)

    def sync(self):
        if self.updating:
            return
        enabled = self.dialog.extent.currentData() == 'distance'
        two = self.dialog.direction.currentData() in ('two', 'symmetric')
        self.widgets['primary'].SetEnabled(enabled)
        self.widgets['opposite'].SetEnabled(enabled and two)
        if not enabled:
            for key in self.widgets: remove_outline_arrow(self.plotter, 'extrude_arrow_' + key)
            return
        if not two and 'extrude_arrow_opposite' in getattr(self.plotter, '_fluent_arrow_meshes', {}):
            remove_outline_arrow(self.plotter, 'extrude_arrow_opposite')
        offset = self.dialog.offset.value()
        distance = self.dialog.distance.value()
        length = max(world_per_pixel(self.plotter.camera, self.plotter.ren_win.GetSize()[1]) * 16, 1e-6)
        for key in ('primary', 'opposite'):
            if key == 'opposite' and not two: continue
            value = distance if key == 'primary' or self.dialog.direction.currentData() == 'symmetric' else self.dialog.second_distance.value()
            direction = self.normal if key == 'primary' else -self.normal
            point = self.anchor + self.normal * offset + direction * value
            visual_direction = direction if value >= 0 else -direction
            self.representations[key].SetWorldPosition(tuple(point))
            add_outline_arrow(self.plotter, 'extrude_arrow_' + key, self.anchor + self.normal * offset,
                              point, visual_direction, length)
        self.plotter.render()

    def start_drag(self, key='primary'):
        field = self.dialog.second_distance if key == 'opposite' and self.dialog.direction.currentData() == 'two' else self.dialog.distance
        direction = self.normal if key == 'primary' else -self.normal
        point = np.asarray(self.representations[key].GetWorldPosition())
        self.drag_start = (key, tuple(self.plotter.iren.get_event_position()), field.value(), axis_drag_projection(self.plotter, point, direction))

    def drag(self, key='primary'):
        if self.drag_start is None or self.drag_start[0] != key:
            return
        _, cursor, initial, (a, b, length) = self.drag_start
        value = initial + projected_axis_delta(cursor, self.plotter.iren.get_event_position(), a, b, length)
        field = self.dialog.second_distance if key == 'opposite' and self.dialog.direction.currentData() == 'two' else self.dialog.distance
        self.updating = True
        try:
            field.setValue(value)
        finally:
            self.updating = False
        schedule_handle_sync(self)

    def end_drag(self, *args):
        self.drag_start = None
        stop_handle_sync(self)
        self.sync()

    def close(self):
        stop_handle_sync(self)
        self._camera_timer.stop()
        self.plotter.camera.RemoveObserver(self._camera_observer)
        for key, widget in self.widgets.items():
            widget.EnabledOff()
            widget.RemoveAllObservers()
            remove_outline_arrow(self.plotter, 'extrude_arrow_' + key)
        self.plotter.render()


class MoveAxisHandles:
    """Three scene handles control the world-space translation fields."""
    def __init__(self, dialog):
        self.dialog, self.plotter = dialog, dialog.manager.plotter
        self.widgets, self.representations = {}, {}
        self.drag_start, self.updating = None, False
        colors = {'X': (0.88, 0.25, 0.22), 'Y': (0.25, 0.72, 0.35), 'Z': (0.22, 0.47, 0.9)}
        for axis in 'XYZ':
            representation = vtk.vtkPointHandleRepresentation3D()
            representation.SetHandleSize(13)
            representation.GetProperty().SetColor(*colors[axis])
            representation.GetSelectedProperty().SetColor(1., 0.8, 0.25)
            representation.GetProperty().SetOpacity(.01)
            representation.GetSelectedProperty().SetOpacity(.01)
            widget = vtk.vtkHandleWidget()
            widget.SetInteractor(self.plotter.iren.interactor)
            widget.SetRepresentation(representation)
            widget.AddObserver('StartInteractionEvent', lambda obj, event, key=axis: self.start_drag(key))
            widget.AddObserver('InteractionEvent', lambda obj, event, key=axis: self.drag(key))
            widget.AddObserver('EndInteractionEvent', lambda obj, event, key=axis: self.end_drag())
            self.widgets[axis], self.representations[axis] = widget, representation
            widget.EnabledOn()
        self.sync()
        observe_arrow_camera(self)

    def sync(self):
        if self.updating:
            return
        shape = self.dialog.manager.current_part
        if self.dialog.scope.currentData() == 'selected':
            shape = list(shape.solids())[self.dialog.selected_body]
        bounds = shape.bounding_box()
        center = np.array([(bounds.min.X + bounds.max.X) / 2,
                           (bounds.min.Y + bounds.max.Y) / 2,
                           (bounds.min.Z + bounds.max.Z) / 2])
        translation = np.array([self.dialog.controls['d' + axis].value() for axis in 'XYZ'])
        center += translation
        height = self.plotter.ren_win.GetSize()[1]
        length = max(world_per_pixel(self.plotter.camera, height) * 80, 1e-6)
        colors = {'X': '#e54138', 'Y': '#40b95a', 'Z': '#3878e5'}
        for i, axis in enumerate('XYZ'):
            direction = np.eye(3)[i]
            point = center + direction * length
            self.representations[axis].SetWorldPosition(tuple(point))
            name = 'move_axis_' + axis
            add_outline_arrow(self.plotter, name, center, point, direction, length * .2, colors[axis])
        self.plotter.render()

    def start_drag(self, axis):
        point = np.asarray(self.representations[axis].GetWorldPosition())
        direction = np.eye(3)['XYZ'.index(axis)]
        self.drag_start = (axis, tuple(self.plotter.iren.get_event_position()),
                           self.dialog.controls['d' + axis].value(), axis_drag_projection(self.plotter, point, direction))

    def drag(self, axis):
        if self.drag_start is None or self.drag_start[0] != axis:
            return
        _, start, initial, (a,b,length) = self.drag_start
        value = initial + projected_axis_delta(start, self.plotter.iren.get_event_position(), a, b, length)
        self.updating = True
        try:
            self.dialog.controls['d' + axis].setValue(value)
        finally:
            self.updating = False
        schedule_handle_sync(self)

    def end_drag(self):
        self.drag_start = None
        stop_handle_sync(self)
        self.sync()

    def close(self):
        stop_handle_sync(self)
        self._camera_timer.stop()
        self.plotter.camera.RemoveObserver(self._camera_observer)
        for axis in 'XYZ':
            self.widgets[axis].EnabledOff()
            self.widgets[axis].RemoveAllObservers()
            remove_outline_arrow(self.plotter, 'move_axis_' + axis)
        self.plotter.render()


class ParameterHandles:
    """Rotation rings or a uniform-scale handle linked to dialog fields."""
    def __init__(self, dialog, mode):
        self.dialog, self.plotter, self.mode = dialog, dialog.manager.plotter, mode
        self.keys = 'XYZ' if mode == 'rotate' else 'S'
        self.widgets, self.representations = {}, {}
        self.drag_start, self.updating = None, False
        self.colors = {'X': (0.88, .25, .22), 'Y': (.25, .72, .35),
                       'Z': (.22, .47, .9), 'S': (1., .65, .1)}
        for key in self.keys:
            rep = vtk.vtkPointHandleRepresentation3D()
            rep.SetHandleSize(12)
            rep.GetProperty().SetColor(*self.colors[key])
            rep.GetSelectedProperty().SetColor(1., .8, .25)
            widget = vtk.vtkHandleWidget()
            widget.SetInteractor(self.plotter.iren.interactor)
            widget.SetRepresentation(rep)
            widget.AddObserver('StartInteractionEvent', lambda obj, event, k=key: self.start_drag(k))
            widget.AddObserver('InteractionEvent', lambda obj, event, k=key: self.drag(k))
            widget.AddObserver('EndInteractionEvent', lambda obj, event: self.end_drag())
            self.widgets[key], self.representations[key] = widget, rep
            widget.EnabledOn()
        self.sync()

    def center_and_size(self):
        shape = self.dialog.manager.current_part
        if self.dialog.scope.currentData() == 'selected':
            shape = list(shape.solids())[self.dialog.selected_body]
        bounds = shape.bounding_box()
        center = np.array([(bounds.min.X + bounds.max.X) / 2, (bounds.min.Y + bounds.max.Y) / 2,
                           (bounds.min.Z + bounds.max.Z) / 2])
        if self.mode == 'rotate':
            center = np.array([self.dialog.controls['p' + axis].value() for axis in 'XYZ'])
            center += np.array([self.dialog.controls['d' + axis].value() for axis in 'XYZ'])
        else:
            center = np.array([self.dialog.controls['p' + axis].value() for axis in 'XYZ'])
        height = self.plotter.ren_win.GetSize()[1]
        length = max(world_per_pixel(self.plotter.camera, height) * 80, 1e-6)
        return center, length

    def sync(self):
        if self.updating:
            return
        center, length = self.center_and_size()
        for key in self.keys:
            name = 'parameter_handle_' + self.mode + key
            self.plotter.remove_actor(name, render=False)
            if self.mode == 'rotate':
                axis = np.eye(3)['XYZ'.index(key)]
                reference = np.eye(3)[('XYZ'.index(key) + 1) % 3]
                angle = np.radians(self.dialog.controls['r' + key].value())
                tangent = np.cross(axis, reference)
                handle_point = center + length * (np.cos(angle) * reference + np.sin(angle) * tangent)
                theta = np.linspace(0, 2 * np.pi, 97)
                mesh = pv.lines_from_points(np.asarray([center + length * (np.cos(t) * reference + np.sin(t) * tangent) for t in theta]))
                self.plotter.add_mesh(mesh, name=name, color=self.colors[key], line_width=3,
                                      pickable=False, reset_camera=False, render=False)
            else:
                factor = self.dialog.controls['factor'].value()
                handle_point = center + np.array([length * max(factor, .05), 0, 0])
                mesh = pv.Line(center, handle_point)
                self.plotter.add_mesh(mesh, name=name, color=self.colors[key], line_width=3,
                                      pickable=False, reset_camera=False, render=False)
            self.representations[key].SetWorldPosition(tuple(handle_point))
        self.plotter.render()

    def cursor_ring_angle(self, key):
        center, length = self.center_and_size()
        axis_index = 'XYZ'.index(key)
        axis = np.eye(3)[axis_index]
        first = np.eye(3)[(axis_index + 1) % 3]
        second = np.cross(axis, first)
        renderer = self.plotter.renderer
        def project(point):
            renderer.SetWorldPoint(float(point[0]), float(point[1]), float(point[2]), 1.0)
            renderer.WorldToDisplay()
            return renderer.GetDisplayPoint()
        return projected_ring_angle(self.plotter.iren.get_event_position(), project(center),
                                    project(center + length * first), project(center + length * second))

    def start_drag(self, key):
        field = self.dialog.controls['r' + key] if self.mode == 'rotate' else self.dialog.controls['factor']
        ring_angle = self.cursor_ring_angle(key) if self.mode == 'rotate' else None
        center, length = self.center_and_size()
        projection = axis_drag_projection(self.plotter, center + np.array([length*max(field.value(),.05),0,0]), np.array([1.,0,0])) if self.mode == 'scale' else None
        self.drag_start = (key, tuple(self.plotter.iren.get_event_position()), field.value(), ring_angle, projection, length)

    def drag(self, key):
        if self.drag_start is None or self.drag_start[0] != key:
            return
        _, start, initial, start_angle, projection, length = self.drag_start
        dy = self.plotter.iren.get_event_position()[1] - start[1]
        field = self.dialog.controls['r' + key] if self.mode == 'rotate' else self.dialog.controls['factor']
        current_angle = self.cursor_ring_angle(key) if self.mode == 'rotate' else None
        if start_angle is not None and current_angle is not None:
            delta = (current_angle - start_angle + 180) % 360 - 180
            value = initial + delta
        elif self.mode == 'rotate':
            value = initial + dy * .5
        else:
            a, b, projected_length = projection
            value = initial + projected_axis_delta(start, self.plotter.iren.get_event_position(), a, b, projected_length) / length
        self.updating = True
        try:
            field.setValue(value)
        finally:
            self.updating = False
        schedule_handle_sync(self)

    def end_drag(self):
        self.drag_start = None
        stop_handle_sync(self)
        self.sync()

    def close(self):
        stop_handle_sync(self)
        for key in self.keys:
            self.widgets[key].EnabledOff()
            self.widgets[key].RemoveAllObservers()
            self.plotter.remove_actor('parameter_handle_' + self.mode + key, render=False)
        self.plotter.render()


class CompositeHandles:
    def __init__(self, *handles): self.handles = handles
    def sync(self):
        for handle in self.handles: handle.sync()
    def close(self):
        for handle in self.handles: handle.close()


class OperationHandle:
    """One model-space arrow or angle ring bound to a numeric dialog field."""
    def __init__(self, dialog, field, axis='Z', angle=False, index=0):
        self.dialog, self.plotter, self.field = dialog, dialog.manager.plotter, field
        self.axis, self.angle, self.index = axis, angle, index
        self.drag_start, self.updating = None, False
        self.name = f'operation_handle_{id(self)}'
        self.rep = vtk.vtkPointHandleRepresentation3D()
        self.rep.SetHandleSize(12)
        self.rep.GetProperty().SetColor(.22, .57, .82)
        self.rep.GetSelectedProperty().SetColor(.65, .85, 1.)
        if not angle:
            self.rep.GetProperty().SetOpacity(.01)
            self.rep.GetSelectedProperty().SetOpacity(.01)
        self.widget = vtk.vtkHandleWidget()
        self.widget.SetInteractor(self.plotter.iren.interactor)
        self.widget.SetRepresentation(self.rep)
        self.widget.AddObserver('StartInteractionEvent', self.start_drag)
        self.widget.AddObserver('InteractionEvent', self.drag)
        self.widget.AddObserver('EndInteractionEvent', self.end_drag)
        try:
            self.widget.EnabledOn()
            self.sync()
            if not self.angle:
                observe_arrow_camera(self)
        except Exception:
            self.widget.EnabledOff()
            self.widget.RemoveAllObservers()
            remove_outline_arrow(self.plotter, self.name)
            raise

    def geometry(self):
        manager, dialog = self.dialog.manager, self.dialog
        if isinstance(self.axis, str) and self.axis == 'revolve':
            direction = manager.u_axis if dialog.axis.currentData() == 'horizontal' else manager.v_axis
            center = np.asarray(manager.active_center, dtype=float)
        else:
            if self.axis == 'pattern':
                axis = dialog.controls['axis'].currentText()
            elif self.axis == 'mirror':
                axis = {'XY': 'Z', 'XZ': 'Y', 'YZ': 'X'}[dialog.controls['plane'].currentText()]
            else:
                axis = self.axis if self.axis != 'radial' else 'X'
            direction = np.eye(3)['XYZ'.index(axis)]
            if hasattr(dialog, 'position'):
                center = np.array([control.value() for control in dialog.position])
            elif getattr(dialog, 'kind', None) == 'circular':
                center = np.array([dialog.controls['p' + key].value() for key in 'XYZ'])
            elif manager.current_part is not None:
                shape = manager.current_part
                if getattr(dialog, 'scope', None) is not None and dialog.scope.currentData() == 'selected':
                    shape = list(shape.solids())[dialog.selected_body]
                if getattr(dialog, 'kind', None) in ('hole', 'shell') and dialog.face_index >= 0:
                    face = list(manager.current_part.faces())[dialog.face_index]
                    center = np.asarray(tuple(face.center()), dtype=float)
                    normal = np.asarray(tuple(face.normal_at()), dtype=float)
                    if self.axis == 'radial':
                        reference = np.array([0., 0., 1.]) if abs(normal[2]) < .9 else np.array([0., 1., 0.])
                        direction = np.cross(normal, reference)
                    else:
                        direction = -normal if dialog.kind == 'hole' else normal
                else:
                    box = shape.bounding_box()
                    center = np.array([(box.min.X + box.max.X) / 2, (box.min.Y + box.max.Y) / 2,
                                       (box.min.Z + box.max.Z) / 2])
            else:
                center = np.zeros(3)
        direction = np.asarray(direction, dtype=float)
        direction /= np.linalg.norm(direction)
        height = self.plotter.ren_win.GetSize()[1]
        length = max(world_per_pixel(self.plotter.camera, height) * (65 + 18 * self.index), 1e-6)
        return center, direction, length

    def ring_basis(self):
        center, axis, length = self.geometry()
        ref = np.array([0., 0., 1.]) if abs(axis[2]) < .9 else np.array([0., 1., 0.])
        first = np.cross(axis, ref)
        first /= np.linalg.norm(first)
        return center, axis, length, first, np.cross(axis, first)

    def sync(self):
        if self.updating:
            return
        enabled = self.field.isEnabled()
        self.widget.SetEnabled(enabled)
        if not enabled:
            remove_outline_arrow(self.plotter, self.name)
            self.plotter.render()
            return
        if self.angle:
            self.plotter.remove_actor(self.name, render=False)
            center, axis, length, first, second = self.ring_basis()
            theta = np.linspace(0, 2 * np.pi, 97)
            mesh = pv.lines_from_points(np.asarray([center + length * (np.cos(t) * first + np.sin(t) * second) for t in theta]))
            value = np.radians(self.field.value())
            point = center + length * (np.cos(value) * first + np.sin(value) * second)
            self.plotter.add_mesh(mesh, name=self.name, color='#f2a13b', line_width=3,
                                  pickable=False, reset_camera=False, render=False)
        else:
            center, direction, length = self.geometry()
            point = center + direction * (length + self.field.value())
            add_outline_arrow(self.plotter, self.name, center, point, direction, length * .22)
        self.rep.SetWorldPosition(tuple(point))
        self.plotter.render()

    def cursor_angle(self):
        center, _, length, first, second = self.ring_basis()
        renderer = self.plotter.renderer
        def project(point):
            renderer.SetWorldPoint(float(point[0]), float(point[1]), float(point[2]), 1.)
            renderer.WorldToDisplay()
            return renderer.GetDisplayPoint()
        return projected_ring_angle(self.plotter.iren.get_event_position(), project(center),
                                    project(center + length * first), project(center + length * second))

    def start_drag(self, *args):
        center, direction, length = self.geometry()
        point = center + direction * (length + self.field.value())
        self.drag_start = (tuple(self.plotter.iren.get_event_position()), self.field.value(),
                           self.cursor_angle() if self.angle else None,
                           None if self.angle else axis_drag_projection(self.plotter, point, direction))

    def drag(self, *args):
        if self.drag_start is None:
            return
        start, initial, start_angle, projection = self.drag_start
        dy = self.plotter.iren.get_event_position()[1] - start[1]
        current = self.cursor_angle() if self.angle else None
        if start_angle is not None and current is not None:
            value = initial + ((current - start_angle + 180) % 360 - 180)
        elif self.angle:
            value = initial + dy * .5
        else:
            a, b, length = projection
            value = initial + projected_axis_delta(start, self.plotter.iren.get_event_position(), a, b, length)
        self.updating = True
        try:
            self.field.setValue(value)
        finally:
            self.updating = False
        schedule_handle_sync(self)

    def end_drag(self, *args):
        self.drag_start = None
        stop_handle_sync(self)
        self.sync()

    def close(self):
        stop_handle_sync(self)
        if hasattr(self, '_camera_observer'):
            self._camera_timer.stop()
            self.plotter.camera.RemoveObserver(self._camera_observer)
        self.widget.EnabledOff()
        self.widget.RemoveAllObservers()
        remove_outline_arrow(self.plotter, self.name)
        self.plotter.render()


def create_operation_handles(dialog, specs):
    handles = []
    try:
        for index, (field, axis, angle) in enumerate(specs):
            handles.append(OperationHandle(dialog, field, axis, angle, index))
        return CompositeHandles(*handles)
    except Exception:
        for handle in handles:
            handle.close()
        raise
