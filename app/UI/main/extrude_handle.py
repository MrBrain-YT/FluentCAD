"""Interactive 3-D handle for the distance of a one-sided extrusion."""
import math
import numpy as np
import pyvista as pv
import vtk


def world_per_pixel(camera, height):
    height = max(int(height), 1)
    if camera.GetParallelProjection():
        return 2.0 * camera.GetParallelScale() / height
    distance = np.linalg.norm(np.asarray(camera.position) - np.asarray(camera.focal_point))
    return 2.0 * distance * math.tan(math.radians(camera.GetViewAngle()) / 2) / height


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
        self.representation = vtk.vtkPointHandleRepresentation3D()
        self.representation.SetHandleSize(14)
        self.representation.GetProperty().SetColor(1.0, 0.58, 0.1)
        self.representation.GetSelectedProperty().SetColor(1.0, 0.8, 0.25)
        self.widget = vtk.vtkHandleWidget()
        self.widget.SetInteractor(self.plotter.iren.interactor)
        self.widget.SetRepresentation(self.representation)
        self.widget.AddObserver('StartInteractionEvent', self.start_drag)
        self.widget.AddObserver('InteractionEvent', self.drag)
        self.widget.AddObserver('EndInteractionEvent', self.end_drag)
        self.widget.EnabledOn()
        self.sync()

    def sync(self):
        if self.updating:
            return
        enabled = self.dialog.extent.currentData() == 'distance'
        self.widget.SetEnabled(enabled)
        self.plotter.remove_actor('extrude_arrow', render=False)
        if not enabled:
            return
        offset = self.dialog.offset.value()
        distance = self.dialog.distance.value()
        point = self.anchor + self.normal * (offset + distance)
        self.representation.SetWorldPosition(tuple(point))
        length = max(self.plotter.camera.parallel_scale * 0.12, 1e-6)
        arrow = pv.Arrow(start=point - self.normal * length, direction=self.normal, scale=length)
        self.plotter.add_mesh(arrow, name='extrude_arrow', color='#f2a13b',
                              pickable=False, reset_camera=False, render=False)
        self.plotter.render()

    def start_drag(self, *args):
        self.drag_start = (self.plotter.iren.get_event_position()[1], self.dialog.distance.value())

    def drag(self, *args):
        if self.drag_start is None:
            return
        y, initial = self.drag_start
        current_y = self.plotter.iren.get_event_position()[1]
        height = self.plotter.ren_win.GetSize()[1]
        value = initial + (current_y - y) * world_per_pixel(self.plotter.camera, height)
        self.updating = True
        try:
            self.dialog.distance.setValue(value)
        finally:
            self.updating = False
        self.sync()

    def end_drag(self, *args):
        self.drag_start = None
        self.sync()

    def close(self):
        self.widget.EnabledOff()
        self.widget.RemoveAllObservers()
        self.plotter.remove_actor('extrude_arrow', render=False)
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
            widget = vtk.vtkHandleWidget()
            widget.SetInteractor(self.plotter.iren.interactor)
            widget.SetRepresentation(representation)
            widget.AddObserver('StartInteractionEvent', lambda obj, event, key=axis: self.start_drag(key))
            widget.AddObserver('InteractionEvent', lambda obj, event, key=axis: self.drag(key))
            widget.AddObserver('EndInteractionEvent', lambda obj, event, key=axis: self.end_drag())
            self.widgets[axis], self.representations[axis] = widget, representation
            widget.EnabledOn()
        self.sync()

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
            self.plotter.remove_actor('move_axis_' + axis, render=False)
            arrow = pv.Arrow(start=center, direction=direction, scale=length)
            self.plotter.add_mesh(arrow, name='move_axis_' + axis, color=colors[axis],
                                  pickable=False, reset_camera=False, render=False)
        self.plotter.render()

    def start_drag(self, axis):
        self.drag_start = (axis, self.plotter.iren.get_event_position()[1],
                           self.dialog.controls['d' + axis].value())

    def drag(self, axis):
        if self.drag_start is None or self.drag_start[0] != axis:
            return
        _, start_y, initial = self.drag_start
        y = self.plotter.iren.get_event_position()[1]
        height = self.plotter.ren_win.GetSize()[1]
        value = initial + (y - start_y) * world_per_pixel(self.plotter.camera, height)
        self.updating = True
        try:
            self.dialog.controls['d' + axis].setValue(value)
        finally:
            self.updating = False
        self.sync()

    def end_drag(self):
        self.drag_start = None
        self.sync()

    def close(self):
        for axis in 'XYZ':
            self.widgets[axis].EnabledOff()
            self.widgets[axis].RemoveAllObservers()
            self.plotter.remove_actor('move_axis_' + axis, render=False)
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
        self.drag_start = (key, self.plotter.iren.get_event_position()[1], field.value(), ring_angle)

    def drag(self, key):
        if self.drag_start is None or self.drag_start[0] != key:
            return
        _, start_y, initial, start_angle = self.drag_start
        dy = self.plotter.iren.get_event_position()[1] - start_y
        field = self.dialog.controls['r' + key] if self.mode == 'rotate' else self.dialog.controls['factor']
        current_angle = self.cursor_ring_angle(key) if self.mode == 'rotate' else None
        if start_angle is not None and current_angle is not None:
            delta = (current_angle - start_angle + 180) % 360 - 180
            value = initial + delta
        else:
            value = initial + (dy * .5 if self.mode == 'rotate' else dy / 80.0)
        self.updating = True
        try:
            field.setValue(value)
        finally:
            self.updating = False
        self.sync()

    def end_drag(self):
        self.drag_start = None
        self.sync()

    def close(self):
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
        self.rep.GetProperty().SetColor(1., .58, .1)
        self.rep.GetSelectedProperty().SetColor(1., .82, .25)
        self.widget = vtk.vtkHandleWidget()
        self.widget.SetInteractor(self.plotter.iren.interactor)
        self.widget.SetRepresentation(self.rep)
        self.widget.AddObserver('StartInteractionEvent', self.start_drag)
        self.widget.AddObserver('InteractionEvent', self.drag)
        self.widget.AddObserver('EndInteractionEvent', self.end_drag)
        try:
            self.widget.EnabledOn()
            self.sync()
        except Exception:
            self.widget.EnabledOff()
            self.widget.RemoveAllObservers()
            self.plotter.remove_actor(self.name, render=False)
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
        self.plotter.remove_actor(self.name, render=False)
        if not enabled:
            self.plotter.render()
            return
        if self.angle:
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
            mesh = pv.Arrow(start=point - direction * length, direction=direction, scale=length)
            self.plotter.add_mesh(mesh, name=self.name, color='#f2a13b',
                                  pickable=False, reset_camera=False, render=False)
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
        self.drag_start = (self.plotter.iren.get_event_position()[1], self.field.value(),
                           self.cursor_angle() if self.angle else None)

    def drag(self, *args):
        if self.drag_start is None:
            return
        y, initial, start_angle = self.drag_start
        dy = self.plotter.iren.get_event_position()[1] - y
        current = self.cursor_angle() if self.angle else None
        if start_angle is not None and current is not None:
            value = initial + ((current - start_angle + 180) % 360 - 180)
        elif self.angle:
            value = initial + dy * .5
        else:
            value = initial + dy * world_per_pixel(self.plotter.camera, self.plotter.ren_win.GetSize()[1])
        self.updating = True
        try:
            self.field.setValue(value)
        finally:
            self.updating = False
        self.sync()

    def end_drag(self, *args):
        self.drag_start = None
        self.sync()

    def close(self):
        self.widget.EnabledOff()
        self.widget.RemoveAllObservers()
        self.plotter.remove_actor(self.name, render=False)
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
