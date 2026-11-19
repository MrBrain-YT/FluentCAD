"""Interactive 3-D handle for the distance of a one-sided extrusion."""
import math
import numpy as np
import pyvista as pv
import vtk
from PySide6 import QtCore


def world_per_pixel(camera, height, point=None):
    height = max(int(height), 1)
    if camera.GetParallelProjection():
        return 2.0 * camera.GetParallelScale() / height
    position = np.asarray(camera.position, dtype=float)
    forward = np.asarray(camera.focal_point, dtype=float) - position
    distance = np.linalg.norm(forward)
    if point is not None and distance > 1e-12:
        distance = max(float(np.dot(np.asarray(point, dtype=float) - position, forward / distance)), 1e-8)
    return 2.0 * distance * math.tan(math.radians(camera.GetViewAngle()) / 2) / height


def projected_axis_delta(cursor_start, cursor_now, screen_start, screen_end, world_length):
    axis = np.asarray(screen_end, dtype=float)[:2] - np.asarray(screen_start, dtype=float)[:2]
    length2 = float(np.dot(axis, axis))
    if length2 < 9.: return 0.0
    # A nearly head-on axis has a very short projection. Limit sensitivity so
    # tiny mouse movements cannot send the model distance to extreme values.
    return float(np.dot(np.asarray(cursor_now, dtype=float)[:2] - np.asarray(cursor_start, dtype=float)[:2], axis) / max(length2, 400.) * world_length)


def axis_drag_projection(plotter, anchor, direction):
    renderer = plotter.renderer
    length = max(world_per_pixel(plotter.camera, plotter.ren_win.GetSize()[1], anchor) * 80, 1e-8)
    def project(point):
        renderer.SetWorldPoint(*map(float, point), 1.)
        renderer.WorldToDisplay()
        return tuple(renderer.GetDisplayPoint())
    return project(anchor), project(anchor + np.asarray(direction)*length), length


def cursor_position(handle):
    position = getattr(handle, '_pointer_position', None)
    return position if position is not None else handle.plotter.iren.get_event_position()


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
    overlay_actors = []
    for mesh, suffix, shade in zip(meshes, ('_guide', '', '_inset'), (color, 'white', color)):
        actor = plotter.add_mesh(mesh, name=name + suffix, color=shade, line_width=2,
                                lighting=False, pickable=False, reset_camera=False, render=False)
        overlay = getattr(plotter, '_fluent_overlay_renderer', None)
        if overlay is not None:
            plotter.renderer.RemoveActor(actor)
            overlay.AddActor(actor)
            overlay_actors.append(actor)
    if overlay_actors:
        actors = getattr(plotter, '_fluent_arrow_actors', None)
        if not isinstance(actors, dict):
            actors = plotter._fluent_arrow_actors = {}
        actors[name] = overlay_actors
    cache[name] = meshes


def remove_outline_arrow(plotter, name):
    cache = getattr(plotter, '_fluent_arrow_meshes', None)
    if isinstance(cache, dict):
        cache.pop(name, None)
    actors = getattr(plotter, '_fluent_arrow_actors', None)
    overlay = getattr(plotter, '_fluent_overlay_renderer', None)
    if isinstance(actors, dict) and overlay is not None:
        for actor in actors.pop(name, []):
            overlay.RemoveActor(actor)
    for suffix in ('', '_guide', '_inset'):
        plotter.remove_actor(name + suffix, render=False)


def update_handle_line(plotter, name, points, color):
    cache = getattr(plotter, '_fluent_handle_lines', None)
    if not isinstance(cache, dict):
        cache = plotter._fluent_handle_lines = {}
    points = np.asarray(points, dtype=float)
    if name in cache:
        cache[name].points = points
    else:
        mesh = pv.lines_from_points(points)
        plotter.add_mesh(mesh, name=name, color=color, line_width=3,
                         pickable=False, reset_camera=False, render=False)
        cache[name] = mesh


def remove_handle_line(plotter, name):
    cache = getattr(plotter, '_fluent_handle_lines', None)
    if isinstance(cache, dict):
        cache.pop(name, None)
    plotter.remove_actor(name, render=False)


def update_ring_knob(plotter, name, point):
    point = np.asarray(point, dtype=float)
    view = np.asarray(plotter.camera.position, dtype=float) - point
    view /= max(np.linalg.norm(view), 1e-12)
    reference = np.array([0., 1., 0.]) if abs(view[1]) < .9 else np.array([1., 0., 0.])
    first = np.cross(view, reference)
    first /= max(np.linalg.norm(first), 1e-12)
    second = np.cross(view, first)
    radius = max(world_per_pixel(plotter.camera, plotter.ren_win.GetSize()[1], point) * 5., 1e-6)
    theta = np.linspace(0., 2 * np.pi, 25)
    points = point + radius * (np.cos(theta)[:, None] * first + np.sin(theta)[:, None] * second)
    update_handle_line(plotter, name, points, '#a5adb2')


def accumulate_ring_angle(handle, initial_angle, current_angle):
    previous = getattr(handle, '_ring_previous', None)
    if previous is None:
        previous = initial_angle
        handle._ring_total = 0.
    handle._ring_total += (current_angle - previous + 180.) % 360. - 180.
    handle._ring_previous = current_angle
    return handle._ring_total


def ring_drag_delta(handle, start_angle, current_angle, start_cursor, cursor):
    """Keep a continuous angle when a projected ring becomes edge-on."""
    if not hasattr(handle, '_ring_last_cursor') or handle._ring_last_cursor is None:
        handle._ring_last_cursor = start_cursor
        handle._ring_previous = start_angle
        handle._ring_total = 0.
    if current_angle is None:
        handle._ring_total += (cursor[1] - handle._ring_last_cursor[1]) * .5
        handle._ring_previous = None
    elif handle._ring_previous is None:
        handle._ring_previous = current_angle
    else:
        accumulate_ring_angle(handle, start_angle, current_angle)
    handle._ring_last_cursor = cursor
    return handle._ring_total


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


def finish_handle_drag(handle):
    stop_handle_sync(handle)
    flush = getattr(getattr(handle, 'dialog', None), 'flush_preview', None)
    if not callable(flush) or not flush():
        handle.sync()


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
        self.normal = np.array(dialog.manager.active_normal, dtype=float)
        self.normal /= np.linalg.norm(self.normal)
        self.drag_start = None
        self.updating = False
        self.value_editors = {}
        if hasattr(self.plotter, 'devicePixelRatioF'):
            from UI.main.scene_value_editor import SceneValueEditor
            self.value_editors = {'primary': SceneValueEditor(self.plotter, dialog.distance),
                                  'opposite': SceneValueEditor(self.plotter, dialog.second_distance)}
        self.widgets, self.representations = {}, {}
        self.hit_filter = None
        if hasattr(self.plotter, 'installEventFilter'):
            from UI.main.arrow_hit_filter import ArrowHitFilter
            self.hit_filter = ArrowHitFilter(self)
        for key in ('primary', 'opposite'):
            rep = vtk.vtkPointHandleRepresentation3D()
            rep.SetHandleSize(11)
            rep.GetProperty().SetColor(.22, .57, .82)
            rep.GetSelectedProperty().SetColor(.65, .85, 1.)
            rep.GetProperty().SetOpacity(0.)
            rep.GetSelectedProperty().SetOpacity(0.)
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

    def sync(self, render=True):
        if self.updating:
            return
        enabled = self.dialog.extent.currentData() == 'distance' and bool(getattr(self.dialog, 'indices', (0,)))
        two = self.dialog.direction.currentData() in ('two', 'symmetric')
        self.widgets['primary'].SetEnabled(enabled)
        self.widgets['opposite'].SetEnabled(enabled and two)
        if not enabled:
            if getattr(self, 'hit_filter', None) is not None: self.hit_filter.segments.clear()
            for key in self.widgets: remove_outline_arrow(self.plotter, 'extrude_arrow_' + key)
            for editor in getattr(self, 'value_editors', {}).values(): editor.hide()
            return
        editors = getattr(self, 'value_editors', {})
        if 'opposite' in editors:
            editors['opposite'].setVisible(two and self.dialog.direction.currentData() == 'two')
        if not two and 'extrude_arrow_opposite' in getattr(self.plotter, '_fluent_arrow_meshes', {}):
            remove_outline_arrow(self.plotter, 'extrude_arrow_opposite')
        if getattr(self, 'hit_filter', None) is not None:
            self.hit_filter.segments.clear()
        offset = self.dialog.offset.value()
        distance = self.dialog.distance.value()
        for key in ('primary', 'opposite'):
            if key == 'opposite' and not two: continue
            value = distance if key == 'primary' or self.dialog.direction.currentData() == 'symmetric' else self.dialog.second_distance.value()
            direction = self.normal if key == 'primary' else -self.normal
            visual_direction = direction if value >= 0 else -direction
            endpoint = self.anchor + self.normal * offset + direction * value
            length = max(world_per_pixel(self.plotter.camera, self.plotter.ren_win.GetSize()[1], endpoint) * 30, 1e-6)
            # Keep the arrowhead beyond the end face, rather than inside the solid.
            point = endpoint + visual_direction * length
            self.representations[key].SetWorldPosition(tuple(point))
            add_outline_arrow(self.plotter, 'extrude_arrow_' + key, self.anchor + self.normal * offset,
                              point, visual_direction, length)
            if getattr(self, 'hit_filter', None) is not None:
                self.hit_filter.segments[key] = (self.anchor + self.normal * offset, point)
            if key in editors and (key == 'primary' or self.dialog.direction.currentData() == 'two'):
                editors[key].set_point(point)
        if render:
            self.plotter.render()

    def start_drag(self, key='primary'):
        field = self.dialog.second_distance if key == 'opposite' and self.dialog.direction.currentData() == 'two' else self.dialog.distance
        direction = self.normal if key == 'primary' else -self.normal
        point = np.asarray(self.representations[key].GetWorldPosition())
        cursor = getattr(self, '_pointer_position', None)
        self.drag_start = (key, tuple(cursor if cursor is not None else cursor_position(self)), field.value(), axis_drag_projection(self.plotter, point, direction))

    def drag(self, key='primary'):
        if self.drag_start is None or self.drag_start[0] != key:
            return
        _, cursor, initial, (a, b, length) = self.drag_start
        position = getattr(self, '_pointer_position', None)
        value = initial + projected_axis_delta(cursor, position if position is not None else cursor_position(self), a, b, length)
        field = self.dialog.second_distance if key == 'opposite' and self.dialog.direction.currentData() == 'two' else self.dialog.distance
        self.updating = True
        try:
            field.setValue(value)
        finally:
            self.updating = False
        schedule_handle_sync(self)

    def end_drag(self, *args):
        self.drag_start = None
        finish_handle_drag(self)

    def close(self):
        stop_handle_sync(self)
        if self.hit_filter is not None: self.hit_filter.dispose()
        self._camera_timer.stop()
        self.plotter.camera.RemoveObserver(self._camera_observer)
        for editor in self.value_editors.values(): editor.dispose()
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
        self.hit_filter = None
        if hasattr(self.plotter, 'installEventFilter'):
            from UI.main.arrow_hit_filter import ArrowHitFilter
            self.hit_filter = ArrowHitFilter(self)
        self.drag_start, self.updating = None, False
        colors = {'X': (0.88, 0.25, 0.22), 'Y': (0.25, 0.72, 0.35), 'Z': (0.22, 0.47, 0.9)}
        for axis in 'XYZ':
            representation = vtk.vtkPointHandleRepresentation3D()
            representation.SetHandleSize(13)
            representation.GetProperty().SetColor(*colors[axis])
            representation.GetSelectedProperty().SetColor(1., 0.8, 0.25)
            representation.GetProperty().SetOpacity(0.)
            representation.GetSelectedProperty().SetOpacity(0.)
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

    def sync(self, render=True):
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
        length = max(world_per_pixel(self.plotter.camera, height, center) * 80, 1e-6)
        colors = {'X': '#e54138', 'Y': '#40b95a', 'Z': '#3878e5'}
        for i, axis in enumerate('XYZ'):
            direction = np.eye(3)[i]
            point = center + direction * length
            self.representations[axis].SetWorldPosition(tuple(point))
            name = 'move_axis_' + axis
            add_outline_arrow(self.plotter, name, center, point, direction, length * .2, colors[axis])
            if self.hit_filter is not None:
                self.hit_filter.segments[axis] = (center, point)
        if render:
            self.plotter.render()

    def start_drag(self, axis):
        point = np.asarray(self.representations[axis].GetWorldPosition())
        direction = np.eye(3)['XYZ'.index(axis)]
        self.drag_start = (axis, tuple(cursor_position(self)),
                           self.dialog.controls['d' + axis].value(), axis_drag_projection(self.plotter, point, direction))

    def drag(self, axis):
        if self.drag_start is None or self.drag_start[0] != axis:
            return
        _, start, initial, (a,b,length) = self.drag_start
        value = initial + projected_axis_delta(start, cursor_position(self), a, b, length)
        self.updating = True
        try:
            self.dialog.controls['d' + axis].setValue(value)
        finally:
            self.updating = False
        schedule_handle_sync(self)

    def end_drag(self):
        self.drag_start = None
        finish_handle_drag(self)

    def close(self):
        stop_handle_sync(self)
        if self.hit_filter is not None: self.hit_filter.dispose()
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
        self.colors = {'X': (.58, .62, .65), 'Y': (.58, .62, .65),
                       'Z': (.58, .62, .65), 'S': (.22, .56, .8)}
        for key in self.keys:
            rep = vtk.vtkPointHandleRepresentation3D()
            rep.SetHandleSize(12)
            rep.GetProperty().SetColor(*self.colors[key])
            rep.GetSelectedProperty().SetColor(.22, .56, .8)
            if self.mode == 'rotate':
                rep.GetProperty().SetOpacity(.01)
                rep.GetSelectedProperty().SetOpacity(.01)
            widget = vtk.vtkHandleWidget()
            widget.SetInteractor(self.plotter.iren.interactor)
            widget.SetRepresentation(rep)
            widget.AddObserver('StartInteractionEvent', lambda obj, event, k=key: self.start_drag(k))
            widget.AddObserver('InteractionEvent', lambda obj, event, k=key: self.drag(k))
            widget.AddObserver('EndInteractionEvent', lambda obj, event: self.end_drag())
            self.widgets[key], self.representations[key] = widget, rep
            widget.EnabledOn()
        self.sync()
        observe_arrow_camera(self)

    def center_and_size(self):
        if self.mode == 'rotate':
            center = np.array([self.dialog.controls['p' + axis].value() for axis in 'XYZ'])
            center += np.array([self.dialog.controls['d' + axis].value() for axis in 'XYZ'])
        else:
            center = np.array([self.dialog.controls['p' + axis].value() for axis in 'XYZ'])
        height = self.plotter.ren_win.GetSize()[1]
        length = max(world_per_pixel(self.plotter.camera, height, center) * 80, 1e-6)
        return center, length

    def sync(self, render=True):
        if self.updating:
            return
        center, length = self.center_and_size()
        for key in self.keys:
            name = 'parameter_handle_' + self.mode + key
            if self.mode == 'rotate':
                axis = np.eye(3)['XYZ'.index(key)]
                reference = np.eye(3)[('XYZ'.index(key) + 1) % 3]
                angle = np.radians(self.dialog.controls['r' + key].value())
                tangent = np.cross(axis, reference)
                handle_point = center + length * (np.cos(angle) * reference + np.sin(angle) * tangent)
                theta = np.linspace(0, 2 * np.pi, 97)
                points = center + length * (np.cos(theta)[:, None] * reference + np.sin(theta)[:, None] * tangent)
            else:
                factor = self.dialog.controls['factor'].value()
                handle_point = center + np.array([length * max(factor, .05), 0, 0])
                points = (center, handle_point)
            update_handle_line(self.plotter, name, points, self.colors[key])
            if self.mode == 'rotate':
                update_ring_knob(self.plotter, name + '_knob', handle_point)
            self.representations[key].SetWorldPosition(tuple(handle_point))
        if render:
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
        return projected_ring_angle(cursor_position(self), project(center),
                                    project(center + length * first), project(center + length * second))

    def start_drag(self, key):
        field = self.dialog.controls['r' + key] if self.mode == 'rotate' else self.dialog.controls['factor']
        ring_angle = self.cursor_ring_angle(key) if self.mode == 'rotate' else None
        self._ring_previous, self._ring_total = ring_angle, 0.
        self._ring_last_cursor = tuple(cursor_position(self))
        center, length = self.center_and_size()
        projection = axis_drag_projection(self.plotter, center + np.array([length*max(field.value(),.05),0,0]), np.array([1.,0,0])) if self.mode == 'scale' else None
        self.drag_start = (key, tuple(cursor_position(self)), field.value(), ring_angle, projection, length)

    def drag(self, key):
        if self.drag_start is None or self.drag_start[0] != key:
            return
        _, start, initial, start_angle, projection, length = self.drag_start
        cursor = tuple(cursor_position(self))
        field = self.dialog.controls['r' + key] if self.mode == 'rotate' else self.dialog.controls['factor']
        current_angle = self.cursor_ring_angle(key) if self.mode == 'rotate' else None
        if self.mode == 'rotate':
            value = initial + ring_drag_delta(self, start_angle, current_angle, start, cursor)
        else:
            a, b, projected_length = projection
            value = initial + projected_axis_delta(start, cursor_position(self), a, b, projected_length) / length
        self.updating = True
        try:
            field.setValue(value)
        finally:
            self.updating = False
        schedule_handle_sync(self)

    def end_drag(self):
        self.drag_start = None
        self._ring_last_cursor = None
        finish_handle_drag(self)

    def close(self):
        stop_handle_sync(self)
        self._camera_timer.stop()
        self.plotter.camera.RemoveObserver(self._camera_observer)
        for key in self.keys:
            self.widgets[key].EnabledOff()
            self.widgets[key].RemoveAllObservers()
            remove_handle_line(self.plotter, 'parameter_handle_' + self.mode + key)
            if self.mode == 'rotate':
                remove_handle_line(self.plotter, 'parameter_handle_' + self.mode + key + '_knob')
        self.plotter.render()


class CompositeHandles:
    def __init__(self, *handles): self.handles = handles
    def sync(self, render=True):
        for handle in self.handles: handle.sync(render=False)
        if render and self.handles:
            self.handles[0].plotter.render()
    def close(self):
        for handle in self.handles: handle.close()


class OperationHandle:
    """One model-space arrow or angle ring bound to a numeric dialog field."""
    def __init__(self, dialog, field, axis='Z', angle=False, index=0):
        self.dialog, self.plotter, self.field = dialog, dialog.manager.plotter, field
        self.axis, self.angle, self.index = axis, angle, index
        self.drag_start, self.updating = None, False
        self.name = f'operation_handle_{id(self)}'
        self.hit_filter = None
        if not angle and hasattr(self.plotter, 'installEventFilter'):
            from UI.main.arrow_hit_filter import ArrowHitFilter
            self.hit_filter = ArrowHitFilter(self)
        self.rep = vtk.vtkPointHandleRepresentation3D()
        self.rep.SetHandleSize(12)
        self.rep.GetProperty().SetColor(.22, .57, .82)
        self.rep.GetSelectedProperty().SetColor(.65, .85, 1.)
        if not angle:
            self.rep.GetProperty().SetOpacity(0.)
            self.rep.GetSelectedProperty().SetOpacity(0.)
        else:
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
            observe_arrow_camera(self)
        except Exception:
            if self.hit_filter is not None:
                self.hit_filter.dispose()
            self.widget.EnabledOff()
            self.widget.RemoveAllObservers()
            remove_outline_arrow(self.plotter, self.name)
            raise

    def geometry(self):
        manager, dialog = self.dialog.manager, self.dialog
        if isinstance(self.axis, str) and self.axis == 'revolve':
            center, direction = manager.revolution_axis_geometry(dialog.index, dialog.current_axis())
            center = np.asarray(center, dtype=float)
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
        direction = np.array(direction, dtype=float)
        direction /= np.linalg.norm(direction)
        height = self.plotter.ren_win.GetSize()[1]
        length = max(world_per_pixel(self.plotter.camera, height, center) * (65 + 18 * self.index), 1e-6)
        return center, direction, length

    def ring_basis(self):
        center, axis, length = self.geometry()
        ref = np.array([0., 0., 1.]) if abs(axis[2]) < .9 else np.array([0., 1., 0.])
        first = np.cross(axis, ref)
        first /= np.linalg.norm(first)
        return center, axis, length, first, np.cross(axis, first)

    def sync(self, render=True):
        if self.updating:
            return
        enabled = self.field.isEnabled()
        self.widget.SetEnabled(enabled)
        if not enabled:
            if self.hit_filter is not None: self.hit_filter.segments.clear()
            remove_outline_arrow(self.plotter, self.name)
            if self.angle:
                remove_handle_line(self.plotter, self.name)
                remove_handle_line(self.plotter, self.name + '_knob')
            if render:
                self.plotter.render()
            return
        if self.angle:
            center, axis, length, first, second = self.ring_basis()
            theta = np.linspace(0, 2 * np.pi, 97)
            points = center + length * (np.cos(theta)[:, None] * first + np.sin(theta)[:, None] * second)
            value = np.radians(self.field.value())
            point = center + length * (np.cos(value) * first + np.sin(value) * second)
            update_handle_line(self.plotter, self.name, points, '#a5adb2')
            update_ring_knob(self.plotter, self.name + '_knob', point)
        else:
            center, direction, length = self.geometry()
            point = center + direction * (length + self.field.value())
            add_outline_arrow(self.plotter, self.name, center, point, direction, length * .22)
            if self.hit_filter is not None:
                self.hit_filter.segments['primary'] = (center, point)
        self.rep.SetWorldPosition(tuple(point))
        if render:
            self.plotter.render()

    def cursor_angle(self):
        center, _, length, first, second = self.ring_basis()
        renderer = self.plotter.renderer
        def project(point):
            renderer.SetWorldPoint(float(point[0]), float(point[1]), float(point[2]), 1.)
            renderer.WorldToDisplay()
            return renderer.GetDisplayPoint()
        return projected_ring_angle(cursor_position(self), project(center),
                                    project(center + length * first), project(center + length * second))

    def start_drag(self, *args):
        center, direction, length = self.geometry()
        point = center + direction * (length + self.field.value())
        ring_angle = self.cursor_angle() if self.angle else None
        self._ring_previous, self._ring_total = ring_angle, 0.
        self._ring_last_cursor = tuple(cursor_position(self))
        self.drag_start = (tuple(cursor_position(self)), self.field.value(),
                           ring_angle,
                           None if self.angle else axis_drag_projection(self.plotter, point, direction))

    def drag(self, *args):
        if self.drag_start is None:
            return
        start, initial, start_angle, projection = self.drag_start
        cursor = tuple(cursor_position(self))
        current = self.cursor_angle() if self.angle else None
        if self.angle:
            value = initial + ring_drag_delta(self, start_angle, current, start, cursor)
        else:
            a, b, length = projection
            value = initial + projected_axis_delta(start, cursor_position(self), a, b, length)
        self.updating = True
        try:
            self.field.setValue(value)
        finally:
            self.updating = False
        schedule_handle_sync(self)

    def end_drag(self, *args):
        self.drag_start = None
        self._ring_last_cursor = None
        finish_handle_drag(self)

    def close(self):
        stop_handle_sync(self)
        if getattr(self, 'hit_filter', None) is not None: self.hit_filter.dispose()
        if hasattr(self, '_camera_observer'):
            self._camera_timer.stop()
            self.plotter.camera.RemoveObserver(self._camera_observer)
        self.widget.EnabledOff()
        self.widget.RemoveAllObservers()
        remove_outline_arrow(self.plotter, self.name)
        if getattr(self, 'angle', False):
            remove_handle_line(self.plotter, self.name)
            remove_handle_line(self.plotter, self.name + '_knob')
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
