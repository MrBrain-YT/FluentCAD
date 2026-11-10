"""Circular and elliptical arc tools based on center and axes."""
import numpy as np
from UI.model.types import SkecthObj
from UI.model.tools.Ellipse import Ellipse


def centered_arc_points(center, start, end, normal):
    center, start, end = (np.asarray(p, dtype=float) for p in (center, start, end))
    normal = np.asarray(normal, dtype=float)
    normal_length = np.linalg.norm(normal)
    if normal_length < 1e-9:
        raise ValueError('Sketch plane has no normal.')
    normal /= normal_length
    first = start - center
    first -= np.dot(first, normal) * normal
    radius = np.linalg.norm(first)
    last = end - center
    last -= np.dot(last, normal) * normal
    if radius < 1e-6 or np.linalg.norm(last) < 1e-6:
        raise ValueError('Arc radius and end direction must be non-zero.')
    x_axis = first / radius
    y_axis = np.cross(normal, x_axis)
    angle = np.arctan2(np.dot(last, y_axis), np.dot(last, x_axis)) % (2 * np.pi)
    if angle < 1e-6 or 2 * np.pi - angle < 1e-6:
        raise ValueError('Arc start and end must be different.')
    middle = center + radius * (np.cos(angle / 2) * x_axis + np.sin(angle / 2) * y_axis)
    final = center + radius * (np.cos(angle) * x_axis + np.sin(angle) * y_axis)
    return [center + first, final, middle]


class CenterArc(SkecthObj):
    def __init__(self, plotter, sketch_data, sketch_instance):
        super().__init__(plotter, sketch_data, sketch_instance, {'points': [], 'lines': []})
        self.type = 'curve'
        self.clicks = []
        self.center = None

    def start(self):
        self.is_drawing = True
        self.plotter.track_click_position(callback=self.on_click, side='left')
        self.sketch_instance.status_callback('Arc: select center, start, then end (counterclockwise).')

    def on_click(self, position):
        if not self.is_drawing:
            return
        point, _ = self.sketch_instance.get_snapped_pos(position)
        self.clicks.append(point)
        if len(self.clicks) < 3:
            return
        try:
            self.points = centered_arc_points(*self.clicks, self.sketch_instance.active_normal)
        except ValueError as exc:
            self.clicks.pop()
            self.sketch_instance.status_callback(str(exc))
            return
        self.center = np.asarray(self.clicks[0])
        self.sketch_instance.redraw_sketch_geometry()
        self.stop_drawing()

    def stop_drawing(self, *args):
        self.is_drawing = False
        self.plotter.disable_picking()


class EllipseArc(Ellipse):
    def __init__(self, plotter, sketch_data, sketch_instance):
        super().__init__(plotter, sketch_data, sketch_instance)
        self.is_closed = False

    def on_click(self, position):
        if not self.is_drawing:
            return
        point, _ = self.sketch_instance.get_snapped_pos(position)
        self.points.append(point)
        if len(self.points) == 3:
            major = self.points[1] - self.points[0]
            minor = self.points[2] - self.points[0]
            if np.linalg.norm(major) < 1e-6 or np.linalg.norm(np.cross(major, minor)) < 1e-6 * np.linalg.norm(major):
                self.points.pop()
                self.sketch_instance.status_callback('Ellipse axes must have non-zero lengths.')
                return
            self.update_ellipse_geometry(*self.points[:3])
            self.sketch_instance.status_callback('Ellipse arc: select start and end directions.')
        elif len(self.points) in (4, 5):
            if np.linalg.norm(self.points[-1] - self.center) < 1e-6:
                self.points.pop()
                return
            angle = self.get_local_angle(self.points[-1])
            self.points[-1] = self.point_at_angle(angle)
            if len(self.points) == 4:
                self.start_angle = angle
            else:
                sweep = (angle - self.start_angle) % (2 * np.pi)
                if sweep < 1e-6 or 2 * np.pi - sweep < 1e-6:
                    self.points.pop()
                    self.sketch_instance.status_callback('Arc start and end must be different.')
                    return
                self.end_angle = self.start_angle + sweep
                self.sketch_instance.redraw_sketch_geometry()
                self.stop_drawing()

    def point_at_angle(self, angle):
        minor_axis = np.cross(self.sketch_instance.active_normal, self.major_axis_vector)
        return (self.center + self.major_radius * np.cos(angle) * self.major_axis_vector +
                self.minor_radius * np.sin(angle) * minor_axis)
