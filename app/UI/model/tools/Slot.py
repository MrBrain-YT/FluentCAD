"""Center-to-center straight slot, stored as exact lines and circular arcs."""
import numpy as np
import build123d as cad
from UI.model.types import SkecthObj


def slot_edges(points, normal, tolerance=1e-6):
    if len(points) < 3:
        return []
    start, end, width_point = (np.asarray(p, dtype=float) for p in points[:3])
    normal = np.asarray(normal, dtype=float)
    direction = end - start
    length = np.linalg.norm(direction)
    if length < tolerance:
        return []
    direction /= length
    perpendicular = np.cross(normal, direction)
    perpendicular /= np.linalg.norm(perpendicular)
    radius = abs(np.dot(width_point - start, perpendicular))
    if radius < tolerance:
        return []
    a, b = start + radius * perpendicular, end + radius * perpendicular
    c, d = end - radius * perpendicular, start - radius * perpendicular
    return [cad.Edge.make_line(tuple(a), tuple(b)),
            cad.Edge.make_three_point_arc(tuple(b), tuple(end + radius * direction), tuple(c)),
            cad.Edge.make_line(tuple(c), tuple(d)),
            cad.Edge.make_three_point_arc(tuple(d), tuple(start - radius * direction), tuple(a))]


def polyline_slot_edges(points, normal, radius, tolerance=1e-6):
    """Boundary of the union of round-ended slots along a polyline."""
    points = [np.asarray(point, dtype=float) for point in points]
    normal = np.asarray(normal, dtype=float)
    if len(points) < 2 or len(points) > 100 or not np.isfinite(radius) or radius <= tolerance:
        raise ValueError('Polyline slot requires 2 to 100 points and a positive radius.')
    faces = []
    for start, end in zip(points, points[1:]):
        direction = end - start
        length = np.linalg.norm(direction)
        if length <= tolerance: raise ValueError('Adjacent slot path points must be distinct.')
        perpendicular = np.cross(normal, direction / length)
        if np.linalg.norm(perpendicular) <= tolerance: raise ValueError('Slot path must lie in its plane.')
        perpendicular /= np.linalg.norm(perpendicular)
        edges = slot_edges((start, end, start + radius * perpendicular), normal, tolerance)
        faces.append(cad.Face(cad.Wire.combine(edges)[0]))
    result = faces[0]
    for face in faces[1:]:
        result = result.fuse(face).clean()
    if not result.is_valid or not result.faces():
        raise ValueError('Unable to construct a valid composite slot.')
    return [edge for face in result.faces() for edge in face.edges()]


def normalize_slot_points(points, normal, mode):
    first, second, width_point = (np.asarray(p, dtype=float) for p in points)
    direction = second - first
    length = np.linalg.norm(direction)
    if length < 1e-6:
        raise ValueError('The slot axis must have a length.')
    direction /= length
    perpendicular = np.cross(normal, direction)
    perpendicular /= np.linalg.norm(perpendicular)
    radius = abs(np.dot(width_point - first, perpendicular))
    if radius < 1e-6:
        raise ValueError('The slot width must be non-zero.')
    if mode == 'center_to_center':
        start, end = first, second
    elif mode == 'overall':
        if length <= 2 * radius + 1e-6:
            raise ValueError('Overall length must exceed the slot width.')
        start, end = first + radius * direction, second - radius * direction
    elif mode == 'center_point':
        start, end = 2 * first - second, second
    else:
        raise ValueError('Unknown slot mode.')
    return [start, end, start + radius * perpendicular]


class Slot(SkecthObj):
    def __init__(self, plotter, sketch_data, sketch_instance, mode='center_to_center'):
        super().__init__(plotter, sketch_data, sketch_instance, {'points': [], 'lines': []})
        self.type = 'slot'
        self.mode = mode

    def start(self):
        self.is_drawing = True
        self.track_sketch_click()
        self.sketch_instance.status_callback('Slot: select two defining points, then set the width.')

    def on_click(self, position):
        if not self.is_drawing:
            return
        point, _ = self.sketch_instance.get_snapped_pos(position)
        self.points.append(point)
        if len(self.points) == 2 and np.linalg.norm(self.points[1] - self.points[0]) < 1e-6:
            self.points.pop()
            return
        if len(self.points) == 3:
            try:
                self.points = normalize_slot_points(self.points, self.sketch_instance.active_normal, self.mode)
            except ValueError as exc:
                self.points.pop()
                self.sketch_instance.status_callback(str(exc))
                return
            self.sketch_instance.redraw_sketch_geometry()
            self.stop_drawing()

    def stop_drawing(self, *args):
        self.is_drawing = False
        self.plotter.disable_picking()
