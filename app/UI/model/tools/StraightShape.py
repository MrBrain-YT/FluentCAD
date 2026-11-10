"""Interactive straight-edged sketch shapes."""
import numpy as np
import pyvista as pv

from UI.model.types import SkecthObj


def shape_points(kind, clicks, center, u_axis, v_axis, sides=6):
    """Return a closed polyline in world space, or None for a degenerate shape."""
    local = [np.array([np.dot(p - center, u_axis), np.dot(p - center, v_axis)])
             for p in map(np.asarray, clicks)]
    a, b = local[:2]
    if kind == 'midpoint_line':
        result = [2 * a - b, b]
    elif kind == 'rectangle':
        result = [a, np.array([b[0], a[1]]), b, np.array([a[0], b[1]])]
    elif kind == 'center_rectangle':
        d = b - a
        result = [a - d, a + [-d[0], d[1]], a + d, a + [d[0], -d[1]]]
    elif kind == 'three_point_rectangle':
        edge = b - a
        length = np.linalg.norm(edge)
        if length < 1e-9:
            return None
        perpendicular = np.array([-edge[1], edge[0]]) / length
        height = np.dot(local[2] - a, perpendicular)
        result = [a, b, b + height * perpendicular, a + height * perpendicular]
    elif kind in ('polygon', 'circumscribed_polygon', 'edge_polygon'):
        if sides < 3:
            raise ValueError('A polygon needs at least three sides.')
        if kind == 'edge_polygon':
            edge = b - a
            length = np.linalg.norm(edge)
            if length < 1e-9:
                return None
            midpoint = (a + b) / 2
            polygon_center = midpoint + np.array([-edge[1], edge[0]]) / (2 * np.tan(np.pi / sides))
            radius_vector = a - polygon_center
        else:
            polygon_center = a
            radius_vector = b - a
        angle = np.arctan2(radius_vector[1], radius_vector[0])
        radius = np.linalg.norm(radius_vector)
        if kind == 'circumscribed_polygon':
            radius /= np.cos(np.pi / sides)
            angle += np.pi / sides
        result = [polygon_center + radius * np.array([np.cos(angle + 2 * np.pi * i / sides),
                                                        np.sin(angle + 2 * np.pi * i / sides)])
                  for i in range(sides)]
    else:
        raise ValueError(kind)
    if any(np.linalg.norm(result[(i + 1) % len(result)] - result[i]) < 1e-9
           for i in range(len(result) if kind != 'midpoint_line' else 1)):
        return None
    if kind != 'midpoint_line':
        result.append(result[0])
    return [center + point[0] * u_axis + point[1] * v_axis for point in result]


class StraightShape(SkecthObj):
    def __init__(self, plotter, sketch_data, sketch_instance, kind, sides=6):
        super().__init__(plotter, sketch_data, sketch_instance, {'points': [], 'lines': []})
        self.type = 'line'
        self.kind = kind
        self.sides = sides
        self.clicks = []

    def start(self):
        self.is_drawing = True
        self.plotter.track_click_position(callback=self.on_click, side='left')

    def on_click(self, position):
        if not self.is_drawing:
            return
        point, _ = self.sketch_instance.get_snapped_pos(position)
        self.clicks.append(point)
        needed = 3 if self.kind == 'three_point_rectangle' else 2
        if len(self.clicks) < needed:
            return
        m = self.sketch_instance
        points = shape_points(self.kind, self.clicks, m.active_center, m.u_axis, m.v_axis, self.sides)
        if points is None:
            self.clicks.pop()
            return
        self.points = points
        self.sketch_instance.redraw_sketch_geometry()
        self.stop_drawing()

    def stop_drawing(self, *args):
        self.is_drawing = False
        self.plotter.disable_picking()
