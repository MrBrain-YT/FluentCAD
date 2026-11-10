"""Circle tools defined by a diameter or three points."""
import numpy as np
from UI.model.tools.Circle import Circle
from utils.utils import find_circle_center


def circle_from_points(points, mode):
    points = [np.asarray(p, dtype=float) for p in points]
    if mode == 'diameter':
        center = (points[0] + points[1]) / 2
        radius = np.linalg.norm(points[1] - points[0]) / 2
    elif mode == 'three_point':
        center, radius, _ = find_circle_center(*points)
    else:
        raise ValueError('Unknown circle mode.')
    if radius < 1e-6:
        raise ValueError('Circle points must be distinct.')
    return center, radius


class PointCircle(Circle):
    def __init__(self, plotter, sketch_data, sketch_instance, mode):
        super().__init__(plotter, sketch_data, sketch_instance)
        self.mode = mode

    def on_click(self, position):
        if not self.is_drawing:
            return
        point, _ = self.sketch_instance.get_snapped_pos(position)
        self.points.append(point)
        needed = 2 if self.mode == 'diameter' else 3
        if len(self.points) < needed:
            return
        try:
            self.center, self.radius = circle_from_points(self.points, self.mode)
        except ValueError:
            self.points.pop()
            return
        self.sketch_instance.redraw_sketch_geometry()
        self.stop_drawing()
