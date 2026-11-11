"""Reference point in a sketch; it does not form a profile."""
from UI.model.types import SkecthObj


class SketchPoint(SkecthObj):
    def __init__(self, plotter, sketch_data, sketch_instance):
        super().__init__(plotter, sketch_data, sketch_instance, {'points': [], 'lines': []})
        self.type = 'point'

    def start(self):
        self.is_drawing = True
        self.track_sketch_click()

    def on_click(self, position):
        if not self.is_drawing:
            return
        point, _ = self.sketch_instance.get_snapped_pos(position)
        self.points.append(point)
        self.render_obj['points'].append(self.plotter.add_points(point, color='blue', point_size=10))
        self.stop_drawing()

    def stop_drawing(self, *args):
        self.is_drawing = False
        self.plotter.disable_picking()
