"""Interactive spline construction on the active sketch plane."""
import numpy as np
import pyvista as pv
from UI.model.types import SkecthObj


class Spline(SkecthObj):
    def __init__(self, plotter, sketch_data, sketch_instance, kind):
        super().__init__(plotter, sketch_data, sketch_instance, {'points':[], 'lines':[]})
        if kind not in ('spline','fit_spline','nurbs','periodic_nurbs'):
            raise ValueError('Unsupported spline type.')
        self.type = kind
        self.degree = 2
        self.weights, self.knots, self.spans = [], [], []

    def start(self):
        self.is_drawing = True
        self.track_sketch_click()
        iren = getattr(self.plotter, 'iren', None)
        self.preview_observer = iren.add_observer('MouseMoveEvent', self.draw_preview) if iren is not None else None
        self.sketch_instance.status_callback('Click spline points. Enter finishes; Esc cancels the tool.')

    def draw_preview(self, *args):
        if not self.is_drawing or not self.points: return
        raw = self.plotter.pick_mouse_position()
        if raw is None: return
        point, _ = self.sketch_instance.get_snapped_pos(raw)
        points = np.asarray([*self.points, point])
        self.plotter.add_mesh(pv.lines_from_points(points), color='#e39832', line_width=1,
                              name='spline_preview', pickable=False, reset_camera=False)

    def sync_parameters(self):
        count = len(self.points)
        if count < 3: return
        self.degree = min(3, count-1)
        self.weights = [1.] * count
        self.knots = [0.]*(self.degree+1) + [i/(count-self.degree) for i in range(1,count-self.degree)] + [1.]*(self.degree+1)
        self.spans = [1.] * count

    def on_click(self, position):
        if not self.is_drawing: return
        point, _ = self.sketch_instance.get_snapped_pos(position)
        if self.points and np.linalg.norm(np.asarray(point)-self.points[-1]) < 1e-6:
            self.sketch_instance.status_callback('Choose a different point.')
            return
        self.points.append(np.asarray(point,dtype=float))
        self.render_obj['points'].append(self.plotter.add_points(point, point_size=10))
        self.sync_parameters()
        minimum = 4 if self.type == 'spline' else 3
        if len(self.points) >= minimum:
            try:
                self.sketch_instance.redraw_sketch_geometry()
            except (ValueError, RuntimeError) as exc:
                self.sketch_instance.status_callback(str(exc))
            if self.type == 'spline' and len(self.points) == 4:
                self.finish_drawing()

    def finish_drawing(self):
        minimum = 4 if self.type == 'spline' else 3
        if len(self.points) < minimum:
            self.sketch_instance.status_callback(f'Spline requires at least {minimum} points.')
            return False
        self.stop_drawing()
        manager = self.sketch_instance
        try:
            if manager.history_edit_id is None:
                manager.promote_current_sketch()
            else:
                manager.flush_history_tools()
            manager.active_tool = None
            manager.status_callback('Spline created. Drag its points or use NURBS parameters.')
            return True
        except Exception as exc:
            manager.status_callback(str(exc))
            return False

    def stop_drawing(self, *args):
        self.is_drawing = False
        observer = getattr(self, 'preview_observer', None)
        if observer is not None:
            self.plotter.iren.remove_observer(observer)
            self.preview_observer = None
        self.plotter.disable_picking()
        if hasattr(self.plotter, 'remove_actor'):
            self.plotter.remove_actor('spline_preview')
