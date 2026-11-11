from pyvistaqt import QtInteractor
import pyvista as pv
import numpy as np

from UI.model.types import SkecthObj

class Line(SkecthObj):
    
    def __init__(self, plotter, sketch_data, sketch_instance):
        super().__init__(
            plotter=plotter,
            sketch_data=sketch_data,
            sketch_instance=sketch_instance,
            render_obj={
                "points" : [],
                "lines" : []
            }
        )
        self.type = "line"
    
    def create_line(self):
        self.is_drawing = True
        self.track_sketch_click()
        self.preview_observer = self.plotter.iren.add_observer('MouseMoveEvent', self.draw_preview_line)
        # self.plotter.track_click_position(callback=self.stop_drawing_line, side='right')
        # self.plotter.add_key_event("Delete", self.sketch_instance.delete_selected)
        # self.plotter.add_key_event("esc", self.stop_drawing_line)
        # self.plotter.track_mouse_position()
        # self.plotter.iren.track_mouse_position(callback=self.draw_preview_line)
        
    def stop_drawing(self, *args):
        self.is_drawing = False
        if getattr(self, 'preview_observer', None) is not None:
            self.plotter.iren.remove_observer(self.preview_observer)
            self.preview_observer = None
        self.plotter.disable_picking()
        self.plotter.remove_actor("preview_line_actor")
        self.plotter.render()
        
    def draw_preview_line(self, *args):
        if not self.is_drawing or not self.points:
            return
        raw_pos = self.plotter.pick_mouse_position()
        if raw_pos is None:
            return
        snapped_pos, _ = self.sketch_instance.get_snapped_pos(raw_pos)
        if np.linalg.norm(self.points[-1] - snapped_pos) < 1e-9:
            self.plotter.remove_actor("preview_line_actor")
            return
        self.plotter.add_mesh(pv.Line(self.points[-1], snapped_pos), color="#f3b842",
                              line_width=2, name="preview_line_actor",
                              pickable=False, reset_camera=False)

    def on_click(self, pos):
        if not self.is_drawing: return

        snapped_pos, sketch_local_pos = self.sketch_instance.get_snapped_pos(pos)
        self.render_obj["points"].append(self.plotter.add_points(snapped_pos, point_size=10.0))
        self.points.append(snapped_pos)
        if len(self.points) > 1:
            self.sketch_instance.redraw_sketch_geometry()
            
