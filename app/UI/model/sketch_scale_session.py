"""Scene selection and center picking for uniform sketch scaling."""
import numpy as np

from UI.model.sketch_move_session import SketchMoveSession
from UI.model.sketch_scale import scale_sketch_selection


class SketchScaleSession(SketchMoveSession):
    command = 'sketch_scale'
    prompt = 'Scale: select geometry, then click Select Center.'

    def __init__(self, manager, factor):
        self.center = None
        self.picking_center = False
        super().__init__(manager, lambda: (self.center, factor()))

    def choose_center(self):
        if not self.selected:
            self.manager.status_callback(self.prompt)
            return False
        self.picking_center = True
        self.center = None
        self.refresh_preview()
        self.manager.status_callback('Click a point on the sketch plane for the scale center.')
        return True

    def can_preview(self, settings):
        center, factor = settings
        return center is not None and factor != 1

    def transform(self, graph, feature_id, settings):
        return scale_sketch_selection(graph, feature_id, self.selected, *settings), False

    def refresh_preview(self, hovered=None):
        previous = self.preview_key
        super().refresh_preview(hovered)
        if self.preview_key != previous and self.center is not None:
            world = self.manager.get_global_pos_in_sketch(self.center)
            self.preview.append(self.manager.plotter.add_points(
                np.asarray([world]), color='#4499c4', point_size=15, pickable=False,
                reset_camera=False, render=False))
            self.manager.plotter.render()

    def hover(self):
        if not self.picking_center:
            super().hover()

    def apply(self):
        if not self.picking_center:
            return super().apply()
        try:
            manager = self.manager
            name = manager._history_handle_at_cursor(named_only=True)
            if name is not None:
                state = manager.history_edit_graph.results[manager.history_edit_id].constraint_status
                self.center = tuple(state.points[name])
            else:
                _, local = manager.get_snapped_pos(manager._screen_to_sketch_plane())
                self.center = tuple(local)
        except (ValueError, RuntimeError) as exc:
            self.manager.status_callback(str(exc))
            return False
        self.picking_center = False
        self.refresh_preview()
        return True

    def confirm(self):
        if self.center is None:
            return self.choose_center()
        if super().confirm():
            self.center = None
            return True
        return False
