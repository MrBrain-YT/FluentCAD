from typing import TYPE_CHECKING

import pyvista as pv
import numpy as np
import build123d as cad
import random

colors = ["red", "yellow", "green", "blue", "darkred"]

from UI.model.types import SkecthObj
from utils.utils import find_circle_center, distance_3d

if TYPE_CHECKING:
    from UI.model.ActionManager import ActionManager

class Curve(SkecthObj):
    
    def __init__(self, plotter, sketch_data, sketch_instance:"ActionManager"):
        super().__init__(
            plotter=plotter,
            sketch_data=sketch_data,
            sketch_instance=sketch_instance,
            render_obj={
                "points" : [],
                "lines" : []
            }
        )
        self.type = "curve"
        self.center = None
    
    def create_curve(self):
        self.is_drawing = True
        self.plotter.track_click_position(callback=self.on_click, side='left')
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
        point, _ = self.sketch_instance.get_snapped_pos(raw_pos)
        if len(self.points) == 1:
            if np.linalg.norm(point - self.points[0]) < 1e-9:
                self.plotter.remove_actor("preview_line_actor")
                return
            mesh = pv.Line(self.points[0], point)
        else:
            try:
                edge = cad.Edge.make_three_point_arc(tuple(self.points[0]), tuple(point), tuple(self.points[1]))
                mesh = pv.lines_from_points(np.asarray([tuple(edge.position_at(t)) for t in np.linspace(0, 1, 65)]))
            except (ValueError, RuntimeError):
                self.plotter.remove_actor("preview_line_actor")
                return
        self.plotter.add_mesh(mesh, color="#f3b842", line_width=2,
                              name="preview_line_actor", pickable=False, reset_camera=False)

    def on_click(self, pos):
        if not self.is_drawing: return
        snapped_pos, sketch_local_pos = self.sketch_instance.get_snapped_pos(pos)
        self.points.append(snapped_pos)
        point = self.plotter.add_points(snapped_pos, point_size=10.0)
        self.render_obj["points"].append(point)
        if len(self.points) > 2:
            try:
                center, radius, normal = find_circle_center(
                    self.points[0], snapped_pos, self.points[1]
                )
            except ValueError:
                self.points.pop()
                self.plotter.remove_actor(self.render_obj["points"].pop())
                return
            self.center = center
            self.sketch_instance.redraw_sketch_geometry()
            self.stop_drawing()
            
    def _build_arc_points(self, point_start, point_end, point_middle, center, resolution):
        """
        Строит точки дуги вручную, гарантируя прохождение через point_middle.
        """
        center = np.array(center)
        p_start  = np.array(point_start)
        p_end    = np.array(point_end)
        p_middle = np.array(point_middle)
        center_arr = np.array(center)

        # Векторы от центра к точкам
        v_start  = p_start  - center_arr
        v_end    = p_end    - center_arr
        v_middle = p_middle - center_arr

        # Нормаль плоскости дуги
        normal = np.cross(v_start, v_end)
        if np.linalg.norm(normal) < 1e-10:
            normal = np.cross(v_start, v_middle)
        normal = normal / np.linalg.norm(normal)

        # Углы точек относительно center в плоскости
        def angle_in_plane(v):
            x = np.dot(v, v_start / np.linalg.norm(v_start))
            y = np.dot(v, np.cross(normal, v_start / np.linalg.norm(v_start)))
            return np.arctan2(y, x)

        angle_start  = 0.0  # по определению
        angle_end    = angle_in_plane(v_end)
        angle_middle = angle_in_plane(v_middle)

        # Нормализуем углы в [0, 2pi]
        def normalize_angle(a):
            return a % (2 * np.pi)

        angle_end    = normalize_angle(angle_end)
        angle_middle = normalize_angle(angle_middle)

        # Проверяем — лежит ли middle между start(0) и end по CCW
        if angle_middle <= angle_end:
            # CCW: от 0 до angle_end, middle внутри
            angles = np.linspace(0, angle_end, resolution)
        else:
            # CW: идём в другую сторону
            angles = np.linspace(0, angle_end - 2 * np.pi, resolution)

        # Строим точки
        radius = np.linalg.norm(v_start)
        e1 = v_start / np.linalg.norm(v_start)
        e2 = np.cross(normal, e1)

        points = np.array([
            center_arr + radius * (np.cos(a) * e1 + np.sin(a) * e2)
            for a in angles
        ])

        return points


    def create_curve_line(self, point_start, point_end, point_middle, invert:bool=False):
        color = random.choice(colors)
        
        center, radius, normal = find_circle_center(
            point_start, point_middle, point_end
        )
        
        if self.center is None:
            self.center = center

        self.middle = point_middle

        if center is None or radius is None or radius < 1e-6:
            print("⚠️ Невозможно построить дугу: точки вырождены или совпадают")
            self.stop_drawing()
            return

        center = np.array(center)
        normal = np.array(normal)
        normal = normal / np.linalg.norm(normal)

        # Векторы
        v_start = np.array(point_start) - center
        v_end   = np.array(point_end) - center
        v_mid   = np.array(point_middle) - center

        radius = np.linalg.norm(v_start)

        # Локальный базис
        e1 = v_start / radius
        e2 = np.cross(normal, e1)

        def angle_of(v):
            x = np.dot(v, e1)
            y = np.dot(v, e2)
            return np.arctan2(y, x)

        a_start = 0.0
        a_end   = angle_of(v_end)
        a_mid   = angle_of(v_mid)

        # Нормализация в [0, 2π)
        def norm_angle(a):
            return a % (2 * np.pi)

        a_end = norm_angle(a_end)
        a_mid = norm_angle(a_mid)

        # Определяем sweep как в _build_arc_points
        if a_mid <= a_end:
            sweep = a_end
        else:
            sweep = a_end - 2 * np.pi

        # ✅ ИНВЕРСИЯ ДУГИ
        if invert:
            if sweep > 0:
                sweep = sweep - 2 * np.pi
            else:
                sweep = sweep + 2 * np.pi

        arc_length = abs(radius * sweep)
        calc_resolution = max(int((arc_length / 0.25) * 64), 3)

        angles = np.linspace(0, sweep, calc_resolution)

        points = np.array([
            center + radius * (
                np.cos(a) * e1 +
                np.sin(a) * e2
            )
            for a in angles
        ])

        if len(points) < 2:
            print("⚠️ Недостаточно точек для построения дуги")
            self.stop_drawing()
            return

        n = len(points)
        lines = np.zeros((n - 1, 3), dtype=int)
        lines[:, 0] = 2
        lines[:, 1] = np.arange(n - 1)
        lines[:, 2] = np.arange(1, n)

        curve = pv.PolyData()
        curve.points = points
        curve.lines = lines

        actor = self.plotter.add_mesh(
            curve,
            color=color,
            line_width=2,
            opacity=1.0
        )

        self.render_obj["lines"].append(actor)
        self.stop_drawing()
            