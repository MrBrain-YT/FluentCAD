from typing import TYPE_CHECKING

import pyvista as pv
import numpy as np
import pprint
import random
from copy import deepcopy

from UI.model.types import SkecthObj
from utils.utils import distance_3d

if TYPE_CHECKING:
    from UI.model.ActionManager import ActionManager

class Circle(SkecthObj):
    
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
        self.type = "circle"
        self.center = None
        self.radius = None
    
    def create_circle(self):
        self.is_drawing = True
        self.plotter.track_click_position(callback=self.on_click, side='left')
        if type(self) is Circle:
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
        if not self.is_drawing or len(self.points) != 1:
            return
        raw_pos = self.plotter.pick_mouse_position()
        if raw_pos is None:
            return
        snapped_pos, _ = self.sketch_instance.get_snapped_pos(raw_pos)
        radius = distance_3d(self.points[0], snapped_pos)
        if radius < 1e-9:
            self.plotter.remove_actor("preview_line_actor")
            return
        angles = np.linspace(0, 2 * np.pi, 97)
        center = self.points[0]
        u, v = self.sketch_instance.u_axis, self.sketch_instance.v_axis
        points = np.asarray([center + radius * (np.cos(t) * u + np.sin(t) * v) for t in angles])
        self.plotter.add_mesh(pv.lines_from_points(points), color="#f3b842", line_width=2,
                              name="preview_line_actor", pickable=False, reset_camera=False)

    def on_click(self, pos):
        if not self.is_drawing: return
        snapped_pos, sketch_local_pos = self.sketch_instance.get_snapped_pos(pos)
        self.points.append(snapped_pos)
        self.render_obj["points"].append(self.plotter.add_points(snapped_pos, point_size=10.0))
        if len(self.points) == 1:
            self.center = snapped_pos
            
        if len(self.points) > 1:
            self.radius = distance_3d(self.points[0], self.points[1])
            if self.radius < 1e-6:
                self.points.pop()
                self.plotter.remove_actor(self.render_obj["points"].pop())
                return
            self.sketch_instance.redraw_sketch_geometry()
            self.stop_drawing()

    def rotation_matrix_from_normals(self, source, target):
        """
        Матрица поворота от source нормали к target нормали
        Формула Родрига
        """
        source = np.array(source, dtype=float)
        target = np.array(target, dtype=float)
        
        # Нормализуем
        source = source / np.linalg.norm(source)
        target = target / np.linalg.norm(target)
        
        # Ось вращения
        v = np.cross(source, target)
        s = np.linalg.norm(v)  # sin угла
        c = np.dot(source, target)  # cos угла
        
        if s < 1e-10:  # векторы параллельны
            if c > 0:
                return np.eye(3)   # одинаковые направления
            else:
                return -np.eye(3)  # противоположные направления
        
        # Кососимметричная матрица
        kmat = np.array([
            [ 0,   -v[2],  v[1]],
            [ v[2],  0,   -v[0]],
            [-v[1],  v[0],  0  ]
        ])
        
        # Формула Родрига: R = I + K + K²*(1-cos)/(sin²)
        R = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))
        return R
            
    def create_circle_lines(self, center, radius, normal, actor_name:str=None, n_points:int=64):
        """
        Создает окружность как набор отрезков в плоскости скетча.
        
        center: центр окружности (3D точка)
        radius: радиус
        normal: нормаль плоскости скетча (чтобы окружность лежала в ней)
        n_points: количество точек (чем больше, тем глаже)
        """
        self.center = center
        calc_resolution = int((radius / 0.25) * 64)
        circle = pv.Circle(radius=radius, resolution=calc_resolution)
        default_normal = np.array(circle.point_normals[0])  # нормаль pv.Circle
        target_normal  = np.array(normal)
        target_normal  = target_normal / np.linalg.norm(target_normal)
        R = self.rotation_matrix_from_normals(default_normal, target_normal)
        
        T = np.eye(4)
        T[:3, :3] = R
        T[:3,  3] = center
        circle.transform(T, inplace=True)

        boundary = circle.extract_feature_edges(
            boundary_edges=True, 
            non_manifold_edges=False, 
            feature_edges=False, 
            manifold_edges=False
        )
        circle_item = self.plotter.add_mesh(
            boundary, 
            color="blue", 
            line_width=2,      # Make the border thicker
            opacity=1.0        # Fully opaque border
        )
        self.render_obj["lines"].append(circle_item)

        self.stop_drawing()
        