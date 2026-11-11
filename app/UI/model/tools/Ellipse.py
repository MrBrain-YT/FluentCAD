import pyvista as pv
import numpy as np
from UI.model.types import SkecthObj
from utils.utils import distance_3d

class Ellipse(SkecthObj):
    def __init__(self, plotter, sketch_data, sketch_instance):
        super().__init__(
            plotter=plotter,
            sketch_data=sketch_data,
            sketch_instance=sketch_instance,
            render_obj={
                "points": [],
                "lines": []
            }
        )
        self.type = "ellipse"
        self.center = None
        self.major_radius = None
        self.minor_radius = None
        self.major_axis_vector = None
        self.start_angle = 0.0
        self.end_angle = 2 * np.pi
        self.is_closed = True

    def create_arc(self):
        self.is_drawing = True
        self.track_sketch_click()
        self.preview_observer = self.plotter.iren.add_observer("MouseMoveEvent", self.draw_preview)

    def draw_preview(self, *args):
        if not self.is_drawing or len(self.points) == 0:
            return

        raw_pos = self.plotter.pick_mouse_position()
        if raw_pos is None: 
            return
        snapped_pos, _ = self.sketch_instance.get_snapped_pos(raw_pos)

        if len(self.points) == 1:  # Тянем большую полуось от центра
            line = pv.Line(self.points[0], snapped_pos)
            self.plotter.add_mesh(line, name="preview", color="yellow")

        elif len(self.points) == 2:  # Тянем малую полуось
            self.update_ellipse_geometry(self.points[0], self.points[1], snapped_pos)
            self.create_arc_mesh(name="preview", color="yellow")

    def on_click(self, pos):
        if not self.is_drawing: 
            return
        snapped_pos, _ = self.sketch_instance.get_snapped_pos(pos)
        
        self.points.append(snapped_pos)
        if len(self.points) == 1:
            self.render_obj["points"].append(self.plotter.add_points(snapped_pos, point_size=10.0))

        if len(self.points) == 3:
            major = self.points[1] - self.points[0]
            minor = self.points[2] - self.points[0]
            if np.linalg.norm(major) < 1e-6 or np.linalg.norm(np.cross(major, minor)) < 1e-6 * np.linalg.norm(major):
                self.points.pop()
                return
            self.update_ellipse_geometry(self.points[0], self.points[1], self.points[2])
            self.sketch_instance.redraw_sketch_geometry()
            self.stop_drawing()

    def update_ellipse_geometry(self, center, p_major, p_minor):
        """
        center - центр эллипса (точка 1)
        p_major - точка на большой оси (точка 2)
        p_minor - точка для определения малой оси (точка 3)
        """
        self.center = np.array(center)
        normal = np.array(self.sketch_instance.active_normal)
        
        # Большая ось определяется вектором от центра к второй точке
        major_vec = np.array(p_major) - self.center
        self.major_radius = np.linalg.norm(major_vec)
        
        if self.major_radius < 1e-6:
            self.major_axis_vector = np.array([1.0, 0.0, 0.0])
            self.major_radius = 1e-6
        else:
            self.major_axis_vector = major_vec / self.major_radius
        
        # Малая ось перпендикулярна большой оси в плоскости скетча
        minor_axis_dir = np.cross(normal, self.major_axis_vector)
        minor_axis_dir = minor_axis_dir / (np.linalg.norm(minor_axis_dir) + 1e-10)
        
        # Проецируем вектор к третьей точке на направление малой оси
        vec_to_minor = np.array(p_minor) - self.center
        self.minor_radius = abs(np.dot(vec_to_minor, minor_axis_dir))
        
        if self.minor_radius < 1e-6:
            self.minor_radius = 1e-6
        center = self.sketch_instance.get_local_pos_in_sketch(self.center)
        # print(f"Updated ellipse geometry: center={center}, major_radius={self.major_radius}, minor_radius={self.minor_radius}, {self.major_axis_vector}")

    def create_arc_mesh(self, name=None, color="blue"):
        """Создает эллипс (замкнутый или дугу)"""
        if self.center is None or self.major_radius is None:
            return
        
        start = self.start_angle
        end = self.end_angle
        
        # Определяем разрешение
        resolution = max(64, int(self.major_radius * 50))
        
        if self.is_closed:
            # Замкнутый эллипс
            t = np.linspace(0, 2 * np.pi, resolution)
        else:
            # Дуга эллипса
            if end <= start:
                end += 2 * np.pi
            t = np.linspace(start, end, resolution)

        # Параметрическое уравнение эллипса
        x = self.major_radius * np.cos(t)
        y = self.minor_radius * np.sin(t)
        z = np.zeros_like(x)
        local_points = np.column_stack((x, y, z))

        # Матрица поворота для преобразования из локальной СК в мировую
        z_axis = np.array(self.sketch_instance.active_normal)
        x_axis = self.major_axis_vector
        y_axis = np.cross(z_axis, x_axis)
        
        rot_mat = np.eye(3)
        rot_mat[:, 0] = x_axis
        rot_mat[:, 1] = y_axis
        rot_mat[:, 2] = z_axis
        
        # Преобразуем точки в мировую СК
        world_points = (rot_mat @ local_points.T).T + self.center
        
        # Создаем линию
        poly = pv.PolyData()
        poly.points = world_points
        
        num_points = len(world_points)
        
        if self.is_closed:
            # Для замкнутой линии добавляем индекс первой точки в конец
            cells = np.arange(0, num_points, dtype=np.int_)
            cells = np.append(cells, 0)  # Замыкаем линию
            cells = np.insert(cells, 0, num_points + 1)  # +1 для замыкания
        else:
            # Для незамкнутой дуги
            cells = np.arange(0, num_points, dtype=np.int_)
            cells = np.insert(cells, 0, num_points)
        
        poly.lines = cells
        
        actor = self.plotter.add_mesh(
            poly, 
            color=color, 
            line_width=2,
            name=name
        )
        
        if name is None:
            self.render_obj["lines"] = [actor]

    def stop_drawing(self, *args):
        self.is_drawing = False
        if getattr(self, "preview_observer", None) is not None:
            self.plotter.iren.remove_observer(self.preview_observer)
            self.preview_observer = None
        self.plotter.disable_picking()
        self.plotter.remove_actor("preview")
        self.plotter.render()
        
    def get_local_angle(self, point_3d):
        """Переводит 3D точку в локальный параметрический угол t эллипса."""
        vec = np.array(point_3d) - self.center
        z_axis = np.array(self.sketch_instance.active_normal)
        x_axis = self.major_axis_vector
        y_axis = np.cross(z_axis, x_axis)

        # Проекция вектора на локальные оси X и Y
        x = np.dot(vec, x_axis)
        y = np.dot(vec, y_axis)

        # В эллипсе x = a*cos(t), y = b*sin(t)
        # Поэтому t = atan2(y/b, x/a)
        angle = np.arctan2(y / self.minor_radius, x / self.major_radius)
        return angle % (2 * np.pi)    
    
    def set_arc_from_intersections(self, intersection_point1, intersection_point2, is_major_arc=False):
        """
        Устанавливает дугу эллипса между двумя точками пересечения.
        
        Args:
            intersection_point1: первая точка пересечения
            intersection_point2: вторая точка пересечения
            is_major_arc: если True, рисует большую дугу, иначе меньшую
        """
        if self.center is None or self.major_radius is None:
            return
        
        # self.is_closed = False
        
        # Получаем углы для точек пересечения
        angle1 = self.get_local_angle(intersection_point1)
        angle2 = self.get_local_angle(intersection_point2)
        
        # Нормализуем углы
        if angle2 < angle1:
            angle2 += 2 * np.pi
        
        # Определяем какую дугу рисовать
        # arc_length = angle2 - angle1
        
        if is_major_arc:
            # Рисуем большую дугу (идем в обратную сторону)
            self.start_angle = angle2
            self.end_angle = angle1 + 2 * np.pi
        else:
            # Рисуем меньшую дугу
            self.start_angle = angle1
            self.end_angle = angle2
