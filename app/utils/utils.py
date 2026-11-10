from typing import Union, TYPE_CHECKING, Callable

from pyvistaqt import QtInteractor
import pyvista as pv

if TYPE_CHECKING:
    from UI.model.ActionManager import ActionManager

class Picking:
    
    def __init__(self, plotter):
        self.plotter:Union[QtInteractor, pv.Plotter] = plotter
    
    def main_picking(self, callback:Callable):
        self.plotter.disable_picking()
        self.plotter.enable_element_picking(
            callback=callback,
            mode='mesh',
            pickable_window=True,
            show_message=False,
            left_clicking=True
        )
        # self.plotter.enable_mesh_picking()
        # self.plotter.enable_element_picking(
        #     callback=None,
        #     mode='edge'
        # )
        # self.plotter.enable_surface_point_picking()
        
    def enable_face_picking(self, callback:Callable):
        self.plotter.enable_cell_picking(
            callback=callback,
            through=False,
            show=False,
            left_clicking=True,
        )
        
    def sketch_picking(self, callback:Callable):
        self.plotter.disable_picking()
        self.plotter.enable_mesh_picking(
            callback=callback, 
            show=False,
            left_clicking=True,
            show_message=False
        )
        
import numpy as np

def angle_between_points(a, b, c):
    """
    Calc angle using 3 points.
    """
    a, b, c = np.array(a), np.array(b), np.array(c)
    
    ba = a - b
    bc = c - b
    
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cos_angle = np.clip(cos_angle, -1, 1)
    
    return np.degrees(np.arccos(cos_angle))

def is_contour_closed_by_angles(points: list) -> bool:
    """
    Проверяет замкнутость через сумму углов поворота.
    Работает для 2D точек.
    """
    if len(points) < 3:
        return False
    
    points.append(points[1])
    points = np.array(points)

    total_angle = 0
    for i in range(1, len(points)-1):
        total_angle += angle_between_points(points[i-1], points[i], points[i+1])

    # Для замкнутого контура сумма ≈ ±2π
    return total_angle == 360


def distance_3d(point_1: np.ndarray, point_2: np.ndarray) -> float:
    """
    Calculate point distance 3D.

    Args:
        a: First point [x, y, z].
        b: Two point [x, y, z].

    Returns:
        Distance
    """
    return float(np.linalg.norm(point_1 - point_2))

def find_circle_center(p1, p2, p3):
    """
    Находит центр и радиус окружности по 3-м точкам в 3D.
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)

    # Векторы сторон треугольника
    v1 = p2 - p1
    v2 = p3 - p1

    # Находим нормаль к плоскости треугольника через кросс-произведение
    # (Она же будет направлением для вашего круга в PyVista)
    normal = np.cross(v1, v2)
    normal_mag_sq = np.dot(normal, normal)

    if normal_mag_sq < 1e-15:
        raise ValueError("Точки лежат на одной прямой!")

    # Длины векторов и скалярные произведения
    v1_sq = np.dot(v1, v1)
    v2_sq = np.dot(v2, v2)
    v1_dot_v2 = np.dot(v1, v2)

    # Вычисление барицентрических координат центра
    # Формула для нахождения центра описанной окружности (circumcenter)
    a = v2_sq * (v1_sq - v1_dot_v2) / (2 * normal_mag_sq)
    b = v1_sq * (v2_sq - v1_dot_v2) / (2 * normal_mag_sq)

    # Вектор центра относительно точки p1
    center_vector = a * v1 + b * v2
    
    # Итоговый центр
    center = p1 + center_vector
    
    # Радиус - это расстояние от центра до любой из точек
    radius = np.linalg.norm(center - p1)
    
    # Нормализуем вектор нормали для дальнейшего использования
    normal = normal / np.sqrt(normal_mag_sq)

    return center, radius, normal

def get_directed_arc_midpoint(p1, p2, center, normal):
    """
    Находит середину дуги, идя от p1 к p2 строго в положительном направлении 
    (против часовой стрелки относительно нормали).
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    center = np.array(center)
    normal = np.array(normal)
    normal = normal / np.linalg.norm(normal) # Нормализуем

    # Векторы от центра
    v1 = p1 - center
    v2 = p2 - center
    radius = np.linalg.norm(v1)

    # Создаем локальный базис в плоскости дуги
    e1 = v1 / radius  # Ось X (направлена на p1)
    e2 = np.cross(normal, e1) # Ось Y (перпендикулярно в плоскости)

    # Находим угол точки p2 относительно p1 в этом базисе
    # x = cos(angle), y = sin(angle)
    x = np.dot(v2, e1)
    y = np.dot(v2, e2)
    
    angle = np.arctan2(y, x)

    # arctan2 дает угол от -pi до pi.
    # Если мы хотим всегда идти "вперед" (в положительном направлении),
    # то отрицательный угол означает, что нам нужно пройти почти полный круг.
    if angle <= 0:
        angle += 2 * np.pi

    # Угол середины дуги
    mid_angle = angle / 2

    # Вычисляем точку через базис
    mid_point = center + radius * (np.cos(mid_angle) * e1 + np.sin(mid_angle) * e2)
    
    return mid_point