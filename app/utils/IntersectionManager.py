from typing import Union, TYPE_CHECKING
import random
import time

from pyvistaqt import QtInteractor
import pyvista as pv
import numpy as np
from scipy.optimize import fsolve
colors = ["red", "yellow", "green", "blue", "darkred"]

from UI.model.tools.Line import Line
from UI.model.tools.Circle import Circle
from UI.model.tools.Curve import Curve
from UI.model.tools.Ellipse import Ellipse

from utils.utils import distance_3d, get_directed_arc_midpoint

class IntersectionManager():

    def __init__(self, item_1:Union['Line', 'Circle', 'Curve', 'Ellipse'], items:list[Union['Line', 'Circle', 'Curve', 'Ellipse']]):
        """
        item_1 it's item with last point\n
        items it's a list of items with which we want to find intersection
        """
        self.item_1 = item_1
        self.items = items
        self.item_1_render_obj = item_1.render_obj
        self.sketch_instance = item_1.sketch_instance
        self.plotter:Union[QtInteractor, pv.Plotter] = item_1.plotter
        self.all_intersections = []

    def find(self, redraw=True) -> list:
        for item_2 in self.items:
            item_1_type = self.item_1.type
            item_2_type = item_2.type
            if item_1_type == "line" and item_2_type == "line":
                self._line_to_line(item_2)
            elif item_1_type == "circle" and item_2_type == "circle":
                self.__circle_to_circle(item_2)
            elif item_1_type == "curve" and item_2_type == "curve":
                self.__curve_to_curve(item_2)
            elif item_1_type == "ellipse" and item_2_type == "ellipse":
                self.__ellipse_to_ellipse(item_2)
            elif item_1_type in ["line", "circle"] and item_2_type in ["line", "circle"]:
                self.__line_to_circle(item_2)
            elif item_1_type in ["line", "curve"] and item_2_type in ["line", "curve"]:
                self.__line_to_curve(item_2)
            elif item_1_type in ["circle", "curve"] and item_2_type in ["circle", "curve"]:
                self.__circle_to_curve(item_2)
            elif item_1_type in ["line", "ellipse"] and item_2_type in ["line", "ellipse"]:
                self.__line_to_ellipse(item_2)
            elif item_1_type in ["circle", "ellipse"] and item_2_type in ["circle", "ellipse"]:
                self.__circle_to_ellipse(item_2)
            # elif item_1_type in ["curve", "ellipse"] and item_2_type in ["curve", "ellipse"]:
            #     self.__curve_to_ellipse(item_2)
                
        if redraw:
            self.__redraw_items(self.all_intersections)
            return None
        else:
            return self.all_intersections

    def _line_to_line(self, item_2) -> None:
        vectors = []
        for point_id in range(1, len(item_2.points)):
            point_start = self.sketch_instance.get_local_pos_in_sketch(item_2.points[point_id - 1].tolist())
            point_end = self.sketch_instance.get_local_pos_in_sketch(item_2.points[point_id].tolist())
            vectors.append(
                {
                    "p_x": point_start[0], # p - point start
                    "p_y": point_start[1],
                    "e_x": point_end[0], # e - point end
                    "e_y": point_end[1]
                }
            )
        vector = {
            "p_x": self.sketch_instance.get_local_pos_in_sketch(self.item_1.points[-2])[0], # p - point start
            "p_y": self.sketch_instance.get_local_pos_in_sketch(self.item_1.points[-2])[1],
            "e_x": self.sketch_instance.get_local_pos_in_sketch(self.item_1.points[-1])[0], # e - point end
            "e_y": self.sketch_instance.get_local_pos_in_sketch(self.item_1.points[-1])[1]
        }
        vector_start = np.array([vector["p_x"], vector["p_y"]])
        vector_end = np.array([vector["e_x"], vector["e_y"]])
        for index, vector_target in enumerate(vectors):
            vector_target_start = np.array([vector_target["p_x"], vector_target["p_y"]])
            vector_target_end = np.array([vector_target["e_x"], vector_target["e_y"]])
            intersect_point = self.find_lines_intersection(vector_start, vector_end, vector_target_start, vector_target_end)
            if intersect_point is not None:
                point_pos_3d = self.sketch_instance.get_global_pos_in_sketch(intersect_point)
                start_target_point_global_pos = self.sketch_instance.get_global_pos_in_sketch(vector_target_start)
                end_target_point_global_pos = self.sketch_instance.get_global_pos_in_sketch(vector_target_end)
                start_point_global_pos = self.sketch_instance.get_global_pos_in_sketch(vector_start)
                end_point_global_pos = self.sketch_instance.get_global_pos_in_sketch(vector_end)
                self.all_intersections.append({
                    "index": index,
                    "old_item_func": item_2,
                    "point_3d": point_pos_3d,
                    "start_3d": start_point_global_pos,
                    "end_3d": end_point_global_pos,
                    "start_target_3d": start_target_point_global_pos,
                    "end_target_3d": end_target_point_global_pos,
                    "distance": distance_3d(start_point_global_pos, point_pos_3d)
                })
                

    def __line_to_circle(self, item_2):
        # Определяем, какой объект линия, а какой окружность
        if self.item_1.type == "line":
            line_item = self.item_1
            circle_item = item_2
        else:
            line_item = item_2
            circle_item = self.item_1

        center_2d = self.sketch_instance.get_local_pos_in_sketch(circle_item.points[0])
        radius = circle_item.radius

        if self.item_1.type == "line":
            l_p1_2d = self.sketch_instance.get_local_pos_in_sketch(line_item.points[-2])
            l_p2_2d = self.sketch_instance.get_local_pos_in_sketch(line_item.points[-1])

            intersection_points_2d = self.find_line_circle_intersections(
                np.array(l_p1_2d), 
                np.array(l_p2_2d), 
                np.array(center_2d), 
                radius
            )

            # for point in intersect_points_2d:
            #     point_3d = self.sketch_instance.get_global_pos_from_sketch(point)
            #     start_line_3d = self.sketch_instance.get_global_pos_from_sketch(l_p1_2d)
                
            #     self.all_intersections.append({
            #         "index": , # индекс сегмента линии
            #         "old_item_func": item_2,
            #         "point_3d": point_3d,
            #         "start_3d": start_line_3d,
            #         "end_3d": self.sketch_instance.get_global_pos_from_sketch(l_p2_2d),
            #         "distance": distance_3d(start_line_3d, point_3d)
            #     })
                
            for point in intersection_points_2d:
                point_3d = self.sketch_instance.get_global_pos_in_sketch(point)
                start_line_3d = self.sketch_instance.get_global_pos_in_sketch(l_p1_2d)
                end_line_3d = self.sketch_instance.get_global_pos_in_sketch(l_p2_2d)
                self.all_intersections.append({
                    "index": len(line_item.points) - 2,
                    "old_item_func": item_2,
                    "point_3d": point_3d,
                    "start_3d": start_line_3d,
                    "end_3d": end_line_3d,
                    "distance": distance_3d(start_line_3d, point_3d)
                })
                
        elif self.item_1.type == "circle":
            intersection_points_2d = []
            for index, point in enumerate(line_item.points):
                if index == 0: continue

                l_p1_2d = self.sketch_instance.get_local_pos_in_sketch(line_item.points[index-1])
                l_p2_2d = self.sketch_instance.get_local_pos_in_sketch(point)

                intersect_points_2d = self.find_line_circle_intersections(
                    np.array(l_p1_2d), 
                    np.array(l_p2_2d), 
                    np.array(center_2d), 
                    radius
                )
                inter_data = {
                    "index": index-1,
                    "start_point": l_p1_2d,
                    "end_point": l_p2_2d,
                    "intersect_points": intersect_points_2d
                }
                intersection_points_2d.append(inter_data)

            for points_data in intersection_points_2d:
                for point in points_data["intersect_points"]:
                    point_3d = self.sketch_instance.get_global_pos_in_sketch(point)
                    start_line_3d = self.sketch_instance.get_global_pos_in_sketch(points_data["start_point"])
                    end_line_3d = self.sketch_instance.get_global_pos_in_sketch(points_data["end_point"])
                    self.all_intersections.append({
                        "index": points_data["index"],
                        "old_item_func": item_2,
                        "point_3d": point_3d,
                        "start_3d": start_line_3d,
                        "end_3d": end_line_3d,
                        "distance": distance_3d(start_line_3d, point_3d)
                    })

    def __line_to_curve(self, item_2):
        if self.item_1.type == "line":
            line_item = self.item_1
            curve_item = item_2
        else:
            line_item = item_2
            curve_item = self.item_1
            
        if self.item_1.type == "line":
            intersection_points_3d = self.find_line_curve_intersections(line_item, curve_item)
            for point in intersection_points_3d:
                point_3d = point
                start_line_3d = line_item.points[-2]
                end_line_3d = line_item.points[-1]
                self.all_intersections.append({
                    "index": -1,
                    "old_item_func": item_2,
                    "point_3d": point_3d,
                    "start_3d": start_line_3d,
                    "end_3d": end_line_3d,
                    "distance": distance_3d(start_line_3d, point_3d)
                })
                
        elif self.item_1.type == "curve":
            for index, point in enumerate(line_item.points):
                if index == 0: continue
                
                temp_line = Line(self.plotter, self.sketch_instance.get_current_sketch_data(), self.sketch_instance)
                temp_line.points = [line_item.points[index-1], line_item.points[index]]
                intersection_points_3d = self.find_line_curve_intersections(temp_line, curve_item)
                for point in intersection_points_3d:
                    point_3d = point
                    start_line_3d = line_item.points[index-1]
                    end_line_3d = line_item.points[index]
                    self.all_intersections.append({
                        "index": index - 1,
                        "old_item_func": item_2,
                        "point_3d": point_3d,
                        "start_3d": start_line_3d,
                        "end_3d": end_line_3d,
                        "distance": distance_3d(start_line_3d, point_3d)
                    })        
    
    def __circle_to_circle(self, item_2):
        intersection_points_3d = self.find_circle_circle_intersections(self.item_1, item_2)
        for point in intersection_points_3d:
            point_3d = point
            self.all_intersections.append({
                "old_item_func": item_2,
                "point_3d": point_3d,
            })

    def __circle_to_curve(self, item_2):
        if self.item_1.type == "circle":
            circle_item = self.item_1
            curve_item = item_2
        else:
            circle_item = item_2
            curve_item = self.item_1
            
        intersection_points_3d = self.find_curve_circle_intersections(curve_item, circle_item)
        for point in intersection_points_3d:
            point_3d = point
            # start_line_3d = line_item.points[-2]
            # end_line_3d = line_item.points[-1]
            self.all_intersections.append({
                # "index": -1,
                "old_item_func": item_2,
                "point_3d": point_3d,
                # "start_3d": start_line_3d,
                # "end_3d": end_line_3d,
                # "distance": distance_3d(start_line_3d, point_3d)
            })

    def __curve_to_curve(self, item_2):
        intersection_points_3d = self.find_curve_curve_intersections(self.item_1, item_2)
        for point in intersection_points_3d:
            point_3d = point
            self.all_intersections.append({
                "old_item_func": item_2,
                "point_3d": point_3d,
            })
            
    def __ellipse_to_ellipse(self, item_2):
        intersection_points_3d = self.find_ellipse_ellipse_intersections(self.item_1, item_2)
        for point in intersection_points_3d:
            point_3d = point
            self.all_intersections.append({
                "old_item_func": item_2,
                "point_3d": point_3d,
            })
            
    def __line_to_ellipse(self, item_2):
        if self.item_1.type == "line":
            line_item = self.item_1
            ellipse_item = item_2
        else:
            line_item = item_2
            ellipse_item = self.item_1
            
        if self.item_1.type == "line":
            intersection_points_3d = self.find_line_ellipse_intersections(line_item, ellipse_item)
            for point in intersection_points_3d:
                point_3d = point
                start_line_3d = line_item.points[-2]
                end_line_3d = line_item.points[-1]
                self.all_intersections.append({
                    "index": -1,
                    "old_item_func": item_2,
                    "point_3d": point_3d,
                    "start_3d": start_line_3d,
                    "end_3d": end_line_3d,
                    "distance": distance_3d(start_line_3d, point_3d)
                })
                
        elif self.item_1.type == "ellipse":
            for index, point in enumerate(line_item.points):
                if index == 0: continue
                
                temp_line = Line(self.plotter, self.sketch_instance.get_current_sketch_data(), self.sketch_instance)
                temp_line.points = [line_item.points[index-1], line_item.points[index]]
                intersection_points_3d = self.find_line_ellipse_intersections(temp_line, ellipse_item)
                print("line to ellipse intersections:", intersection_points_3d)
                for point in intersection_points_3d:
                    point_3d = point
                    start_line_3d = line_item.points[index-1]
                    end_line_3d = line_item.points[index]
                    self.all_intersections.append({
                        "index": index - 1,
                        "old_item_func": item_2,
                        "point_3d": point_3d,
                        "start_3d": start_line_3d,
                        "end_3d": end_line_3d,
                        "distance": distance_3d(start_line_3d, point_3d)
                    })        
        
    def __circle_to_ellipse(self, item_2):
        if self.item_1.type == "circle":
            circle_item = self.item_1
            ellipse_item = item_2
        else:
            circle_item = item_2
            ellipse_item = self.item_1
            
        ellipse_circle = Ellipse(self.plotter, self.sketch_instance.get_current_sketch_data(), self.sketch_instance)
        ellipse_circle.center = circle_item.center
        ellipse_circle.major_radius = circle_item.radius
        ellipse_circle.minor_radius = circle_item.radius
        ellipse_circle.major_axis_vector = self.sketch_instance.active_normal
        intersection_points_3d = self.find_ellipse_ellipse_intersections(ellipse_item, ellipse_circle)
        for point in intersection_points_3d:
            point_3d = point
            self.all_intersections.append({
                "old_item_func": circle_item,
                "point_3d": point_3d,
            })

    def __redraw_items(self, intersections:list):
        sketch_functions = self.sketch_instance.get_current_sketch_data()["sketch_func"]
        
        if self.item_1.type == "line":
            if intersections is None or len(intersections) == 0:
                # draw new_line without intersections points
                line = pv.Line(self.item_1.points[-2], self.item_1.points[-1])
                actor = self.plotter.add_mesh(line, color="blue", line_width=2, pickable=True)
                self.item_1_render_obj["lines"].append(actor)
                return 

            for intersection in intersections:
                # draw points
                if "point_3d" in intersection.keys():
                    point = self.plotter.add_points(intersection["point_3d"], point_size=10.0)
                    self.item_1_render_obj["points"].insert(-1, point)

            # draw new_line with intersections points
            _intersections = intersections.copy()
            _intersections.sort(key=lambda x: x["distance"])
            old_point = intersection["start_3d"]
            for intersection_data in _intersections:
                self.item_1.points.insert(-1, intersection_data["point_3d"])
                self.item_1.points.insert(-1, intersection_data["point_3d"])
                line_1 = pv.Line(old_point, intersection_data["point_3d"])
                actor_1 = self.plotter.add_mesh(line_1, color="blue", line_width=2, pickable=True)
                self.item_1_render_obj["lines"].append(actor_1)
                old_point = intersection_data["point_3d"]
                
            line_last = pv.Line(old_point, intersection["end_3d"])
            actor_last = self.plotter.add_mesh(line_last, color="blue", line_width=2, pickable=True)
            self.item_1_render_obj["lines"].append(actor_last)

            # redraw old item
            intersections.sort(key=lambda x: x["index"], reverse=True)
            intersection_type = None
            intersection_points_type_index = 0
            _curve_intersections = None
            for inter_index, intersection in enumerate(intersections):
                if intersection_type != intersection["old_item_func"].type:
                    intersection_type = intersection["old_item_func"].type
                    intersection_points_type_index = inter_index
                if intersection["old_item_func"].type == "line":
                    self.plotter.remove_actor(intersection["old_item_func"].render_obj["lines"][intersection["index"]])
                    line1 = pv.Line(intersection["start_target_3d"], intersection["point_3d"])
                    line2 = pv.Line(intersection["point_3d"], intersection["end_target_3d"])
                    actor_line_1 = self.plotter.add_mesh(line1, color=random.choice(colors), line_width=2, pickable=True)
                    actor_line_2 = self.plotter.add_mesh(line2, color=random.choice(colors), line_width=2, pickable=True)
                    intersection["old_item_func"].render_obj["lines"][intersection["index"]] = actor_line_2
                    intersection["old_item_func"].render_obj["lines"].insert(intersection["index"], actor_line_1)
                    intersection["old_item_func"].points.insert(intersection["index"]+1, intersection["point_3d"])
                elif intersection["old_item_func"].type == "circle":
                    self.plotter.remove_actor(intersection["old_item_func"].render_obj["lines"][0])
                    # Draw points
                    point = self.plotter.add_points(intersection["point_3d"], point_size=10.0)
                    if inter_index == 0:
                        self.plotter.remove_actor(intersection["old_item_func"].render_obj["points"][-1])
                        intersection["old_item_func"].render_obj["points"][-1] = point
                    else:
                        intersection["old_item_func"].render_obj["points"].append(point)

                    # Draw curves
                    next_intersect_point = None
                    next_item = intersections[inter_index + 1] if inter_index + 1 < len(intersections) else None
                    if next_item is not None:
                        next_intersect_point = intersections[inter_index + 1]["point_3d"]
                    
                    if next_item is None:
                        end_curve = Curve(self.plotter, self.sketch_instance.get_current_sketch_data(), self.sketch_instance)
                        point_start = intersection["point_3d"]
                        point_end = intersections[intersection_points_type_index]["point_3d"]
                        point_middle = get_directed_arc_midpoint(point_start, point_end, intersection["old_item_func"].points[0], self.sketch_instance.active_normal)
                        end_curve.points = [point_start, point_end, point_middle]
                        end_curve.center = intersection["old_item_func"].points[0]
                        end_curve.create_curve_line(point_start, point_end, point_middle)
                        end_curve.stop_drawing()
                        sketch_functions.append(end_curve)
                        # Delete circle item
                        del sketch_functions[sketch_functions.index(intersection["old_item_func"])]
                    elif intersections[inter_index + 1]["old_item_func"] == intersection["old_item_func"]:
                        new_curve = Curve(self.plotter, self.sketch_instance.get_current_sketch_data(), self.sketch_instance)
                        point_start = intersection["point_3d"]
                        point_end = next_intersect_point
                        point_middle = get_directed_arc_midpoint(point_start, point_end, intersection["old_item_func"].points[0], self.sketch_instance.active_normal)
                        new_curve.points = [point_start, point_end, point_middle]
                        new_curve.center = intersection["old_item_func"].points[0]
                        new_curve.create_curve_line(point_start, point_end, point_middle)
                        new_curve.stop_drawing()
                        sketch_functions.append(new_curve)
                        
                elif intersection["old_item_func"].type == "curve":
                    # skip next intersections
                    if inter_index - 1 >= intersection_points_type_index:
                        if intersections[inter_index - 1]["old_item_func"].type == "curve":
                            continue
                    
                    next_item = intersections[inter_index + 1] if inter_index + 1 < len(intersections) else None
                    next_intersect_point = None
                    if next_item is not None:
                        next_intersect_point = intersections[inter_index + 1]["point_3d"]
                        
                    if next_item is not None:
                        if intersections[inter_index + 1]["old_item_func"] == intersection["old_item_func"]:
                            _curve_intersections = [intersection["point_3d"], intersections[inter_index + 1]["point_3d"]]
                        else:
                            _curve_intersections = [intersection["point_3d"]]
                    else:
                        _curve_intersections = [intersection["point_3d"]]         
                        
                    # Remove curve actor
                    self.plotter.remove_actor(intersection["old_item_func"].render_obj["lines"][0])
                    # 3d to 2d points for curve splitting
                    old_curve_points_2d = [
                        self.sketch_instance.get_local_pos_in_sketch(pt) for pt in intersection["old_item_func"].points
                    ]
                    inter_curve_points = [
                        self.sketch_instance.get_local_pos_in_sketch(pt) for pt in _curve_intersections
                    ]
                    new_curves = self.split_arc(old_curve_points_2d, inter_curve_points)
                    
                    # Draw curves
                    new_functions = []
                    for points in new_curves:
                        points = [self.sketch_instance.get_global_pos_in_sketch(pt) for pt in points]
                        curve = Curve(self.plotter, self.sketch_instance.get_current_sketch_data(), self.sketch_instance)
                        point_start = points[0]
                        point_end = points[1]
                        curve.points = [point_start, point_end, points[2]]
                        curve.center = intersection["old_item_func"].center
                        curve.create_curve_line(point_start, point_end, points[2])
                        curve.stop_drawing()
                        new_functions.append(curve)
                        old_point = point_end
                    
                    # Delete curve item
                    del sketch_functions[sketch_functions.index(intersection["old_item_func"])]
                    sketch_functions.extend(new_functions)
                    _curve_intersections = None 
                    

        elif self.item_1.type == "circle":
            if intersections is None or len(intersections) <= 1:
                # draw new_circle without intersections points
                normal = self.sketch_instance.active_normal
                self.item_1.create_circle_lines(self.item_1.points[0], self.item_1.radius, normal)
                if len(intersections) == 1:
                    point = self.plotter.add_points(intersections[0]["point_3d"], point_size=10.0)
                    self.plotter.remove_actor(self.item_1_render_obj["points"][-1])
                    self.item_1_render_obj["points"][-1] = point

                return 
            
            intersections = self.sort_intersections_by_angle(
                intersections, 
                self.item_1.points[0],
                self.sketch_instance.active_normal
            )

            intersection_type = None
            intersection_points_type_index = 0
            for inter_index, intersection in enumerate(intersections):
                if intersection_type != intersection["old_item_func"].type:
                    intersection_type = intersection["old_item_func"].type
                    intersection_points_type_index = inter_index

                # Draw points
                point = self.plotter.add_points(intersections[inter_index]["point_3d"], point_size=10.0)
                if inter_index == 0:
                    self.plotter.remove_actor(self.item_1_render_obj["points"][-1])
                    self.item_1_render_obj["points"][-1] = point
                else:
                    self.item_1_render_obj["points"].append(point)
                    
                # Draw curves
                next_intersect_point = None
                next_item = intersections[inter_index + 1] if inter_index + 1 < len(intersections) else None
                if next_item is not None:
                    next_intersect_point = intersections[inter_index + 1]["point_3d"]
                
                if next_item is None:
                    end_curve = Curve(self.plotter, self.sketch_instance.get_current_sketch_data(), self.sketch_instance)
                    point_start = intersection["point_3d"]
                    point_end = intersections[intersection_points_type_index]["point_3d"]
                    point_midle = get_directed_arc_midpoint(point_start, point_end, self.item_1.points[0], self.sketch_instance.active_normal)
                    end_curve.points = [point_start, point_end, point_midle]
                    end_curve.center = intersection["old_item_func"].points[0]
                    end_curve.create_curve_line(point_start, point_end, point_midle)
                    end_curve.stop_drawing()
                    sketch_functions.append(end_curve)
                    # Delete circle item
                    del sketch_functions[sketch_functions.index(self.item_1)]

                elif intersections[inter_index + 1]["old_item_func"] == intersection["old_item_func"]:
                    new_curve = Curve(self.plotter, self.sketch_instance.get_current_sketch_data(), self.sketch_instance)
                    point_start = intersection["point_3d"]
                    point_end = next_intersect_point
                    point_midle = get_directed_arc_midpoint(point_start, point_end, self.item_1.points[0], self.sketch_instance.active_normal)
                    new_curve.points = [point_start, point_end, point_midle]
                    new_curve.center = self.item_1.points[0]
                    new_curve.create_curve_line(point_start, point_end, point_midle)
                    new_curve.stop_drawing()
                    sketch_functions.append(new_curve)

                if intersection["old_item_func"].type == "line":
                    # Redraw line
                    self.plotter.remove_actor(intersection["old_item_func"].render_obj["lines"][intersection["index"]])
                    line1 = pv.Line(intersection["start_3d"], intersection["point_3d"])
                    line2 = pv.Line(intersection["point_3d"], intersection["end_3d"])
                    actor_line_1 = self.plotter.add_mesh(line1, color=random.choice(colors), line_width=2, pickable=True)
                    actor_line_2 = self.plotter.add_mesh(line2, color=random.choice(colors), line_width=2, pickable=True)
                    intersection["old_item_func"].render_obj["lines"][intersection["index"]] = actor_line_2
                    intersection["old_item_func"].render_obj["lines"].insert(intersection["index"], actor_line_1)
                    intersection["old_item_func"].points.insert(intersection["index"]+1, intersection["point_3d"])

                elif intersection["old_item_func"].type == "circle":
                    # Draw points
                    point = self.plotter.add_points(intersection["point_3d"], point_size=10.0)
                    # if inter_index == 0:
                    #     self.plotter.remove_actor(intersection["old_item_func"].render_obj["points"][-1])
                    #     intersection["old_item_func"].render_obj["points"][-1] = point
                    # else:
                    intersection["old_item_func"].render_obj["points"].append(point)

                    # Draw curves
                    next_intersect_point = None
                    next_item = intersections[inter_index + 1] if inter_index + 1 < len(intersections) else None
                    if next_item is not None:
                        next_intersect_point = intersections[inter_index + 1]["point_3d"]
                    
                    if next_item is None or next_item["old_item_func"] != intersection["old_item_func"]:
                        end_curve = Curve(self.plotter, self.sketch_instance.get_current_sketch_data(), self.sketch_instance)
                        point_start = intersection["point_3d"]
                        point_end = intersections[intersection_points_type_index]["point_3d"]
                        point_middle = get_directed_arc_midpoint(point_start, point_end, intersection["old_item_func"].points[0], self.sketch_instance.active_normal)
                        end_curve.points = [point_start, point_end, point_middle]
                        end_curve.center = intersection["old_item_func"].points[0]
                        end_curve.create_curve_line(point_start, point_end, point_middle)
                        end_curve.stop_drawing()
                        sketch_functions.append(end_curve)
                        # Delete circle item
                        self.plotter.remove_actor(intersection["old_item_func"].render_obj["lines"][0])
                        del sketch_functions[sketch_functions.index(intersection["old_item_func"])]

                    elif intersections[inter_index + 1]["old_item_func"] == intersection["old_item_func"]:
                        new_curve = Curve(self.plotter, self.sketch_instance.get_current_sketch_data(), self.sketch_instance)
                        point_start = intersection["point_3d"]
                        point_end = next_intersect_point
                        point_middle = get_directed_arc_midpoint(point_start, point_end, intersection["old_item_func"].points[0], self.sketch_instance.active_normal)
                        new_curve.points = [point_start, point_end, point_middle]
                        new_curve.center = intersection["old_item_func"].points[0]
                        new_curve.create_curve_line(point_start, point_end, point_middle)
                        new_curve.stop_drawing()
                        sketch_functions.append(new_curve)
                
                elif intersection["old_item_func"].type == "curve":
                    # skip next intersections
                    if inter_index - 1 >= intersection_points_type_index:
                        if intersections[inter_index - 1]["old_item_func"].type == "curve":
                            continue
                    
                    next_item = intersections[inter_index + 1] if inter_index + 1 < len(intersections) else None
                    next_intersect_point = None
                    if next_item is not None:
                        next_intersect_point = intersections[inter_index + 1]["point_3d"]
                        
                    if next_item is not None:
                        if intersections[inter_index + 1]["old_item_func"] == intersection["old_item_func"]:
                            _curve_intersections = [intersection["point_3d"], intersections[inter_index + 1]["point_3d"]]
                        else:
                            _curve_intersections = [intersection["point_3d"]]
                    else:
                        _curve_intersections = [intersection["point_3d"]]         
                        
                    # Remove curve actor
                    self.plotter.remove_actor(intersection["old_item_func"].render_obj["lines"][0])
                    # 3d to 2d points for curve splitting
                    old_curve_points_2d = [
                        self.sketch_instance.get_local_pos_in_sketch(pt) for pt in intersection["old_item_func"].points
                    ]
                    inter_curve_points = [
                        self.sketch_instance.get_local_pos_in_sketch(pt) for pt in _curve_intersections
                    ]
                    new_curves = self.split_arc(old_curve_points_2d, inter_curve_points)
                    
                    # Draw curves
                    new_functions = []
                    for points in new_curves:
                        points = [self.sketch_instance.get_global_pos_in_sketch(pt) for pt in points]
                        curve = Curve(self.plotter, self.sketch_instance.get_current_sketch_data(), self.sketch_instance)
                        point_start = points[0]
                        point_end = points[1]
                        curve.points = [point_start, point_end, points[2]]
                        curve.center = intersection["old_item_func"].center
                        curve.create_curve_line(point_start, point_end, points[2])
                        curve.stop_drawing()
                        new_functions.append(curve)
                        old_point = point_end
                    
                    # Delete curve item
                    del sketch_functions[sketch_functions.index(intersection["old_item_func"])]
                    sketch_functions.extend(new_functions)
                    _curve_intersections = None 
                    
        elif self.item_1.type == "curve":
            if intersections is None or len(intersections) == 0:
                # draw new_curve without intersections points
                point_start = self.item_1.points[0]
                point_end = self.item_1.points[1]
                point_middle = self.item_1.points[2]
                self.item_1.create_curve_line(point_start, point_end, point_middle)
                return
            
            intersection_type = None
            intersection_points_type_index = 0
            point_start = self.item_1.points[0]
            for inter_index, intersection in enumerate(intersections):
                if intersection_type != intersection["old_item_func"].type:
                    intersection_type = intersection["old_item_func"].type
                    intersection_points_type_index = inter_index
            
                # Draw points
                point = self.plotter.add_points(intersections[inter_index]["point_3d"], point_size=10.0)
                if inter_index == 0:
                    self.plotter.remove_actor(self.item_1_render_obj["points"][-1])
                    self.item_1_render_obj["points"][-1] = point
                else:
                    self.item_1_render_obj["points"].append(point)
                    
                skip = False
                if inter_index - 1 >= intersection_points_type_index:
                    if intersections[inter_index - 1]["old_item_func"].type == intersections[inter_index]["old_item_func"].type:
                        skip = True
                        
                if not skip:
                    next_item = intersections[inter_index + 1] if inter_index + 1 < len(intersections) else None
                    next_intersect_point = None
                    if next_item is not None:
                        next_intersect_point = intersections[inter_index + 1]["point_3d"]
                        
                    if next_item is not None:
                        if intersections[inter_index + 1]["old_item_func"] == intersection["old_item_func"]:
                            _curve_intersections = [intersection["point_3d"], intersections[inter_index + 1]["point_3d"]]
                        else:
                            _curve_intersections = [intersection["point_3d"]]
                    else:
                        _curve_intersections = [intersection["point_3d"]]
                    
                    # Draw curves
                    old_curve_points_2d = [
                        self.sketch_instance.get_local_pos_in_sketch(pt) for pt in self.item_1.points
                    ]
                    inter_curve_points = [
                        self.sketch_instance.get_local_pos_in_sketch(pt) for pt in _curve_intersections
                    ]
                    new_curves = self.split_arc(old_curve_points_2d, inter_curve_points)
                    del sketch_functions[-1]
                    
                    for points in new_curves:
                        points = [self.sketch_instance.get_global_pos_in_sketch(pt) for pt in points]
                        curve = Curve(self.plotter, self.sketch_instance.get_current_sketch_data(), self.sketch_instance)
                        point_start = points[0]
                        point_end = points[1]
                        curve.center = self.item_1.center
                        curve.points = [point_start, point_end, points[2]]
                        curve.create_curve_line(point_start, point_end, points[2])
                        curve.stop_drawing()
                        sketch_functions.append(curve)
                        old_point = point_end

                if intersection["old_item_func"].type == "line":
                    self.all_intersections.sort(key=lambda x: x["distance"])
                    # draw new_curve with intersections points
                    old_start_point = intersection["old_item_func"].points[intersection["index"]]
                    old_end_point = intersection["old_item_func"].points[intersection["index"] + 1]
                    for intersection_data in intersections:
                        intersection["old_item_func"].points.insert(intersection["index"] + 1, intersection_data["point_3d"])
                        line_1 = pv.Line(old_start_point, intersection_data["point_3d"])
                        actor_line_1 = self.plotter.add_mesh(line_1, color="blue", line_width=2, pickable=True)
                        intersection["old_item_func"].render_obj["lines"].append(actor_line_1)
                        old_start_point = intersection_data["point_3d"]
                        
                    line = pv.Line(old_start_point, old_end_point)
                    actor_line = self.plotter.add_mesh(line, color="blue", line_width=2, pickable=True)
                    intersection["old_item_func"].render_obj["lines"].append(actor_line)
                    old_point = intersection_data["point_3d"]
                        
                elif intersection["old_item_func"].type == "circle":
                    if not skip:
                        new_curve = Curve(self.plotter, self.sketch_instance.get_current_sketch_data(), self.sketch_instance)
                        point_start = _curve_intersections[0]
                        point_end = _curve_intersections[1]
                        point_middle = get_directed_arc_midpoint(point_start, point_end, intersection["old_item_func"].center, self.sketch_instance.active_normal)
                        new_curve.points = [point_start, point_end, point_middle]
                        new_curve.center = intersection["old_item_func"].center
                        new_curve.create_curve_line(point_start, point_end, point_middle)
                        new_curve.stop_drawing()
                        sketch_functions.append(new_curve)
                        
                        new_curve = Curve(self.plotter, self.sketch_instance.get_current_sketch_data(), self.sketch_instance)
                        point_start = _curve_intersections[1]
                        point_end = _curve_intersections[0]
                        point_middle = get_directed_arc_midpoint(point_start, point_end, intersection["old_item_func"].center, self.sketch_instance.active_normal)
                        new_curve.points = [point_start, point_end, point_middle]
                        new_curve.center = intersection["old_item_func"].center
                        new_curve.create_curve_line(point_start, point_end, point_middle)
                        new_curve.stop_drawing()
                        sketch_functions.append(new_curve)
                        
                        self.plotter.remove_actor(intersection["old_item_func"].render_obj["lines"][0])
                        del sketch_functions[sketch_functions.index(intersection["old_item_func"])]
                
                elif intersection["old_item_func"].type == "curve":
                    if not skip:
                        next_item = intersections[inter_index + 1] if inter_index + 1 < len(intersections) else None
                        next_intersect_point = None
                        if next_item is not None:
                            next_intersect_point = intersections[inter_index + 1]["point_3d"]
                            
                        if next_item is not None:
                            if intersections[inter_index + 1]["old_item_func"] == intersection["old_item_func"]:
                                _curve_intersections = [intersection["point_3d"], intersections[inter_index + 1]["point_3d"]]
                            else:
                                _curve_intersections = [intersection["point_3d"]]
                        else:
                            _curve_intersections = [intersection["point_3d"]]    
                        
                        # Draw curves
                        old_curve_points_2d = [
                            self.sketch_instance.get_local_pos_in_sketch(pt) for pt in intersection["old_item_func"].points
                        ]
                        new_curves = self.split_arc(old_curve_points_2d, inter_curve_points)
                        
                        # Draw curves
                        new_functions = []
                        for points in new_curves:
                            points = [self.sketch_instance.get_global_pos_in_sketch(pt) for pt in points]
                            curve = Curve(self.plotter, self.sketch_instance.get_current_sketch_data(), self.sketch_instance)
                            point_start = points[0]
                            point_end = points[1]
                            curve.points = [point_start, point_end, points[2]]
                            curve.center = intersection["old_item_func"].center
                            curve.create_curve_line(point_start, point_end, points[2])
                            curve.stop_drawing()
                            new_functions.append(curve)
                            old_point = point_end
                            
                        # Delete curve item
                        self.plotter.remove_actor(intersection["old_item_func"].render_obj["lines"][0])
                        del sketch_functions[sketch_functions.index(intersection["old_item_func"])]
                        sketch_functions.extend(new_functions)
                        
                _curve_intersections = None 
        
        elif self.item_1.type == "ellipse":
            if intersections is None or len(intersections) == 0:
                # draw new ellipse without intersections points
                self.item_1.create_arc_mesh(color="blue")
                return
            
            intersection_type = None
            intersection_points_type_index = 0
            _ellipse_intersections = []

            for inter_index, intersection in enumerate(intersections):
                if intersection_type != intersection["old_item_func"].type:
                    intersection_type = intersection["old_item_func"].type
                    intersection_points_type_index = inter_index
                # Draw points
                point = self.plotter.add_points(intersection["point_3d"], point_size=10.0,)
                # print("/////", intersections[inter_index]["point_3d"])
                self.item_1_render_obj["points"].append(point)
                    
                skip = False
                if inter_index - 1 >= intersection_points_type_index:
                    if intersections[inter_index - 1]["old_item_func"] == intersections[inter_index]["old_item_func"]:
                        skip = True
                    else:
                        _ellipse_intersections = []
                        
                if not skip:
                    for add_inter_index in range(inter_index, len(intersections)):
                        next_item = intersections[add_inter_index] if add_inter_index < len(intersections) else None
                        next_intersect_point = None
                        if next_item is not None:
                            next_intersect_point = intersections[add_inter_index]
                            _ellipse_intersections.append(next_intersect_point)
                            
                        if add_inter_index + 1 <= len(intersections) - 1:
                            if intersections[add_inter_index + 1]["old_item_func"] != intersection["old_item_func"]:
                                break
                
                
                ellipse_arc = Ellipse(self.plotter, self.sketch_instance.get_current_sketch_data(), self.sketch_instance)
                ellipse_arc.center = self.item_1.center
                ellipse_arc.major_radius = self.item_1.major_radius
                ellipse_arc.minor_radius = self.item_1.minor_radius
                ellipse_arc.major_axis_vector = self.item_1.major_axis_vector
                ellipse_arc.is_closed = False
                next_inter_index = inter_index + 1 if inter_index + 1 < len(intersections) else 0
                
                sorted_intersections = self.sort_intersections_by_angle(
                    intersections, 
                    self.item_1.center,
                    self.sketch_instance.active_normal
                )
                
                if inter_index == 0:
                    if len(intersections) > 1:
                        del sketch_functions[-1]

                ellipse_arc.set_arc_from_intersections(
                    sorted_intersections[inter_index]["point_3d"],  # [3.5, 2.0, 0]
                    sorted_intersections[next_inter_index]["point_3d"],  # [3.5, -2.0, 0]
                    is_major_arc=False    # False = меньшая дуга, True = большая дуга
                )
                ellipse_arc.create_arc_mesh(color=random.choice(colors))
                sketch_functions.append(ellipse_arc)
                
                if intersection["old_item_func"].type == "line":
                    if not skip:
                        _ellipse_intersections.sort(key=lambda x: x["distance"])
                        _ellipse_intersections.reverse()
                        # draw new_curve with intersections points
                        old_start_point = intersection["old_item_func"].points[intersection["index"]]
                        old_end_point = intersection["old_item_func"].points[intersection["index"] + 1]
                        for intersection_data in _ellipse_intersections:
                            intersection["old_item_func"].points.insert(intersection["index"] + 1, intersection_data["point_3d"])
                            line_1 = pv.Line(old_start_point, intersection_data["point_3d"])
                            actor_line_1 = self.plotter.add_mesh(line_1, color="blue", line_width=2, pickable=True)
                            intersection["old_item_func"].render_obj["lines"].append(actor_line_1)
                            old_start_point = intersection_data["point_3d"]
                            
                        line = pv.Line(old_start_point, old_end_point)
                        actor_line = self.plotter.add_mesh(line, color="blue", line_width=2, pickable=True)
                        intersection["old_item_func"].render_obj["lines"].append(actor_line)
                        old_point = intersection_data["point_3d"]
                        
                elif intersection["old_item_func"].type == "circle":
                    if not skip:
                        sorted_circle_intersections = self.sort_intersections_by_angle(
                            _ellipse_intersections, 
                            intersection["old_item_func"].center,
                            self.sketch_instance.active_normal
                        )
                        self.plotter.remove_actor(intersection["old_item_func"].render_obj["lines"][0])
                        del sketch_functions[sketch_functions.index(intersection["old_item_func"])]
                        for inter_i, inter in enumerate(sorted_circle_intersections):
                            if inter_i == 0:
                                pass
                            
                            # draw curves
                            new_curve = Curve(self.plotter, self.sketch_instance.get_current_sketch_data(), self.sketch_instance)
                            point_start = sorted_circle_intersections[inter_i - 1]["point_3d"]
                            point_end = inter["point_3d"]
                            point_middle = get_directed_arc_midpoint(point_start, point_end, intersection["old_item_func"].center, self.sketch_instance.active_normal)
                            new_curve.points = [point_start, point_end, point_middle]
                            new_curve.center = intersection["old_item_func"].center
                            new_curve.create_curve_line(point_start, point_end, point_middle)
                            sketch_functions.append(new_curve)
                            
                        point_start = sorted_circle_intersections[inter_i]["point_3d"]
                        point_end = sorted_circle_intersections[0]["point_3d"]
                        point_middle = get_directed_arc_midpoint(point_start, point_end, intersection["old_item_func"].center, self.sketch_instance.active_normal)
                        new_curve.points = [point_start, point_end, point_middle]
                        new_curve.center = intersection["old_item_func"].center
                        new_curve.create_curve_line(point_start, point_end, point_middle)
                        sketch_functions.append(new_curve)
                                
                        

    #  addition functions
    @staticmethod
    def find_lines_intersection(
        a1: np.ndarray,  # начало отрезка 1
        a2: np.ndarray,  # конец отрезка 1
        b1: np.ndarray,  # начало отрезка 2
        b2: np.ndarray,  # конец отрезка 2
        eps: float = 1e-10
    ):
        """
        Находит точку пересечения двух отрезков в 2D.

        Args:
            a1, a2: Начало и конец первого отрезка.
            b1, b2: Начало и конец второго отрезка.
            eps: Порог параллельности.

        Returns:
            Точка пересечения или None.
        """
        connection_point_start = a1.copy()
        connection_point_end = a2.copy()
        
        a1, a2, b1, b2 = map(np.asarray, (a1, a2, b1, b2))
        

        d1 = a2 - a1  # направление отрезка 1
        d2 = b2 - b1  # направление отрезка 2

        # Решаем: a1 + t*d1 = b1 + s*d2
        cross = d1[0] * d2[1] - d1[1] * d2[0]  # псевдоскалярное произведение

        if abs(cross) < eps:
            return None  # параллельные или совпадающие

        delta = b1 - a1
        t = (delta[0] * d2[1] - delta[1] * d2[0]) / cross
        s = (delta[0] * d1[1] - delta[1] * d1[0]) / cross

        # Проверяем что пересечение лежит внутри обоих отрезков [0, 1]
        if 0 <= t <= 1 and 0 <= s <= 1:
            result = a1 + t * d1
            if np.allclose(result, connection_point_start, atol=eps) or np.allclose(result, connection_point_end, atol=eps):
                return None
            return result

        return None  # отрезки не пересекаются (пересекаются их продолжения)
    
    @staticmethod
    def find_line_circle_intersections(
        p1: np.ndarray, # Старт отрезка [x, y]
        p2: np.ndarray, # Конец отрезка [x, y]
        center: np.ndarray, # Центр окружности [x, y]
        radius: float,
        eps: float = 1e-10
    ) -> list:
        """Возвращает список точек пересечения (от 0 до 2)"""
        # Смещаем систему координат так, чтобы центр окружности был в (0,0)
        d = p2 - p1
        f = p1 - center

        # Решаем уравнение At² + Bt + C = 0
        a = np.dot(d, d)
        b = 2 * np.dot(f, d)
        c = np.dot(f, f) - radius**2

        discriminant = b**2 - 4*a*c

        if discriminant < 0:
            return [] # Нет пересечений

        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2 * a)
        t2 = (-b + discriminant) / (2 * a)

        results = []
        # Проверяем, лежат ли корни t в пределах отрезка [0, 1]
        for t in [t1, t2]:
            if 0 <= t <= 1:
                # Исключаем точки, которые являются концами отрезка (как в вашем find_intersection)
                if eps < t < 1 - eps:
                    results.append(p1 + t * d)
        
        return results
    
    @staticmethod
    def sort_intersections_by_angle(intersections, center, normal):
        """
        Сортирует точки пересечения по углу вокруг центра окружности
        """
        center = np.array(center)
        normal = np.array(normal)
        normal = normal / np.linalg.norm(normal)

        # Строим локальную систему координат в плоскости окружности
        v_first = np.array(intersections[0]["point_3d"]) - center
        x_axis = v_first / np.linalg.norm(v_first)
        y_axis = np.cross(normal, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        def get_angle(intersection):
            v = np.array(intersection["point_3d"]) - center
            x = np.dot(v, x_axis)
            y = np.dot(v, y_axis)
            return np.arctan2(y, x)

        return sorted(intersections, key=get_angle)

    def find_line_curve_intersections(self, line_obj:Union['Line'], curve_obj:Union['Curve']) -> list:
        """
        Возвращает список точек пересечения отрезка line_obj с дугой curve_obj.
        Результат: [] / [p] / [p1, p2]
        """
        
        if len(line_obj.points) < 2 or len(curve_obj.points) < 3:
            return []

        # ---------- line ----------
        a = np.asarray(self.sketch_instance.get_local_pos_in_sketch(line_obj.points[-2]), dtype=float)
        b = np.asarray(self.sketch_instance.get_local_pos_in_sketch(line_obj.points[-1]), dtype=float)

        # ---------- curve ----------
        p_start = np.asarray(self.sketch_instance.get_local_pos_in_sketch(curve_obj.points[0]), dtype=float)
        p_end   = np.asarray(self.sketch_instance.get_local_pos_in_sketch(curve_obj.points[1]), dtype=float)
        p_mid   = np.asarray(self.sketch_instance.get_local_pos_in_sketch(curve_obj.points[2]), dtype=float)
        
        center = self.sketch_instance.get_local_pos_in_sketch(curve_obj.center)
        radius = np.linalg.norm(p_start - center)
        if center is None or radius is None or radius < 1e-6:
            return []

        center = np.asarray(center, dtype=float)

        # Параметрическое уравнение отрезка:
        # P(t) = a + t * (b - a), t in [0, 1]
        d = b - a
        A = np.dot(d, d)

        if A < 1e-6:
            return []

        f = a - center

        # Решаем |a + t*d - center|^2 = r^2
        B = 2.0 * np.dot(f, d)
        C = np.dot(f, f) - radius * radius

        discriminant = B * B - 4.0 * A * C

        if discriminant < -1e-6:
            return []

        ts = []

        if abs(discriminant) <= 1e-6:
            # касание
            t = -B / (2.0 * A)
            ts.append(t)
        else:
            sqrt_disc = np.sqrt(max(discriminant, 0.0))
            t1 = (-B - sqrt_disc) / (2.0 * A)
            t2 = (-B + sqrt_disc) / (2.0 * A)
            ts.extend([t1, t2])

        result = []

        for t in ts:
            # точка должна лежать на ОТРЕЗКЕ, а не на бесконечной линии
            if -1e-6 <= t <= 1.0 + 1e-6:
                q = a + t * d

                # проверяем, что точка лежит именно на дуге
                if self.__point_on_arc(q, p_start, p_mid, p_end, center):
                    # убрать дубликаты
                    if not any(np.linalg.norm(q - p) <= 1e-6 for p in result):
                        result.append(q)
                        
        result = [self.sketch_instance.get_global_pos_in_sketch(p) for p in result]

        return result
    
    
    def split_arc(self, arc, intersection_points):
        """
        Разделяет дугу точками пересечения на новые дуги.
        
        Args:
            arc: список из 3 точек [start, end, middle] - исходная дуга
            intersection_points: список из 1 или 2 точек пересечения
        
        Returns:
            список новых дуг, каждая в формате [start, end, middle]
        """
        if not intersection_points or len(intersection_points) == 0:
            return [arc]
        
        start = np.array(arc[0])
        end = np.array(arc[1])
        middle = np.array(arc[2])
        
        # Находим центр и радиус дуги
        center = self.get_arc_center(start, middle, end)
        radius = np.linalg.norm(start - center)
        
        # Определяем направление дуги (по часовой или против)
        is_clockwise = self.get_arc_direction(center, start, middle, end)
        
        # Параметризуем точки пересечения
        points_with_params = []
        for point in intersection_points:
            point = np.array(point)
            t = self.get_parameter_on_arc(center, start, end, point, is_clockwise)
            points_with_params.append((t, point))
        
        # Сортируем по параметру
        points_with_params.sort(key=lambda x: x[0])
        
        # Создаем упорядоченный список точек
        ordered_points = [start] + [p[1] for p in points_with_params] + [end]
        
        # Создаем новые дуги
        new_arcs = []
        for i in range(len(ordered_points) - 1):
            arc_start = ordered_points[i]
            arc_end = ordered_points[i + 1]
            
            # Вычисляем среднюю точку новой дуги
            arc_middle = self.get_arc_middle_point(center, radius, arc_start, arc_end, is_clockwise)
            
            new_arcs.append([
                arc_start.tolist() if isinstance(arc_start, np.ndarray) else arc_start,
                arc_end.tolist() if isinstance(arc_end, np.ndarray) else arc_end,
                arc_middle.tolist()
            ])
        
        return new_arcs

    @staticmethod
    def get_arc_center(p1, p2, p3):
        """
        Находит центр окружности по трем точкам на дуге.
        """
        p1 = np.array(p1)
        p2 = np.array(p2)
        p3 = np.array(p3)
        
        ax, ay = p1
        bx, by = p2
        cx, cy = p3
        
        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        
        if abs(d) < 1e-10:
            raise ValueError("Точки лежат на одной прямой")
        
        ux = ((ax**2 + ay**2) * (by - cy) + 
            (bx**2 + by**2) * (cy - ay) + 
            (cx**2 + cy**2) * (ay - by)) / d
        
        uy = ((ax**2 + ay**2) * (cx - bx) + 
            (bx**2 + by**2) * (ax - cx) + 
            (cx**2 + cy**2) * (bx - ax)) / d
        
        return np.array([ux, uy])


    @staticmethod
    def get_arc_direction(center, start, middle, end):
        """
        Определяет направление дуги (по часовой стрелке или против).
        
        Returns:
            bool: True если по часовой стрелке, False если против
        """
        # Используем векторное произведение для определения направления
        # от start к middle и от middle к end
        
        v1 = middle - start
        v2 = end - middle
        
        # Векторное произведение (z-компонента в 2D)
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        
        # Альтернативный метод через углы
        def get_angle(p):
            vec = p - center
            return np.arctan2(vec[1], vec[0])
        
        angle_start = get_angle(start)
        angle_middle = get_angle(middle)
        angle_end = get_angle(end)
        
        # Нормализуем углы в диапазон [0, 2π)
        angle_start = angle_start % (2 * np.pi)
        angle_middle = angle_middle % (2 * np.pi)
        angle_end = angle_end % (2 * np.pi)
        
        # Вычисляем угловые расстояния
        def angular_distance(from_angle, to_angle, clockwise):
            """Вычисляет угловое расстояние"""
            if clockwise:
                if to_angle <= from_angle:
                    return from_angle - to_angle
                else:
                    return from_angle + (2 * np.pi - to_angle)
            else:
                if to_angle >= from_angle:
                    return to_angle - from_angle
                else:
                    return (2 * np.pi - from_angle) + to_angle
        
        # Проверяем, лежит ли middle между start и end по часовой
        dist_cw_to_middle = angular_distance(angle_start, angle_middle, True)
        dist_cw_to_end = angular_distance(angle_start, angle_end, True)
        
        # Проверяем, лежит ли middle между start и end против часовой
        dist_ccw_to_middle = angular_distance(angle_start, angle_middle, False)
        dist_ccw_to_end = angular_distance(angle_start, angle_end, False)
        
        # Если middle находится между start и end по часовой, и это кратчайший путь
        if dist_cw_to_middle < dist_cw_to_end and dist_cw_to_end < np.pi:
            return True
        elif dist_ccw_to_middle < dist_ccw_to_end and dist_ccw_to_end < np.pi:
            return False
        
        # Для больших дуг (> 180°)
        if dist_cw_to_middle < dist_cw_to_end:
            return True
        else:
            return False

    @staticmethod
    def get_parameter_on_arc(center, start, end, point, is_clockwise):
        """
        Находит параметр t для точки на дуге (от 0 до 1).
        """
        def get_angle(p):
            vec = p - center
            return np.arctan2(vec[1], vec[0])
        
        angle_start = get_angle(start)
        angle_end = get_angle(end)
        angle_point = get_angle(point)
        
        # Нормализуем в [0, 2π)
        angle_start = angle_start % (2 * np.pi)
        angle_end = angle_end % (2 * np.pi)
        angle_point = angle_point % (2 * np.pi)
        
        if is_clockwise:
            # По часовой стрелке
            if angle_end > angle_start:
                angle_end -= 2 * np.pi
            if angle_point > angle_start:
                angle_point -= 2 * np.pi
            if angle_point < angle_end:
                angle_point += 2 * np.pi
                
            total_angle = angle_start - angle_end
            point_angle = angle_start - angle_point
        else:
            # Против часовой стрелки
            if angle_end < angle_start:
                angle_end += 2 * np.pi
            if angle_point < angle_start:
                angle_point += 2 * np.pi
            if angle_point > angle_end:
                angle_point -= 2 * np.pi
                
            total_angle = angle_end - angle_start
            point_angle = angle_point - angle_start
        
        if abs(total_angle) < 1e-10:
            return 0.5
        
        t = point_angle / total_angle
        return np.clip(t, 0, 1)

    @staticmethod
    def get_arc_middle_point(center, radius, start, end, is_clockwise):
        """
        Вычисляет среднюю точку для новой дуги с учетом направления.
        """
        def get_angle(p):
            vec = p - center
            return np.arctan2(vec[1], vec[0])
        
        angle_start = get_angle(start)
        angle_end = get_angle(end)
        
        # Нормализуем в [0, 2π)
        angle_start = angle_start % (2 * np.pi)
        angle_end = angle_end % (2 * np.pi)
        
        if is_clockwise:
            # По часовой стрелке
            if angle_end > angle_start:
                angle_end -= 2 * np.pi
            angle_mid = (angle_start + angle_end) / 2
        else:
            # Против часовой стрелки
            if angle_end < angle_start:
                angle_end += 2 * np.pi
            angle_mid = (angle_start + angle_end) / 2
        
        # Вычисляем среднюю точку
        middle_point = center + radius * np.array([np.cos(angle_mid), np.sin(angle_mid)])
        
        return middle_point
    
    def find_curve_circle_intersections(self, curve_obj: Union['Curve'], circle_obj: Union['Circle']) -> list:
        """
        Возвращает список точек пересечения дуги curve_obj с окружностью circle_obj.
        
        Args:
            curve_obj: дуга в формате [start, end, middle]
            circle_obj: окружность в формате [center_point, additional_point]
        
        Returns:
            список точек пересечения: [] / [p] / [p1, p2]
        """
        
        if len(curve_obj.points) < 3:
            return []
        
        if len(circle_obj.points) < 2:
            return []
        
        # ---------- curve (дуга) ----------
        p_start = np.asarray(self.sketch_instance.get_local_pos_in_sketch(curve_obj.points[0]), dtype=float)
        p_end   = np.asarray(self.sketch_instance.get_local_pos_in_sketch(curve_obj.points[1]), dtype=float)
        p_mid   = np.asarray(self.sketch_instance.get_local_pos_in_sketch(curve_obj.points[2]), dtype=float)
        
        center1 = self.sketch_instance.get_local_pos_in_sketch(curve_obj.center)
        radius1 = np.linalg.norm(p_start - center1)
        
        if center1 is None or radius1 is None or radius1 < 1e-6:
            return []
        
        center1 = np.asarray(center1, dtype=float)
        
        # ---------- circle (окружность) ----------
        center2 = np.asarray(self.sketch_instance.get_local_pos_in_sketch(circle_obj.points[0]), dtype=float)  # start point = центр
        additional_point = np.asarray(self.sketch_instance.get_local_pos_in_sketch(circle_obj.points[1]), dtype=float)
        radius2 = np.linalg.norm(additional_point - center2)
        
        if radius2 < 1e-6:
            return []
        
        # Расстояние между центрами
        d = np.linalg.norm(center2 - center1)
        
        # Проверка на возможность пересечения
        # Окружности не пересекаются, если:
        # 1) слишком далеко друг от друга: d > r1 + r2
        # 2) одна внутри другой: d < |r1 - r2|
        
        if d > radius1 + radius2 + 1e-6:
            return []  # слишком далеко
        
        if d < abs(radius1 - radius2) - 1e-6:
            return []  # одна внутри другой
        
        # Совпадающие окружности
        if d < 1e-6 and abs(radius1 - radius2) < 1e-6:
            return []  # бесконечно много точек пересечения
        
        # Вычисляем точки пересечения двух окружностей
        # Используем формулу из аналитической геометрии
        
        # Расстояние от center1 до линии, соединяющей точки пересечения
        a = (radius1**2 - radius2**2 + d**2) / (2.0 * d)
        
        # Проверка на касание
        h_squared = radius1**2 - a**2
        
        if h_squared < -1e-6:
            return []  # нет пересечений
        
        # Точка на линии между центрами
        direction = (center2 - center1) / d
        p2 = center1 + a * direction
        
        result = []
        
        if abs(h_squared) <= 1e-6:
            # Касание (одна точка)
            q = p2
            
            if self.__point_on_arc(q, p_start, p_mid, p_end, center1):
                result.append(self.sketch_instance.get_global_pos_in_sketch(q))
        else:
            # Два пересечения
            h = np.sqrt(h_squared)
            
            # Перпендикулярное направление
            perp = np.array([-direction[1], direction[0]])
            
            # Две точки пересечения
            q1 = p2 + h * perp
            q2 = p2 - h * perp
            
            # Проверяем, лежат ли точки на дуге
            if self.__point_on_arc(q1, p_start, p_mid, p_end, center1):
                result.append(q1)
            
            if self.__point_on_arc(q2, p_start, p_mid, p_end, center1):
                # Проверка на дубликаты
                if not any(np.linalg.norm(q2 - p) <= 1e-6 for p in result):
                    result.append(q2)
        
        result = [self.sketch_instance.get_global_pos_in_sketch(point) for point in result]
        return result
    
    
    def __point_on_arc(self, point, p_start, p_mid, p_end, center):
        """
        Проверяет, лежит ли точка на дуге (а не на полной окружности).
        
        Args:
            point: проверяемая точка (numpy array)
            p_start: начало дуги (numpy array)
            p_mid: средняя точка дуги (numpy array)
            p_end: конец дуги (numpy array)
            center: центр дуги (numpy array)
        
        Returns:
            bool: True если точка на дуге
        """
        point = np.asarray(point, dtype=float)
        
        # Проверяем, что точка лежит на окружности
        dist_to_center = np.linalg.norm(point - center)
        radius = np.linalg.norm(p_start - center)
        
        if abs(dist_to_center - radius) > 1e-6:
            return False
        
        def get_angle(p):
            """Вычисляет угол от центра до точки"""
            vec = p - center
            return np.arctan2(vec[1], vec[0])
        
        def is_angle_between(angle, start_angle, end_angle, is_ccw):
            """
            Проверяет, находится ли угол между start_angle и end_angle
            в заданном направлении (по или против часовой стрелки)
            """
            # Нормализуем все углы относительно start_angle
            def normalize_relative(a, reference):
                diff = a - reference
                while diff < 0:
                    diff += 2 * np.pi
                while diff >= 2 * np.pi:
                    diff -= 2 * np.pi
                return diff
            
            angle_norm = normalize_relative(angle, start_angle)
            end_norm = normalize_relative(end_angle, start_angle)
            
            if is_ccw:
                # Против часовой стрелки: угол должен быть между 0 и end_norm
                return -1e-6 <= angle_norm <= end_norm + 1e-6
            else:
                # По часовой стрелке: угол должен быть между end_norm и 2π
                return angle_norm >= end_norm - 1e-6 or angle_norm <= 1e-6
        
        # Получаем углы всех точек
        angle_start = get_angle(p_start)
        angle_mid = get_angle(p_mid)
        angle_end = get_angle(p_end)
        angle_point = get_angle(point)
        
        # Определяем направление дуги через векторное произведение
        # Вектор от центра к start
        v_start = p_start - center
        # Вектор от центра к mid
        v_mid = p_mid - center
        
        # Векторное произведение (для 2D это скаляр - z-компонента)
        cross_product = v_start[0] * v_mid[1] - v_start[1] * v_mid[0]
        
        # Если cross_product > 0, то дуга идет против часовой стрелки
        # Если cross_product < 0, то дуга идет по часовой стрелке
        is_ccw = cross_product > 0
        
        # Проверяем, находится ли точка на дуге
        return is_angle_between(angle_point, angle_start, angle_end, is_ccw)
    


    def find_circle_circle_intersections(self, circle_obj1: Union['Circle'], circle_obj2: Union['Circle']) -> list:
        """
        Возвращает список точек пересечения окружности circle_obj1 с окружностью circle_obj2.
        
        Args:
            circle_obj1: первая окружность в формате [center_point, additional_point]
            circle_obj2: вторая окружность в формате [center_point, additional_point]
        
        Returns:
            список точек пересечения: [] / [p] / [p1, p2]
        """
        
        if len(circle_obj1.points) < 2:
            return []
        
        if len(circle_obj2.points) < 2:
            return []
        
        # ---------- circle 1 ----------
        center1 = np.asarray(self.sketch_instance.get_local_pos_in_sketch(circle_obj1.points[0]), dtype=float)
        additional_point1 = np.asarray(self.sketch_instance.get_local_pos_in_sketch(circle_obj1.points[1]), dtype=float)
        radius1 = np.linalg.norm(additional_point1 - center1)
        
        if radius1 < 1e-6:
            return []
        
        # ---------- circle 2 ----------
        center2 = np.asarray(self.sketch_instance.get_local_pos_in_sketch(circle_obj2.points[0]), dtype=float)
        additional_point2 = np.asarray(self.sketch_instance.get_local_pos_in_sketch(circle_obj2.points[1]), dtype=float)
        radius2 = np.linalg.norm(additional_point2 - center2)
        
        if radius2 < 1e-6:
            return []
        
        # Расстояние между центрами
        d = np.linalg.norm(center2 - center1)
        
        # Проверка на возможность пересечения
        # Окружности не пересекаются, если:
        # 1) слишком далеко друг от друга: d > r1 + r2
        # 2) одна внутри другой: d < |r1 - r2|
        
        if d > radius1 + radius2 + 1e-6:
            return []  # слишком далеко
        
        if d < abs(radius1 - radius2) - 1e-6:
            return []  # одна внутри другой
        
        # Совпадающие окружности
        if d < 1e-6 and abs(radius1 - radius2) < 1e-6:
            return []  # бесконечно много точек пересечения (совпадают)
        
        # Проверка на концентрические окружности (одинаковый центр, разные радиусы)
        if d < 1e-6:
            return []  # нет пересечений
        
        # Вычисляем точки пересечения двух окружностей
        # Используем формулу из аналитической геометрии
        
        # Расстояние от center1 до линии, соединяющей точки пересечения
        a = (radius1**2 - radius2**2 + d**2) / (2.0 * d)
        
        # Проверка на касание
        h_squared = radius1**2 - a**2
        
        if h_squared < -1e-6:
            return []  # нет пересечений (не должно произойти после предыдущих проверок)
        
        # Точка на линии между центрами
        direction = (center2 - center1) / d
        p2 = center1 + a * direction
        
        result = []
        
        if abs(h_squared) <= 1e-6:
            # Касание (одна точка)
            q = p2
            result.append(self.sketch_instance.get_global_pos_in_sketch(q))
        else:
            # Два пересечения
            h = np.sqrt(h_squared)
            
            # Перпендикулярное направление
            perp = np.array([-direction[1], direction[0]])
            
            # Две точки пересечения
            q1 = p2 + h * perp
            q2 = p2 - h * perp
            
            result.append(self.sketch_instance.get_global_pos_in_sketch(q1))
            result.append(self.sketch_instance.get_global_pos_in_sketch(q2))
        
        return result
    
    def find_curve_curve_intersections(self, curve_obj1: Curve, curve_obj2: Curve) -> list:
        """
        Возвращает список точек пересечения дуги curve_obj1 с дугой curve_obj2.
        
        Args:
            curve_obj1: первая дуга в формате [start, end, middle]
            curve_obj2: вторая дуга в формате [start, end, middle]
        
        Returns:
            список точек пересечения: [] / [p] / [p1, p2]
        """
        
        if len(curve_obj1.points) < 3:
            return []
        
        if len(curve_obj2.points) < 3:
            return []
        
        # ---------- curve 1 (первая дуга) ----------
        p1_start = np.asarray(self.sketch_instance.get_local_pos_in_sketch(curve_obj1.points[0]), dtype=float)
        p1_end   = np.asarray(self.sketch_instance.get_local_pos_in_sketch(curve_obj1.points[1]), dtype=float)
        p1_mid   = np.asarray(self.sketch_instance.get_local_pos_in_sketch(curve_obj1.points[2]), dtype=float)
        
        center1 = self.sketch_instance.get_local_pos_in_sketch(curve_obj1.center)
        radius1 = np.linalg.norm(p1_start - center1)
        
        if center1 is None or radius1 is None or radius1 < 1e-6:
            return []
        
        center1 = np.asarray(center1, dtype=float)
        
        # ---------- curve 2 (вторая дуга) ----------
        p2_start = np.asarray(self.sketch_instance.get_local_pos_in_sketch(curve_obj2.points[0]), dtype=float)
        p2_end   = np.asarray(self.sketch_instance.get_local_pos_in_sketch(curve_obj2.points[1]), dtype=float)
        p2_mid   = np.asarray(self.sketch_instance.get_local_pos_in_sketch(curve_obj2.points[2]), dtype=float)
        
        center2 = self.sketch_instance.get_local_pos_in_sketch(curve_obj2.center)
        radius2 = np.linalg.norm(p2_start - center2)
        
        if center2 is None or radius2 is None or radius2 < 1e-6:
            return []
        
        center2 = np.asarray(center2, dtype=float)
        
        # Расстояние между центрами
        d = np.linalg.norm(center2 - center1)
        
        # Проверка на возможность пересечения окружностей
        if d > radius1 + radius2 + 1e-6:
            return []  # слишком далеко
        
        if d < abs(radius1 - radius2) - 1e-6:
            return []  # одна внутри другой
        
        # Совпадающие окружности
        if d < 1e-6 and abs(radius1 - radius2) < 1e-6:
            return []  # бесконечно много точек пересечения
        
        # Проверка на концентрические окружности
        if d < 1e-6:
            return []  # нет пересечений
        
        # Вычисляем точки пересечения двух окружностей
        a = (radius1**2 - radius2**2 + d**2) / (2.0 * d)
        h_squared = radius1**2 - a**2
        
        if h_squared < -1e-6:
            return []  # нет пересечений
        
        # Точка на линии между центрами
        direction = (center2 - center1) / d
        p_mid = center1 + a * direction
        
        result = []
        
        if abs(h_squared) <= 1e-6:
            # Касание (одна точка)
            q = p_mid
            
            # Проверяем, что точка лежит на обеих дугах
            if (self.__point_on_arc(q, p1_start, p1_mid, p1_end, center1) and
                self.__point_on_arc(q, p2_start, p2_mid, p2_end, center2)):
                result.append(q)
        else:
            # Два пересечения окружностей
            h = np.sqrt(h_squared)
            
            # Перпендикулярное направление
            perp = np.array([-direction[1], direction[0]])
            
            # Две точки пересечения окружностей
            q1 = p_mid + h * perp
            q2 = p_mid - h * perp
            
            # Проверяем, лежит ли первая точка на обеих дугах
            if (self.__point_on_arc(q1, p1_start, p1_mid, p1_end, center1) and
                self.__point_on_arc(q1, p2_start, p2_mid, p2_end, center2)):
                result.append(q1)
            
            # Проверяем, лежит ли вторая точка на обеих дугах
            if (self.__point_on_arc(q2, p1_start, p1_mid, p1_end, center1) and
                self.__point_on_arc(q2, p2_start, p2_mid, p2_end, center2)):
                # Проверка на дубликаты
                if not any(np.linalg.norm(q2 - p) <= 1e-6 for p in result):
                    result.append(q2)
        
        # Преобразуем обратно в глобальные координаты
        result = [self.sketch_instance.get_global_pos_in_sketch(point) for point in result]
        return result
    
    def find_ellipse_ellipse_intersections(
        self,
        ellipse_obj1: Ellipse,
        ellipse_obj2: Ellipse,
    ) -> list:
        """
        Находит точки пересечения двух эллипсов алгебраическим методом.
        Возвращает список точек в глобальных координатах.
        """
        
        def _angle_in_arc(t: float, start: float, end: float, is_closed: bool) -> bool:
            """Проверяет, лежит ли угол t в пределах дуги [start, end]"""
            if is_closed:
                return True
            
            t = t % (2 * np.pi)
            start = start % (2 * np.pi)
            end = end % (2 * np.pi)
            
            if start <= end:
                return start <= t <= end
            else:  # дуга переходит через 0
                return t >= start or t <= end
    
        # elipse1 data
        elipse1_center = self.sketch_instance.get_local_pos_in_sketch(ellipse_obj1.center)
        local_vector1 = self.sketch_instance.transform_to_local_vector_in_sketch(ellipse_obj1.major_axis_vector)
        elipse1_angle = np.degrees(np.arctan2(local_vector1[1], local_vector1[0]))

        # elipse2 data
        elipse2_center = self.sketch_instance.get_local_pos_in_sketch(ellipse_obj2.center)
        local_vector2 = self.sketch_instance.transform_to_local_vector_in_sketch(ellipse_obj2.major_axis_vector)
        elipse2_angle = np.degrees(np.arctan2(local_vector2[1], local_vector2[0]))
        
        found_points = []
        
        if ellipse_obj1.is_closed:
            points = np.linspace(0, 2*np.pi, 200)
        else:
            start1 = ellipse_obj1.start_angle
            end1 = ellipse_obj1.end_angle
            if end1 <= start1:
                end1 += 2 * np.pi
            points = np.linspace(start1, end1, 200)
            
        old_f = None
        for index, t in enumerate(points):
            x1 = elipse1_center[0] + ellipse_obj1.major_radius * np.cos(t) * np.cos(np.radians(elipse1_angle)) - ellipse_obj1.minor_radius * np.sin(t) * np.sin(np.radians(elipse1_angle))
            y1 = elipse1_center[1] + ellipse_obj1.major_radius * np.cos(t) * np.sin(np.radians(elipse1_angle)) + ellipse_obj1.minor_radius * np.sin(t) * np.cos(np.radians(elipse1_angle))
            
            f = ((x1 - elipse2_center[0]) * np.cos(np.radians(elipse2_angle)) + (y1 - elipse2_center[1]) * np.sin(np.radians(elipse2_angle)))**2 / ellipse_obj2.major_radius**2 + \
                (-(x1 - elipse2_center[0]) * np.sin(np.radians(elipse2_angle)) + (y1 - elipse2_center[1]) * np.cos(np.radians(elipse2_angle)))**2 / ellipse_obj2.minor_radius**2 - 1

            if old_f is not None:
                if (f < 0) != (old_f < 0):
                    old_f = f
                    minimum_point = None
                    old_f2 = None
                    for t2 in np.linspace(points[index-1], t, 200):
                        x1 = elipse1_center[0] + ellipse_obj1.major_radius * np.cos(t2) * np.cos(np.radians(elipse1_angle)) - ellipse_obj1.minor_radius * np.sin(t2) * np.sin(np.radians(elipse1_angle))
                        y1 = elipse1_center[1] + ellipse_obj1.major_radius * np.cos(t2) * np.sin(np.radians(elipse1_angle)) + ellipse_obj1.minor_radius * np.sin(t2) * np.cos(np.radians(elipse1_angle))

                        # f(t) = ((x(t)-cx2)·cos θ2 + (y(t)-cy2)·sin θ2)² / a2² + (-(x(t)-cx2)·sin θ2 + (y(t)-cy2)·cos θ2)² / b2²  − 1,
                        f2 = ((x1 - elipse2_center[0]) * np.cos(np.radians(elipse2_angle)) + (y1 - elipse2_center[1]) * np.sin(np.radians(elipse2_angle)))**2 / ellipse_obj2.major_radius**2 + \
                            (-(x1 - elipse2_center[0]) * np.sin(np.radians(elipse2_angle)) + (y1 - elipse2_center[1]) * np.cos(np.radians(elipse2_angle)))**2 / ellipse_obj2.minor_radius**2 - 1
                        # print(f)
                        if old_f2 is not None:
                            if (f2 < 0) != (old_f2 < 0):
                                old_f2 = f2
                                minimum_point = (float(x1), float(y1))
                        if old_f2 is None:
                            old_f2 = f2
                    
                    # Найти угол точки относительно второго эллипса
                    dx = minimum_point[0] - elipse2_center[0]
                    dy = minimum_point[1] - elipse2_center[1]
                    
                    # Перевести в локальную СК второго эллипса
                    angle2 = np.radians(elipse2_angle)
                    local_x =  dx * np.cos(angle2) + dy * np.sin(angle2)
                    local_y = -dx * np.sin(angle2) + dy * np.cos(angle2)
                    
                    t2_point = np.arctan2(local_y / ellipse_obj2.minor_radius, 
                                        local_x / ellipse_obj2.major_radius)
                    
                    if _angle_in_arc(t2_point, ellipse_obj2.start_angle, 
                                        ellipse_obj2.end_angle, ellipse_obj2.is_closed):
                        found_points.append(minimum_point)
                     
            # print(f"t={t:.4f}, f={f:.6f}")   
            if old_f is None:
                old_f = f
                
            old_f = f
        
        # deduplicate
        tol = 1e-6
        unique_points = []
        for p in found_points:
            if not any(np.linalg.norm(np.array(p) - np.array(u)) < tol for u in unique_points):
                unique_points.append(p)
        
        # convert to 3d points
        found_points_3d = [
            self.sketch_instance.get_global_pos_in_sketch(p)
            for p in found_points
        ]
        return found_points_3d
    
    def find_line_ellipse_intersections(
        self,
        line_obj: Line,
        ellipse_obj: Ellipse,
    ) -> list:
        """
        Находит точки пересечения эллипса и прямой алгебраическим методом.
        Возвращает список точек в глобальных координатах.
        """
    
        # elipse1 data
        elipse1_center = self.sketch_instance.get_local_pos_in_sketch(ellipse_obj.center)
        local_vector1 = self.sketch_instance.transform_to_local_vector_in_sketch(ellipse_obj.major_axis_vector)
        elipse1_angle = np.degrees(np.arctan2(local_vector1[1], local_vector1[0]))

        # line data
        p1 = self.sketch_instance.get_local_pos_in_sketch(line_obj.points[0])
        p2 = self.sketch_instance.get_local_pos_in_sketch(line_obj.points[1])
        
        found_points = []
        
        if ellipse_obj.is_closed:
            points = np.linspace(0, 2*np.pi, 200)
        else:
            start1 = ellipse_obj.start_angle
            end1 = ellipse_obj.end_angle
            if end1 <= start1:
                end1 += 2 * np.pi
            points = np.linspace(start1, end1, 200)
            
        old_f = None
        for index, t in enumerate(points):
            x1 = elipse1_center[0] + ellipse_obj.major_radius * np.cos(t) * np.cos(np.radians(elipse1_angle)) - ellipse_obj.minor_radius * np.sin(t) * np.sin(np.radians(elipse1_angle))
            y1 = elipse1_center[1] + ellipse_obj.major_radius * np.cos(t) * np.sin(np.radians(elipse1_angle)) + ellipse_obj.minor_radius * np.sin(t) * np.cos(np.radians(elipse1_angle))
            dx_line = p2[0] - p1[0]
            dy_line = p2[1] - p1[1]
            f = (x1 - p1[0]) * dy_line - (y1 - p1[1]) * dx_line
            
            if old_f is not None:
                if (f < 0) != (old_f < 0):
                    old_f = f
                    minimum_point = None
                    old_f2 = None
                    for t2 in np.linspace(points[index-1], t, 200):
                        x1 = elipse1_center[0] + ellipse_obj.major_radius * np.cos(t2) * np.cos(np.radians(elipse1_angle)) - ellipse_obj.minor_radius * np.sin(t2) * np.sin(np.radians(elipse1_angle))
                        y1 = elipse1_center[1] + ellipse_obj.major_radius * np.cos(t2) * np.sin(np.radians(elipse1_angle)) + ellipse_obj.minor_radius * np.sin(t2) * np.cos(np.radians(elipse1_angle))
                        dx_line2 = p2[0] - p1[0]
                        dy_line2 = p2[1] - p1[1]
                        f2 = (x1 - p1[0]) * dy_line2 - (y1 - p1[1]) * dx_line2

                        if old_f2 is not None:
                            if (f2 < 0) != (old_f2 < 0):
                                old_f2 = f2
                                minimum_point = (float(x1), float(y1))
                        if old_f2 is None:
                            old_f2 = f2
                    
                    # Найти угол точки относительно второго эллипса
                    len2 = dx_line**2 + dy_line**2
                    u = ((minimum_point[0] - p1[0]) * dx_line + 
                        (minimum_point[1] - p1[1]) * dy_line) / len2

                    if 0.0 <= u <= 1.0:  # точка лежит внутри отрезка
                        found_points.append(minimum_point)
                     
            # print(f"t={t:.4f}, f={f:.6f}")   
            if old_f is None:
                old_f = f
                
            old_f = f
        
        # deduplicate
        tol = 1e-6
        unique_points = []
        for p in found_points:
            if not any(np.linalg.norm(np.array(p) - np.array(u)) < tol for u in unique_points):
                unique_points.append(p)
        
        # convert to 3d points
        found_points_3d = [
            self.sketch_instance.get_global_pos_in_sketch(p)
            for p in found_points
        ]
        return found_points_3d