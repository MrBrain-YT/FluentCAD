from typing import Union
import secrets

from pyvistaqt import QtInteractor
import pyvista as pv
import numpy as np
import build123d as cad_core
import vtk

from utils.utils import Picking
from UI.model.ModelConverter import b123d_to_pro_mesh
from utils.utils import is_contour_closed_by_angles
from UI.model.tools.Line import Line
from UI.model.tools.Circle import Circle
from UI.model.tools.PointCircle import PointCircle
from UI.model.tools.SketchPoint import SketchPoint
from UI.model.tools.Slot import Slot
from UI.model.tools.ArcTools import CenterArc, EllipseArc
from UI.model.tools.Curve import Curve
from UI.model.tools.Ellipse import Ellipse
from UI.model.tools.StraightShape import StraightShape
from UI.model.tools.Spline import Spline
from UI.model.profiles import build_profiles, sketch_edges, split_edges
from UI.model.solid_operations import build_extrusion, build_revolution, build_primitive
from UI.model.edge_selection import nearest_screen_edge, face_edge_indices
from UI.model.sketch_constraints import line_points, add_automatic_coincidences
from UI.model.transforms import body_index_for_face
from UI.model.feature_graph import Feature, FeatureGraph
from UI.model.timeline import Timeline

class ActionManager():
    
    def __init__(self, plotter):
        self.plotter:Union[QtInteractor, pv.Plotter] = plotter
        self.init_parameters()
        
    def init_parameters(self):
        self.start_helper_planes = []
        self.construction_planes = []
        self.sketchs_data = {}
        self.activated_sketch_id:int = None # editing sketch id
        
        self.active_tool = None
        self.current_part = None
        self.undo_stack = []
        self.redo_stack = []
        self.timeline = Timeline()
        self.parametric_graph = None
        self.parametric_sketch_id = None
        self.history_edit_id = None
        self.history_edit_graph = None
        self.history_origin_is_new = False
        self.history_edit_actors = []
        self.history_point_actors = {}
        self.history_drag_name = None
        self.history_drag_start = None
        self.previous_sketch_style = None
        self.sketch_plane_support = None
        self.graph_undo = []
        self.graph_redo = []
        self.legacy_migration_redo = None
        self.timeline_callback = lambda: None
        self.sketch_editing = False
        self.pending_sketch = False
        self.construction_mode = False
        self.profiles = []
        self.sketch_edge_actors = []
        self.sketch_intersection_points = []
        self.constraint_glyph_actor = None
        self.selecting_sketch_plane = False
        self.profile_selection_callback = None
        self.selected_profile_index = None
        self.edge_selection_mode = False
        self.interaction_locked = False
        self.selected_edge_indices = set()
        self.edge_selection_actors = []
        self.status_callback = lambda message: None
        self._adjusting_clipping = False
        self.active_normal = np.array([0, 0, 1]) # Normal for connect z axis for drow sketch
        self.active_center = np.array([0, 0, 0]) # center point in active sketch
        self.u_axis = np.array([1., 0., 0.])
        self.v_axis = np.array([0., 1., 0.])
        self.grid_step = 1.0  # fixed model-space grid and snap spacing
        self.grid_visible = True
        self.snap_to_grid = True
        self.snap_to_points = True

        self.sketch_grid_observer_id:int # object for run callback for change grid size
        
        self.selected_actor_name = None  # Имя текущего выделенного объекта
        
        self.sketch_profiles = []    # Список мешей (актеров) профилей на сцене
        self.selected_profiles = set() # Множество имен выбранных профилей
        
        self.cell_picker = vtk.vtkCellPicker()
        self.cell_picker.SetTolerance(0.0005)
        self.cell_picker.PickFromListOn()
        self.hover_id = -1
        self.selected_id = -1
        
        self.highlight_actor = None
        self.selected_face_id = -1
        
    def activate_sketch_mode(self):
        self.sketch_plane_support = None
        self.clear_selected_edges()
        self.edge_selection_mode = False
        self.profile_selection_callback = None
        self.selecting_sketch_plane = True
        self.selected_face_id = -1
        self._clear_highlight()
        for profile in self.profiles:
            self.plotter.remove_actor(profile["name"])
        self.profiles = []
        self.selected_profile_index = None
        self.show_origin_planes()
        self.set_construction_planes_visible(True)
        self.status_callback("Select an origin plane, construction plane, or planar face.")

    def cancel_pending_selection(self):
        if self.selecting_sketch_plane:
            self.selecting_sketch_plane = False
            for name in self.start_helper_planes:
                self.plotter.remove_actor(name)
            self.start_helper_planes.clear()
            self.set_construction_planes_visible(False)
            self.status_callback("Plane selection cancelled.")
            return True
        if self.profile_selection_callback is not None:
            self.profile_selection_callback = None
            self.status_callback("Profile selection cancelled.")
            return True
        if self.edge_selection_mode:
            self.edge_selection_mode = False
            self.status_callback("Edge selection finished.")
            return True
        return False

    def request_profile_selection(self, callback):
        self.profile_selection_callback = callback
        self.status_callback("Click a sketch profile to continue.")

    def _pick_actor(self, actors):
        actors = [actor for actor in actors if actor is not None]
        if not actors:
            return None
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.0005)
        picker.PickFromListOn()
        for actor in actors:
            picker.AddPickList(actor)
        x, y = self.plotter.iren.get_event_position()
        picker.Pick(x, y, 0, self.plotter.renderer)
        return picker if picker.GetCellId() >= 0 else None

    def _select_sketch_plane(self):
        actors = [self.plotter.actors.get(name) for name in self.start_helper_planes]
        actors += [self.plotter.actors.get(plane["name"]) for plane in self.construction_planes]
        body_actor = self.plotter.actors.get("main_part")
        picker = self._pick_actor(actors + [body_actor])
        if picker is None:
            return
        actor = picker.GetActor()
        if body_actor is not None and actor == body_actor:
            data = actor.GetMapper().GetInput().GetCellData().GetArray("face_id")
            if data is None:
                return
            face_id = data.GetValue(picker.GetCellId())
            faces = list(self.current_part.faces())
            if face_id >= len(faces) or faces[face_id].geom_type != cad_core.GeomType.PLANE:
                self.status_callback("Choose a planar face; curved faces are not supported.")
                return
            face = faces[face_id]
            if self.parametric_graph is not None:
                from UI.model.geometry_references import make_reference
                source_id = next((item.id for item in reversed(self.parametric_graph.features)
                                  if item.id in self.parametric_graph.results and self.parametric_graph.results[item.id] is self.current_part), None)
                if source_id is None:
                    self.status_callback('Cannot identify the parametric face source.')
                    return
                try:
                    reference = make_reference(source_id, self.current_part, 'face', face_id)
                except ValueError as exc:
                    self.status_callback(str(exc))
                    return
                self.sketch_plane_support = (source_id, reference)
                center = np.array(tuple(face.center()))
            else:
                center = np.array(picker.GetPickPosition())
            self._start_sketch(np.array(tuple(face.normal_at())), center)
        else:
            mesh = pv.wrap(actor.GetMapper().GetInput())
            self.on_plane_selected(mesh)

    def _select_profile(self):
        actors = [self.plotter.actors.get(profile["name"]) for profile in self.profiles]
        picker = self._pick_actor(actors)
        if picker is None:
            return False
        actor = picker.GetActor()
        index = next((i for i, candidate in enumerate(actors) if candidate == actor), None)
        if index is None:
            return False
        self.selected_profile_index = index
        for i, candidate in enumerate(actors):
            if candidate is not None:
                candidate.prop.color = "#f39c12" if i == index else "#3498db"
                candidate.prop.opacity = 0.7 if i == index else 0.45
        self.plotter.render()
        self.status_callback("Profile selected. Choose a solid operation.")
        callback = self.profile_selection_callback
        if callback is not None:
            self.profile_selection_callback = None
            callback(index)
        return True

    def get_active_sketch_points(self, mode:str="raw_numpy") -> list:
        """Get activ sketch points

        Args:
            mode (str, optional): defaults to "raw_numpy".
            "raw" - list of points
            "raw_numpy" - list of numpy array points
            "complete" - lists of points (splitted by tools)
            "complete_local" - lists of points in local position in sketch

        Returns:
            list: points
        """
        current_sketch = self.sketchs_data.get(self.activated_sketch_id) or {"sketch_func": []}
        points = []
        for sketch_points_items in current_sketch["sketch_func"]:
            if mode == "complete":
                points.append(sketch_points_items.points)
            if mode == "complete_local":
                local_points = []
                for point in sketch_points_items.points:
                    local_points.append(self.get_local_pos_in_sketch(point))
                points.append(local_points)
            elif mode == "raw":
                points.extend(sketch_points_items.points)
            elif mode == "raw_numpy":
                points.extend(np.asarray(point) for point in sketch_points_items.points)
                
        return points
    
    def set_active_sketch_points(self, new_points:list[list[np.ndarray]]) -> list:
        """
        Set active sketch points
        """
        points = self.sketchs_data.get(self.activated_sketch_id, [])["sketch_func"]
        for tool_id in range(len(new_points)):
            new_pt = []
            for point in new_points[tool_id]:
                new_pt.append(point.tolist())
            points[tool_id].points.clear()
            points[tool_id].points.extend(new_pt)
    
    def get_active_sketch_profiles(self) -> list[pv.Actor]:
        current_sketch = self.sketchs_data.get(self.activated_sketch_id, [])
        return current_sketch["sketch_profiles"]
    
    def get_local_pos_in_sketch(self, p_on_plane, normal=None):
        return self.transform_to_local_vector_in_sketch(np.asarray(p_on_plane) - self.active_center, normal)

    def transform_to_local_vector_in_sketch(self, vector, normal=None):
        return np.array([np.dot(vector, self.u_axis), np.dot(vector, self.v_axis)])

    def get_global_pos_in_sketch(self, point_2d, normal=None):
        return self.active_center + point_2d[0] * self.u_axis + point_2d[1] * self.v_axis

    def get_current_sketch_data(self):
        return self.sketchs_data.get(self.activated_sketch_id)

    def _get_face_under_mouse(self):
        # Исправляем координаты для 4K мониторов и масштабирования Windows
        # (Важно: берем координаты напрямую из интеркатора)
        x, y = self.plotter.iren.get_event_position()
        
        # Обновляем "белый список" объектов для поиска
        main_actor = self.plotter.actors.get("main_part")
        if not main_actor: return -1, None
        
        self.cell_picker.InitializePickList()
        self.cell_picker.AddPickList(main_actor) # Ищем ТОЛЬКО в этой детали
        
        # Выполняем поиск
        self.cell_picker.Pick(x, y, 0, self.plotter.renderer)
        
        cell_id = self.cell_picker.GetCellId()
        
        # Если не попали
        if cell_id < 0:
            return -1, None

        # Получаем данные
        mesh = main_actor.GetMapper().GetInput()
        face_arr = mesh.GetCellData().GetArray("face_id")
        if not face_arr: return -1, None

        return face_arr.GetValue(cell_id), mesh

    def update_plotter(self, plotter):
        self.plotter: Union[QtInteractor, pv.Plotter] = plotter
        self.plotter.iren.add_observer(
            "MouseWheelForwardEvent", self.update_grid_lod
        )
        self.plotter.iren.add_observer(
            "MouseWheelBackwardEvent", self.update_grid_lod
        )

        # Создаем второй рендерер для подсветки
        self.highlight_renderer = vtk.vtkRenderer()
        self.highlight_renderer.SetLayer(1)
        self.highlight_renderer.SetInteractive(0)
        self.highlight_renderer.SetBackgroundAlpha(0)
        self.highlight_renderer.SetBackground(0, 0, 0)

        render_window = self.plotter.ren_win
        render_window.SetNumberOfLayers(2)
        render_window.AddRenderer(self.highlight_renderer)

        # Синхронизируем камеру (ВАЖНО: после AddRenderer)
        self.highlight_renderer.SetActiveCamera(self.plotter.renderer.GetActiveCamera())

        # Add event listener
        self.plotter.iren.add_observer("MouseMoveEvent", self._on_hover)
        self.plotter.iren.add_observer("MouseMoveEvent", self._preview_history_drag)
        self.plotter.iren.add_observer("LeftButtonPressEvent", self._on_click)
        self.plotter.iren.add_observer("LeftButtonReleaseEvent", self._finish_history_drag)
        self.plotter.camera.AddObserver('ModifiedEvent', self.refresh_camera_clipping)

    def refresh_camera_clipping(self, *args):
        if self._adjusting_clipping or self.plotter is None: return
        self._adjusting_clipping = True
        try:
            bounds = self.plotter.renderer.ComputeVisiblePropBounds()
            if not all(np.isfinite(bounds)) or any(bounds[i] > bounds[i+1] for i in (0,2,4)):
                return
            self.plotter.renderer.ResetCameraClippingRange()
            near, far = self.plotter.camera.GetClippingRange()
            if np.isfinite(far) and far > 0:
                self.plotter.camera.SetClippingRange(max(1e-8, min(near, far * 1e-5)), far * 1.2)
        finally:
            self._adjusting_clipping = False

    def control_modifier_active(self):
        iren = getattr(self.plotter, 'iren', None)
        interactor = getattr(iren, 'interactor', None)
        return bool(interactor.GetControlKey()) if interactor is not None else False

    def lock_sketch_camera(self):
        iren = getattr(self.plotter, 'iren', None)
        if iren is None or not hasattr(iren, 'style') or self.previous_sketch_style is not None:
            return
        from UI.model.sketch_navigation import SketchCameraStyle
        self.previous_sketch_style = iren.style
        iren.style = SketchCameraStyle(self)

    def restore_camera_style(self):
        if self.previous_sketch_style is None: return
        self.plotter.iren.style = self.previous_sketch_style
        self.previous_sketch_style = None

    def _pick_face(self):
        main_actor = self.plotter.actors.get("main_part")
        if not main_actor:
            return -1, None

        x, y = self.plotter.iren.get_event_position()

        self.cell_picker.InitializePickList()
        self.cell_picker.AddPickList(main_actor)
        self.cell_picker.Pick(x, y, 0, self.plotter.renderer)

        cell_id = self.cell_picker.GetCellId()
        if cell_id < 0:
            return -1, None

        vtk_mesh = main_actor.GetMapper().GetInput()
        face_arr = vtk_mesh.GetCellData().GetArray("face_id")
        if not face_arr:
            return -1, None

        face_id = face_arr.GetValue(cell_id)

        # Конвертируем в PyVista, вырезаем грань, конвертируем обратно в PolyData
        pv_mesh = pv.wrap(vtk_mesh)
        mask = pv_mesh.cell_data["face_id"] == face_id
        
        # extract_cells -> UnstructuredGrid -> extract_surface -> PolyData
        face_unstructured = pv_mesh.extract_cells(np.where(mask)[0])
        face_poly = face_unstructured.extract_surface(algorithm="dataset_surface")

        return face_id, face_poly

    def _on_hover(self, *args):
        """Белая подсветка при наведении"""
        # Sketch interaction and profile picking have priority over solid highlighting.
        if self.interaction_locked or self.sketch_editing or self.selecting_sketch_plane or self.profile_selection_callback or self.edge_selection_mode:
            return
        if self.selected_face_id != -1:
            return

        face_id, face_mesh = self._pick_face()

        # Оптимизация: не перерисовываем одну и ту же грань
        if face_id == self.hover_id:
            return
        self.hover_id = face_id

        if face_id == -1:
            self._clear_highlight()
        else:
            self._draw_highlight(face_mesh, color=[255,255,255], opacity=0.1)

    def _on_click(self, *args):
        if self.interaction_locked:
            return
        if self.selecting_sketch_plane:
            self._select_sketch_plane()
            return
        if self.sketch_editing:
            if self.control_modifier_active():
                return
            if self.history_edit_id is not None and (self.active_tool is None or not self.active_tool.is_drawing):
                self.active_tool = None
                self._begin_history_drag()
            return
        if self.edge_selection_mode:
            self.select_edge_at_cursor()
            return
        if self._select_profile():
            return
        if self.profile_selection_callback is not None:
            return
        face_id, face_mesh = self._pick_face()
        if face_id == -1:
            self.selected_face_id = -1
            self.hover_id = -1
            self._clear_highlight()
            return
        self.selected_face_id = face_id
        self._draw_highlight(face_mesh, color=[101, 164, 252], opacity=0.3)

    def selected_body_index(self):
        if self.current_part is None:
            return None
        return body_index_for_face(self.current_part, self.selected_face_id)

    def delete_selected_body(self):
        if self.pending_sketch or self.sketch_editing or self.selecting_sketch_plane:
            raise ValueError("Finish the current sketch and solid operation first.")
        index = self.selected_body_index()
        if index is None:
            raise ValueError("Select a body by clicking one of its faces.")
        previous = self.current_part
        solids = list(previous.solids())
        remaining = solids[:index] + solids[index + 1:]
        if remaining:
            self.commit_result(cad_core.Part(remaining), "Delete body", {"Body index": index + 1})
        else:
            self._restore_result(None)
            self.undo_stack.append(previous)
            self.redo_stack.clear()
            self.timeline.append("Delete body", {"Body index": index + 1})
            self.timeline_callback()
        self.status_callback("Selected body deleted. Use Undo to restore it.")

    def start_edge_selection(self):
        if self.current_part is None:
            raise ValueError("Create or import a solid body first.")
        if self.pending_sketch or self.sketch_editing or self.selecting_sketch_plane:
            raise ValueError("Finish the current sketch and solid operation first.")
        self.edge_selection_mode = True
        self.selected_face_id = -1
        self.hover_id = -1
        self._clear_highlight()
        self.status_callback("Click edges to select or deselect them; press Esc to finish selection.")

    def clear_selected_edges(self):
        for actor in self.edge_selection_actors:
            self.plotter.remove_actor(actor, render=False)
        self.edge_selection_actors = []
        self.selected_edge_indices.clear()
        if self.plotter is not None:
            self.plotter.render()

    def select_edge_at_cursor(self):
        face_index, _ = self._get_face_under_mouse()
        if face_index < 0 or self.current_part is None:
            return
        edges, candidates = face_edge_indices(self.current_part, face_index)
        if not candidates:
            return
        renderer = self.plotter.renderer
        def project(point):
            renderer.SetWorldPoint(float(point[0]), float(point[1]), float(point[2]), 1.0)
            renderer.WorldToDisplay()
            return renderer.GetDisplayPoint()
        index = nearest_screen_edge(edges, candidates, self.plotter.iren.get_event_position(), project)
        if index is None:
            self.status_callback("Click closer to an edge.")
            return
        if index in self.selected_edge_indices:
            self.selected_edge_indices.remove(index)
        else:
            self.selected_edge_indices.add(index)
        self.redraw_selected_edges()
        self.status_callback(f"Selected edges: {len(self.selected_edge_indices)}. Press Esc to finish.")

    def redraw_selected_edges(self):
        for actor in self.edge_selection_actors:
            self.plotter.remove_actor(actor, render=False)
        self.edge_selection_actors = []
        if self.current_part is not None:
            edges = list(self.current_part.edges())
            for index in sorted(self.selected_edge_indices):
                if index >= len(edges):
                    continue
                edge = edges[index]
                count = max(16, min(257, int(edge.length * 8) + 2))
                points = np.asarray([tuple(edge.position_at(t)) for t in np.linspace(0, 1, count)])
                self.edge_selection_actors.append(self.plotter.add_mesh(
                    pv.lines_from_points(points), color="#ff9d00", line_width=6,
                    render_lines_as_tubes=True, pickable=False, reset_camera=False, render=False))
        self.plotter.render()

    def _draw_highlight(self, face_mesh, color=[255,255,255], opacity=0.4):
        self._clear_highlight(render=False)

        r = color[0]
        g = color[1]
        b = color[2]

        # face_mesh уже приходит как PolyData из _pick_face
        # НЕ вызываем extract_surface повторно!
        
        # Проверяем тип и конвертируем ТОЛЬКО если это не PolyData
        if isinstance(face_mesh, pv.PolyData):
            vtk_data = face_mesh
        elif isinstance(face_mesh, pv.UnstructuredGrid):
            vtk_data = face_mesh.extract_surface(algorithm="dataset_surface")
        else:
            surface_filter = vtk.vtkDataSetSurfaceFilter()
            surface_filter.SetInputData(face_mesh)
            surface_filter.Update()
            vtk_data = pv.wrap(surface_filter.GetOutput())

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(vtk_data)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        # ГЛАВНЫЙ ФИКС: настраиваем свойства правильно
        prop = actor.GetProperty()
        prop.SetColor(r, g, b)
        prop.SetOpacity(opacity)
        prop.SetLighting(False)
        
        # Убираем красную обратную сторону
        prop.SetBackfaceCulling(False)
        backface = vtk.vtkProperty()
        backface.SetColor(r, g, b)  # Тот же цвет!
        backface.SetOpacity(opacity)
        backface.SetLighting(False)
        actor.SetBackfaceProperty(backface)

        self.highlight_renderer.AddActor(actor)
        self.highlight_actor = actor
        self.plotter.render()

    def _clear_highlight(self, render=True):
        """Убирает подсветку из второго слоя"""
        if self.highlight_actor:
            self.highlight_renderer.RemoveActor(self.highlight_actor)
            self.highlight_actor = None
        if render:
            self.plotter.render()
    
    def exit_sketch_mode(self):
        if not self.sketch_editing:
            return
        if self.history_edit_id is not None:
            self.finish_history_sketch()
            return
        graph = None
        sketch_feature = None
        if self.parametric_graph is not None or self.current_part is None:
            from UI.model.sketch_capture import capture_sketch
            definition = capture_sketch(self.get_current_sketch_data(), self.active_center, self.u_axis, self.v_axis)
            plane_args = {'origin':self.active_center.tolist(), 'normal':self.active_normal.tolist(), 'x_dir':self.u_axis.tolist()}
            support_id = None
            if self.sketch_plane_support:
                support_id, plane_args['face'] = self.sketch_plane_support
            plane = Feature.create('Sketch plane', 'datum_plane', plane_args, support_id)
            sketch_feature = Feature.create('Sketch', 'planar_sketch', definition, plane.id)
            previous = self.parametric_graph
            graph = FeatureGraph([*(previous.features if previous else []), plane, sketch_feature], previous.parameters if previous else {})
        self.plotter.remove_actor("sketch_grid")
        if self.constraint_glyph_actor is not None:
            self.plotter.remove_actor(self.constraint_glyph_actor)
            self.constraint_glyph_actor = None
        self.plotter.disable_picking()
        for tool in self.get_current_sketch_data()["sketch_func"]:
            tool.is_drawing = False
        self.sketch_editing = False
        self.restore_camera_style()
        if graph is not None:
            self.install_parametric_graph(graph)
            self.parametric_sketch_id = sketch_feature.id
            self.show_parametric_profiles(sketch_feature.id)
        else:
            self.parametric_sketch_id = None
            self.create_profile_mesh()

    def start_history_sketch(self, feature_id):
        if self.sketch_editing or self.pending_sketch or self.selecting_sketch_plane:
            raise ValueError('Finish or cancel the current sketch first.')
        if self.parametric_graph is None:
            raise ValueError('Open a parametric model first.')
        candidate = FeatureGraph.from_data(self.parametric_graph.to_data())
        feature = next((item for item in candidate.features if item.id == feature_id), None)
        sketch = candidate.results.get(feature_id)
        if feature is None or feature.kind != 'planar_sketch' or sketch is None or sketch.constraint_status is None:
            raise ValueError('This sketch needs active named points for scene editing.')
        for profile in self.profiles:
            self.plotter.remove_actor(profile['name'])
        self.profiles = []
        self.selected_profile_index = None
        for actor in self.sketch_edge_actors:
            self.plotter.remove_actor(actor, render=False)
        self.sketch_edge_actors = []
        for previous_sketch in self.sketchs_data.values():
            for tool in previous_sketch['sketch_func']:
                for actor in tool.render_obj['points']:
                    self.plotter.remove_actor(actor, render=False)
        self.history_edit_id, self.history_edit_graph = feature_id, candidate
        self.history_origin_is_new = False
        self.active_center = np.asarray(tuple(sketch.plane.origin), dtype=float)
        self.active_normal = np.asarray(tuple(sketch.plane.z_dir), dtype=float)
        self.u_axis = np.asarray(tuple(sketch.plane.x_dir), dtype=float)
        self.v_axis = np.asarray(tuple(sketch.plane.y_dir), dtype=float)
        self.activated_sketch_id = max(self.sketchs_data.keys(), default=0) + 1
        self.sketchs_data[self.activated_sketch_id] = {'sketch_func':[], 'sketch_profiles':[], 'sketch_points':[],
                                                        'sketch_render_obj':{'points':[], 'lines':[]}}
        self.sketch_editing = True
        self.pending_sketch = True
        self.active_tool = None
        body = self.plotter.actors.get('main_part')
        if body is not None: body.SetVisibility(False)
        try:
            self.lock_sketch_camera()
            self.align_camera_to_normal(self.active_normal)
            self.update_grid_lod()
            self.render_history_sketch()
        except Exception:
            self.cancel_history_sketch()
            raise
        self.status_callback('Drag sketch points, add geometry, then finish. Esc cancels when no tool is active.')

    def promote_current_sketch(self):
        if not self.sketch_editing or self.history_edit_id is not None:
            raise ValueError('Start a new sketch first.')
        if self.parametric_graph is None and self.current_part is not None:
            raise ValueError('Convert the legacy model to parametric before using this command.')
        from UI.model.sketch_capture import capture_sketch
        sketch_data = self.get_current_sketch_data()
        definition = capture_sketch(sketch_data, self.active_center, self.u_axis, self.v_axis)
        if not definition['points']:
            raise ValueError('Draw sketch geometry before using this command.')
        args = {'origin':self.active_center.tolist(), 'normal':self.active_normal.tolist(), 'x_dir':self.u_axis.tolist()}
        support = None
        if self.sketch_plane_support: support, args['face'] = self.sketch_plane_support
        plane = Feature.create('Sketch plane','datum_plane',args,support)
        feature = Feature.create('Sketch','planar_sketch',definition,plane.id)
        previous = self.parametric_graph
        candidate = FeatureGraph([*(previous.features if previous else []),plane,feature], previous.parameters if previous else {})
        self.history_edit_id, self.history_edit_graph = feature.id, candidate
        self.history_origin_is_new = True
        try:
            self.render_history_sketch()
        except Exception:
            self.clear_history_sketch_actors()
            self.history_edit_id = self.history_edit_graph = None
            self.history_origin_is_new = False
            raise
        for actor in self.sketch_edge_actors: self.plotter.remove_actor(actor, render=False)
        self.sketch_edge_actors = []
        for tool in sketch_data['sketch_func']:
            for actor in tool.render_obj['points']: self.plotter.remove_actor(actor, render=False)
        sketch_data['sketch_func'] = []
        sketch_data.pop('constraints', None)
        sketch_data.pop('constraint_known_points', None)
        if self.constraint_glyph_actor is not None:
            self.plotter.remove_actor(self.constraint_glyph_actor, render=False)
            self.constraint_glyph_actor = None
        body = self.plotter.actors.get('main_part')
        if body is not None: body.SetVisibility(False)
        self.plotter.render()

    def clear_history_sketch_actors(self):
        for actor in self.history_edit_actors:
            self.plotter.remove_actor(actor, render=False)
        self.history_edit_actors = []
        self.history_point_actors = {}

    def render_history_sketch(self):
        self.clear_history_sketch_actors()
        sketch = self.history_edit_graph.results[self.history_edit_id]
        for tool in sketch.tools:
            for edge in sketch_edges([tool], self.active_normal):
                count = max(2, min(257, int(edge.length * 8) + 2))
                mesh = pv.lines_from_points(np.asarray([tuple(edge.position_at(t)) for t in np.linspace(0, 1, count)]))
                actor = self.plotter.add_mesh(mesh, color='#43889d' if tool.is_construction else 'blue',
                                              line_width=2, pickable=False, reset_camera=False, render=False)
                self.history_edit_actors.append(actor)
            if tool.type in ('spline','fit_spline','nurbs','periodic_nurbs') and len(tool.points) >= 2:
                controls = [np.asarray(point, dtype=float) for point in tool.points]
                if tool.type == 'periodic_nurbs': controls.append(controls[0])
                vertices, cells = [], []
                for a, b in zip(controls, controls[1:]):
                    for i in range(0, 8, 2):
                        start = len(vertices)
                        vertices.extend((a+(b-a)*i/8, a+(b-a)*(i+1)/8))
                        cells.extend((2, start, start+1))
                mesh = pv.PolyData(np.asarray(vertices)); mesh.lines = np.asarray(cells, dtype=np.int64)
                actor = self.plotter.add_mesh(mesh, color='#94a5af', line_width=1,
                                              pickable=False, reset_camera=False, render=False)
                self.history_edit_actors.append(actor)
        for name, point in sketch.constraint_status.points.items():
            world = self.get_global_pos_in_sketch(point)
            actor = self.plotter.add_points(world, color='#1785ae', point_size=12, pickable=True,
                                            reset_camera=False, render=False)
            self.history_edit_actors.append(actor)
            self.history_point_actors[name] = actor
        from UI.model.parametric_sketch import circle_radius_name
        feature = next(item for item in self.history_edit_graph.features if item.id == self.history_edit_id)
        for index, entity in enumerate(feature.parameters.get('entities', [])):
            if entity.get('type') != 'circle' or not isinstance(entity.get('center'), str): continue
            center = sketch.constraint_status.points[entity['center']]
            radius = sketch.constraint_status.radii[circle_radius_name(index)]
            world = self.get_global_pos_in_sketch(center + np.array([radius, 0.]))
            actor = self.plotter.add_points(world, color='#e39832', point_size=12, pickable=True,
                                            reset_camera=False, render=False)
            self.history_edit_actors.append(actor)
            self.history_point_actors[('radius', index, entity['center'])] = actor
        glyphs, labels = [], []
        for constraint in feature.parameters.get('constraints', []):
            refs = constraint.get('entities', [])
            first = refs[0] if refs else None
            if isinstance(first, (list, tuple)) and first: first = first[0]
            if not isinstance(first, str) or first not in sketch.constraint_status.points: continue
            glyphs.append(self.get_global_pos_in_sketch(sketch.constraint_status.points[first]))
            value = constraint.get('value')
            labels.append(str(constraint.get('kind', 'constraint')) + (f' {value}' if value is not None else ''))
        if glyphs and hasattr(self.plotter, 'add_point_labels'):
            actor = self.plotter.add_point_labels(np.asarray(glyphs), labels, font_size=11,
                show_points=False, shape=None, always_visible=True, pickable=False)
            self.history_edit_actors.append(actor)
        self.plotter.render()

    def _begin_history_drag(self):
        self.history_drag_name = self._history_handle_at_cursor()
        self.history_drag_start = self.plotter.iren.get_event_position() if self.history_drag_name is not None else None

    def _history_handle_at_cursor(self):
        from UI.model.parametric_sketch import circle_radius_name
        state = self.history_edit_graph.results[self.history_edit_id].constraint_status
        cursor = np.asarray(self.plotter.iren.get_event_position(), dtype=float)
        renderer = self.plotter.renderer
        nearest, best = None, 12.0
        for name in self.history_point_actors:
            if isinstance(name, tuple):
                point = state.points[name[2]] + np.array([state.radii[circle_radius_name(name[1])], 0.])
            else:
                point = state.points[name]
            world = self.get_global_pos_in_sketch(point)
            renderer.SetWorldPoint(float(world[0]), float(world[1]), float(world[2]), 1.)
            renderer.WorldToDisplay()
            display = renderer.GetDisplayPoint()
            if not 0 <= display[2] <= 1: continue
            distance = float(np.linalg.norm(np.asarray(display[:2]) - cursor))
            if distance <= best:
                nearest, best = name, distance
        return nearest

    def _preview_history_drag(self, *args):
        if self.history_drag_name is None or self.history_edit_id is None: return
        try:
            world = self._screen_to_sketch_plane()
            state = self.history_edit_graph.results[self.history_edit_id].constraint_status
            name = self.history_drag_name
            if isinstance(name, tuple):
                from UI.model.parametric_sketch import circle_radius_name
                original = state.points[name[2]] + np.array([state.radii[circle_radius_name(name[1])], 0.])
            else:
                original = state.points[name]
            delta = world - self.get_global_pos_in_sketch(original)
            self.history_point_actors[name].SetPosition(*delta)
            self.plotter.render()
        except (ValueError, KeyError):
            pass

    def _screen_to_sketch_plane(self):
        x, y = self.plotter.iren.get_event_position()
        renderer = self.plotter.renderer
        points = []
        for depth in (0., 1.):
            renderer.SetDisplayPoint(x, y, depth)
            renderer.DisplayToWorld()
            world = np.asarray(renderer.GetWorldPoint(), dtype=float)
            if abs(world[3]) < 1e-12: raise ValueError('Cannot project pointer to the sketch plane.')
            points.append(world[:3] / world[3])
        direction = points[1] - points[0]
        denominator = np.dot(direction, self.active_normal)
        if abs(denominator) < 1e-12: raise ValueError('View is parallel to the sketch plane.')
        return points[0] + direction * (np.dot(self.active_center-points[0], self.active_normal) / denominator)

    def _finish_history_drag(self, *args):
        name, self.history_drag_name = self.history_drag_name, None
        start, self.history_drag_start = self.history_drag_start, None
        if name is None or self.history_edit_id is None: return
        try:
            if start is not None and np.linalg.norm(np.asarray(self.plotter.iren.get_event_position()) - np.asarray(start)) < 3:
                self.history_point_actors[name].SetPosition(0, 0, 0)
                return
            position = self.get_local_pos_in_sketch(self._screen_to_sketch_plane())
            if isinstance(name, tuple):
                from UI.model.sketch_drag import set_circle_radius
                center = self.history_edit_graph.results[self.history_edit_id].constraint_status.points[name[2]]
                self.history_edit_graph = set_circle_radius(self.history_edit_graph, self.history_edit_id, name[1],
                                                            float(np.linalg.norm(position - center)))
                self.render_history_sketch()
                self.status_callback('Circle radius updated. Finish the sketch to apply.')
            else:
                self.move_history_point(name, position)
        except Exception as exc:
            if name in self.history_point_actors:
                self.history_point_actors[name].SetPosition(0, 0, 0)
                self.plotter.render()
            self.status_callback(str(exc))

    def move_history_point(self, name, position):
        from UI.model.sketch_drag import move_sketch_point
        candidate = move_sketch_point(self.history_edit_graph, self.history_edit_id, name, position)
        self.history_edit_graph = candidate
        self.render_history_sketch()
        self.status_callback('Sketch point updated. Finish the sketch to apply.')

    def merged_history_graph(self):
        from copy import deepcopy
        from UI.model.sketch_capture import capture_sketch
        feature_id = self.history_edit_id
        candidate = FeatureGraph.from_data(self.history_edit_graph.to_data())
        new = capture_sketch(self.get_current_sketch_data(), self.active_center, self.u_axis, self.v_axis)
        if new['points'] or new['entities'] or new['constraints']:
            feature = next(item for item in candidate.features if item.id == feature_id)
            params = deepcopy(feature.parameters)
            mapping = {}
            for name in new['points']:
                number = 1
                while f'N{number}_{name}' in params['points']: number += 1
                mapping[name] = f'N{number}_{name}'
            def remap(value):
                if isinstance(value, str): return mapping.get(value, value)
                if isinstance(value, (tuple, list)): return [remap(item) for item in value]
                return value
            params['points'].update({mapping[name]: point for name, point in new['points'].items()})
            old_points = candidate.results[feature_id].constraint_status.points
            for name, point in new['points'].items():
                matching = [old for old, position in old_points.items() if np.linalg.norm(np.asarray(point)-position) < 1e-7]
                if matching:
                    params.setdefault('constraints', []).append({'kind':'coincident', 'entities':[mapping[name], sorted(matching)[0]]})
            for entity in new['entities']:
                entity = deepcopy(entity)
                if 'points' in entity: entity['points'] = remap(entity['points'])
                if 'center' in entity: entity['center'] = remap(entity['center'])
                params['entities'].append(entity)
            for constraint in new['constraints']:
                constraint = deepcopy(constraint); constraint['entities'] = remap(constraint['entities'])
                params.setdefault('constraints', []).append(constraint)
            candidate.edit(feature_id, parameters=params)
        return candidate

    def flush_history_tools(self):
        candidate = self.merged_history_graph()
        sketch = self.get_current_sketch_data()
        for actor in self.sketch_edge_actors: self.plotter.remove_actor(actor, render=False)
        self.sketch_edge_actors = []
        for tool in sketch['sketch_func']:
            for actor in tool.render_obj['points']: self.plotter.remove_actor(actor, render=False)
        sketch['sketch_func'] = []
        sketch.pop('constraints', None)
        sketch.pop('constraint_known_points', None)
        if self.constraint_glyph_actor is not None:
            self.plotter.remove_actor(self.constraint_glyph_actor, render=False)
            self.constraint_glyph_actor = None
        self.history_edit_graph = candidate
        self.render_history_sketch()

    def finish_history_sketch(self):
        feature_id = self.history_edit_id
        candidate = self.merged_history_graph()
        self.install_parametric_graph(candidate, allow_empty=True)
        self.clear_history_sketch_actors()
        for actor in self.sketch_edge_actors: self.plotter.remove_actor(actor, render=False)
        self.sketch_edge_actors = []
        for tool in self.get_current_sketch_data()['sketch_func']:
            for actor in tool.render_obj['points']: self.plotter.remove_actor(actor, render=False)
        self.plotter.remove_actor('sketch_grid')
        if self.constraint_glyph_actor is not None:
            self.plotter.remove_actor(self.constraint_glyph_actor); self.constraint_glyph_actor = None
        self.plotter.disable_picking()
        self.sketch_editing = False
        self.pending_sketch = False
        self.restore_camera_style()
        self.history_edit_id = self.history_edit_graph = None
        self.history_origin_is_new = False
        self.history_drag_name = self.history_drag_start = None
        self.sketchs_data.pop(self.activated_sketch_id, None)
        self.activated_sketch_id = None
        self.show_parametric_profiles(feature_id)

    def cancel_history_sketch(self):
        if self.history_edit_id is None: return False
        feature_id = self.history_edit_id
        was_new = self.history_origin_is_new
        self.clear_history_sketch_actors()
        self.plotter.remove_actor('sketch_grid')
        if self.constraint_glyph_actor is not None:
            self.plotter.remove_actor(self.constraint_glyph_actor, render=False)
            self.constraint_glyph_actor = None
        for actor in self.sketch_edge_actors: self.plotter.remove_actor(actor, render=False)
        self.sketch_edge_actors = []
        for tool in self.get_current_sketch_data()['sketch_func']:
            for actor in tool.render_obj['points']: self.plotter.remove_actor(actor, render=False)
        self.plotter.disable_picking()
        self.sketchs_data.pop(self.activated_sketch_id, None)
        self.activated_sketch_id = None
        self.sketch_editing = self.pending_sketch = False
        self.restore_camera_style()
        self.history_edit_id = self.history_edit_graph = self.history_drag_name = self.history_drag_start = None
        self.history_origin_is_new = False
        body = self.plotter.actors.get('main_part')
        if body is not None: body.SetVisibility(True)
        if not was_new: self.show_parametric_profiles(feature_id)
        self.plotter.render()
        self.status_callback('Sketch edits cancelled.')
        return True

    def show_parametric_profiles(self, feature_id):
        sketch = self.parametric_graph.results.get(feature_id)
        if sketch is None or not hasattr(sketch, 'profiles'):
            raise ValueError('The parametric sketch is unavailable.')
        for profile in self.profiles:
            self.plotter.remove_actor(profile['name'])
        self.profiles, self.selected_profile_index = [], None
        self.active_normal = np.asarray(tuple(sketch.plane.z_dir), dtype=float)
        self.active_center = np.asarray(tuple(sketch.plane.origin), dtype=float)
        self.u_axis = np.asarray(tuple(sketch.plane.x_dir), dtype=float)
        self.v_axis = np.asarray(tuple(sketch.plane.y_dir), dtype=float)
        self.parametric_sketch_id = feature_id
        for index, face in enumerate(sketch.profiles):
            name = f'parametric_profile_{feature_id}_{index}'
            actor = self.plotter.add_mesh(b123d_to_pro_mesh(face), color='#3498db', opacity=.45,
                                          name=name, pickable=True, reset_camera=False)
            actor.SetPosition(*(self.active_normal * 1e-4))
            self.profiles.append({'name':name, 'face':face, 'sketch_id':feature_id, 'profile_index':index})
        return self.profiles

    def create_offset_plane(self, base, offset, origin=None, normal=None):
        normals = {"XY": (0, 0, 1), "XZ": (0, 1, 0), "YZ": (1, 0, 0)}
        if not np.isfinite(offset):
            raise ValueError("Offset must be finite.")
        if base in normals:
            normal = np.asarray(normals[base], dtype=float)
            origin = np.zeros(3)
        elif base == "face":
            origin = np.asarray(origin, dtype=float)
            normal = np.asarray(normal, dtype=float)
            if (origin.shape != (3,) or normal.shape != (3,) or
                    not np.all(np.isfinite(origin)) or not np.all(np.isfinite(normal)) or
                    np.linalg.norm(normal) < 1e-9):
                raise ValueError("Choose a valid planar face.")
            normal = normal / np.linalg.norm(normal)
        else:
            raise ValueError("Choose a valid base plane.")
        center = origin + normal * float(offset)
        name = f"construction_plane_{secrets.token_hex(8)}"
        mesh = pv.Plane(center=center, direction=normal, i_size=20, j_size=20)
        self.plotter.add_mesh(mesh, color="#d8a24a", opacity=0.3, name=name,
                              pickable=True, reset_camera=False)
        plane = {"name": name, "base": base, "offset": float(offset)}
        if base == "face":
            plane.update(origin=origin.tolist(), normal=normal.tolist())
        self.construction_planes.append(plane)
        self.status_callback(f"Created offset plane at distance {offset:g}.")
        return plane

    def create_plane_from_selected_face(self, offset):
        if self.current_part is None or self.selected_face_id < 0:
            raise ValueError("Select a planar face of the solid first.")
        faces = list(self.current_part.faces())
        if self.selected_face_id >= len(faces):
            raise ValueError("The selected face is no longer available.")
        face = faces[self.selected_face_id]
        if face.geom_type != cad_core.GeomType.PLANE:
            raise ValueError("Select a planar face; curved faces are not supported.")
        return self.create_offset_plane("face", offset, tuple(face.center()), tuple(face.normal_at()))

    def set_construction_planes_visible(self, visible):
        for plane in self.construction_planes:
            actor = self.plotter.actors.get(plane["name"])
            if actor is not None:
                actor.SetVisibility(visible)
                actor.SetPickable(visible)

    def clear_construction_planes(self):
        for plane in self.construction_planes:
            self.plotter.remove_actor(plane["name"])
        self.construction_planes = []

    def show_origin_planes(self):
        """Рисует три прозрачные плоскости в начале координат"""
        size = 20
        # Создаем плоскости
        planes = {
            "XY": pv.Plane(direction=(0, 0, 1), i_size=size, j_size=size),
            "XZ": pv.Plane(direction=(0, 1, 0), i_size=size, j_size=size),
            "YZ": pv.Plane(direction=(1, 0, 0), i_size=size, j_size=size)
        }
        
        colors = {"XY": "red", "XZ": "green", "YZ": "blue"}
        
        for name, mesh in planes.items():
            actor = self.plotter.add_mesh(
                mesh, color=colors[name], opacity=0.3, 
                name=f"helper_{name}", label=name
            )
            self.start_helper_planes.append(f"helper_{name}")

    def on_plane_selected(self, picked_mesh):
        if picked_mesh is None:
            return
        bounds = picked_mesh.bounds
        center = np.array([(bounds.x_max + bounds.x_min) / 2,
                           (bounds.y_max + bounds.y_min) / 2,
                           (bounds.z_max + bounds.z_min) / 2])
        self._start_sketch(np.array(picked_mesh.cell_normals[0]), center)

    def _start_sketch(self, normal, center):
        self.selecting_sketch_plane = False
        self.plotter.disable_picking()
        for name in self.start_helper_planes:
            self.plotter.remove_actor(name)
        self.start_helper_planes.clear()
        self.set_construction_planes_visible(False)
        self._clear_highlight()
        self.active_center = np.asarray(center, dtype=float)
        self.align_camera_to_normal(normal)
        central_point = self.active_center

        self.activated_sketch_id = len(self.sketchs_data.keys()) + 1
        self.sketchs_data[self.activated_sketch_id] = {
            "sketch_func" : [],
            "sketch_profiles": [],
            "sketch_points": [], # only points rendering
            "sketch_render_obj": {
                "points" : [],
                "lines" : []
            } # only rendering
        }
        
        self.sketch_editing = True
        self.pending_sketch = True
        self.lock_sketch_camera()
        self.sketch_intersection_points = []
        self.active_normal = np.asarray(normal, dtype=float)
        self.active_center = np.array(central_point)
        # Создаем базис плоскости (U и V оси)
        normal = np.asarray(normal, dtype=float)
        # Ищем любой вектор, не параллельный нормали
        ref = np.array([0, 1, 0]) if abs(normal[1]) < 0.9 else np.array([1, 0, 0])
        self.u_axis = np.cross(normal, ref)
        self.u_axis /= np.linalg.norm(self.u_axis)
        self.v_axis = np.cross(normal, self.u_axis)
        self.active_normal = normal
        self.active_center = np.array(central_point)
        self.update_grid_lod()
        self.status_callback("Sketch active. Choose a drawing tool.")
        
    def update_grid_lod(self, *args):
        """Extend the grid on zoom without changing its model-space spacing."""
        if self.sketch_editing:
            self.redraw_grid()

    def set_grid_step(self, value):
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Grid spacing must be positive and finite.")
        self.grid_step = float(value)
        self.update_grid_lod()

    def set_grid_visible(self, visible):
        self.grid_visible = bool(visible)
        if self.sketch_editing:
            self.redraw_grid()

    def redraw_grid(self):
        self.plotter.remove_actor("sketch_grid")
        if not self.grid_visible:
            return
        # Grid vertices and snapping use the same fixed basis, origin and step.
        # Bound the amount of geometry for extreme zoom-out.
        radius = min(250, max(10, int(np.ceil(self.plotter.camera.parallel_scale * 2 / self.grid_step))))
        extent = radius * self.grid_step
        points = []
        lines = []
        for index in range(-radius, radius + 1):
            offset = index * self.grid_step
            for a, b in (((offset, -extent), (offset, extent)),
                         ((-extent, offset), (extent, offset))):
                start = len(points)
                points.extend((self.get_global_pos_in_sketch(a), self.get_global_pos_in_sketch(b)))
                lines.extend((2, start, start + 1))
        grid = pv.PolyData(np.asarray(points))
        grid.lines = np.asarray(lines)
        self.plotter.add_mesh(grid, color="#555555", opacity=0.45,
                              name="sketch_grid", pickable=False, reset_camera=False)

    def align_camera_to_normal(self, normal):
        """Разворачивает камеру лицом к нормали"""
        n = np.array(normal)
        
        distance = 100 
        self.plotter.camera.position = self.active_center + n * distance
        self.plotter.camera.focal_point = self.active_center
        if np.allclose(np.abs(n), [0, 0, 1]):
            self.plotter.camera.up = (0, 1, 0)
        else:
            self.plotter.camera.up = (0, 0, 1)
            
        self.plotter.enable_parallel_projection()
        self.plotter.reset_camera()
        self.plotter.render()
        
    def get_snapped_pos(self, raw_pos) -> tuple[np.ndarray, np.ndarray]:
        """Возвращает точку с привязкой к сетке или точкам скетча"""
        p = np.array(raw_pos)
        
        # 1. ПРОЕКЦИЯ МЫШИ НА ПЛОСКОСТЬ (Чтобы не было погрешностей глубины)
        v = p - self.active_center
        dist_to_plane = np.dot(v, self.active_normal)
        p_on_plane = p - dist_to_plane * self.active_normal
        
        # Existing points take priority; choose the nearest, not the first one.
        if self.snap_to_points:
            points = self.get_active_sketch_points() + self.sketch_intersection_points
            if self.history_edit_graph is not None and self.history_edit_id is not None:
                state = self.history_edit_graph.results[self.history_edit_id].constraint_status
                points += [self.get_global_pos_in_sketch(point) for point in state.points.values()]
            if points:
                nearest = min(points, key=lambda point: np.linalg.norm(p_on_plane - point))
                point_radius = self.plotter.camera.parallel_scale * 0.025
                if np.linalg.norm(p_on_plane - nearest) < point_radius:
                    return nearest, self.get_local_pos_in_sketch(nearest)

        if not self.snap_to_grid:
            return p_on_plane, self.get_local_pos_in_sketch(p_on_plane)
        snap_radius = min(self.grid_step * 0.4, self.plotter.camera.parallel_scale * 0.025)

        # 3. ПРИВЯЗКА К СЕТКЕ (Grid Intersection Snap) - Приоритет №2
        # Переводим 3D координаты в локальные 2D (U, V)
        rel_p = p_on_plane - self.active_center
        u_coord = np.dot(rel_p, self.u_axis)
        v_coord = np.dot(rel_p, self.v_axis)
        
        # Округляем до ближайшего шага сетки
        u_snapped = round(u_coord / self.grid_step) * self.grid_step
        v_snapped = round(v_coord / self.grid_step) * self.grid_step
        
        # Переводим обратно в 3D
        grid_snapped_3d = self.active_center + (u_snapped * self.u_axis) + (v_snapped * self.v_axis)
        
        # Проверяем, достаточно ли мы близко к узлу сетки, чтобы "магнитить"
        if np.linalg.norm(p_on_plane - grid_snapped_3d) < snap_radius:
            return grid_snapped_3d, self.get_local_pos_in_sketch(grid_snapped_3d)

        # Если никуда не примагнитились — возвращаем просто точку на плоскости
        return p_on_plane, self.get_local_pos_in_sketch(p_on_plane)
        
    def redraw_sketch_geometry(self):
        """Render the OCC-split edges without changing the source sketch objects."""
        sketch = self.get_current_sketch_data()
        if not sketch:
            return
        tools = sketch["sketch_func"]
        points_for_constraints = line_points(self)
        add_automatic_coincidences(sketch, points_for_constraints)
        regular = [tool for tool in tools if not getattr(tool, "is_construction", False)]
        construction = [tool for tool in tools if getattr(tool, "is_construction", False)]
        regular_edges = sketch_edges(regular, self.active_normal)
        construction_edges = sketch_edges(construction, self.active_normal)
        parts = split_edges(regular_edges)
        construction_parts = split_edges(construction_edges)
        all_parts = split_edges(regular_edges + construction_edges)
        for actor in self.sketch_edge_actors:
            self.plotter.remove_actor(actor, render=False)
        self.sketch_edge_actors = []
        self.sketch_intersection_points = []
        for edge in all_parts:
            self.sketch_intersection_points.extend((np.array(tuple(edge.position_at(0))),
                                                    np.array(tuple(edge.position_at(1)))))
        for edge, is_construction in [(e, False) for e in parts] + [(e, True) for e in construction_parts]:
            count = max(2, min(257, int(edge.length * 8) + 2))
            points = np.asarray([tuple(edge.position_at(t)) for t in np.linspace(0, 1, count)])
            if is_construction:
                # Separate line cells leave visible gaps in construction geometry.
                mesh = pv.PolyData()
                mesh.points = points
                cells = []
                for index in range(0, len(points) - 1, 2):
                    cells.extend((2, index, index + 1))
                mesh.lines = np.asarray(cells, dtype=np.int64)
            else:
                mesh = pv.lines_from_points(points)
            self.sketch_edge_actors.append(self.plotter.add_mesh(
                mesh, color="#43889d" if is_construction else "blue", line_width=2,
                pickable=False, reset_camera=False, render=False))
        if self.constraint_glyph_actor is not None:
            self.plotter.remove_actor(self.constraint_glyph_actor, render=False)
            self.constraint_glyph_actor = None
        glyph_positions, glyph_labels = [], []
        symbols = {'coincident': 'Coincident', 'horizontal': 'Horizontal', 'vertical': 'Vertical',
                   'fixed': 'Fixed', 'distance': 'D', 'horizontal_distance': 'DX', 'vertical_distance': 'DY'}
        for constraint in sketch.get('constraints', []):
            first = constraint.entities[0]
            first = first[0] if isinstance(first, tuple) else first
            if first in points_for_constraints:
                glyph_positions.append(self.get_global_pos_in_sketch(points_for_constraints[first][2]))
                label = symbols.get(constraint.kind, constraint.kind)
                if constraint.kind in ('distance', 'horizontal_distance', 'vertical_distance'):
                    label += f' {constraint.value:g}'
                glyph_labels.append(label)
        if glyph_positions and hasattr(self.plotter, 'add_point_labels'):
            self.constraint_glyph_actor = self.plotter.add_point_labels(
                np.asarray(glyph_positions), glyph_labels, name='constraint_glyphs',
                font_size=11, show_points=False, shape=None, always_visible=True, pickable=False)
        self.plotter.render()

    def create_profile_mesh(self):
        for profile in self.profiles:
            self.plotter.remove_actor(profile["name"])
        self.profiles = []
        self.selected_profile_index = None
        sketch = self.get_current_sketch_data()
        if not sketch:
            return []
        faces = build_profiles(sketch["sketch_func"], self.active_normal)
        for index, face in enumerate(faces):
            name = f"sketch_profile_{self.activated_sketch_id}_{index}"
            mesh = b123d_to_pro_mesh(face)
            actor = self.plotter.add_mesh(mesh, color="#3498db", opacity=0.45,
                                          name=name, pickable=True, reset_camera=False)
            # Avoid z-fighting when the sketch lies on a solid face.
            actor.SetPosition(*(np.asarray(self.active_normal) * 1e-4))
            self.profiles.append({"name": name, "face": face})
        return self.profiles

    def prepare_selected_face_profile(self):
        if self.current_part is None or self.selected_face_id < 0:
            return False
        faces = list(self.current_part.faces())
        if self.selected_face_id >= len(faces):
            return False
        face = faces[self.selected_face_id]
        if face.geom_type != cad_core.GeomType.PLANE:
            raise ValueError("Select a planar face for extrusion.")
        self.active_normal = np.asarray(tuple(face.normal_at()), dtype=float)
        self.profiles = [{"name": "temporary_selected_face", "face": face, "temporary": True}]
        self.selected_profile_index = 0
        return True

    def clear_temporary_face_profile(self):
        if self.profiles and self.profiles[0].get("temporary"):
            self.profiles = []
            self.selected_profile_index = None

    def build_extrusion(self, index, amount, operation="new_body", offset=0.0,
                        taper=0.0, symmetric=False, second_distance=None, extent="distance"):
        if not 0 <= index < len(self.profiles):
            raise ValueError("Select a closed profile.")
        return build_extrusion(self.profiles[index]["face"], self.active_normal, amount,
                               operation, getattr(self, "current_part", None), offset, taper, symmetric, second_distance, extent)

    def commit_parametric_extrusion(self, index, distance, operation='new_body', offset=0.0,
                                    taper=0.0, symmetric=False, second_distance=None, extent='distance'):
        if self.parametric_graph is None or not 0 <= index < len(self.profiles):
            raise ValueError('Select a parametric sketch profile.')
        profile = self.profiles[index]
        sketch_id = profile.get('sketch_id')
        if not sketch_id or sketch_id not in self.parametric_graph.results:
            raise ValueError('The selected profile is not part of a parametric sketch.')
        base_id = next((item.id for item in reversed(self.parametric_graph.features)
                        if item.id in self.parametric_graph.results and
                        hasattr(self.parametric_graph.results[item.id], 'solids') and
                        self.parametric_graph.results[item.id].solids()), None)
        args = {'distance':float(distance), 'profile_index':profile['profile_index'], 'operation':operation,
                'profile_count':len(self.parametric_graph.results[sketch_id].profiles),
                'offset':float(offset), 'taper':float(taper), 'symmetric':bool(symmetric),
                'second_distance':float(second_distance) if second_distance is not None else None, 'extent':extent}
        keys = self.parametric_graph.results[sketch_id].profile_keys
        key = keys[profile['profile_index']]
        if key is not None:
            args['profile_key'] = key
        elif len(keys) > 1:
            raise ValueError('The selected profile cannot be identified unambiguously after rebuild.')
        if base_id: args['base_id'] = base_id
        feature = Feature.create('Extrude', 'extrude', args, sketch_id)
        graph = FeatureGraph([*self.parametric_graph.features, feature], self.parametric_graph.parameters)
        self.install_parametric_graph(graph)
        self.parametric_sketch_id = None
        return feature.id

    def extrude_profile(self, index, amount, operation="new_body", offset=0.0,
                        taper=0.0, symmetric=False, second_distance=None, extent="distance"):
        result = self.build_extrusion(index, amount, operation, offset, taper, symmetric, second_distance, extent)
        self.commit_result(result)
        return result

    def build_revolution(self, index, axis, angle, operation="new_body"):
        if not 0 <= index < len(self.profiles):
            raise ValueError("Select a closed profile.")
        direction = self.u_axis if axis == "horizontal" else self.v_axis
        return build_revolution(self.profiles[index]["face"], self.active_center, direction,
                                angle, operation, getattr(self, "current_part", None))

    def build_primitive(self, kind, dimensions, position, operation):
        return build_primitive(kind, dimensions, position, operation, self.current_part)

    def convert_legacy_model(self):
        if self.parametric_graph is not None: raise ValueError('This model is already parametric.')
        if self.sketch_editing or self.pending_sketch or self.selecting_sketch_plane:
            raise ValueError('Finish or cancel the current sketch first.')
        from UI.model.legacy_conversion import convert_snapshot
        graph = convert_snapshot(self.current_part, self.undo_stack, self.redo_stack,
                                 {'entries': self.timeline.entries, 'cursor': self.timeline.cursor})
        self.install_parametric_graph(graph, reset_history=True)
        self.legacy_migration_redo = None
        self.status_callback('Model converted; existing solid is a frozen base. Undo restores the snapshot history.')

    def install_parametric_graph(self, graph, record=True, reset_history=False, allow_empty=False):
        # Rebuild has already succeeded; choose the last available solid result.
        solid = next((graph.results[item.id] for item in reversed(graph.features)
                      if item.id in graph.results and hasattr(graph.results[item.id], 'solids')
                      and graph.results[item.id].solids()), None)
        if solid is None and self.current_part is not None and not allow_empty:
            raise ValueError('The graph has no active solid result.')
        previous = self.parametric_graph.to_data() if self.parametric_graph is not None else None
        if solid is not None:
            self.display_result(solid)
        else:
            self.plotter.remove_actor('main_part')
            self.current_part = None
            self.pending_sketch = False
            for profile in self.profiles:
                self.plotter.remove_actor(profile['name'])
            self.profiles = []
            self.selected_profile_index = None
        self.parametric_graph = graph
        if reset_history or previous is None:
            self.graph_undo.clear()
            self.graph_redo.clear()
        elif record and previous != graph.to_data():
            self.graph_undo.append(previous)
            self.graph_redo.clear()
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.timeline = Timeline()
        self.timeline_callback()

    def commit_result(self, part, title="Solid operation", parameters=None):
        if self.parametric_graph is not None:
            raise ValueError('This operation is not yet supported in parametric mode; use the Timeline editor.')
        # Validate metadata before modifying the model.
        from UI.model.timeline import validate_entry
        record = validate_entry({'title': title, 'parameters': parameters or {}})
        previous = self.current_part
        self.display_result(part)
        self.undo_stack.append(previous)
        self.redo_stack.clear()
        self.legacy_migration_redo = None
        self.timeline.append(record['title'], record['parameters'])
        self.timeline_callback()

    def _restore_result(self, part):
        if part is not None:
            self.display_result(part)
        else:
            self.clear_selected_edges()
            self.edge_selection_mode = False
            self.current_part = None
            self.selected_face_id = -1
            self.hover_id = -1
            self._clear_highlight(render=False)
            self.plotter.remove_actor("main_part")
            self.plotter.render()

    def undo_solid(self):
        if self.sketch_editing or self.selecting_sketch_plane or self.pending_sketch:
            self.status_callback('Finish or cancel the current sketch before undo.')
            return False
        if self.parametric_graph is not None:
            if not self.graph_undo:
                if len(self.parametric_graph.features) == 1 and self.parametric_graph.features[0].kind == 'frozen_body' and 'legacy_backup' in self.parametric_graph.features[0].parameters:
                    from UI.model.legacy_conversion import restore_snapshot
                    graph_data = self.parametric_graph.to_data()
                    body, undo, redo, timeline = restore_snapshot(self.parametric_graph)
                    self.display_result(body)
                    self.parametric_graph = None
                    self.undo_stack, self.redo_stack = undo, redo
                    self.timeline = Timeline()
                    self.timeline.entries, self.timeline.cursor = timeline['entries'], timeline['cursor']
                    self.legacy_migration_redo = graph_data
                    self.timeline_callback()
                    self.status_callback('Legacy model and its snapshot history restored.')
                    return True
                self.status_callback("Nothing to undo.")
                return False
            previous = self.parametric_graph.to_data()
            target = FeatureGraph.from_data(self.graph_undo[-1])
            self.install_parametric_graph(target, record=False, allow_empty=True)
            if self.current_part is None:
                sketch_id = next((item.id for item in reversed(target.features)
                                  if item.kind == 'planar_sketch' and item.id in target.results), None)
                if sketch_id: self.show_parametric_profiles(sketch_id)
            self.graph_undo.pop()
            self.graph_redo.append(previous)
            self.status_callback("Parametric edit undone.")
            return True
        if self.sketch_editing or self.selecting_sketch_plane or self.pending_sketch or self.profiles:
            self.status_callback("Finish or cancel the current sketch before undoing a solid operation.")
            return False
        if not self.undo_stack:
            self.status_callback("Nothing to undo.")
            return False
        target = self.undo_stack[-1]
        current = self.current_part
        self._restore_result(target)
        self.legacy_migration_redo = None
        self.undo_stack.pop()
        self.redo_stack.append(current)
        self.timeline.undo()
        self.timeline_callback()
        self.status_callback("Solid operation undone.")
        return True

    def redo_solid(self):
        if self.sketch_editing or self.selecting_sketch_plane or self.pending_sketch:
            self.status_callback('Finish or cancel the current sketch before redo.')
            return False
        if self.parametric_graph is not None:
            if not self.graph_redo:
                self.status_callback("Nothing to redo.")
                return False
            previous = self.parametric_graph.to_data()
            target = FeatureGraph.from_data(self.graph_redo[-1])
            self.install_parametric_graph(target, record=False, allow_empty=True)
            if self.current_part is None:
                sketch_id = next((item.id for item in reversed(target.features)
                                  if item.kind == 'planar_sketch' and item.id in target.results), None)
                if sketch_id: self.show_parametric_profiles(sketch_id)
            self.graph_redo.pop()
            self.graph_undo.append(previous)
            self.status_callback("Parametric edit redone.")
            return True
        if self.legacy_migration_redo is not None:
            graph = FeatureGraph.from_data(self.legacy_migration_redo)
            self.install_parametric_graph(graph, reset_history=True)
            self.legacy_migration_redo = None
            self.status_callback('Legacy model converted to a frozen parametric base.')
            return True
        if self.sketch_editing or self.selecting_sketch_plane or self.pending_sketch or self.profiles:
            self.status_callback("Finish or cancel the current sketch before redoing a solid operation.")
            return False
        if not self.redo_stack:
            self.status_callback("Nothing to redo.")
            return False
        target = self.redo_stack[-1]
        current = self.current_part
        self._restore_result(target)
        self.redo_stack.pop()
        self.undo_stack.append(current)
        self.timeline.redo()
        self.timeline_callback()
        self.status_callback("Solid operation redone.")
        return True

    def display_result(self, part):
        # Prepare the display mesh before committing or removing the existing scene.
        pv_mesh = b123d_to_pro_mesh(part, 0.001, 0.05)
        self.selected_face_id = -1
        self.hover_id = -1
        self._clear_highlight(render=False)
        self.clear_selected_edges()
        self.edge_selection_mode = False
        self.current_part = part
        self.pending_sketch = False
        self.profile_selection_callback = None
        self.selected_profile_index = None
        for profile in self.profiles:
            self.plotter.remove_actor(profile["name"])
        self.profiles = []
        for edge_actor in self.sketch_edge_actors:
            self.plotter.remove_actor(edge_actor)
        self.sketch_edge_actors = []
        for sketch in self.sketchs_data.values():
            for tool in sketch["sketch_func"]:
                for point_actor in tool.render_obj["points"]:
                    self.plotter.remove_actor(point_actor)
        self.status_callback("Select a planar face to create another sketch.")

        for name in list(self.plotter.actors.keys()):
            if any(prefix in name for prefix in ["line_", "profile_", "point_", "sketch_grid"]):
                self.plotter.remove_actor(name)

        actor = self.plotter.add_mesh(
            pv_mesh,
            color="#b0b0b0",
            pbr=True,
            metallic=0.3,
            roughness=0.4,
            smooth_shading=True,
            show_edges=False,
            split_sharp_edges=True,
            name="main_part",
            reset_camera=False,
        )
        
        # ФИКС КРАСНОЙ ГРАНИ: три строки которые решают всё
        actor.GetProperty().SetBackfaceCulling(False)
        actor.SetBackfaceProperty(None)
        self.plotter.renderer.SetTwoSidedLighting(True)
        
        self.plotter.view_isometric()
        self.plotter.reset_camera()
        
    def on_cell_picked(self, picked_data):
        """Подсвечивает всю грань, к которой принадлежит выбранный треугольник"""
        if picked_data is None:
            return

        # 1. Узнаем ID грани, в которую мы попали
        cell_id = picked_data["orig_extract_id"]
        
        # Получаем меш детали
        main_mesh = self.plotter.actors["main_part"].mapper.dataset
        
        # Смотрим, к какой грани принадлежит этот треугольник
        clicked_face_id = main_mesh.cell_data["face_id"][cell_id]

        # 2. Находим ВСЕ треугольники этой грани
        mask = main_mesh.cell_data["face_id"] == clicked_face_id
        
        # 3. Извлекаем подмеш (только эту грань)
        face_mesh = main_mesh.extract_cells(np.where(mask)[0])

        # 4. Рисуем подсветку поверх основной детали
        self.plotter.remove_actor("face_highlight")
        highlight_actor = self.plotter.add_mesh(
            face_mesh,
            color="#95d5ff",      # Синяя подсветка как во Fusion
            opacity=0.6,
            smooth_shading=True,
            name="face_highlight",
            pickable=False        # Чтобы подсветка сама не "ловила" клики
        )
        
        # 5. Сохраняем выбранную грань для дальнейших операций
        self.selected_face_id = clicked_face_id
        
        highlight_actor.GetMapper().SetRelativeCoincidentTopologyPolygonOffsetParameters(-10, -10)
        
        # Сдвигаем на 0.02 мм, чтобы гарантированно не было пересечений под углом
        try:
            n = face_mesh.cell_normals[0]
            highlight_actor.SetPosition(n[0]*0.02, n[1]*0.02, n[2]*0.02)
        except: pass

        self.selected_face_id = clicked_face_id
        self.plotter.render()
        
    def activate_line_mode(self):
        line = Line(self.plotter, self.sketchs_data[self.activated_sketch_id], self)
        self.sketchs_data[self.activated_sketch_id]["sketch_func"].append(line)
        line.is_construction = self.construction_mode
        self.active_tool = line
        line.create_line()

    def activate_spline_mode(self, kind):
        tool = Spline(self.plotter, self.get_current_sketch_data(), self, kind)
        self.get_current_sketch_data()['sketch_func'].append(tool)
        tool.is_construction = self.construction_mode
        self.active_tool = tool
        tool.start()
        
    def activate_circle_mode(self):
        circle = Circle(self.plotter, self.sketchs_data[self.activated_sketch_id], self)
        self.sketchs_data[self.activated_sketch_id]["sketch_func"].append(circle)
        circle.is_construction = self.construction_mode
        self.active_tool = circle
        circle.create_circle()

    def activate_curve_mode(self):
        curve = Curve(self.plotter, self.sketchs_data[self.activated_sketch_id], self)
        self.sketchs_data[self.activated_sketch_id]["sketch_func"].append(curve)
        curve.is_construction = self.construction_mode
        self.active_tool = curve
        curve.create_curve()
        
    def activate_ellipse_arc_mode(self):
        ellipse_arc = Ellipse(self.plotter, self.sketchs_data[self.activated_sketch_id], self)
        self.sketchs_data[self.activated_sketch_id]["sketch_func"].append(ellipse_arc)
        ellipse_arc.is_construction = self.construction_mode
        self.active_tool = ellipse_arc
        ellipse_arc.create_arc()
    def activate_straight_shape(self, kind, sides=6):
        shape = StraightShape(self.plotter, self.get_current_sketch_data(), self, kind, sides)
        self.get_current_sketch_data()["sketch_func"].append(shape)
        shape.is_construction = self.construction_mode
        self.active_tool = shape
        shape.start()

    def activate_point_circle(self, mode):
        tool = PointCircle(self.plotter, self.get_current_sketch_data(), self, mode)
        self.get_current_sketch_data()["sketch_func"].append(tool)
        tool.is_construction = self.construction_mode
        self.active_tool = tool
        tool.create_circle()

    def activate_sketch_point(self):
        tool = SketchPoint(self.plotter, self.get_current_sketch_data(), self)
        self.get_current_sketch_data()["sketch_func"].append(tool)
        tool.is_construction = self.construction_mode
        self.active_tool = tool
        tool.start()

    def activate_slot(self, mode='center_to_center'):
        tool = Slot(self.plotter, self.get_current_sketch_data(), self, mode)
        self.get_current_sketch_data()["sketch_func"].append(tool)
        tool.is_construction = self.construction_mode
        self.active_tool = tool
        tool.start()

    def activate_center_arc(self):
        tool = CenterArc(self.plotter, self.get_current_sketch_data(), self)
        self.get_current_sketch_data()["sketch_func"].append(tool)
        tool.is_construction = self.construction_mode
        self.active_tool = tool
        tool.start()

    def activate_elliptical_arc(self):
        tool = EllipseArc(self.plotter, self.get_current_sketch_data(), self)
        self.get_current_sketch_data()["sketch_func"].append(tool)
        tool.is_construction = self.construction_mode
        self.active_tool = tool
        tool.create_arc()
        self.status_callback("Ellipse arc: select center, two axes, start and end.")
