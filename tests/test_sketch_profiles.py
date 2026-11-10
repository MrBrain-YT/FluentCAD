import sys
import math
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from build123d import extrude, Box, Cylinder, Sphere, Face, Plane, Align, Pos, Rot, import_step, import_stl
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from UI.model.profiles import build_profiles, sketch_edges, split_edges
from UI.model.ActionManager import ActionManager
from UI.model.ModelConverter import b123d_to_pro_mesh
from UI.model.solid_operations import build_extrusion, build_revolution, build_primitive, import_step_body
from UI.model.export import export_body
from UI.model.snapshot import save_snapshot, load_snapshot
from UI.model.timeline import Timeline, validate_timeline
from UI.model.feature_graph import Feature, FeatureGraph, RebuildError, resolve_parameters
from UI.model.sketch_drag import move_sketch_point, set_circle_radius, add_sketch_constraint
from UI.model.geometry_references import make_reference, resolve_reference
from UI.model.measurement import measure_face, measure_model, measure_edges
from UI.model.expressions import evaluate_expression
from UI.model.constraints import Constraint, solve_constraints
from UI.model.sketch_constraints import line_points, add_automatic_coincidences, apply_constraint
from UI.model.transforms import move_copy, scale_model, mirror_model, rectangular_pattern, circular_pattern, body_index_for_face, apply_to_body
from UI.model.surface_operations import fillet_all, chamfer_all, shell_face, offset_solid, hole_at_face_center, fillet_edges, chamfer_edges
from UI.model.edge_selection import nearest_screen_edge, face_edge_indices
from UI.main.content import LayoutManager
from UI.main.extrude_handle import world_per_pixel, ExtrudeHandle, MoveAxisHandles, ParameterHandles, OperationHandle, projected_ring_angle
from UI.model.tools.StraightShape import shape_points
from UI.model.tools.PointCircle import circle_from_points
from UI.model.tools.Slot import slot_edges, normalize_slot_points
from UI.model.tools.ArcTools import centered_arc_points, EllipseArc
from UI.model.tools.Line import Line
from UI.model.tools.Circle import Circle
from UI.model.tools.Curve import Curve


class SketchProfilesTest(unittest.TestCase):
    normal = np.array([0., 0., 1.])

    def line(self, points):
        return SimpleNamespace(type='line', points=points)

    def test_closed_segments_create_extrudable_face(self):
        tools = [self.line([(0, 0, 0), (2, 0, 0)]),
                 self.line([(2, 0, 0), (2, 2, 0), (0, 2, 0), (0, 0, 0)])]
        faces = build_profiles(tools, self.normal)
        self.assertEqual(len(faces), 1)
        self.assertAlmostEqual(faces[0].area, 4)
        self.assertAlmostEqual(extrude(faces[0], amount=-3, dir=tuple(self.normal)).volume, 12)

    def test_open_contour_is_not_a_profile(self):
        self.assertEqual(build_profiles([self.line([(0, 0, 0), (1, 0, 0)])], self.normal), [])

    def test_ellipse_is_exact(self):
        ellipse = SimpleNamespace(type='ellipse', points=[], center=np.zeros(3),
                                  major_radius=2., minor_radius=1.,
                                  major_axis_vector=np.array([1., 0., 0.]), is_closed=True)
        self.assertAlmostEqual(build_profiles([ellipse], self.normal)[0].area, 2 * np.pi)

    def test_coordinates_round_trip_on_offset_plane(self):
        manager = ActionManager(None)
        manager.active_center = np.array([2., 3., 4.])
        manager.u_axis = np.array([0., 1., 0.])
        manager.v_axis = np.array([0., 0., 1.])
        point = np.array([2., 8., 10.])
        np.testing.assert_allclose(manager.get_global_pos_in_sketch(manager.get_local_pos_in_sketch(point)), point)


class GridAndShapesTest(unittest.TestCase):
    def manager(self):
        manager = ActionManager(None)
        manager.active_center = np.array([2., 3., 4.])
        manager.active_normal = np.array([0., 0., 1.])
        manager.u_axis = np.array([1., 0., 0.])
        manager.v_axis = np.array([0., 1., 0.])
        manager.sketch_editing = True
        manager.activated_sketch_id = 1
        manager.sketchs_data[1] = {"sketch_func": []}
        class Plotter:
            camera = SimpleNamespace(parallel_scale=5.)
            def remove_actor(self, name): pass
            def add_mesh(self, mesh, **kwargs): self.mesh = mesh
        manager.plotter = Plotter()
        return manager

    def test_grid_and_snap_stay_fixed_at_different_zooms(self):
        manager = self.manager()
        for scale in (2., 10., 100.):
            manager.plotter.camera.parallel_scale = scale
            manager.update_grid_lod()
            self.assertEqual(manager.grid_step, 1.)
            local = np.array([manager.get_local_pos_in_sketch(p) for p in manager.plotter.mesh.points])
            self.assertTrue(np.allclose(local, np.round(local)))
            point, _ = manager.get_snapped_pos(np.array([4.03, 6.02, 4.]))
            np.testing.assert_allclose(point, [4., 6., 4.])

    def test_independent_grid_settings_and_nearest_point(self):
        manager = self.manager()
        manager.set_grid_step(0.5)
        manager.update_grid_lod()
        local = np.array([manager.get_local_pos_in_sketch(p) for p in manager.plotter.mesh.points])
        np.testing.assert_allclose(local / 0.5, np.round(local / 0.5))
        manager.sketch_intersection_points = [np.array([4.15, 6., 4.]), np.array([4.09, 6., 4.])]
        point, _ = manager.get_snapped_pos(np.array([4.08, 6., 4.]))
        np.testing.assert_allclose(point, [4.09, 6., 4.])
        manager.snap_to_points = False
        point, _ = manager.get_snapped_pos(np.array([4.04, 6., 4.]))
        np.testing.assert_allclose(point, [4., 6., 4.])
        manager.snap_to_grid = False
        point, _ = manager.get_snapped_pos(np.array([4.04, 6., 4.]))
        np.testing.assert_allclose(point, [4.04, 6., 4.])
        manager.set_grid_visible(False)
        self.assertFalse(manager.grid_visible)
        with self.assertRaises(ValueError):
            manager.set_grid_step(0)
        self.assertEqual(manager.grid_step, 0.5)

    def test_rectangle_and_polygon_make_profiles(self):
        center = np.zeros(3)
        u, v = np.array([1., 0., 0.]), np.array([0., 1., 0.])
        for kind, clicks, expected_area in [
            ('rectangle', [(0, 0, 0), (4, 3, 0)], 12),
            ('center_rectangle', [(0, 0, 0), (2, 1, 0)], 8),
            ('three_point_rectangle', [(0, 0, 0), (4, 0, 0), (2, 3, 0)], 12),
        ]:
            points = shape_points(kind, clicks, center, u, v)
            face = build_profiles([SimpleNamespace(type='line', points=points)], np.array([0., 0., 1.]))[0]
            self.assertAlmostEqual(face.area, expected_area)
        polygon = shape_points('polygon', [(0, 0, 0), (1, 0, 0)], center, u, v, 6)
        self.assertEqual(len(polygon), 7)
        self.assertIsNone(shape_points('rectangle', [(0, 0, 0), (0, 2, 0)], center, u, v))


class TopologyTest(unittest.TestCase):
    normal = np.array([0., 0., 1.])

    def circle(self, x, radius):
        return SimpleNamespace(type='circle', points=[], center=np.array([x, 0., 0.]), radius=radius)

    def line(self, a, b):
        return SimpleNamespace(type='line', points=[np.array(a, dtype=float), np.array(b, dtype=float)])

    def ellipse(self):
        return SimpleNamespace(type='ellipse', points=[], center=np.zeros(3), major_radius=2.,
                               minor_radius=1., major_axis_vector=np.array([1., 0., 0.]), is_closed=True)

    def test_ellipse_intersected_by_line_preserves_exact_curve(self):
        ellipse, line = self.ellipse(), self.line((-3, 0, 0), (3, 0, 0))
        edges = split_edges(sketch_edges([ellipse, line], self.normal))
        self.assertEqual(len(edges), 5)
        curved = [edge for edge in edges if edge.geom_type.name == 'ELLIPSE']
        self.assertEqual(len(curved), 2)
        for edge in curved:
            x, y, _ = tuple(edge.position_at(0.5))
            self.assertAlmostEqual((x / 2) ** 2 + y ** 2, 1)
        faces = build_profiles([ellipse, line], self.normal)
        self.assertEqual(len(faces), 2)
        for face in faces:
            self.assertAlmostEqual(face.area, np.pi)
        self.assertEqual(len(line.points), 2)
        self.assertTrue(ellipse.is_closed)

    def test_mixed_curve_pairs_and_offset_plane(self):
        arc = SimpleNamespace(type='curve', points=[np.array([-2., 0., 0.]),
                              np.array([2., 0., 0.]), np.array([0., 2., 0.])])
        cases = [
            [arc, self.line((0, -3, 0), (0, 3, 0))],
            [arc, self.circle(1., 1.5)],
            [arc, self.ellipse()],
            [self.circle(1., 1.5), self.ellipse()],
            [self.ellipse(), SimpleNamespace(type='ellipse', points=[], center=np.array([1., 0., 0.]),
                major_radius=2., minor_radius=1., major_axis_vector=np.array([1., 0., 0.]), is_closed=True)],
        ]
        for tools in cases:
            with self.subTest(types=[tool.type for tool in tools]):
                edges = sketch_edges(tools, self.normal)
                self.assertGreater(len(split_edges(edges)), len(edges))
                build_profiles(tools, self.normal)
        offset_circle = self.circle(0., 2.)
        offset_circle.center = np.array([4., 0., 0.])
        faces = build_profiles([offset_circle], np.array([1., 0., 0.]))
        self.assertEqual(len(faces), 1)
        self.assertAlmostEqual(faces[0].area, 4 * np.pi)

    def test_nested_circles_have_annulus_and_disk(self):
        faces = build_profiles([self.circle(0., 3.), self.circle(0., 1.)], self.normal)
        self.assertEqual(len(faces), 2)
        areas = sorted(face.area for face in faces)
        self.assertAlmostEqual(areas[0], np.pi)
        self.assertAlmostEqual(areas[1], 8 * np.pi)
        self.assertAlmostEqual(extrude(max(faces, key=lambda f: f.area), amount=2).volume, 16 * np.pi)

    def test_overlapping_and_tangent_circles(self):
        overlap = build_profiles([self.circle(0., 2.), self.circle(2., 2.)], self.normal)
        self.assertEqual(len(overlap), 3)
        self.assertAlmostEqual(sum(f.area for f in overlap), 8 * np.pi - (8 * np.pi / 3 - 2 * np.sqrt(3)))
        self.assertEqual(len(build_profiles([self.circle(0., 1.), self.circle(2., 1.)], self.normal)), 2)

    def test_crossed_open_lines_do_not_form_face(self):
        tools = [self.line((-1, 0, 0), (1, 0, 0)), self.line((0, -1, 0), (0, 1, 0))]
        self.assertEqual(len(split_edges(sketch_edges(tools, self.normal))), 4)
        self.assertEqual(build_profiles(tools, self.normal), [])


class SelectionTest(unittest.TestCase):
    def test_profile_pick_highlights_and_invokes_extrude(self):
        manager = ActionManager(None)
        first = Mock()
        first.prop = SimpleNamespace(color=None, opacity=None)
        second = Mock()
        second.prop = SimpleNamespace(color=None, opacity=None)
        manager.profiles = [{"name": "one"}, {"name": "two"}]
        manager.plotter = SimpleNamespace(actors={"one": first, "two": second}, render=Mock())
        manager._pick_actor = Mock(return_value=SimpleNamespace(GetActor=lambda: second))
        callback = Mock()
        manager.request_profile_selection(callback)
        self.assertTrue(manager._select_profile())
        self.assertEqual(manager.selected_profile_index, 1)
        self.assertEqual(second.prop.color, "#f39c12")
        callback.assert_called_once_with(1)
        self.assertIsNone(manager.profile_selection_callback)

    def test_planar_solid_face_uses_exact_normal_and_pick_position(self):
        manager = ActionManager(None)
        manager.current_part = Box(2, 4, 6)
        mesh = b123d_to_pro_mesh(manager.current_part)
        mapper = SimpleNamespace(GetInput=lambda: mesh)
        actor = SimpleNamespace(GetMapper=lambda: mapper)
        manager.plotter = SimpleNamespace(actors={"main_part": actor})
        position = (1., 0.5, 0.2)
        picker = SimpleNamespace(GetActor=lambda: actor, GetCellId=lambda: 0, GetPickPosition=lambda: position)
        manager._pick_actor = Mock(return_value=picker)
        manager._start_sketch = Mock()
        manager._select_sketch_plane()
        normal, center = manager._start_sketch.call_args.args
        face_id = mesh.cell_data["face_id"][0]
        np.testing.assert_allclose(normal, tuple(list(manager.current_part.faces())[face_id].normal_at()))
        np.testing.assert_allclose(center, position)

    def test_curved_solid_face_is_rejected(self):
        manager = ActionManager(None)
        manager.current_part = Cylinder(2, 4)
        faces = list(manager.current_part.faces())
        face_id = next(i for i, face in enumerate(faces) if face.geom_type.name != "PLANE")
        mesh = b123d_to_pro_mesh(manager.current_part)
        cell_id = int(np.flatnonzero(mesh.cell_data["face_id"] == face_id)[0])
        actor = SimpleNamespace(GetMapper=lambda: SimpleNamespace(GetInput=lambda: mesh))
        manager.plotter = SimpleNamespace(actors={"main_part": actor})
        manager._pick_actor = Mock(return_value=SimpleNamespace(GetActor=lambda: actor, GetCellId=lambda: cell_id))
        manager._start_sketch = Mock()
        manager.status_callback = Mock()
        manager._select_sketch_plane()
        manager._start_sketch.assert_not_called()
        manager.status_callback.assert_called_once()


class ExtrusionOperationsTest(unittest.TestCase):
    def setUp(self):
        self.body = Box(4, 4, 4, align=(Align.CENTER, Align.CENTER, Align.MIN))
        self.face = Face.make_rect(2, 2, Plane(origin=(0, 0, 4)))
        self.normal = (0, 0, 1)

    def test_new_body_preserves_existing_body(self):
        result = build_extrusion(self.face, self.normal, 2, 'new_body', self.body)
        self.assertEqual(len(result.solids()), 2)
        self.assertAlmostEqual(result.volume, 72)
        self.assertAlmostEqual(self.body.volume, 64)

    def test_join_cut_and_intersect(self):
        self.assertAlmostEqual(build_extrusion(self.face, self.normal, 2, 'join', self.body).volume, 72)
        self.assertAlmostEqual(build_extrusion(self.face, self.normal, -2, 'cut', self.body).volume, 56)
        self.assertAlmostEqual(build_extrusion(self.face, self.normal, -2, 'intersect', self.body).volume, 8)

    def test_invalid_operations_do_not_mutate_body(self):
        for operation, distance in [('join', -20), ('cut', 2), ('intersect', 2)]:
            if operation == 'join':
                with self.assertRaises(ValueError):
                    build_extrusion(self.face, self.normal, distance, operation, self.body, offset=100)
            else:
                with self.assertRaises(ValueError):
                    build_extrusion(self.face, self.normal, distance, operation, self.body)
        self.assertAlmostEqual(self.body.volume, 64)

    def test_selected_planar_face_can_be_extruded(self):
        manager = ActionManager(None)
        manager.current_part = self.body
        faces = list(self.body.faces())
        manager.selected_face_id = next(i for i, face in enumerate(faces)
            if abs(face.center().Z - 4) < 1e-7)
        self.assertTrue(manager.prepare_selected_face_profile())
        result = manager.build_extrusion(0, 1, 'join')
        self.assertAlmostEqual(result.volume, 80)
        manager.clear_temporary_face_profile()
        self.assertEqual(manager.profiles, [])

    def test_symmetric_and_offset(self):
        result = build_extrusion(self.face, self.normal, 2, offset=3, symmetric=True)
        self.assertAlmostEqual(result.volume, 16)
        self.assertAlmostEqual(result.bounding_box().min.Z, 5)
        self.assertAlmostEqual(result.bounding_box().max.Z, 9)


class NewSketchToolsTest(unittest.TestCase):
    def test_circles_from_two_and_three_points(self):
        center, radius = circle_from_points([(0, 0, 0), (4, 0, 0)], 'diameter')
        np.testing.assert_allclose(center, (2, 0, 0))
        self.assertAlmostEqual(radius, 2)
        center, radius = circle_from_points([(1, 0, 0), (0, 1, 0), (-1, 0, 0)], 'three_point')
        np.testing.assert_allclose(center, (0, 0, 0), atol=1e-9)
        self.assertAlmostEqual(radius, 1)
        with self.assertRaises(ValueError):
            circle_from_points([(0, 0, 0), (1, 0, 0), (2, 0, 0)], 'three_point')

    def test_polygon_modes(self):
        center = np.zeros(3)
        u, v = np.array([1., 0., 0.]), np.array([0., 1., 0.])
        inscribed = shape_points('polygon', [(0, 0, 0), (1, 0, 0)], center, u, v, 4)
        circumscribed = shape_points('circumscribed_polygon', [(0, 0, 0), (1, 0, 0)], center, u, v, 4)
        edge = shape_points('edge_polygon', [(0, 0, 0), (2, 0, 0)], center, u, v, 4)
        areas = [build_profiles([SimpleNamespace(type='line', points=points)], (0, 0, 1))[0].area
                 for points in (inscribed, circumscribed, edge)]
        np.testing.assert_allclose(areas, [2, 4, 4], atol=1e-7)
        np.testing.assert_allclose(edge[0], [0, 0, 0], atol=1e-9)
        np.testing.assert_allclose(edge[1], [2, 0, 0], atol=1e-9)

    def test_two_sided_extrusion(self):
        result = build_extrusion(Face.make_rect(2, 2), (0, 0, 1), 3, second_distance=2)
        self.assertAlmostEqual(result.volume, 20)
        self.assertAlmostEqual(result.bounding_box().min.Z, -2)
        self.assertAlmostEqual(result.bounding_box().max.Z, 3)
        with self.assertRaises(ValueError):
            build_extrusion(Face.make_rect(2, 2), (0, 0, 1), -3, second_distance=2)


class RevolutionTest(unittest.TestCase):
    def test_full_and_partial_revolution(self):
        face = Face.make_rect(2, 2, Plane(origin=(3, 0, 0)))
        full = build_revolution(face, (0, 0, 0), (0, 1, 0), 360)
        half = build_revolution(face, (0, 0, 0), (0, 1, 0), 180)
        self.assertAlmostEqual(full.volume, 24 * np.pi)
        self.assertAlmostEqual(half.volume, 12 * np.pi)
        with self.assertRaises(ValueError):
            build_revolution(face, (0, 0, 0), (0, 1, 0), 361)


class SlotAndPreviewTest(unittest.TestCase):
    def test_slot_profile_and_extrusion(self):
        points = [(0, 0, 0), (4, 0, 0), (0, 1, 0)]
        slot = SimpleNamespace(type='slot', points=points)
        self.assertEqual(len(slot_edges(points, (0, 0, 1))), 4)
        faces = build_profiles([slot], (0, 0, 1))
        self.assertEqual(len(faces), 1)
        self.assertAlmostEqual(faces[0].area, 8 + np.pi)
        self.assertAlmostEqual(extrude(faces[0], amount=2).volume, 2 * (8 + np.pi))
        crossing = SimpleNamespace(type='line', points=[(2, -2, 0), (2, 2, 0)])
        self.assertEqual(len(build_profiles([slot, crossing], (0, 0, 1))), 2)
        self.assertEqual(slot_edges([(0, 0, 0), (0, 0, 0), (1, 0, 0)], (0, 0, 1)), [])

    def test_slot_creation_modes(self):
        normal = np.array([0., 0., 1.])
        for mode, points in [
            ('center_to_center', [(0, 0, 0), (4, 0, 0), (0, 1, 0)]),
            ('overall', [(-1, 0, 0), (5, 0, 0), (-1, 1, 0)]),
            ('center_point', [(2, 0, 0), (4, 0, 0), (2, 1, 0)]),
        ]:
            normalized = normalize_slot_points(points, normal, mode)
            face = build_profiles([SimpleNamespace(type='slot', points=normalized)], normal)[0]
            self.assertAlmostEqual(face.area, 8 + np.pi)
        with self.assertRaises(ValueError):
            normalize_slot_points([(0, 0, 0), (1, 0, 0), (0, 1, 0)], normal, 'overall')

    def test_line_and_circle_preview_cleanup(self):
        iren = SimpleNamespace(add_observer=Mock(return_value=10), remove_observer=Mock())
        plotter = SimpleNamespace(iren=iren, track_click_position=Mock(), disable_picking=Mock(),
                                  remove_actor=Mock(), render=Mock(), add_mesh=Mock(),
                                  pick_mouse_position=Mock(return_value=np.array([2., 0., 0.])))
        sketch = SimpleNamespace(get_snapped_pos=lambda p: (p, p[:2]),
                                 u_axis=np.array([1., 0., 0.]), v_axis=np.array([0., 1., 0.]))
        for cls, method in [(Line, 'create_line'), (Circle, 'create_circle')]:
            tool = cls(plotter, {}, sketch)
            getattr(tool, method)()
            tool.points = [np.zeros(3)]
            tool.draw_preview_line()
            self.assertEqual(len(tool.points), 1)
            tool.stop_drawing()
            iren.remove_observer.assert_called_with(10)
            self.assertFalse(tool.is_drawing)


class ConstructionAndExportTest(unittest.TestCase):
    def test_construction_geometry_does_not_create_or_split_profiles(self):
        rectangle = SimpleNamespace(type='line', is_construction=False,
            points=[(0, 0, 0), (4, 0, 0), (4, 4, 0), (0, 4, 0), (0, 0, 0)])
        reference = SimpleNamespace(type='line', is_construction=True,
            points=[(2, -1, 0), (2, 5, 0)])
        faces = build_profiles([rectangle, reference], (0, 0, 1))
        self.assertEqual(len(faces), 1)
        self.assertAlmostEqual(faces[0].area, 16)
        self.assertEqual(build_profiles([reference], (0, 0, 1)), [])
        reference.is_construction = False
        self.assertEqual(len(build_profiles([rectangle, reference], (0, 0, 1))), 2)

    def test_intersection_with_construction_remains_snappable(self):
        manager = ActionManager(None)
        manager.active_normal = np.array([0., 0., 1.])
        manager.activated_sketch_id = 1
        regular = SimpleNamespace(type='line', points=[(-2, 0, 0), (2, 0, 0)])
        reference = SimpleNamespace(type='line', is_construction=True, points=[(0, -2, 0), (0, 2, 0)])
        manager.sketchs_data[1] = {'sketch_func': [regular, reference]}
        meshes = []
        manager.plotter = SimpleNamespace(remove_actor=Mock(), render=Mock(),
            add_mesh=lambda mesh, **kwargs: meshes.append(mesh) or object())
        manager.redraw_sketch_geometry()
        self.assertTrue(any(np.linalg.norm(point) < 1e-7 for point in manager.sketch_intersection_points))
        self.assertEqual(len(meshes), 2)
        self.assertEqual(meshes[1].n_verts, 0)

    def test_step_and_stl_export(self):
        body = Box(2, 3, 4)
        with tempfile.TemporaryDirectory() as directory:
            step = Path(directory) / 'model.step'
            stl = Path(directory) / 'model.stl'
            export_body(body, step, 'step')
            export_body(body, stl, 'stl')
            self.assertAlmostEqual(import_step(step).volume, 24)
            self.assertGreater(len(import_stl(stl).faces()), 0)
        with self.assertRaises(ValueError):
            export_body(None, 'unused.step', 'step')


class SolidHistoryTest(unittest.TestCase):
    def test_undo_redo_and_new_branch(self):
        manager = ActionManager(None)
        first, second, third = Box(1, 1, 1), Box(2, 1, 1), Box(3, 1, 1)
        manager.display_result = lambda shape: setattr(manager, 'current_part', shape)
        manager._restore_result = lambda shape: setattr(manager, 'current_part', shape)
        manager.commit_result(first)
        manager.commit_result(second)
        self.assertTrue(manager.undo_solid())
        self.assertIs(manager.current_part, first)
        self.assertTrue(manager.redo_solid())
        self.assertIs(manager.current_part, second)
        self.assertTrue(manager.undo_solid())
        manager.commit_result(third)
        self.assertFalse(manager.redo_solid())
        self.assertTrue(manager.undo_solid())
        self.assertIs(manager.current_part, first)
        self.assertTrue(manager.undo_solid())
        self.assertIsNone(manager.current_part)
        self.assertFalse(manager.undo_solid())
        self.assertTrue(manager.redo_solid())
        self.assertIs(manager.current_part, first)

    def test_history_does_not_discard_open_sketch(self):
        manager = ActionManager(None)
        manager.undo_stack.append(None)
        manager.sketch_editing = True
        self.assertFalse(manager.undo_solid())
        self.assertEqual(manager.undo_stack, [None])
        manager.sketch_editing = False
        manager.profiles = [{'name': 'pending'}]
        self.assertFalse(manager.undo_solid())
        self.assertEqual(manager.undo_stack, [None])

    def test_failed_commit_does_not_change_history(self):
        manager = ActionManager(None)
        manager.display_result = Mock(side_effect=ValueError('failed'))
        with self.assertRaises(ValueError):
            manager.commit_result(Box(1, 1, 1))
        self.assertEqual(manager.undo_stack, [])


class ModelSnapshotTest(unittest.TestCase):
    def test_solid_and_history_round_trip(self):
        current, old, future = Box(2, 3, 4), Box(1, 2, 3), Box(3, 3, 4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.fcad'
            save_snapshot(path, current, [None, old], [future])
            loaded, undo, redo = load_snapshot(path)
            self.assertAlmostEqual(loaded.volume, 24)
            self.assertEqual(len(undo), 2)
            self.assertIsNone(undo[0])
            self.assertAlmostEqual(undo[1].volume, 6)
            self.assertAlmostEqual(redo[0].volume, 36)
            self.assertTrue(loaded.is_valid)
            self.assertGreater(b123d_to_pro_mesh(loaded).n_cells, 0)

    def test_invalid_file_and_failed_save_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.fcad'
            path.write_bytes(b'existing')
            with self.assertRaises(ValueError):
                save_snapshot(path, None, [], [])
            self.assertEqual(path.read_bytes(), b'existing')
            with self.assertRaises(ValueError):
                load_snapshot(path)


class ConstructionPlaneTest(unittest.TestCase):
    def plotter(self):
        class Actor:
            def __init__(self, mesh):
                self.mesh, self.visible, self.pickable = mesh, True, True
            def SetVisibility(self, value): self.visible = value
            def SetPickable(self, value): self.pickable = value
            def GetMapper(self): return SimpleNamespace(GetInput=lambda: self.mesh)
        class Plotter:
            def __init__(self): self.actors = {}
            def add_mesh(self, mesh, **kwargs):
                actor = Actor(mesh)
                self.actors[kwargs['name']] = actor
                return actor
            def remove_actor(self, name): self.actors.pop(name, None)
        return Plotter()

    def test_plane_positions_and_selection_visibility(self):
        manager = ActionManager(self.plotter())
        for base, offset, axis in [('XY', 4., 2), ('XZ', -3., 1), ('YZ', 2., 0)]:
            plane = manager.create_offset_plane(base, offset)
            actor = manager.plotter.actors[plane['name']]
            np.testing.assert_allclose(actor.mesh.points[:, axis], offset, atol=1e-6)
        manager.set_construction_planes_visible(False)
        self.assertTrue(all(not a.visible and not a.pickable for a in manager.plotter.actors.values()))
        manager.set_construction_planes_visible(True)
        actor = manager.plotter.actors[manager.construction_planes[0]['name']]
        manager._pick_actor = Mock(return_value=SimpleNamespace(GetActor=lambda: actor))
        manager.on_plane_selected = Mock()
        manager._select_sketch_plane()
        self.assertEqual(manager.on_plane_selected.call_count, 1)
        manager.clear_construction_planes()
        self.assertEqual(manager.plotter.actors, {})

    def test_offset_from_tilted_face_and_snapshot(self):
        manager = ActionManager(self.plotter())
        manager.current_part = Rot(25, 35, 10) * Box(4, 6, 8)
        manager.selected_face_id = 0
        source = list(manager.current_part.faces())[0]
        plane = manager.create_plane_from_selected_face(3.5)
        mesh = manager.plotter.actors[plane['name']].mesh
        normal = np.array(tuple(source.normal_at()))
        origin = np.array(tuple(source.center()))
        distances = (np.asarray(mesh.points) - origin) @ normal
        np.testing.assert_allclose(distances, 3.5, atol=2e-6)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.fcad'
            save_snapshot(path, manager.current_part, [], [], manager.construction_planes)
            _, _, _, loaded = load_snapshot(path, include_planes=True)
            restored_manager = ActionManager(self.plotter())
            restored = loaded[0]
            restored_manager.create_offset_plane(restored['base'], restored['offset'], restored['origin'], restored['normal'])
            actor = next(iter(restored_manager.plotter.actors.values()))
            np.testing.assert_allclose((np.asarray(actor.mesh.points) - origin) @ normal, 3.5, atol=2e-6)

    def test_curved_face_cannot_define_offset_plane(self):
        manager = ActionManager(self.plotter())
        manager.current_part = Cylinder(2, 4)
        manager.selected_face_id = next(i for i, face in enumerate(manager.current_part.faces())
                                        if face.geom_type.name != 'PLANE')
        with self.assertRaises(ValueError):
            manager.create_plane_from_selected_face(1)
        self.assertEqual(manager.construction_planes, [])

    def test_plane_data_roundtrip_and_legacy_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.fcad'
            body = Box(1, 2, 3)
            planes = [{'base': 'XY', 'offset': 7.5}, {'base': 'YZ', 'offset': -2}]
            save_snapshot(path, body, [], [], planes)
            _, _, _, loaded_planes = load_snapshot(path, include_planes=True)
            self.assertEqual(loaded_planes, planes)
            save_snapshot(path, body, [], [])
            self.assertEqual(load_snapshot(path, include_planes=True)[3], [])
            with self.assertRaises(ValueError):
                save_snapshot(path, body, [], [], [{'base': 'XY', 'offset': float('nan')}])
            self.assertAlmostEqual(load_snapshot(path)[0].volume, 6)

    def test_cancel_plane_selection(self):
        manager = ActionManager(self.plotter())
        manager.create_offset_plane('XY', 2)
        manager.selecting_sketch_plane = True
        self.assertTrue(manager.cancel_pending_selection())
        self.assertFalse(manager.selecting_sketch_plane)
        actor = manager.plotter.actors[manager.construction_planes[0]['name']]
        self.assertFalse(actor.pickable)


class PrimitiveAndImportTest(unittest.TestCase):
    def test_primitives_have_expected_volume_and_position(self):
        box = build_primitive('box', (2, 3, 4), (5, 6, 7))
        cylinder = build_primitive('cylinder', (2, 5), (0, 0, 0))
        sphere = build_primitive('sphere', (3,), (0, 0, 0))
        self.assertAlmostEqual(box.volume, 24)
        self.assertAlmostEqual(cylinder.volume, 20 * np.pi)
        self.assertAlmostEqual(sphere.volume, 36 * np.pi)
        np.testing.assert_allclose(tuple(box.center()), (5, 6, 7), atol=1e-7)

    def test_primitive_operations_and_invalid_dimensions(self):
        body = Box(10, 10, 10)
        cut = build_primitive('sphere', (2,), (0, 0, 0), 'cut', body)
        self.assertAlmostEqual(cut.volume, 1000 - 32 * np.pi / 3)
        result = build_primitive('box', (2, 2, 2), (20, 0, 0), 'new_body', body)
        self.assertEqual(len(result.solids()), 2)
        with self.assertRaises(ValueError):
            build_primitive('box', (0, 2, 2))
        self.assertAlmostEqual(body.volume, 1000)

    def test_import_step_keeps_existing_solid(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'box.step'
            export_body(Box(2, 3, 4), path, 'step')
            result = import_step_body(path, Box(1, 1, 1))
            self.assertEqual(len(result.solids()), 2)
            self.assertAlmostEqual(result.volume, 25)
            with self.assertRaises(ValueError):
                import_step_body(Path(directory) / 'missing.step')


class ThroughAllAndMeasureTest(unittest.TestCase):
    def test_through_all_cuts_both_sides_of_profile(self):
        body = Box(10, 10, 10)
        profile = Face.make_rect(2, 2, Plane(origin=(0, 0, 2)))
        result = build_extrusion(profile, (0, 0, 1), 0, 'cut', body, extent='all')
        self.assertAlmostEqual(result.volume, 960)
        self.assertAlmostEqual(body.volume, 1000)
        intersection = build_extrusion(profile, (0, 0, 1), 0, 'intersect', body, extent='all')
        self.assertAlmostEqual(intersection.volume, 40)

    def test_through_all_on_rotated_body(self):
        rotation = Rot(30, 25, 5)
        body = rotation * Box(10, 10, 10)
        profile = rotation * Face.make_rect(2, 2, Plane(origin=(0, 0, 5)))
        normal = np.asarray(tuple(profile.normal_at()))
        result = build_extrusion(profile, normal, 1, 'cut', body, extent='all')
        self.assertAlmostEqual(result.volume, 960, places=5)
        with self.assertRaises(ValueError):
            build_extrusion(profile, normal, 1, 'new_body', body, extent='all')
        with self.assertRaises(ValueError):
            build_extrusion(profile, normal, 1, 'cut', None, extent='all')

    def test_inspect_model_and_face(self):
        body = Box(2, 3, 4)
        data = measure_model(body)
        self.assertAlmostEqual(data['volume'], 24)
        self.assertAlmostEqual(data['area'], 52)
        np.testing.assert_allclose(data['dimensions'], (2, 3, 4))
        face = Face.make_rect(2, 3)
        details = measure_face(face)
        self.assertAlmostEqual(details['area'], 6)
        self.assertAlmostEqual(details['perimeter'], 10)
        with self.assertRaises(ValueError):
            measure_model(None)


class ArcToolsTest(unittest.TestCase):
    def test_center_arc_crosses_zero_and_uses_input_radius(self):
        normal = np.array([0., 0., 1.])
        start = (np.cos(np.deg2rad(300)), np.sin(np.deg2rad(300)), 0)
        end = (2 * np.cos(np.deg2rad(60)), 2 * np.sin(np.deg2rad(60)), 0)
        points = centered_arc_points((0, 0, 0), start, end, normal)
        np.testing.assert_allclose(points[1], np.array(end) / 2, atol=1e-8)
        np.testing.assert_allclose(points[2], (1, 0, 0), atol=1e-8)
        self.assertAlmostEqual(np.linalg.norm(normal), 1)
        with self.assertRaises(ValueError):
            centered_arc_points((0, 0, 0), (1, 0, 0), (2, 0, 0), normal)

    def test_ellipse_arc_endpoints_on_both_axis_orderings(self):
        normal = np.array([0., 0., 1.])
        for major, minor in [(3., 1.), (1., 3.)]:
            arc = SimpleNamespace(type='ellipse', points=[(0, 0, 0)] * 5, center=np.zeros(3),
                major_radius=major, minor_radius=minor, major_axis_vector=np.array([1., 0., 0.]),
                is_closed=False, start_angle=np.deg2rad(300), end_angle=np.deg2rad(420))
            edge = sketch_edges([arc], normal)[0]
            for parameter, angle in [(0, 300), (1, 420)]:
                expected = (major * np.cos(np.deg2rad(angle)), minor * np.sin(np.deg2rad(angle)), 0)
                np.testing.assert_allclose(tuple(edge.position_at(parameter)), expected, atol=1e-7)
            a, b = tuple(edge.position_at(0)), tuple(edge.position_at(1))
            chord = SimpleNamespace(type='line', points=[a, b])
            faces = build_profiles([arc, chord], normal)
            self.assertEqual(len(faces), 1)
            self.assertGreater(extrude(faces[0], amount=2).volume, 0)

    def test_elliptical_arc_click_sequence(self):
        iren = SimpleNamespace(remove_observer=Mock())
        plotter = SimpleNamespace(iren=iren, disable_picking=Mock(), remove_actor=Mock(), render=Mock())
        sketch = SimpleNamespace(active_normal=np.array([0., 0., 1.]),
            get_snapped_pos=lambda p: (np.asarray(p, dtype=float), None),
            get_local_pos_in_sketch=lambda p: np.asarray(p)[:2],
            status_callback=Mock(), redraw_sketch_geometry=Mock())
        tool = EllipseArc(plotter, {}, sketch)
        tool.is_drawing = True
        for point in [(0, 0, 0), (3, 0, 0), (0, 2, 0), (3, 0, 0), (0, 2, 0)]:
            tool.on_click(point)
        self.assertFalse(tool.is_drawing)
        self.assertEqual(len(sketch_edges([tool], sketch.active_normal)), 1)
        sketch.redraw_sketch_geometry.assert_called_once()

    def test_three_point_arc_preview_cleanup(self):
        iren = SimpleNamespace(add_observer=Mock(return_value=42), remove_observer=Mock())
        plotter = SimpleNamespace(iren=iren, track_click_position=Mock(), disable_picking=Mock(),
            remove_actor=Mock(), render=Mock(), add_mesh=Mock(),
            pick_mouse_position=Mock(return_value=np.array([0., 1., 0.])))
        sketch = SimpleNamespace(get_snapped_pos=lambda p: (p, p[:2]))
        tool = Curve(plotter, {}, sketch)
        tool.create_curve()
        tool.points = [np.array([-1., 0., 0.]), np.array([1., 0., 0.])]
        tool.draw_preview_line()
        self.assertEqual(plotter.add_mesh.call_count, 1)
        tool.stop_drawing()
        iren.remove_observer.assert_called_once_with(42)

    def test_incomplete_elliptical_arc_has_no_geometry(self):
        arc = SimpleNamespace(type='ellipse', points=[(0, 0, 0)] * 4, center=np.zeros(3),
            major_radius=3., minor_radius=1., major_axis_vector=np.array([1., 0., 0.]), is_closed=False)
        self.assertEqual(sketch_edges([arc], (0, 0, 1)), [])


class SketchToolCancellationTest(unittest.TestCase):
    def test_unfinished_tool_removed_but_completed_line_retained(self):
        manager = ActionManager(None)
        manager.plotter = SimpleNamespace(remove_actor=Mock())
        manager.activated_sketch_id = 1
        sketch = {'sketch_func': []}
        manager.sketchs_data[1] = sketch
        ui = LayoutManager.__new__(LayoutManager)
        ui.action_manager = manager
        unfinished = SimpleNamespace(type='line', points=[np.zeros(3)],
                                     render_obj={'points': ['marker'], 'lines': []}, stop_drawing=Mock())
        sketch['sketch_func'].append(unfinished)
        manager.active_tool = unfinished
        ui.disable_active_tool()
        self.assertEqual(sketch['sketch_func'], [])
        manager.plotter.remove_actor.assert_called_once_with('marker')
        completed = SimpleNamespace(type='line', points=[np.zeros(3), np.array([1., 0., 0.])],
                                    render_obj={'points': [], 'lines': []}, stop_drawing=Mock())
        sketch['sketch_func'].append(completed)
        manager.active_tool = completed
        ui.disable_active_tool()
        self.assertIs(sketch['sketch_func'][0], completed)
        self.assertIsNone(manager.active_tool)


class MorePrimitivesAndViewsTest(unittest.TestCase):
    def test_cone_frustum_and_torus(self):
        cone = build_primitive('cone', (3, 0, 4))
        frustum = build_primitive('cone', (3, 1, 4))
        torus = build_primitive('torus', (5, 2))
        self.assertAlmostEqual(cone.volume, 12 * np.pi)
        self.assertAlmostEqual(frustum.volume, 52 * np.pi / 3)
        self.assertAlmostEqual(torus.volume, 40 * np.pi ** 2)
        with self.assertRaises(ValueError):
            build_primitive('cone', (0, 0, 2))
        with self.assertRaises(ValueError):
            build_primitive('torus', (2, 2))
        self.assertEqual(len(build_primitive('torus', (5, 2), (20, 0, 0), 'new_body', cone).solids()), 2)

    def test_standard_views_and_protected_sketch(self):
        manager = ActionManager(None)
        ui = LayoutManager.__new__(LayoutManager)
        ui.action_manager = manager
        ui.plotter = SimpleNamespace(view_vector=Mock(), view_isometric=Mock())
        expected = {'top': (0, 0, 1), 'bottom': (0, 0, -1), 'front': (0, -1, 0),
                    'back': (0, 1, 0), 'right': (1, 0, 0), 'left': (-1, 0, 0)}
        for name, vector in expected.items():
            ui.set_standard_view(name)
            self.assertEqual(ui.plotter.view_vector.call_args.args[0], vector)
        ui.set_standard_view('isometric')
        ui.plotter.view_isometric.assert_called_once()
        manager.sketch_editing = True
        ui.plotter.view_vector.reset_mock()
        ui.set_standard_view('top')
        ui.plotter.view_vector.assert_not_called()

    def test_look_at_tilted_plane_and_reject_curved_face(self):
        manager = ActionManager(None)
        manager.current_part = Rot(25, 35, 10) * Box(4, 6, 8)
        manager.selected_face_id = 0
        face = list(manager.current_part.faces())[0]
        camera = SimpleNamespace(position=(10., 10., 10.), focal_point=(0., 0., 0.), up=(0., 0., 1.))
        plotter = SimpleNamespace(camera=camera, enable_parallel_projection=Mock(), render=Mock())
        ui = LayoutManager.__new__(LayoutManager)
        ui.action_manager, ui.plotter = manager, plotter
        self.assertTrue(ui.look_at_selection())
        np.testing.assert_allclose(camera.focal_point, tuple(face.center()), atol=1e-7)
        direction = np.asarray(camera.position) - np.asarray(camera.focal_point)
        direction /= np.linalg.norm(direction)
        np.testing.assert_allclose(direction, tuple(face.normal_at()), atol=1e-7)
        self.assertAlmostEqual(np.dot(camera.up, direction), 0, places=7)
        manager.current_part = Cylinder(2, 4)
        manager.selected_face_id = next(i for i, f in enumerate(manager.current_part.faces()) if f.geom_type.name != 'PLANE')
        self.assertFalse(ui.look_at_selection())


class SolidTransformsTest(unittest.TestCase):
    def setUp(self):
        self.body = Pos(3, 0, 0) * Box(2, 2, 2)

    def test_move_rotate_and_copy_do_not_change_original(self):
        result = move_copy(self.body, (1, 0, 0), (0, 0, 90), (1, 0, 0))
        np.testing.assert_allclose(tuple(result.center()), (2, 2, 0), atol=1e-7)
        np.testing.assert_allclose(tuple(self.body.center()), (3, 0, 0), atol=1e-7)
        copy = move_copy(self.body, (10, 0, 0), copy=True)
        self.assertEqual(len(copy.solids()), 2)
        self.assertAlmostEqual(copy.volume, 16)

    def test_scale_about_center_and_mirror(self):
        result = scale_model(self.body, 2, (1, 0, 0))
        self.assertAlmostEqual(result.volume, 64)
        np.testing.assert_allclose(tuple(result.center()), (5, 0, 0), atol=1e-7)
        mirrored = mirror_model(self.body, 'YZ', 1, copy=False)
        np.testing.assert_allclose(tuple(mirrored.center()), (-1, 0, 0), atol=1e-7)
        self.assertEqual(len(mirror_model(self.body, 'YZ', 0, copy=True).solids()), 2)
        with self.assertRaises(ValueError):
            scale_model(self.body, 0)

    def test_rectangular_and_circular_patterns(self):
        rectangle = rectangular_pattern(self.body, 3, 2, 10, 10)
        self.assertEqual(len(rectangle.solids()), 6)
        self.assertAlmostEqual(rectangle.volume, 48)
        ring = circular_pattern(self.body, 4, 360)
        self.assertEqual(len(ring.solids()), 4)
        centers = sorted((round(s.center().X), round(s.center().Y)) for s in ring.solids())
        self.assertEqual(centers, [(-3, 0), (0, -3), (0, 3), (3, 0)])
        partial = circular_pattern(self.body, 2, 90)
        self.assertEqual(sorted((round(s.center().X), round(s.center().Y)) for s in partial.solids()), [(0, 3), (3, 0)])
        with self.assertRaises(ValueError):
            rectangular_pattern(self.body, 101, 1, 1, 1)
        with self.assertRaises(ValueError):
            circular_pattern(self.body, 2, 0)

    def test_multiple_input_bodies_and_snapshot(self):
        compound = move_copy(self.body, (10, 0, 0), copy=True)
        result = rectangular_pattern(compound, 2, 1, 30, 1)
        self.assertEqual(len(result.solids()), 4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'pattern.fcad'
            save_snapshot(path, result, [self.body], [])
            restored, history, _ = load_snapshot(path)
            self.assertEqual(len(restored.solids()), 4)
            self.assertAlmostEqual(history[0].volume, 8)


class SolidFeatureTest(unittest.TestCase):
    def setUp(self):
        self.body = Box(10, 10, 10)
        self.top = next(i for i, face in enumerate(self.body.faces()) if face.center().Z > 4)

    def test_all_edge_modifications(self):
        filleted = fillet_all(self.body, 1)
        chamfered = chamfer_all(self.body, 1)
        self.assertTrue(filleted.is_valid and chamfered.is_valid)
        self.assertLess(filleted.volume, self.body.volume)
        self.assertLess(chamfered.volume, self.body.volume)
        self.assertAlmostEqual(self.body.volume, 1000)
        with self.assertRaises(ValueError):
            fillet_all(self.body, 0)
        with self.assertRaises(ValueError):
            chamfer_all(self.body, -1)

    def test_shell_and_offset(self):
        shell = shell_face(self.body, self.top, -1)
        expanded = offset_solid(self.body, 1)
        self.assertAlmostEqual(shell.volume, 424)
        self.assertGreater(expanded.volume, 1000)
        with self.assertRaises(ValueError):
            shell_face(self.body, -1, -1)
        with self.assertRaises(ValueError):
            offset_solid(self.body, 0)
        compound = move_copy(self.body, (20, 0, 0), copy=True)
        with self.assertRaises(ValueError):
            shell_face(compound, 0, -1)

    def test_center_hole_blind_through_and_outside(self):
        blind = hole_at_face_center(self.body, self.top, 2, 5)
        through = hole_at_face_center(self.body, self.top, 2, 0, through_all=True)
        self.assertAlmostEqual(blind.volume, 1000 - 5 * np.pi)
        self.assertAlmostEqual(through.volume, 1000 - 10 * np.pi)
        with self.assertRaises(ValueError):
            hole_at_face_center(self.body, self.top, 2, 5, offset_u=20)
        self.assertAlmostEqual(self.body.volume, 1000)

    def test_hole_on_rotated_face(self):
        rotated = Rot(20, 30, 10) * self.body
        result = hole_at_face_center(rotated, self.top, 2, 5)
        self.assertAlmostEqual(result.volume, 1000 - 5 * np.pi, places=5)


class EdgeSelectionTest(unittest.TestCase):
    def test_visible_face_edges_and_screen_tolerance(self):
        body = Box(10, 10, 10)
        edges, indices = face_edge_indices(body, 0)
        self.assertEqual(len(indices), 4)
        edge_index = next(i for i in indices if abs(edges[i].position_at(0).Y - edges[i].position_at(1).Y) > 1)
        midpoint = tuple(edges[edge_index].position_at(0.5))
        project = lambda p: (p[1] * 10, p[2] * 10, 0)
        self.assertEqual(nearest_screen_edge(edges, indices, (midpoint[1] * 10, midpoint[2] * 10), project, tolerance=3), edge_index)
        self.assertIsNone(nearest_screen_edge(edges, indices, (500, 500), project, tolerance=3))
        self.assertEqual(face_edge_indices(body, 100)[1], [])

    def test_selected_edge_fillet_and_chamfer(self):
        body = Box(10, 10, 10)
        rounded = fillet_edges(body, [0], 1)
        beveled = chamfer_edges(body, [0], 1)
        self.assertTrue(rounded.is_valid and beveled.is_valid)
        self.assertGreater(rounded.volume, fillet_all(body, 1).volume)
        self.assertGreater(beveled.volume, chamfer_all(body, 1).volume)
        self.assertAlmostEqual(body.volume, 1000)
        for indices in ([], [-1], [999]):
            with self.assertRaises(ValueError):
                fillet_edges(body, indices, 1)
        compound = move_copy(body, (20, 0, 0), copy=True)
        self.assertEqual(len(fillet_edges(compound, [0], 1).solids()), 2)

    def test_measure_selected_edges(self):
        body = Box(10, 10, 10)
        result = measure_edges(body, {0, 1})
        self.assertEqual(result['count'], 2)
        self.assertAlmostEqual(result['length'], 20)
        self.assertEqual(result['types'], ['LINE'])
        with self.assertRaises(ValueError):
            measure_edges(body, set())

    def test_click_selects_and_toggles_edge(self):
        body = Box(10, 10, 10)
        face_index = next(i for i, face in enumerate(body.faces()) if abs(face.normal_at().X) > .9)
        edges, candidates = face_edge_indices(body, face_index)
        target = next(i for i in candidates if abs(edges[i].position_at(0).Y - edges[i].position_at(1).Y) > 1)
        midpoint = tuple(edges[target].position_at(.5))
        class Renderer:
            def SetWorldPoint(self, x, y, z, w): self.point = (y * 10, z * 10, 0)
            def WorldToDisplay(self): pass
            def GetDisplayPoint(self): return self.point
        plotter = SimpleNamespace(renderer=Renderer(), iren=SimpleNamespace(get_event_position=lambda: (midpoint[1] * 10, midpoint[2] * 10)),
            add_mesh=Mock(return_value=object()), remove_actor=Mock(), render=Mock())
        manager = ActionManager(plotter)
        manager.current_part = body
        manager._get_face_under_mouse = Mock(return_value=(face_index, None))
        manager.select_edge_at_cursor()
        self.assertEqual(manager.selected_edge_indices, {target})
        manager.select_edge_at_cursor()
        self.assertEqual(manager.selected_edge_indices, set())
        self.assertEqual(manager.edge_selection_actors, [])


class IndividualBodySelectionTest(unittest.TestCase):
    def setUp(self):
        self.first = Box(2, 2, 2)
        self.second = Pos(20, 0, 0) * Box(4, 4, 4)
        self.model = __import__('build123d').Part([self.first, self.second])
        self.index = 1
        self.face_index = next(i for i, face in enumerate(self.model.faces())
            if any(face.is_same(candidate) for candidate in list(self.model.solids())[1].faces()))

    def test_face_identifies_body(self):
        self.assertEqual(body_index_for_face(self.model, self.face_index), 1)
        self.assertIsNone(body_index_for_face(self.model, -1))
        manager = ActionManager(None)
        manager.current_part, manager.selected_face_id = self.model, self.face_index
        self.assertEqual(manager.selected_body_index(), 1)

    def test_five_transforms_leave_other_body_untouched(self):
        operations = [
            lambda target: move_copy(target, (10, 0, 0)),
            lambda target: scale_model(target, 2, (20, 0, 0)),
            lambda target: mirror_model(target, 'YZ', copy=False),
            lambda target: rectangular_pattern(target, 2, 1, 10, 1),
            lambda target: circular_pattern(target, 2, 180),
        ]
        original_first = list(self.model.solids())[0]
        for operation in operations:
            with self.subTest(operation=operation):
                result = apply_to_body(self.model, self.index, operation)
                self.assertTrue(any(solid.is_same(original_first) for solid in result.solids()))
                self.assertAlmostEqual(self.first.volume, 8)
        self.assertAlmostEqual(self.model.volume, 72)
        with self.assertRaises(ValueError):
            apply_to_body(self.model, 2, operations[0])

    def test_delete_selected_body_and_undo(self):
        manager = ActionManager(None)
        manager.current_part = self.model
        manager.selected_face_id = self.face_index
        manager.display_result = lambda shape: setattr(manager, 'current_part', shape)
        manager._restore_result = lambda shape: setattr(manager, 'current_part', shape)
        manager.delete_selected_body()
        self.assertEqual(len(manager.current_part.solids()), 1)
        self.assertAlmostEqual(manager.current_part.volume, 8)
        self.assertTrue(manager.undo_solid())
        self.assertAlmostEqual(manager.current_part.volume, 72)
        self.assertTrue(manager.redo_solid())
        self.assertAlmostEqual(manager.current_part.volume, 8)

    def test_delete_last_body_and_restore(self):
        manager = ActionManager(None)
        manager.current_part = self.first
        manager.selected_face_id = 0
        manager._restore_result = lambda shape: setattr(manager, 'current_part', shape)
        manager.delete_selected_body()
        self.assertIsNone(manager.current_part)
        self.assertTrue(manager.undo_solid())
        self.assertAlmostEqual(manager.current_part.volume, 8)
        self.assertTrue(manager.redo_solid())
        self.assertIsNone(manager.current_part)

    def test_failed_transformation_does_not_modify_model(self):
        with self.assertRaises(ValueError):
            apply_to_body(self.model, 1, lambda target: scale_model(target, 0))
        self.assertAlmostEqual(self.model.volume, 72)
        self.assertEqual(len(self.model.solids()), 2)


class ExpressionInputTest(unittest.TestCase):
    def test_arithmetic_and_named_values(self):
        self.assertEqual(evaluate_expression('(width / 2 + 3) * 2', {'width': 10}), 16)
        self.assertEqual(evaluate_expression('2 ** 3 + -1'), 7)

    def test_rejects_unsafe_and_invalid_expressions(self):
        for value in ['1 / 0', 'unknown + 1', '__import__("os")', '1e999', '2 ** 100', '1 +']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                evaluate_expression(value)


class ConstraintEngineTest(unittest.TestCase):
    def test_fixed_horizontal_dimension_and_expression(self):
        points = {'a': (0, 0), 'b': (4, 1)}
        constraints = [Constraint('fixed', ('a',), (0, 0)),
                       Constraint('horizontal', (('a', 'b'),)),
                       Constraint('distance', ('a', 'b'), 'width / 2')]
        result = solve_constraints(points, {}, constraints, {'width': 20})
        self.assertEqual(result.status, 'fully_constrained')
        np.testing.assert_allclose(result.points['a'], (0, 0), atol=1e-6)
        np.testing.assert_allclose(result.points['b'], (10, 0), atol=1e-6)
        self.assertEqual(points['b'], (4, 1))

    def test_underconstrained_and_conflicting(self):
        points = {'a': (0, 0), 'b': (2, 1)}
        result = solve_constraints(points, {}, [Constraint('horizontal', (('a', 'b'),))])
        self.assertEqual(result.status, 'underconstrained')
        self.assertGreater(result.degrees_of_freedom, 0)
        constraints = [Constraint('fixed', ('a',), (0, 0)), Constraint('fixed', ('a',), (1, 0))]
        failed = solve_constraints(points, {}, constraints)
        self.assertEqual(failed.status, 'conflict')
        np.testing.assert_allclose(failed.points['a'], points['a'])

    def test_circle_diameter_and_tangency(self):
        points = {'a': (-2, 0), 'b': (2, 0), 'c': (0, 3)}
        circle = ('c', 'r')
        constraints = [Constraint('fixed', ('a',), (-2, 0)), Constraint('fixed', ('b',), (2, 0)),
                       Constraint('fixed', ('c',), (0, 2)), Constraint('diameter', (circle,), 4),
                       Constraint('tangent_line_circle', (('a', 'b'), circle))]
        result = solve_constraints(points, {'r': 1}, constraints)
        self.assertEqual(result.status, 'fully_constrained')
        self.assertAlmostEqual(result.radii['r'], 2, places=6)

    def test_midpoint_and_equal_length(self):
        points = {'a': (0, 0), 'b': (4, 0), 'm': (1, 2), 'c': (0, 1), 'd': (3, 1)}
        constraints = [Constraint('fixed', ('a',), (0, 0)), Constraint('fixed', ('b',), (4, 0)),
                       Constraint('fixed', ('c',), (0, 1)), Constraint('horizontal', (('c', 'd'),)),
                       Constraint('midpoint', ('m', ('a', 'b'))),
                       Constraint('equal_length', (('a', 'b'), ('c', 'd')))]
        result = solve_constraints(points, {}, constraints)
        self.assertEqual(result.status, 'fully_constrained')
        np.testing.assert_allclose(result.points['m'], (2, 0), atol=1e-6)
        self.assertAlmostEqual(np.linalg.norm(result.points['d'] - result.points['c']), 4, places=6)


class EditorConstraintBridgeTest(unittest.TestCase):
    def manager(self):
        manager = ActionManager(None)
        manager.activated_sketch_id = 1
        first = SimpleNamespace(type='line', points=[np.array([0.,0.,0.]), np.array([4.,1.,0.])])
        second = SimpleNamespace(type='line', points=[np.array([4.,1.,0.]), np.array([6.,3.,0.])])
        manager.sketchs_data[1] = {'sketch_func': [first, second]}
        return manager

    def test_automatic_coincidence_follows_dimension(self):
        manager = self.manager()
        points = line_points(manager)
        add_automatic_coincidences(manager.get_current_sketch_data(), points)
        constraints = manager.get_current_sketch_data()['constraints']
        self.assertEqual(len(constraints), 1)
        self.assertEqual(constraints[0].kind, 'coincident')
        add_automatic_coincidences(manager.get_current_sketch_data(), points)
        self.assertEqual(len(constraints), 1)
        apply_constraint(manager, Constraint('fixed', ('L1.P1',), (0, 0)))
        apply_constraint(manager, Constraint('horizontal', (('L1.P1', 'L1.P2'),)))
        apply_constraint(manager, Constraint('distance', ('L1.P1', 'L1.P2'), 10))
        points = line_points(manager)
        np.testing.assert_allclose(points['L1.P2'][2], (10, 0), atol=1e-6)
        np.testing.assert_allclose(points['L2.P1'][2], (10, 0), atol=1e-6)

    def test_conflict_does_not_move_geometry_or_add_constraint(self):
        manager = self.manager()
        apply_constraint(manager, Constraint('fixed', ('L1.P1',), (0, 0)))
        before = {name: value[2].copy() for name, value in line_points(manager).items()}
        count = len(manager.get_current_sketch_data()['constraints'])
        with self.assertRaises(ValueError):
            apply_constraint(manager, Constraint('fixed', ('L1.P1',), (5, 0)))
        for name, value in line_points(manager).items():
            np.testing.assert_allclose(value[2], before[name])
        self.assertEqual(len(manager.get_current_sketch_data()['constraints']), count)


class ExtrusionHandleTest(unittest.TestCase):
    def test_scale_follows_zoom_and_projection(self):
        camera = SimpleNamespace(GetParallelProjection=lambda: True, GetParallelScale=lambda: 10.)
        self.assertAlmostEqual(world_per_pixel(camera, 1000), .02)
        camera.GetParallelScale = lambda: 5.
        self.assertAlmostEqual(world_per_pixel(camera, 1000), .01)
        camera = SimpleNamespace(GetParallelProjection=lambda: False, position=(0, 0, 10),
            focal_point=(0, 0, 0), GetViewAngle=lambda: 90.)
        self.assertAlmostEqual(world_per_pixel(camera, 1000), .02)

    def test_drag_updates_numeric_field(self):
        handle = ExtrudeHandle.__new__(ExtrudeHandle)
        camera = SimpleNamespace(GetParallelProjection=lambda: True, GetParallelScale=lambda: 10.)
        plotter = SimpleNamespace(camera=camera, iren=SimpleNamespace(get_event_position=lambda: (0, 150)),
                                  ren_win=SimpleNamespace(GetSize=lambda: (1000, 1000)))
        field = SimpleNamespace(setValue=Mock())
        handle.plotter, handle.dialog = plotter, SimpleNamespace(distance=field)
        handle.drag_start, handle.updating, handle.sync = (100, 3.), False, Mock()
        handle.drag()
        field.setValue.assert_called_once_with(4.)
        handle.sync.assert_called_once()
        self.assertFalse(handle.updating)


class MoveHandleTest(unittest.TestCase):
    def test_each_axis_updates_only_its_translation_field(self):
        camera = SimpleNamespace(GetParallelProjection=lambda: True, GetParallelScale=lambda: 10.)
        plotter = SimpleNamespace(camera=camera, iren=SimpleNamespace(get_event_position=lambda: (0, 150)),
                                  ren_win=SimpleNamespace(GetSize=lambda: (1000, 1000)))
        fields = {'d' + axis: SimpleNamespace(setValue=Mock()) for axis in 'XYZ'}
        handle = MoveAxisHandles.__new__(MoveAxisHandles)
        handle.plotter, handle.dialog = plotter, SimpleNamespace(controls=fields)
        handle.updating, handle.sync = False, Mock()
        for axis in 'XYZ':
            handle.drag_start = (axis, 100, 3.)
            handle.drag(axis)
            fields['d' + axis].setValue.assert_called_once_with(4.)
        self.assertEqual(handle.sync.call_count, 3)

    def test_scene_selection_is_locked_during_operation(self):
        manager = ActionManager(None)
        manager.interaction_locked = True
        manager._select_profile = Mock()
        manager._on_click()
        manager._select_profile.assert_not_called()


class RotationAndScaleHandleTest(unittest.TestCase):
    def test_projected_ring_angle_and_edge_on(self):
        self.assertAlmostEqual(projected_ring_angle((0, 20), (0, 0), (10, 0), (0, 20)), 90)
        self.assertAlmostEqual(projected_ring_angle((-10, 0), (0, 0), (10, 0), (0, 20)), 180)
        self.assertIsNone(projected_ring_angle((1, 1), (0, 0), (10, 0), (20, 0)))

    def test_rotate_and_scale_update_fields(self):
        plotter = SimpleNamespace(iren=SimpleNamespace(get_event_position=lambda: (0, 120)))
        rotation = SimpleNamespace(setValue=Mock())
        scale = SimpleNamespace(setValue=Mock())
        handle = ParameterHandles.__new__(ParameterHandles)
        handle.plotter, handle.dialog = plotter, SimpleNamespace(controls={'rZ': rotation, 'factor': scale})
        handle.updating, handle.sync = False, Mock()
        handle.mode, handle.drag_start = 'rotate', ('Z', 100, 10., 350.)
        handle.cursor_ring_angle = Mock(return_value=10.)
        handle.drag('Z')
        rotation.setValue.assert_called_once_with(30.)
        handle.mode, handle.drag_start = 'scale', ('S', 100, 1., None)
        handle.drag('S')
        scale.setValue.assert_called_once_with(1.25)


class OperationHandleTest(unittest.TestCase):
    def test_linear_handle_updates_field_and_respects_zoom(self):
        camera = SimpleNamespace(GetParallelProjection=lambda: True, GetParallelScale=lambda: 10.)
        plotter = SimpleNamespace(camera=camera, iren=SimpleNamespace(get_event_position=lambda: (0, 150)),
                                  ren_win=SimpleNamespace(GetSize=lambda: (1000, 1000)))
        handle = OperationHandle.__new__(OperationHandle)
        handle.plotter, handle.field, handle.angle = plotter, SimpleNamespace(setValue=Mock()), False
        handle.drag_start, handle.updating, handle.sync = (100, 3., None), False, Mock()
        handle.drag()
        handle.field.setValue.assert_called_once_with(4.)
        handle.sync.assert_called_once()

    def test_angle_handle_wraps_and_updates_field(self):
        handle = OperationHandle.__new__(OperationHandle)
        handle.plotter = SimpleNamespace(iren=SimpleNamespace(get_event_position=lambda: (0, 150)))
        handle.field, handle.angle = SimpleNamespace(setValue=Mock()), True
        handle.drag_start, handle.updating, handle.sync = (100, 10., 350.), False, Mock()
        handle.cursor_angle = Mock(return_value=10.)
        handle.drag()
        handle.field.setValue.assert_called_once_with(30.)

    def test_close_disables_widget_and_removes_actor(self):
        handle = OperationHandle.__new__(OperationHandle)
        handle.widget = SimpleNamespace(EnabledOff=Mock(), RemoveAllObservers=Mock())
        handle.plotter = SimpleNamespace(remove_actor=Mock(), render=Mock())
        handle.name = 'temporary'
        handle.close()
        handle.widget.EnabledOff.assert_called_once()
        handle.plotter.remove_actor.assert_called_once_with('temporary', render=False)


class TimelineTest(unittest.TestCase):
    def test_undo_redo_and_branch_truncation(self):
        timeline = Timeline()
        timeline.append('Box', {'Width': 10.})
        timeline.append('Extrude', {'Distance': 5.})
        timeline.undo()
        self.assertEqual(timeline.cursor, 1)
        timeline.redo()
        self.assertEqual(timeline.cursor, 2)
        timeline.undo()
        timeline.append('Hole', {'Diameter': 2.})
        self.assertEqual([item['title'] for item in timeline.entries], ['Box', 'Hole'])
        self.assertEqual(timeline.cursor, 2)

    def test_rename_preserves_operation_data_and_rejects_blank(self):
        timeline = Timeline()
        timeline.append('Box', {'Width': 10.})
        timeline.rename(0, '  Housing  ')
        self.assertEqual(timeline.entries[0], {'title': 'Housing', 'parameters': {'Width': 10.}})
        with self.assertRaises(ValueError):
            timeline.rename(0, '   ')
        self.assertEqual(timeline.entries[0]['title'], 'Housing')

    def test_snapshot_preserves_parameters_and_cursor(self):
        timeline = {'entries': [{'title': 'Box', 'parameters': {'Length': 2.}},
                                {'title': 'Extrude', 'parameters': {'Operation': 'Cut', 'Distance': -3.}}], 'cursor': 1}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'timeline.fcad'
            save_snapshot(path, Box(2, 2, 2), [None], [Box(3, 3, 3)], timeline=timeline)
            current, undo, redo, restored = load_snapshot(path, include_timeline=True)
            self.assertEqual(restored, timeline)
            self.assertEqual(len(undo), 1)
            self.assertEqual(len(redo), 1)
            with self.assertRaises(ValueError):
                save_snapshot(path, current, undo, redo, timeline={'entries': [], 'cursor': 0})
            self.assertEqual(load_snapshot(path, include_timeline=True)[3], timeline)

    def test_legacy_timeline_uses_placeholder_entries(self):
        restored = validate_timeline(None, 2, 1)
        self.assertEqual(restored['cursor'], 2)
        self.assertEqual(len(restored['entries']), 3)


class ParametricFeatureGraphTest(unittest.TestCase):
    def graph(self):
        box = Feature.create('Box', 'primitive', {'type': 'box', 'dimensions': ['width', 2, 3]})
        move = Feature.create('Move', 'move', {'translation': ['width', 0, 0]}, box.id)
        return FeatureGraph([box, move], {'width': '10', 'half': 'width / 2'}), box, move

    def test_parameter_change_rebuilds_dependents(self):
        graph, box, move = self.graph()
        self.assertAlmostEqual(graph.results[box.id].volume, 60)
        graph.set_parameters({'width': '20', 'half': 'width / 2'})
        self.assertAlmostEqual(graph.results[box.id].volume, 120)
        self.assertAlmostEqual(graph.results[move.id].center().X, 20)
        self.assertEqual(resolve_parameters(graph.parameters)['half'], 10)

    def test_suppress_and_restore_dependency(self):
        graph, box, move = self.graph()
        graph.edit(box.id, suppressed=True)
        self.assertNotIn(move.id, graph.results)
        self.assertIn(move.id, graph.blocked)
        graph.edit(box.id, suppressed=False)
        self.assertIn(move.id, graph.results)

    def test_invalid_reorder_or_edit_is_transactional(self):
        graph, box, move = self.graph()
        with self.assertRaises(RebuildError):
            graph.reorder(move.id, 0)
        self.assertEqual([item.id for item in graph.features], [box.id, move.id])
        with self.assertRaises(RebuildError):
            graph.set_parameters({'width': '-1'})
        self.assertAlmostEqual(graph.results[box.id].volume, 60)
        self.assertEqual(graph.parameters['width'], '10')
        graph.edit(box.id, name='Housing')
        self.assertEqual(graph.features[0].name, 'Housing')

    def test_external_changes_cannot_bypass_rebuild(self):
        graph, box, _ = self.graph()
        graph.features[0].parameters['dimensions'][0] = 999
        external_parameters = graph.parameters
        external_parameters['width'] = '-10'
        self.assertEqual(graph.parameters['width'], '10')
        self.assertEqual(graph.features[0].parameters['dimensions'][0], 'width')
        self.assertAlmostEqual(graph.results[box.id].volume, 60)

    def test_parameter_cycles_reported(self):
        with self.assertRaises(RebuildError):
            resolve_parameters({'a': 'b + 1', 'b': 'a + 1'})


class ParametricSketchChainTest(unittest.TestCase):
    def test_plane_sketch_extrude_rebuild_and_snapshot(self):
        plane = Feature.create('Plane', 'datum_plane', {'origin': (0, 0, 'offset')})
        sketch = Feature.create('Rectangle', 'rectangle_sketch', {'width': 'width', 'height': 2}, plane.id)
        extrude_feature = Feature.create('Extrude', 'extrude', {'distance': 'depth'}, sketch.id)
        graph = FeatureGraph([plane, sketch, extrude_feature], {'width': '4', 'offset': '3', 'depth': '5'})
        self.assertAlmostEqual(graph.results[extrude_feature.id].volume, 40)
        graph.set_parameters({'width': '8', 'offset': '7', 'depth': '5'})
        self.assertAlmostEqual(graph.results[extrude_feature.id].volume, 80)
        self.assertAlmostEqual(graph.results[extrude_feature.id].bounding_box().min.Z, 7)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'graph.fcad'
            save_snapshot(path, graph.results[extrude_feature.id], [], [], feature_graph=graph)
            model, _, _, restored = load_snapshot(path, include_feature_graph=True)
            self.assertAlmostEqual(model.volume, 80)
            self.assertEqual([f.id for f in restored.features], [plane.id, sketch.id, extrude_feature.id])
            restored.set_parameters({'width': '10', 'offset': '7', 'depth': '5'})
            self.assertAlmostEqual(restored.results[extrude_feature.id].volume, 100)

    def test_missing_plane_and_invalid_change_preserve_last_result(self):
        plane = Feature.create('Plane', 'datum_plane', {})
        sketch = Feature.create('Sketch', 'rectangle_sketch', {'width': 2, 'height': 3}, plane.id)
        extrude_feature = Feature.create('Extrude', 'extrude', {'distance': 4}, sketch.id)
        graph = FeatureGraph([plane, sketch, extrude_feature])
        with self.assertRaises(RebuildError):
            graph.edit(sketch.id, parameters={'width': -1, 'height': 3})
        self.assertAlmostEqual(graph.results[extrude_feature.id].volume, 24)
        graph.edit(plane.id, suppressed=True)
        self.assertIn(sketch.id, graph.blocked)
        self.assertIn(extrude_feature.id, graph.blocked)
        graph.edit(plane.id, suppressed=False)
        self.assertAlmostEqual(graph.results[extrude_feature.id].volume, 24)


class ParametricEditorIntegrationTest(unittest.TestCase):
    def test_install_and_edit_graph_keeps_valid_model(self):
        plane = Feature.create('Plane', 'datum_plane', {})
        sketch = Feature.create('Rectangle', 'rectangle_sketch', {'width': 'width', 'height': 2}, plane.id)
        solid = Feature.create('Extrude', 'extrude', {'distance': 5}, sketch.id)
        graph = FeatureGraph([plane, sketch, solid], {'width': 4})
        manager = ActionManager(None)
        manager.display_result = lambda part: setattr(manager, 'current_part', part)
        manager.timeline_callback = Mock()
        manager.install_parametric_graph(graph)
        self.assertAlmostEqual(manager.current_part.volume, 40)
        candidate = FeatureGraph.from_data(graph.to_data())
        candidate.set_parameters({'width': 8})
        manager.install_parametric_graph(candidate)
        self.assertAlmostEqual(manager.current_part.volume, 80)
        self.assertEqual(manager.parametric_graph.features[1].id, sketch.id)
        broken = FeatureGraph.from_data(candidate.to_data())
        broken.edit(plane.id, suppressed=True)
        with self.assertRaises(ValueError):
            manager.install_parametric_graph(broken)
        self.assertAlmostEqual(manager.current_part.volume, 80)
        with self.assertRaises(ValueError):
            manager.commit_result(Box(1, 1, 1))

    def test_legacy_snapshots_remain_compatible(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'old.fcad'
            save_snapshot(path, Box(2, 2, 2), [], [])
            current, _, _, graph = load_snapshot(path, include_feature_graph=True)
            self.assertIsNone(graph)
            self.assertAlmostEqual(current.volume, 8)


class ParametricReferencesAndUndoTest(unittest.TestCase):
    def test_edge_reference_rebuilds_after_resize(self):
        box = Feature.create('Box', 'primitive', {'type': 'box', 'dimensions': ['width', 10, 10]})
        initial = FeatureGraph([box], {'width': 10})
        ref = make_reference(box.id, initial.results[box.id], 'edge', 0)
        fillet = Feature.create('Fillet', 'fillet', {'edges': [ref], 'radius': 1}, box.id)
        graph = FeatureGraph([box, fillet], {'width': 10})
        graph.set_parameters({'width': 12})
        self.assertTrue(graph.results[fillet.id].is_valid)
        self.assertLess(graph.results[fillet.id].volume, 1200)
        self.assertGreaterEqual(resolve_reference(ref, graph.results[box.id], box.id), 0)
        with self.assertRaises(ValueError):
            resolve_reference(ref, graph.results[box.id], 'wrong_feature')

    def test_face_reference_and_graph_history_roundtrip(self):
        box = Feature.create('Box', 'primitive', {'type': 'box', 'dimensions': ['width', 10, 10]})
        base = FeatureGraph([box], {'width': 10})
        face_index = next(i for i, face in enumerate(base.results[box.id].faces()) if face.center().Z > 4)
        ref = make_reference(box.id, base.results[box.id], 'face', face_index)
        hole = Feature.create('Hole', 'hole', {'face': ref, 'diameter': 2, 'depth': 5}, box.id)
        graph = FeatureGraph([box, hole], {'width': 10})
        manager = ActionManager(None)
        manager.display_result = lambda part: setattr(manager, 'current_part', part)
        manager.install_parametric_graph(graph)
        modified = FeatureGraph.from_data(graph.to_data())
        modified.set_parameters({'width': 12})
        manager.install_parametric_graph(modified)
        self.assertTrue(manager.undo_solid())
        self.assertAlmostEqual(manager.current_part.volume, 1000 - 5 * np.pi)
        self.assertTrue(manager.redo_solid())
        self.assertAlmostEqual(manager.current_part.volume, 1200 - 5 * np.pi)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'history.fcad'
            save_snapshot(path, manager.current_part, [], [], feature_graph=manager.parametric_graph,
                          graph_history={'undo': manager.graph_undo, 'redo': manager.graph_redo})
            _, _, _, restored, history = load_snapshot(path, include_feature_graph=True, include_graph_history=True)
            self.assertEqual(len(history['undo']), 1)
            self.assertAlmostEqual(restored.results[hole.id].volume, 1200 - 5 * np.pi)

    def test_topology_change_rejected_without_mutating_feature_graph(self):
        box = Feature.create('Box', 'primitive', {'type': 'box', 'dimensions': [10, 10, 10]})
        initial = FeatureGraph([box])
        reference = make_reference(box.id, initial.results[box.id], 'edge', 0)
        fillet = Feature.create('Fillet', 'fillet', {'edges': [reference], 'radius': 1}, box.id)
        graph = FeatureGraph([box, fillet])
        old_volume = graph.results[fillet.id].volume
        with self.assertRaises(RebuildError):
            graph.edit(box.id, parameters={'type': 'cylinder', 'dimensions': [5, 10]})
        self.assertAlmostEqual(graph.results[fillet.id].volume, old_volume)
        self.assertEqual(graph.features[0].parameters['type'], 'box')

    def test_corrupted_and_legacy_reference_handling(self):
        shape = Box(10, 10, 10)
        reference = make_reference('box', shape, 'face', 0)
        self.assertEqual(reference['version'], 2)
        corrupted = dict(reference, topology={'solids': 1, 'faces': 99, 'edges': 12})
        with self.assertRaisesRegex(ValueError, 'topology changed'):
            resolve_reference(corrupted, shape, 'box')
        legacy = {key: reference[key] for key in ('feature_id', 'kind', 'geometry', 'center')}
        self.assertEqual(resolve_reference(legacy, shape, 'box'), 0)

    def test_unmatched_geometry_is_rejected(self):
        ref = make_reference('box', Box(10, 10, 10), 'edge', 0)
        with self.assertRaises(ValueError):
            resolve_reference(ref, Sphere(5), 'box')


class EditableParametricSketchTest(unittest.TestCase):
    def test_circle_sketch_rebuild_and_persistence(self):
        plane = Feature.create('Plane', 'datum_plane', {'origin': [0, 0, 'offset']})
        sketch = Feature.create('Sketch', 'planar_sketch', {'entities': [
            {'type': 'circle', 'center': [0, 0], 'radius': 'radius'},
            {'type': 'line', 'points': [[-5, 0], [5, 0]], 'construction': True}]}, plane.id)
        solid = Feature.create('Extrude', 'extrude', {'distance': 3}, sketch.id)
        graph = FeatureGraph([plane, sketch, solid], {'radius': 2, 'offset': 4})
        self.assertAlmostEqual(graph.results[solid.id].volume, 12 * np.pi)
        graph.set_parameters({'radius': 3, 'offset': 5})
        self.assertAlmostEqual(graph.results[solid.id].volume, 27 * np.pi)
        self.assertAlmostEqual(graph.results[solid.id].bounding_box().min.Z, 5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sketch.fcad'
            save_snapshot(path, graph.results[solid.id], [], [], feature_graph=graph)
            _, _, _, restored = load_snapshot(path, include_feature_graph=True)
            self.assertEqual(restored.features[1].id, sketch.id)
            self.assertAlmostEqual(restored.results[solid.id].volume, 27 * np.pi)

    def test_edit_entities_is_transactional(self):
        plane = Feature.create('Plane', 'datum_plane', {})
        sketch = Feature.create('Sketch', 'planar_sketch', {'entities': [
            {'type': 'line', 'points': [[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]]}]}, plane.id)
        solid = Feature.create('Extrude', 'extrude', {'distance': 2}, sketch.id)
        graph = FeatureGraph([plane, sketch, solid])
        self.assertAlmostEqual(graph.results[solid.id].volume, 32)
        with self.assertRaises(RebuildError):
            graph.edit(sketch.id, parameters={'entities': [{'type': 'line', 'points': [[0, 0], [1, 0]]}]})
        self.assertAlmostEqual(graph.results[solid.id].volume, 32)
        graph.edit(sketch.id, parameters={'entities': [{'type': 'circle', 'center': [0, 0], 'radius': 2}]})
        self.assertAlmostEqual(graph.results[solid.id].volume, 8 * np.pi)

    def test_parametric_revolve(self):
        plane = Feature.create('Plane', 'datum_plane', {})
        sketch = Feature.create('Sketch', 'planar_sketch', {'entities': [
            {'type': 'line', 'points': [[2,-1], [4,-1], [4,1], [2,1], [2,-1]]}]}, plane.id)
        revolve = Feature.create('Revolve', 'revolve', {'axis_direction': [0,1,0], 'angle': 360}, sketch.id)
        graph = FeatureGraph([plane, sketch, revolve])
        self.assertAlmostEqual(graph.results[revolve.id].volume, 24 * np.pi)


class ParametricSketchConstraintsTest(unittest.TestCase):
    def test_constrained_rectangle_rebuild_and_reject_conflict(self):
        plane = Feature.create('Plane', 'datum_plane', {})
        definition = {'points': {'a':[0,0], 'b':[4,0], 'c':[4,2], 'd':[0,2]},
            'entities':[{'type':'line', 'points':['a','b','c','d','a']}],
            'constraints':[
                {'kind':'fixed','entities':['a'],'value':[0,0]},
                {'kind':'fixed','entities':['d'],'value':[0,2]},
                {'kind':'horizontal','entities':[['a','b']]},
                {'kind':'vertical','entities':[['b','c']]},
                {'kind':'horizontal','entities':[['c','d']]},
                {'kind':'distance','entities':['a','b'],'value':'width'}]}
        sketch = Feature.create('Sketch', 'planar_sketch', definition, plane.id)
        solid = Feature.create('Extrude', 'extrude', {'distance':3}, sketch.id)
        graph = FeatureGraph([plane, sketch, solid], {'width':4})
        self.assertAlmostEqual(graph.results[solid.id].volume, 24, places=5)
        graph.set_parameters({'width':8})
        self.assertAlmostEqual(graph.results[solid.id].volume, 48, places=5)
        bad = dict(definition)
        bad['constraints'] = definition['constraints'] + [{'kind':'fixed','entities':['b'],'value':[1,0]}]
        with self.assertRaises(RebuildError):
            graph.edit(sketch.id, parameters=bad)
        self.assertAlmostEqual(graph.results[solid.id].volume, 48, places=5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'constrained.fcad'
            save_snapshot(path, graph.results[solid.id], [], [], feature_graph=graph)
            _, _, _, restored = load_snapshot(path, include_feature_graph=True)
            self.assertAlmostEqual(restored.results[solid.id].volume, 48, places=5)


class ParametricSketchDragTest(unittest.TestCase):
    def graph(self, fixed=False):
        plane = Feature.create('Plane', 'datum_plane', {})
        constraints = [{'kind': 'fixed', 'entities': ['b'], 'value': [4, 0]}] if fixed else []
        sketch = Feature.create('Sketch', 'planar_sketch', {'points': {'a':[0,0], 'b':[4,0], 'c':[4,4], 'd':[0,4]},
            'entities':[{'type':'line', 'points':['a','b','c','d','a']}], 'constraints': constraints}, plane.id)
        solid = Feature.create('Extrude', 'extrude', {'distance':2}, sketch.id)
        return FeatureGraph([plane, sketch, solid]), sketch, solid

    def test_drag_rebuilds_without_mutating_original(self):
        graph, sketch, solid = self.graph()
        updated = move_sketch_point(graph, sketch.id, 'b', (6, 0))
        self.assertAlmostEqual(graph.results[solid.id].volume, 32)
        self.assertAlmostEqual(updated.results[solid.id].volume, 40)
        self.assertEqual(updated.features[1].parameters['points']['b'], [6., 0.])

    def test_fixed_point_cannot_be_dragged(self):
        graph, sketch, solid = self.graph(fixed=True)
        with self.assertRaises(RebuildError):
            move_sketch_point(graph, sketch.id, 'b', (6, 0))
        self.assertAlmostEqual(graph.results[solid.id].volume, 32)

    def test_circle_radius_rebuild_and_invalid_radius(self):
        plane = Feature.create('Plane', 'datum_plane', {})
        sketch = Feature.create('Sketch', 'planar_sketch', {'points': {'c': [0, 0]},
            'entities': [{'type': 'circle', 'center': 'c', 'radius': 2}]}, plane.id)
        solid = Feature.create('Extrude', 'extrude', {'distance': 3}, sketch.id)
        graph = FeatureGraph([plane, sketch, solid])
        updated = set_circle_radius(graph, sketch.id, 0, 3)
        self.assertAlmostEqual(graph.results[solid.id].volume, 12 * math.pi, places=5)
        self.assertAlmostEqual(updated.results[solid.id].volume, 27 * math.pi, places=5)
        with self.assertRaises(ValueError):
            set_circle_radius(graph, sketch.id, 0, 0)

    def test_constraint_is_transactional(self):
        graph, sketch, solid = self.graph()
        updated = add_sketch_constraint(graph, sketch.id,
            {'kind': 'fixed', 'entities': ['b'], 'value': [4, 0]})
        self.assertEqual(len(updated.features[1].parameters['constraints']), 1)
        self.assertEqual(graph.features[1].parameters['constraints'], [])
        with self.assertRaises(RebuildError):
            add_sketch_constraint(updated, sketch.id,
                {'kind': 'fixed', 'entities': ['b'], 'value': [6, 0]})
        self.assertAlmostEqual(updated.results[solid.id].volume, 32)

    def test_arc_control_point_rebuilds_solid(self):
        plane = Feature.create('Plane', 'datum_plane', {})
        sketch = Feature.create('Sketch', 'planar_sketch', {
            'points': {'a': [-2, 0], 'b': [2, 0], 'm': [0, 2]},
            'entities': [{'type': 'arc', 'points': ['a', 'b', 'm']},
                         {'type': 'line', 'points': ['b', 'a']}]}, plane.id)
        solid = Feature.create('Extrude', 'extrude', {'distance': 2}, sketch.id)
        graph = FeatureGraph([plane, sketch, solid])
        updated = move_sketch_point(graph, sketch.id, 'm', (0, 3))
        self.assertGreater(updated.results[solid.id].volume, graph.results[solid.id].volume)


if __name__ == '__main__':
    unittest.main()
