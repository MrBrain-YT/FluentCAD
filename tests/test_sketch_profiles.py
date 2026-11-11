import sys
import math
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from build123d import extrude, Box, Cylinder, Sphere, Face, Plane, Align, Pos, Rot, import_step, import_stl
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from UI.model.profiles import build_profiles, sketch_edges, split_edges
from UI.model.ActionManager import ActionManager
from UI.model.ModelConverter import b123d_to_pro_mesh
from UI.model.solid_operations import build_extrusion, build_revolution, build_primitive, import_step_body
from UI.model.export import export_body
from UI.model.snapshot import save_snapshot, load_snapshot
from UI.model.timeline import Timeline, validate_timeline
from UI.model.feature_graph import Feature, FeatureGraph, RebuildError, resolve_parameters
from UI.model.sketch_drag import move_sketch_point, set_circle_radius, add_sketch_constraint, remove_sketch_constraint, edit_nurbs_parameters
from UI.model.geometry_references import make_reference, resolve_reference
from UI.model.measurement import measure_face, measure_model, measure_edges
from UI.model.expressions import evaluate_expression
from UI.model.constraints import Constraint, solve_constraints
from UI.model.sketch_constraints import line_points, add_automatic_coincidences, apply_constraint
from UI.model.transforms import move_copy, scale_model, mirror_model, rectangular_pattern, circular_pattern, body_index_for_face, apply_to_body
from UI.model.surface_operations import fillet_all, chamfer_all, shell_face, offset_solid, hole_at_face_center, fillet_edges, chamfer_edges
from UI.model.edge_selection import nearest_screen_edge, face_edge_indices
from UI.main.content import LayoutManager
from UI.main.extrude_handle import world_per_pixel, ExtrudeHandle, MoveAxisHandles, ParameterHandles, OperationHandle, projected_ring_angle, projected_axis_delta
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
    def test_projected_axis_follows_direction_and_ignores_perpendicular_motion(self):
        self.assertAlmostEqual(projected_axis_delta((0, 0), (50, 20), (0, 0), (100, 0), 2), 1)
        self.assertAlmostEqual(projected_axis_delta((0, 0), (50, 20), (0, 0), (-100, 0), 2), -1)
        self.assertAlmostEqual(projected_axis_delta((0, 0), (10, 10), (0, 0), (20, 20), 4), 2)
        self.assertEqual(projected_axis_delta((0, 0), (10, 10), (0, 0), (1, 1), 4), 0)

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
        handle.plotter, handle.dialog = plotter, SimpleNamespace(distance=field, direction=SimpleNamespace(currentData=lambda: 'one'))
        handle.drag_start, handle.updating, handle.sync = ('primary', (0, 100), 3., ((0, 0), (0, 100), 2.)), False, Mock()
        handle.drag()
        field.setValue.assert_called_once_with(4.)
        handle.sync.assert_called_once()
        self.assertFalse(handle.updating)

    def test_two_sided_extrude_controls_each_distance(self):
        handle = ExtrudeHandle.__new__(ExtrudeHandle)
        first, second = SimpleNamespace(setValue=Mock()), SimpleNamespace(setValue=Mock())
        handle.dialog = SimpleNamespace(distance=first, second_distance=second,
                                        direction=SimpleNamespace(currentData=lambda: 'two'))
        handle.plotter = SimpleNamespace(iren=SimpleNamespace(get_event_position=lambda: (-50, 0)))
        handle.drag_start = ('opposite', (0, 0), 3., ((0, 0), (-100, 0), 2.))
        handle.updating, handle.sync = False, Mock()
        handle.drag('opposite')
        second.setValue.assert_called_once_with(4.)
        first.setValue.assert_not_called()

    def test_two_sided_and_symmetric_show_second_handle(self):
        handle = ExtrudeHandle.__new__(ExtrudeHandle)
        mode = ['two']
        handle.dialog = SimpleNamespace(extent=SimpleNamespace(currentData=lambda: 'distance'),
            direction=SimpleNamespace(currentData=lambda: mode[0]),
            offset=SimpleNamespace(value=lambda: 0.), distance=SimpleNamespace(value=lambda: 3.),
            second_distance=SimpleNamespace(value=lambda: 5.), manager=SimpleNamespace())
        handle.anchor, handle.normal = np.zeros(3), np.array([1., 0., 0.])
        handle.widgets = {key: SimpleNamespace(SetEnabled=Mock()) for key in ('primary', 'opposite')}
        handle.representations = {key: SimpleNamespace(SetWorldPosition=Mock()) for key in handle.widgets}
        camera = SimpleNamespace(GetParallelProjection=lambda: True, GetParallelScale=lambda: 10., position=(0, 0, 10))
        handle.plotter = SimpleNamespace(camera=camera, ren_win=SimpleNamespace(GetSize=lambda: (800, 600)),
                                         remove_actor=Mock(), add_mesh=Mock(), render=Mock())
        handle.updating = False
        handle.sync()
        handle.widgets['opposite'].SetEnabled.assert_called_with(True)
        handle.representations['opposite'].SetWorldPosition.assert_called_with((-5., 0., 0.))
        mode[0] = 'symmetric'
        handle.sync()
        handle.representations['opposite'].SetWorldPosition.assert_called_with((-3., 0., 0.))
        mode[0] = 'one'
        handle.sync()
        handle.widgets['opposite'].SetEnabled.assert_called_with(False)


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
            handle.drag_start = (axis, (0, 100), 3., ((0, 0), (0, 100), 2.))
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
        handle.mode, handle.drag_start = 'rotate', ('Z', (0, 100), 10., 350., None, 1.)
        handle.cursor_ring_angle = Mock(return_value=10.)
        handle.drag('Z')
        rotation.setValue.assert_called_once_with(30.)
        handle.mode, handle.drag_start = 'scale', ('S', (0, 100), 1., None, ((0, 0), (0, 80), 1.), 1.)
        handle.drag('S')
        scale.setValue.assert_called_once_with(1.25)


class OperationHandleTest(unittest.TestCase):
    def test_linear_handle_updates_field_and_respects_zoom(self):
        camera = SimpleNamespace(GetParallelProjection=lambda: True, GetParallelScale=lambda: 10.)
        plotter = SimpleNamespace(camera=camera, iren=SimpleNamespace(get_event_position=lambda: (0, 150)),
                                  ren_win=SimpleNamespace(GetSize=lambda: (1000, 1000)))
        handle = OperationHandle.__new__(OperationHandle)
        handle.plotter, handle.field, handle.angle = plotter, SimpleNamespace(setValue=Mock()), False
        handle.drag_start, handle.updating, handle.sync = ((0, 100), 3., None, ((0, 0), (0, 100), 2.)), False, Mock()
        handle.drag()
        handle.field.setValue.assert_called_once_with(4.)
        handle.sync.assert_called_once()

    def test_angle_handle_wraps_and_updates_field(self):
        handle = OperationHandle.__new__(OperationHandle)
        handle.plotter = SimpleNamespace(iren=SimpleNamespace(get_event_position=lambda: (0, 150)))
        handle.field, handle.angle = SimpleNamespace(setValue=Mock()), True
        handle.drag_start, handle.updating, handle.sync = ((0, 100), 10., 350., None), False, Mock()
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
        self.assertEqual(handle.plotter.remove_actor.call_count, 3)
        handle.plotter.remove_actor.assert_any_call('temporary', render=False)


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
    def test_scene_spline_tools_capture_all_types(self):
        from UI.model.tools.Spline import Spline
        from UI.model.sketch_capture import capture_sketch
        points = [np.array([0.,0.,0.]), np.array([2.,0.,0.]), np.array([3.,1.,0.]), np.array([4.,2.,0.])]
        for kind in ('spline','fit_spline','nurbs','periodic_nurbs'):
            plotter = SimpleNamespace(track_click_position=Mock(), add_points=Mock(side_effect=lambda *a,**k:Mock()), disable_picking=Mock())
            owner = SimpleNamespace(get_snapped_pos=lambda value:(value,value[:2]), redraw_sketch_geometry=Mock(),
                                    status_callback=Mock(), control_modifier_active=lambda:False)
            tool = Spline(plotter, {}, owner, kind); tool.is_construction=True; tool.start()
            for point in points: tool.on_click(point)
            tool.stop_drawing()
            captured = capture_sketch({'sketch_func':[tool]}, np.zeros(3), np.array([1.,0,0]), np.array([0.,1,0]))
            self.assertEqual(captured['entities'][0]['type'], kind)
            plane = Feature.create('Plane','datum_plane',{})
            sketch = Feature.create('Sketch','planar_sketch',captured,plane.id)
            graph = FeatureGraph([plane,sketch])
            self.assertEqual(len(graph.results[sketch.id].constraint_status.points),4)

    def test_promoted_new_sketch_cancel_does_not_commit(self):
        manager = ActionManager(None)
        actor = Mock()
        manager.plotter = SimpleNamespace(actors={}, remove_actor=Mock(), render=Mock(), disable_picking=Mock(),
                                          add_mesh=Mock(return_value=actor))
        manager.render_history_sketch = Mock()
        manager.sketch_editing = manager.pending_sketch = True
        manager.active_center=np.zeros(3); manager.active_normal=np.array([0.,0,1])
        manager.u_axis=np.array([1.,0,0]); manager.v_axis=np.array([0.,1,0])
        manager.activated_sketch_id=1
        manager.sketchs_data[1]={'sketch_func':[SimpleNamespace(type='line', points=[np.array([0.,0,0]),np.array([2.,0,0])],
            is_construction=False, render_obj={'points':[], 'lines':[]})]}
        manager.promote_current_sketch()
        self.assertIsNone(manager.parametric_graph)
        self.assertIsNotNone(manager.history_edit_graph)
        self.assertTrue(manager.cancel_history_sketch())
        self.assertIsNone(manager.parametric_graph)
        self.assertFalse(manager.sketch_editing)

    def test_promote_render_failure_keeps_live_sketch(self):
        manager = ActionManager(None)
        manager.plotter = SimpleNamespace(actors={}, remove_actor=Mock(), render=Mock())
        manager.render_history_sketch = Mock(side_effect=ValueError('render failed'))
        manager.sketch_editing = manager.pending_sketch = True
        manager.active_center=np.zeros(3); manager.active_normal=np.array([0.,0,1])
        manager.u_axis=np.array([1.,0,0]); manager.v_axis=np.array([0.,1,0])
        tool = SimpleNamespace(type='line', points=[np.array([0.,0,0]),np.array([2.,0,0])],
                               is_construction=False, render_obj={'points':[], 'lines':[]})
        manager.activated_sketch_id=1; manager.sketchs_data[1]={'sketch_func':[tool]}
        with self.assertRaisesRegex(ValueError, 'render failed'): manager.promote_current_sketch()
        self.assertIsNone(manager.history_edit_graph)
        self.assertIsNone(manager.parametric_graph)
        self.assertIs(manager.get_current_sketch_data()['sketch_func'][0],tool)
        self.assertTrue(manager.sketch_editing)

    def test_drawn_bezier_is_immediately_editable_in_scene_candidate(self):
        manager = ActionManager(None)
        manager.plotter = SimpleNamespace(actors={}, remove_actor=Mock(), render=Mock(), disable_picking=Mock(),
            track_click_position=Mock(), add_points=Mock(side_effect=lambda *a,**k:Mock()), add_mesh=Mock(return_value=Mock()))
        manager.render_history_sketch = Mock()
        manager.sketch_editing = manager.pending_sketch = True
        manager.active_center=np.zeros(3); manager.active_normal=np.array([0.,0,1])
        manager.u_axis=np.array([1.,0,0]); manager.v_axis=np.array([0.,1,0])
        manager.activated_sketch_id=1; manager.sketchs_data[1]={'sketch_func':[]}
        manager.get_snapped_pos=lambda point:(np.asarray(point,dtype=float),np.asarray(point,dtype=float)[:2])
        manager.activate_spline_mode('spline')
        tool=manager.active_tool
        for point in ([0,0,0],[1,0,0],[2,1,0],[3,1,0]): tool.on_click(point)
        self.assertIsNone(manager.parametric_graph)
        self.assertIsNotNone(manager.history_edit_id)
        self.assertIsNone(manager.active_tool)
        manager.move_history_point('SP1.P2', (1,.5))
        np.testing.assert_allclose(manager.history_edit_graph.results[manager.history_edit_id].constraint_status.points['SP1.P2'],[1,.5],atol=1e-6)
        self.assertIsNone(manager.parametric_graph)

    def test_sketch_camera_style_rotates_only_with_control(self):
        from UI.model.sketch_navigation import SketchCameraStyle
        manager = SimpleNamespace(_finish_history_drag=Mock())
        style = SketchCameraStyle(manager)
        interactor = SimpleNamespace(GetControlKey=Mock(return_value=0), GetEventPosition=lambda:(20,30))
        with patch.object(style, 'GetInteractor', return_value=interactor), \
             patch.object(style, 'FindPokedRenderer') as find, \
             patch.object(style, 'GetCurrentRenderer', return_value=object()), \
             patch.object(style, 'StartRotate') as rotate, patch.object(style, 'EndRotate') as end:
            style._press()
            rotate.assert_not_called()
            interactor.GetControlKey.return_value = 1
            style._press()
            find.assert_called_once_with(20,30)
            rotate.assert_called_once()
            style._release()
            end.assert_called_once()
            manager._finish_history_drag.assert_called_once()

    def test_sketch_camera_style_restored(self):
        manager = ActionManager(None)
        original = object()
        manager.plotter = SimpleNamespace(iren=SimpleNamespace(style=original))
        manager.lock_sketch_camera()
        self.assertIsNot(manager.plotter.iren.style, original)
        manager.restore_camera_style()
        self.assertIs(manager.plotter.iren.style, original)

    def test_control_click_does_not_create_sketch_point(self):
        from UI.model.tools.Line import Line
        plotter = SimpleNamespace(track_click_position=Mock())
        owner = SimpleNamespace(control_modifier_active=Mock(return_value=True))
        tool = Line(plotter, {}, owner)
        tool.on_click = Mock()
        tool.track_sketch_click()
        callback = plotter.track_click_position.call_args.kwargs['callback']
        callback((1,2,3))
        tool.on_click.assert_not_called()
        owner.control_modifier_active.return_value = False
        callback((1,2,3))
        tool.on_click.assert_called_once_with((1,2,3))

    def test_scene_point_hit_uses_pixel_distance(self):
        manager, sketch, _ = self.scene_history_manager()
        manager.history_edit_id, manager.history_edit_graph = sketch.id, manager.parametric_graph
        manager.history_point_actors = {'a':Mock(), 'b':Mock()}
        position = [0.,0.,0.,1.]
        renderer = SimpleNamespace(SetWorldPoint=lambda *p: position.__setitem__(slice(None),p),
                                   WorldToDisplay=lambda:None,
                                   GetDisplayPoint=lambda:(position[0]*10,position[1]*10,.5))
        manager.plotter.renderer = renderer
        manager.plotter.iren = SimpleNamespace(get_event_position=lambda:(43,1))
        self.assertEqual(manager._history_handle_at_cursor(), 'b')
        manager.plotter.iren.get_event_position = lambda:(100,100)
        self.assertIsNone(manager._history_handle_at_cursor())

    def scene_history_manager(self, fixed=False):
        plane = Feature.create('Plane','datum_plane',{})
        constraints = [{'kind':'fixed','entities':['b'],'value':[4,0]}] if fixed else []
        sketch = Feature.create('Sketch','planar_sketch', {'points':{'a':[0,0],'b':[4,0],'c':[4,4],'d':[0,4]},
            'entities':[{'type':'line','points':['a','b','c','d','a']}], 'constraints':constraints}, plane.id)
        solid = Feature.create('Extrude','extrude',{'distance':2},sketch.id)
        graph = FeatureGraph([plane,sketch,solid])
        body = Mock(); actor = Mock(); actor.prop = SimpleNamespace(color=None, opacity=None)
        manager = ActionManager(None)
        manager.plotter = SimpleNamespace(actors={'main_part':body}, remove_actor=Mock(), disable_picking=Mock(),
                                           add_mesh=Mock(return_value=actor), render=Mock())
        manager.display_result = lambda part: setattr(manager, 'current_part', part)
        manager.parametric_graph = graph
        manager.current_part = graph.results[solid.id]
        manager.align_camera_to_normal = Mock()
        manager.update_grid_lod = Mock()
        manager.render_history_sketch = Mock()
        return manager, sketch, solid

    def test_scene_history_edit_cancel_commit_and_undo(self):
        manager, sketch, solid = self.scene_history_manager()
        manager.start_history_sketch(sketch.id)
        manager.move_history_point('b', (6,0))
        self.assertAlmostEqual(manager.history_edit_graph.results[solid.id].volume, 40, places=5)
        self.assertAlmostEqual(manager.parametric_graph.results[solid.id].volume, 32, places=5)
        self.assertTrue(manager.cancel_history_sketch())
        self.assertAlmostEqual(manager.parametric_graph.results[solid.id].volume, 32, places=5)
        manager.start_history_sketch(sketch.id)
        manager.move_history_point('b', (6,0))
        manager.finish_history_sketch()
        self.assertAlmostEqual(manager.parametric_graph.results[solid.id].volume, 40, places=5)
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / 'scene_edit.fcad'
            save_snapshot(filename, manager.current_part, [], [], feature_graph=manager.parametric_graph)
            _, _, _, restored = load_snapshot(filename, include_feature_graph=True)
            self.assertAlmostEqual(restored.results[solid.id].volume, 40, places=5)
        self.assertTrue(manager.undo_solid())
        self.assertAlmostEqual(manager.parametric_graph.results[solid.id].volume, 32, places=5)

    def test_scene_history_edit_rejects_constraint_conflict(self):
        manager, sketch, solid = self.scene_history_manager(fixed=True)
        manager.start_history_sketch(sketch.id)
        before = manager.history_edit_graph.to_data()
        with self.assertRaises(RebuildError): manager.move_history_point('b', (6,0))
        self.assertEqual(manager.history_edit_graph.to_data(), before)
        self.assertAlmostEqual(manager.parametric_graph.results[solid.id].volume, 32, places=5)

    def test_scene_history_point_drag_event_is_transactional(self):
        manager, sketch, solid = self.scene_history_manager()
        manager.start_history_sketch(sketch.id)
        actor = Mock()
        manager.history_point_actors = {'b':actor}
        manager._history_handle_at_cursor = Mock(return_value='b')
        manager.plotter.iren = SimpleNamespace(get_event_position=Mock(side_effect=[(10,10),(30,10)]),
            interactor=Mock())
        manager._screen_to_sketch_plane = Mock(return_value=np.array([6.,0.,0.]))
        manager._begin_history_drag()
        manager._finish_history_drag()
        self.assertAlmostEqual(manager.history_edit_graph.results[solid.id].volume, 40, places=5)
        self.assertAlmostEqual(manager.parametric_graph.results[solid.id].volume, 32, places=5)

    def test_scene_history_renders_circle_radius_handle(self):
        plane = Feature.create('Plane','datum_plane',{})
        sketch = Feature.create('Sketch','planar_sketch',{'points':{'c':[0,0]},
            'entities':[{'type':'circle','center':'c','radius':2}]},plane.id)
        solid = Feature.create('Extrude','extrude',{'distance':2},sketch.id)
        graph = FeatureGraph([plane,sketch,solid])
        manager = ActionManager(None)
        manager.plotter = SimpleNamespace(remove_actor=Mock(), add_mesh=Mock(side_effect=lambda *a,**k: Mock()),
                                          add_points=Mock(side_effect=lambda *a,**k: Mock()), render=Mock())
        manager.history_edit_id, manager.history_edit_graph = sketch.id, graph
        manager.active_center = np.zeros(3); manager.u_axis=np.array([1.,0,0]); manager.v_axis=np.array([0.,1,0])
        manager.render_history_sketch()
        self.assertIn('c', manager.history_point_actors)
        self.assertIn(('radius',0,'c'), manager.history_point_actors)

    def test_scene_history_adds_geometry_without_replacing_original(self):
        manager, sketch, solid = self.scene_history_manager()
        manager.start_history_sketch(sketch.id)
        tool = SimpleNamespace(type='line', points=[np.array([0.,0.,0.]),np.array([1.,1.,0.])],
                               is_construction=True, is_drawing=False, render_obj={'points':[], 'lines':[]})
        manager.get_current_sketch_data()['sketch_func'].append(tool)
        manager.finish_history_sketch()
        feature = next(item for item in manager.parametric_graph.features if item.id == sketch.id)
        self.assertEqual(len(feature.parameters['entities']), 2)
        self.assertEqual(feature.parameters['entities'][0]['points'], ['a','b','c','d','a'])
        self.assertTrue(feature.parameters['entities'][1]['construction'])
        self.assertTrue(any(item['kind'] == 'coincident' for item in feature.parameters['constraints']))
        self.assertAlmostEqual(manager.parametric_graph.results[solid.id].volume, 32, places=5)

    def test_scene_spline_flush_constraints_cancel_and_persist(self):
        manager, sketch, solid = self.scene_history_manager()
        manager.start_history_sketch(sketch.id)
        spline = SimpleNamespace(type='nurbs', points=[np.array([1.,1.,0.]),np.array([2.,1.,0.]),np.array([3.,2.,0.])],
            degree=2, weights=[1.,1.,1.], knots=[0.,0.,0.,1.,1.,1.], is_construction=True,
            render_obj={'points':[], 'lines':[]})
        manager.get_current_sketch_data()['sketch_func'].append(spline)
        manager.flush_history_tools()
        self.assertEqual(len(manager.history_edit_graph.features[1].parameters['entities']),2)
        self.assertEqual(len(manager.parametric_graph.features[1].parameters['entities']),1)
        entity = manager.history_edit_graph.features[1].parameters['entities'][1]
        point = entity['points'][0]
        manager.history_edit_graph = add_sketch_constraint(manager.history_edit_graph,sketch.id,
            {'kind':'fixed','entities':[point],'value':[1.,1.]})
        manager.history_edit_graph = edit_nurbs_parameters(manager.history_edit_graph,sketch.id,1,2,
            [1.,.7,1.],[0.,0.,0.,1.,1.,1.])
        manager.finish_history_sketch()
        saved = manager.parametric_graph.features[1].parameters
        self.assertEqual(saved['entities'][1]['weights'], [1.,.7,1.])
        self.assertTrue(any(c['kind']=='fixed' and c['entities']==[point] for c in saved['constraints']))
        self.assertAlmostEqual(manager.parametric_graph.results[solid.id].volume,32,places=5)
        self.assertTrue(manager.undo_solid())
        self.assertEqual(len(manager.parametric_graph.features[1].parameters['entities']),1)

    def test_capture_elliptical_arc_build_and_roundtrip(self):
        from UI.model.sketch_capture import capture_sketch
        origin = np.zeros(3); u = np.array([1.,0,0]); v = np.array([0.,1,0])
        arc = SimpleNamespace(type='ellipse', center=origin, major_axis_vector=u, major_radius=4., minor_radius=2.,
                              is_closed=False, start_angle=0., end_angle=math.pi/2,
                              points=[origin, [4,0,0], [0,2,0], [4,0,0], [0,2,0]], is_construction=False)
        line = SimpleNamespace(type='line', points=[np.array([0.,2.,0.]), origin, np.array([4.,0.,0.])], is_construction=False)
        data = capture_sketch({'sketch_func':[arc,line]}, origin,u,v)
        self.assertFalse(data['entities'][0]['is_closed'])
        plane = Feature.create('Plane','datum_plane',{})
        sketch = Feature.create('Sketch','planar_sketch',data,plane.id)
        solid = Feature.create('Extrude','extrude',{'distance':2},sketch.id)
        graph = FeatureGraph([plane,sketch,solid])
        self.assertAlmostEqual(graph.results[solid.id].volume, 4*math.pi, places=5)
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / 'ellipse_arc.fcad'
            save_snapshot(filename, graph.results[solid.id], [], [], feature_graph=graph)
            _, _, _, restored = load_snapshot(filename, include_feature_graph=True)
            self.assertAlmostEqual(restored.results[solid.id].volume, 4*math.pi, places=5)

    def test_legacy_migration_preserves_snapshot_undo_and_redo(self):
        manager = ActionManager(None)
        manager.plotter = SimpleNamespace(remove_actor=Mock())
        manager.display_result = lambda part: setattr(manager, 'current_part', part)
        manager.current_part = Box(2,2,2)
        manager.undo_stack = [None]
        manager.timeline.append('Box', {'Size':'2'})
        manager.convert_legacy_model()
        self.assertEqual(manager.parametric_graph.features[0].kind, 'frozen_body')
        self.assertAlmostEqual(manager.current_part.volume, 8)
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / 'converted.fcad'
            save_snapshot(filename, manager.current_part, [], [], feature_graph=manager.parametric_graph)
            _, _, _, restored = load_snapshot(filename, include_feature_graph=True)
            from UI.model.legacy_conversion import restore_snapshot
            body, undo, redo, timeline = restore_snapshot(restored)
            self.assertAlmostEqual(body.volume, 8)
            self.assertEqual(undo, [None])
            self.assertEqual(len(timeline['entries']), 1)
        self.assertTrue(manager.undo_solid())
        self.assertIsNone(manager.parametric_graph)
        self.assertEqual(manager.undo_stack, [None])
        self.assertTrue(manager.redo_solid())
        self.assertEqual(manager.parametric_graph.features[0].kind, 'frozen_body')

    def test_interactive_capture_preserves_axes_and_constraints(self):
        from UI.model.sketch_capture import capture_sketch
        origin = np.array([5., 2., 3.]); u = np.array([0., 1., 0.]); v = np.array([0., 0., 1.])
        world = lambda x,y: origin + x*u + y*v
        line = SimpleNamespace(type='line', points=[world(0,0), world(4,0)], is_construction=False)
        circle = SimpleNamespace(type='circle', points=[world(2,2),world(3,2)], center=world(2,2), radius=1., is_construction=False)
        data = capture_sketch({'sketch_func':[line,circle],
                               'constraints':[Constraint('horizontal', (('L1.P1','L1.P2'),)),
                                              Constraint('fixed', ('L1.P1',), (np.float64(0), np.float64(0)))]}, origin,u,v)
        np.testing.assert_allclose(data['points']['L1.P2'], [4,0])
        np.testing.assert_allclose(data['points']['C2.center'], [2,2])
        plane = Feature.create('Plane','datum_plane',{'origin':origin.tolist(), 'normal':[1,0,0], 'x_dir':u.tolist()})
        sketch = Feature.create('Sketch','planar_sketch',data,plane.id)
        graph = FeatureGraph([plane,sketch])
        self.assertEqual(len(graph.results[sketch.id].profiles), 1)
        center = graph.results[sketch.id].profiles[0].center()
        np.testing.assert_allclose(tuple(center), world(2,2), atol=1e-6)

    def test_face_supported_sketch_and_join_rebuild(self):
        box = Feature.create('Box','primitive',{'type':'box','dimensions':[4,4,4], 'operation':'new_body'})
        initial = FeatureGraph([box])
        body = initial.results[box.id]
        face_index = next(i for i, face in enumerate(body.faces()) if tuple(face.normal_at())[2] > .9)
        ref = make_reference(box.id, body, 'face', face_index)
        plane = Feature.create('Plane','datum_plane',{'face':ref, 'x_dir':[1,0,0]},box.id)
        sketch = Feature.create('Sketch','planar_sketch', {'points':{'c':[0,0]},
            'entities':[{'type':'circle','center':'c','radius':1}]},plane.id)
        extrude_feature = Feature.create('Extrude','extrude',{'distance':1,'operation':'join',
            'base_id':box.id,'profile_index':0,'profile_count':1},sketch.id)
        graph = FeatureGraph([box,plane,sketch,extrude_feature])
        self.assertAlmostEqual(graph.results[extrude_feature.id].volume, 64+math.pi, places=5)
        graph.edit(box.id, parameters={'type':'box','dimensions':[6,4,4], 'operation':'new_body'})
        self.assertAlmostEqual(graph.results[extrude_feature.id].volume, 96+math.pi, places=5)

    def test_interactive_sketch_capture_and_selected_profile_extrusion(self):
        manager = ActionManager(None)
        actors = {}
        def add_mesh(mesh, **kwargs):
            actor = Mock(); actor.prop = SimpleNamespace(color=None, opacity=None)
            actors[kwargs['name']] = actor
            return actor
        manager.plotter = SimpleNamespace(actors=actors, add_mesh=add_mesh, remove_actor=Mock(),
                                          disable_picking=Mock(), render=Mock())
        manager.display_result = lambda part: setattr(manager, 'current_part', part)
        manager.active_center = np.array([0.,0.,0.]); manager.active_normal = np.array([0.,0.,1.])
        manager.u_axis = np.array([1.,0.,0.]); manager.v_axis = np.array([0.,1.,0.])
        def line(points):
            return SimpleNamespace(type='line', points=[np.array([x,y,0.]) for x,y in points],
                                   is_construction=False, render_obj={'points':[], 'lines':[]}, is_drawing=False)
        manager.activated_sketch_id = 1
        manager.sketchs_data[1] = {'sketch_func':[line([(0,0),(2,0),(2,2),(0,2),(0,0)]),
                                                   line([(5,0),(8,0),(8,3),(5,3),(5,0)])]}
        manager.sketch_editing = True
        manager.exit_sketch_mode()
        self.assertEqual(len(manager.profiles), 2)
        sketch_id = manager.parametric_sketch_id
        manager._pick_actor = Mock(return_value=SimpleNamespace(GetActor=lambda: actors[manager.profiles[1]['name']]))
        self.assertTrue(manager._select_profile())
        self.assertEqual(manager.selected_profile_index, 1)
        selected_area = manager.profiles[1]['face'].area
        feature_id = manager.commit_parametric_extrusion(1, 3)
        self.assertAlmostEqual(manager.current_part.volume, selected_area*3, places=5)
        feature = next(item for item in manager.parametric_graph.features if item.id == feature_id)
        self.assertEqual(feature.input_id, sketch_id)
        self.assertEqual(feature.parameters['profile_index'], 1)
        self.assertEqual(feature.parameters['profile_count'], 2)
        self.assertTrue(manager.undo_solid())
        self.assertIsNone(manager.current_part)
        self.assertEqual(len(manager.profiles), 2)
        self.assertTrue(manager.redo_solid())
        self.assertAlmostEqual(manager.current_part.volume, selected_area*3, places=5)
        restored = FeatureGraph.from_data(manager.parametric_graph.to_data())
        self.assertAlmostEqual(restored.results[feature_id].volume, selected_area*3, places=5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scene_extrude.fcad'
            save_snapshot(path, manager.current_part, [], [], feature_graph=restored)
            _, _, _, loaded = load_snapshot(path, include_feature_graph=True)
            self.assertAlmostEqual(loaded.results[feature_id].volume, selected_area*3, places=5)
        params = next(item for item in restored.features if item.id == sketch_id).parameters
        params['points']['L2.P3'][1] = 4
        params['points']['L2.P4'][1] = 4
        restored.edit(sketch_id, parameters=params)
        self.assertNotAlmostEqual(restored.results[feature_id].volume, selected_area*3, places=5)
        manager.show_parametric_profiles(sketch_id)
        second_area = manager.profiles[0]['face'].area
        second = manager.commit_parametric_extrusion(0, 2, operation='new_body')
        self.assertAlmostEqual(manager.parametric_graph.results[second].volume, selected_area*3 + second_area*2, places=5)
        second_feature = next(item for item in manager.parametric_graph.features if item.id == second)
        self.assertEqual(second_feature.parameters['base_id'], feature_id)

    def test_extrusion_rejects_profile_count_change(self):
        plane = Feature.create('Plane','datum_plane',{})
        sketch = Feature.create('Sketch','planar_sketch', {'points':{'a':[0,0],'b':[2,0],'c':[2,2],'d':[0,2]},
            'entities':[{'type':'line','points':['a','b','c','d','a']}]}, plane.id)
        solid = Feature.create('Extrude','extrude',{'distance':3,'profile_index':0,'profile_count':1},sketch.id)
        graph = FeatureGraph([plane,sketch,solid])
        changed = dict(sketch.parameters)
        changed['points'] = dict(changed['points'], e=[1,-1], f=[1,3])
        changed['entities'] = changed['entities'] + [{'type':'line','points':['e','f']}]
        with self.assertRaisesRegex(RebuildError, 'topology changed'):
            graph.edit(sketch.id, parameters=changed)
        self.assertAlmostEqual(graph.results[solid.id].volume, 12)

    def test_profile_identity_survives_unrelated_topology_change(self):
        plane = Feature.create('Plane','datum_plane',{})
        sketch = Feature.create('Sketch','planar_sketch', {'points':{'a':[0,0],'b':[5,0]},
            'entities':[{'type':'circle','center':'a','radius':1}, {'type':'circle','center':'b','radius':2}]}, plane.id)
        initial = FeatureGraph([plane,sketch])
        result = initial.results[sketch.id]
        selected = next(i for i, face in enumerate(result.profiles) if face.center().X > 3)
        key = result.profile_keys[selected]
        self.assertIsNotNone(key)
        solid = Feature.create('Extrude','extrude',{'distance':3,'profile_index':selected,
                                                     'profile_count':2,'profile_key':key},sketch.id)
        graph = FeatureGraph([plane,sketch,solid])
        changed = dict(sketch.parameters)
        changed['points'] = dict(changed['points'], c=[-5,0])
        changed['entities'] = changed['entities'] + [{'type':'circle','center':'c','radius':.5}]
        graph.edit(sketch.id, parameters=changed)
        self.assertAlmostEqual(graph.results[solid.id].volume, 12*math.pi, places=5)
        missing = dict(changed, entities=[changed['entities'][0], changed['entities'][2]])
        with self.assertRaisesRegex(RebuildError, 'missing or ambiguous'):
            graph.edit(sketch.id, parameters=missing)
        self.assertAlmostEqual(graph.results[solid.id].volume, 12*math.pi, places=5)

    def test_split_profile_reports_missing_identity(self):
        plane = Feature.create('Plane','datum_plane',{})
        sketch = Feature.create('Sketch','planar_sketch', {'points':{'c':[0,0]},
            'entities':[{'type':'circle','center':'c','radius':2}]}, plane.id)
        initial = FeatureGraph([plane,sketch])
        key = initial.results[sketch.id].profile_keys[0]
        self.assertIsNotNone(key)
        solid = Feature.create('Extrude','extrude',{'distance':2,'profile_index':0,'profile_count':1,'profile_key':key},sketch.id)
        graph = FeatureGraph([plane,sketch,solid])
        changed = dict(sketch.parameters)
        changed['points'] = dict(changed['points'], a=[-3,0], b=[3,0])
        changed['entities'] = changed['entities'] + [{'type':'line','points':['a','b']}]
        with self.assertRaisesRegex(RebuildError, 'missing or ambiguous'):
            graph.edit(sketch.id, parameters=changed)
        self.assertAlmostEqual(graph.results[solid.id].volume, 8*math.pi, places=5)

    def test_reselect_profile_after_suppressed_sketch_split(self):
        plane = Feature.create('Plane','datum_plane',{})
        sketch = Feature.create('Sketch','planar_sketch', {'points':{'c':[0,0]},
            'entities':[{'type':'circle','center':'c','radius':2}]}, plane.id)
        start = FeatureGraph([plane, sketch])
        solid = Feature.create('Extrude','extrude',{'distance':2,'profile_index':0,'profile_count':1,
            'profile_key':start.results[sketch.id].profile_keys[0]},sketch.id)
        graph = FeatureGraph([plane,sketch,solid])
        graph.edit(solid.id, suppressed=True)
        changed = dict(sketch.parameters)
        changed['points'] = dict(changed['points'], a=[-3,0], b=[3,0])
        changed['entities'] = changed['entities'] + [{'type':'line','points':['a','b']}]
        graph.edit(sketch.id, parameters=changed)
        manager = ActionManager(None)
        actor = Mock(); actor.prop = SimpleNamespace(color=None, opacity=None)
        manager.plotter = SimpleNamespace(add_mesh=Mock(return_value=actor), remove_actor=Mock(), render=Mock())
        manager.parametric_graph = graph
        ui = SimpleNamespace(action_manager=manager, window=None)
        LayoutManager.reselect_extrusion_profile(ui, solid.id)
        self.assertEqual(len(manager.profiles), 2)
        manager.profile_selection_callback(0)
        chosen = next(item for item in manager.parametric_graph.features if item.id == solid.id)
        self.assertTrue(chosen.suppressed)
        self.assertEqual(chosen.parameters['profile_count'], 2)
        resumed = FeatureGraph.from_data(manager.parametric_graph.to_data())
        resumed.edit(solid.id, suppressed=False)
        self.assertAlmostEqual(resumed.results[solid.id].volume, 4*math.pi, places=5)

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
    def test_general_curvature_circle_ellipse_and_conflict(self):
        plane = Feature.create('Plane','datum_plane',{})
        sketch = Feature.create('Sketch','planar_sketch',{
            'points':{'c':[0,0]},
            'entities':[{'type':'circle','center':'c','radius':2},
                        {'type':'ellipse','center':'c','radius_x':2,'radius_y':2,'construction':True}]},plane.id)
        solid = Feature.create('Extrude','extrude',{'distance':2},sketch.id)
        graph = FeatureGraph([plane, sketch, solid], {'contact':0})
        relation = {'kind':'curvature_curves','entities':[['@curve',0],['@curve',1]],'value':['contact',0]}
        updated = add_sketch_constraint(graph, sketch.id, relation)
        self.assertLess(updated.results[sketch.id].constraint_status.residual, 1e-7)
        with self.assertRaises(RebuildError): updated.set_parameters({'contact':.25})
        self.assertAlmostEqual(updated.results[solid.id].volume, 8*math.pi, places=5)

    def test_general_curvature_nurbs_and_arc(self):
        plane = Feature.create('Plane','datum_plane',{})
        sketch = Feature.create('Sketch','planar_sketch',{
            'points':{'a':[1,0],'b':[1,1],'c':[0,1], 'm':[2**-.5,2**-.5]},
            'entities':[{'type':'nurbs','points':['a','b','c'],'degree':2,'weights':[1,2**-.5,1],
                         'knots':[0,0,0,1,1,1], 'construction':True},
                        {'type':'arc','points':['a','c','m'], 'construction':True}]},plane.id)
        graph = FeatureGraph([plane, sketch])
        updated = add_sketch_constraint(graph, sketch.id, {'kind':'curvature_curves',
            'entities':[['@curve',0],['@curve',1]], 'value':[.5,.5]})
        self.assertLess(updated.results[sketch.id].constraint_status.residual, 1e-7)
        restored = FeatureGraph.from_data(updated.to_data())
        self.assertLess(restored.results[sketch.id].constraint_status.residual, 1e-7)

    def test_nurbs_parameters_transactional_and_linked(self):
        plane = Feature.create('Plane','datum_plane',{})
        sketch = Feature.create('Sketch','planar_sketch',{
            'points':{'a':[1,0],'b':[1,1],'c':[0,1],'o':[0,0]},
            'entities':[{'type':'nurbs','points':['a','b','c'],'degree':2,'weights':[1,1,1],
                         'knots':[0,0,0,1,1,1]}, {'type':'line','points':['c','o','a']}]},plane.id)
        solid = Feature.create('Extrude','extrude',{'distance':2},sketch.id)
        graph = FeatureGraph([plane, sketch, solid], {'w':2**-.5})
        updated = edit_nurbs_parameters(graph,sketch.id,0,2,[1,'w',1],[0,0,0,1,1,1])
        self.assertAlmostEqual(updated.results[solid.id].volume, math.pi/2, places=5)
        with self.assertRaises(RebuildError): edit_nurbs_parameters(updated,sketch.id,0,2,[1,-1,1],[0,0,0,1,1,1])
        self.assertEqual(updated.features[1].parameters['entities'][0]['weights'], [1,'w',1])
    def test_composite_slot_with_bezier_and_nurbs_path(self):
        from UI.model.composite_slot import composite_contact, composite_slot_edges
        import build123d as cad
        points = {'z':[-2,0], 'a':[0,0], 'b':[1,0], 'c':[2,1], 'd':[3,2]}
        for kind in ('spline', 'nurbs'):
            source = {'type':kind, 'points':['a','b','c','d'], 'construction':True}
            degree, knots, weights = 0, [], []
            if kind == 'nurbs':
                degree, knots, weights = 3, [0,0,0,0,1,1,1,1], [1,1,1,1]
                source.update(degree=degree, knots=knots, weights=weights)
            segments = [('line', [points['z'], points['a']], 0, [], []),
                        (kind, [points[n] for n in 'abcd'], degree, knots, weights)]
            edges = composite_slot_edges(segments, cad.Plane.XY, .2)
            contact, _, _ = composite_contact(segments, .2, 'segment:1:left', .4)
            vertex = cad.Vertex(float(contact[0]), float(contact[1]), 0)
            self.assertLess(min(vertex.distance_to(edge) for edge in edges), 1e-5)
            plane = Feature.create('Plane', 'datum_plane', {})
            sketch = Feature.create('Sketch', 'planar_sketch', {'points':points,
                'entities':[{'type':'line','points':['z','a'],'construction':True}, source,
                            {'type':'composite_path_slot','curves':[0,1],'radius':.2}]}, plane.id)
            solid = Feature.create('Extrude', 'extrude', {'distance':2}, sketch.id)
            self.assertGreater(FeatureGraph([plane, sketch, solid]).results[solid.id].volume, 0)

    def test_composite_curve_slot_rebuild_constraint_and_save(self):
        plane = Feature.create('Plane', 'datum_plane', {})
        points = {'a':[0,0], 'b':[2,0], 'c':[4,2], 'd':[4,4],
                  'm':[2+2**.5, 2-2**.5],
                  's0':[1,.25], 's1':[2,.25], 's2':[3,.25], 's3':[4,.25]}
        entities = [{'type':'line', 'points':['a','b'], 'construction':True},
                    {'type':'arc', 'points':['b','c','m'], 'construction':True},
                    {'type':'line', 'points':['c','d'], 'construction':True},
                    {'type':'composite_path_slot', 'curves':[0,1,2], 'radius':'r'},
                    {'type':'spline', 'points':['s0','s1','s2','s3'], 'construction':True}]
        sketch = Feature.create('Sketch', 'planar_sketch', {'points':points, 'entities':entities,
            'constraints':[{'kind':'curvature_spline_slot',
                'entities':[['s0','s1','s2','s3'],'s0',['@composite_path_slot',3],'segment:0:left'], 'value':.5}]}, plane.id)
        solid = Feature.create('Extrude', 'extrude', {'distance':2}, sketch.id)
        graph = FeatureGraph([plane, sketch, solid], {'r':.25})
        self.assertAlmostEqual(graph.results[solid.id].volume, 2*(.5*(4+math.pi) + math.pi*.25**2), places=5)
        self.assertLess(graph.results[sketch.id].constraint_status.residual, 1e-7)
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / 'composite.fcad'
            save_snapshot(filename, graph.results[solid.id], [], [], feature_graph=graph)
            _, _, _, restored = load_snapshot(filename, include_feature_graph=True)
            self.assertAlmostEqual(restored.results[solid.id].volume, graph.results[solid.id].volume, places=5)
        original = graph.results[solid.id].volume
        updated = move_sketch_point(graph, sketch.id, 'd', (5, 4))
        self.assertNotAlmostEqual(updated.results[solid.id].volume, original, places=5)
        self.assertAlmostEqual(graph.results[solid.id].volume, original, places=5)

    def test_composite_slot_curvature_joint_is_not_smooth(self):
        from UI.model.composite_slot import composite_contact
        segments = [('line', [[0,0],[2,0]], 0, [], []),
                    ('arc', [[2,0],[4,2],[2+2**.5,2-2**.5]], 0, [], [])]
        left = composite_contact(segments, .25, 'segment:0:left', 1)
        right = composite_contact(segments, .25, 'segment:1:left', 0)
        np.testing.assert_allclose(left[0], right[0], atol=1e-8)
        np.testing.assert_allclose(left[1], right[1], atol=1e-8)
        self.assertGreater(np.linalg.norm(left[2]-right[2]), .1)
        plane = Feature.create('Plane', 'datum_plane', {})
        sketch = Feature.create('Sketch', 'planar_sketch', {
            'points': {'a':[0,0], 'b':[2,0], 'c':[4,2], 'm':[2+2**.5,2-2**.5],
                       's0':[2,.25], 's1':[3,.25], 's2':[4,.25], 's3':[5,.25]},
            'entities': [{'type':'line','points':['a','b'],'construction':True},
                         {'type':'arc','points':['b','c','m'],'construction':True},
                         {'type':'composite_path_slot','curves':[0,1],'radius':.25},
                         {'type':'spline','points':['s0','s1','s2','s3'],'construction':True}]}, plane.id)
        graph = FeatureGraph([plane, sketch])
        with self.assertRaisesRegex(RebuildError, 'Curvature is discontinuous'):
            add_sketch_constraint(graph, sketch.id, {'kind':'curvature_spline_slot',
                'entities':[['s0','s1','s2','s3'],'s0',['@composite_path_slot',2],'segment:0:left'], 'value':1})

    def test_point_at_composite_slot_parameter(self):
        plane = Feature.create('Plane', 'datum_plane', {})
        points = {'a':[0,0], 'b':[2,0], 'c':[4,2], 'm':[2+2**.5,2-2**.5], 'p':[1,2]}
        sketch = Feature.create('Sketch', 'planar_sketch', {'points':points,
            'entities':[{'type':'line','points':['a','b'],'construction':True},
                        {'type':'arc','points':['b','c','m'],'construction':True},
                        {'type':'composite_path_slot','curves':[0,1],'radius':.25}],
            'constraints':[{'kind':'fixed','entities':[name],'value':points[name]} for name in ('a','b','c','m')] +
                          [{'kind':'point_on_slot','entities':['p',['@composite_path_slot',2],'segment:0:left'],'value':.5}]}, plane.id)
        graph = FeatureGraph([plane, sketch])
        np.testing.assert_allclose(graph.results[sketch.id].constraint_status.points['p'], [1,.25], atol=1e-6)
    def test_periodic_nurbs_repeated_knots(self):
        from UI.model.nurbs import periodic_nurbs_contact, periodic_nurbs_edge
        points = [[2,0], [1,2], [-1,2], [-2,0], [-1,-2], [1,-2]]
        spans = [1, 0, 1, 1, 1, 1]
        weights = [1] * len(points)
        left = periodic_nurbs_contact(points, 3, weights, 0, spans)
        right = periodic_nurbs_contact(points, 3, weights, 1, spans)
        for a, b in zip(left, right): np.testing.assert_allclose(a, b, atol=1e-8)
        self.assertTrue(periodic_nurbs_edge([(*p,0) for p in points], 3, weights, spans).is_closed)
        seam_spans = [1, 1, 1, 1, 1, 0]
        left = periodic_nurbs_contact(points, 3, weights, 0, seam_spans)
        right = periodic_nurbs_contact(points, 3, weights, 1, seam_spans)
        for a, b in zip(left, right): np.testing.assert_allclose(a, b, atol=1e-8)
        with self.assertRaises(ValueError):
            periodic_nurbs_edge([(*p,0) for p in points], 3, weights, [1,0,0,0,1,1])

    def test_point_at_spline_parameter_rebuilds_and_rejects_invalid_parameter(self):
        plane = Feature.create('Plane', 'datum_plane', {})
        points = {'a':[0,0], 'b':[2,0], 'c':[3,1], 'd':[4,2], 'p':[1,3]}
        sketch = Feature.create('Sketch', 'planar_sketch', {
            'points':points,
            'entities':[{'type':'fit_spline', 'points':['a','b','c','d'], 'construction':True}],
            'constraints':[{'kind':'fixed', 'entities':[name], 'value':points[name]} for name in 'abcd'] +
                          [{'kind':'point_on_spline', 'entities':['p',['@fit','a','b','c','d']], 'value':'t'}]}, plane.id)
        graph = FeatureGraph([plane, sketch], {'t':.5})
        from UI.model.spline_slot import spline_path_contact
        solved = graph.results[sketch.id].constraint_status.points
        target = spline_path_contact([solved[n] for n in 'abcd'], .5, 'fit_spline')[0]
        np.testing.assert_allclose(solved['p'], target, atol=1e-6)
        with self.assertRaises(RebuildError): graph.set_parameters({'t':1.2})

    def test_slot_for_fit_and_nurbs_curves(self):
        from UI.model.spline_slot import spline_slot_edges, spline_slot_contact
        import build123d as cad
        points = [[0,0],[2,0],[3,1],[4,2]]
        paths = [('fit_spline', None, [], []), ('nurbs', 3, [0,0,0,0,1,1,1,1], [1,1,1,1])]
        for kind, degree, knots, weights in paths:
            edges = spline_slot_edges([(*p,0) for p in points], .25, kind, degree, knots, weights)
            contact, _, _ = spline_slot_contact(points, .25, 'left', .4, kind, degree, knots, weights)
            vertex = cad.Vertex(float(contact[0]), float(contact[1]), 0)
            self.assertLess(min(vertex.distance_to(edge) for edge in edges), 1e-5, kind)
            plane = Feature.create('Plane', 'datum_plane', {})
            source = {'type': kind, 'points': list('abcd'), 'construction': True}
            if kind == 'nurbs': source.update(degree=degree, knots=knots, weights=weights)
            sketch = Feature.create('Sketch', 'planar_sketch', {'points': dict(zip('abcd', points)),
                'entities': [source, {'type':'spline_slot', 'curve':0, 'radius':.25}]}, plane.id)
            solid = Feature.create('Extrude', 'extrude', {'distance':2}, sketch.id)
            graph = FeatureGraph([plane, sketch, solid])
            self.assertGreater(graph.results[solid.id].volume, 0)
            restored = FeatureGraph.from_data(graph.to_data())
            self.assertAlmostEqual(restored.results[solid.id].volume, graph.results[solid.id].volume, places=5)
    def test_nonuniform_periodic_nurbs_rebuild_and_round_trip(self):
        from UI.model.nurbs import periodic_nurbs_contact, periodic_nurbs_edge
        control = [[1,0], [0,1], [-1,0], [0,-1]]
        weights, spans = [1,2,1,1], [1,2,3,4]
        a = periodic_nurbs_contact(control, 2, weights, 0, spans)
        b = periodic_nurbs_contact(control, 2, weights, 1, spans)
        np.testing.assert_allclose(a[0], b[0], atol=1e-10)
        np.testing.assert_allclose(a[1], b[1], atol=1e-10)
        self.assertTrue(periodic_nurbs_edge([(*p, 0) for p in control], 2, weights, spans).is_closed)
        with self.assertRaises(ValueError): periodic_nurbs_contact(control, 2, weights, .5, [1, 0, 0, 1])
        plane = Feature.create('Plane', 'datum_plane', {})
        sketch = Feature.create('Sketch', 'planar_sketch', {
            'points': dict(zip('abcd', control)),
            'entities': [{'type': 'periodic_nurbs', 'points': list('abcd'), 'degree': 2,
                          'weights': weights, 'spans': [1, 'span', 3, 4]}]}, plane.id)
        solid = Feature.create('Extrude', 'extrude', {'distance': 2}, sketch.id)
        graph = FeatureGraph([plane, sketch, solid], {'span': 2})
        old = graph.results[solid.id].volume
        graph.set_parameters({'span': 1})
        self.assertNotAlmostEqual(old, graph.results[solid.id].volume, places=5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'nonuniform.fcad'
            save_snapshot(path, graph.results[solid.id], [], [], feature_graph=graph)
            _, _, _, restored = load_snapshot(path, include_feature_graph=True)
            self.assertAlmostEqual(restored.results[solid.id].volume, graph.results[solid.id].volume, places=5)

    def test_spline_slot_boundary_constraint_and_rebuild(self):
        from UI.model.spline_slot import spline_slot_edges, spline_slot_contact
        import build123d as cad
        path = [[0,0], [2,0], [3,1], [4,2]]
        edges = spline_slot_edges([(*p, 0) for p in path], .5)
        contact, tangent, curvature = spline_slot_contact(path, .5, 'left', .4)
        vertex = cad.Vertex(float(contact[0]), float(contact[1]), 0)
        self.assertLess(min(vertex.distance_to(edge) for edge in edges), 1e-5)
        plane = Feature.create('Plane', 'datum_plane', {})
        sketch = Feature.create('Sketch', 'planar_sketch', {
            'points': dict(zip('abcd', path)),
            'entities': [{'type': 'spline', 'points': list('abcd'), 'construction': True},
                         {'type': 'spline_slot', 'curve': 0, 'radius': 'r'}]}, plane.id)
        solid = Feature.create('Extrude', 'extrude', {'distance': 2}, sketch.id)
        graph = FeatureGraph([plane, sketch, solid], {'r': .5})
        old = graph.results[solid.id].volume
        graph.set_parameters({'r': .4})
        self.assertLess(graph.results[solid.id].volume, old)
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / 'spline_slot.fcad'
            save_snapshot(filename, graph.results[solid.id], [], [], feature_graph=graph)
            _, _, _, restored = load_snapshot(filename, include_feature_graph=True)
            self.assertAlmostEqual(restored.results[solid.id].volume, graph.results[solid.id].volume, places=5)

    def test_spline_slot_curvature_constraint_on_cap(self):
        plane = Feature.create('Plane', 'datum_plane', {})
        sketch = Feature.create('Sketch', 'planar_sketch', {
            'points': {'a':[0,0], 'b':[2,0], 'c':[3,1], 'd':[4,2],
                       'x':[-.5,0], 'y':[0,.5], 'z':[-.5 * 2**-.5,.5 * 2**-.5]},
            'entities': [{'type':'spline', 'points':['a','b','c','d'], 'construction':True},
                         {'type':'spline_slot', 'curve':0, 'radius':.5},
                         {'type':'arc', 'points':['x','y','z'], 'construction':True}],
            'constraints': [{'kind':'curvature_arc_slot',
                             'entities':[['x','y','z'],'x',['@spline_slot',1],'start_cap'], 'value':.5}]}, plane.id)
        graph = FeatureGraph([plane, sketch])
        self.assertLess(graph.results[sketch.id].constraint_status.residual, 1e-7)

    def test_periodic_nurbs_closed_curvature_and_save(self):
        from UI.model.nurbs import periodic_nurbs_contact, periodic_nurbs_edge
        control = [[1, 0], [0, 1], [-1, 0], [0, -1]]
        start = periodic_nurbs_contact(control, 2, [1]*4, 0)
        end = periodic_nurbs_contact(control, 2, [1]*4, 1)
        for a, b in zip(start, end): np.testing.assert_allclose(a, b, atol=1e-10)
        self.assertTrue(periodic_nurbs_edge([(*p, 0) for p in control], 2, [1]*4).is_closed)
        plane = Feature.create('Plane', 'datum_plane', {})
        sketch = Feature.create('Sketch', 'planar_sketch', {
            'points': {'a': [1,0], 'b': [0,1], 'c': [-1,0], 'd': [0,-1],
                       's0': [.5,.5], 's1': [-.5,1.5], 's2': [-3,1], 's3': [-3,0]},
            'entities': [{'type': 'periodic_nurbs', 'points': ['a','b','c','d'], 'degree': 2, 'weights': [1,1,1,1]},
                         {'type': 'spline', 'points': ['s0','s1','s2','s3'], 'construction': True}],
            'constraints': [{'kind': 'curvature_spline_periodic_nurbs',
                             'entities': [['s0','s1','s2','s3'], 's0', ['@periodic_nurbs', 0]], 'value': 0}]}, plane.id)
        solid = Feature.create('Extrude', 'extrude', {'distance': 2}, sketch.id)
        graph = FeatureGraph([plane, sketch, solid])
        self.assertLess(graph.results[sketch.id].constraint_status.residual, 1e-7)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'periodic.fcad'
            save_snapshot(path, graph.results[solid.id], [], [], feature_graph=graph)
            _, _, _, restored = load_snapshot(path, include_feature_graph=True)
            self.assertAlmostEqual(restored.results[solid.id].volume, graph.results[solid.id].volume, places=5)

    def test_arc_slot_profile_constraint_and_round_trip(self):
        from UI.model.arc_slot import arc_slot_contact
        plane = Feature.create('Plane', 'datum_plane', {})
        points = {'a': [5, 0], 'b': [0, 5], 'm': [5 * 2**-.5, 5 * 2**-.5],
                  's0': [6, 0], 's1': [6, 1], 's2': [5.75, 2], 's3': [5, 3]}
        sketch = Feature.create('Sketch', 'planar_sketch', {
            'points': points,
            'entities': [{'type': 'arc_slot', 'points': ['a', 'b', 'm'], 'radius': 'width'},
                         {'type': 'spline', 'points': ['s0', 's1', 's2', 's3'], 'construction': True}],
            'constraints': [{'kind': 'curvature_spline_slot',
                'entities': [['s0', 's1', 's2', 's3'], 's0', ['@arc_slot', 0], 'outer'], 'value': 0}]}, plane.id)
        solid = Feature.create('Extrude', 'extrude', {'distance': 2}, sketch.id)
        graph = FeatureGraph([plane, sketch, solid], {'width': 1})
        self.assertAlmostEqual(graph.results[solid.id].volume, 12 * math.pi, places=5)
        contact, _, curvature = arc_slot_contact([points['a'], points['b'], points['m']], 1, 'outer', 0)
        np.testing.assert_allclose(contact, [6, 0], atol=1e-8)
        np.testing.assert_allclose(curvature, [-1/6, 0], atol=1e-8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'arc_slot.fcad'
            save_snapshot(path, graph.results[solid.id], [], [], feature_graph=graph)
            _, _, _, restored = load_snapshot(path, include_feature_graph=True)
            self.assertAlmostEqual(restored.results[solid.id].volume, 12 * math.pi, places=5)
        with self.assertRaises(RebuildError):
            graph.set_parameters({'width': 5})

    def test_arc_slot_clockwise_and_invalid_radius(self):
        from UI.model.arc_slot import arc_slot_edges, arc_slot_contact
        import build123d as cad
        points = [(5, 0, 0), (0, -5, 0), (5 * 2**-.5, -5 * 2**-.5, 0)]
        edges = arc_slot_edges(points, (0, 0, 1), 1)
        self.assertAlmostEqual(cad.Face(cad.Wire.combine(edges)[0]).area, 6 * math.pi, places=5)
        p, _, _ = arc_slot_contact([point[:2] for point in points], 1, 'outer', .5)
        np.testing.assert_allclose(p, [6 * 2**-.5, -6 * 2**-.5], atol=1e-7)
        with self.assertRaises(ValueError):
            arc_slot_edges(points, (0, 0, 1), 5)

    def test_polyline_slot_rebuild_and_round_trip(self):
        plane = Feature.create('Plane', 'datum_plane', {})
        sketch = Feature.create('Sketch', 'planar_sketch', {
            'points': {'a': [0, 0], 'b': [4, 0], 'c': [4, 3]},
            'entities': [{'type': 'polyline_slot', 'points': ['a', 'b', 'c'], 'radius': 'width / 2'}]}, plane.id)
        solid = Feature.create('Extrude', 'extrude', {'distance': 2}, sketch.id)
        graph = FeatureGraph([plane, sketch, solid], {'width': 2})
        self.assertAlmostEqual(graph.results[solid.id].volume, 2 * (13 + 5 * math.pi / 4), places=5)
        updated = move_sketch_point(graph, sketch.id, 'c', (4, 4))
        self.assertGreater(updated.results[solid.id].volume, graph.results[solid.id].volume)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'slot.fcad'
            save_snapshot(path, updated.results[solid.id], [], [], feature_graph=updated)
            _, _, _, restored = load_snapshot(path, include_feature_graph=True)
            self.assertAlmostEqual(restored.results[solid.id].volume, updated.results[solid.id].volume, places=5)
        with self.assertRaises(RebuildError):
            move_sketch_point(graph, sketch.id, 'c', (4, 0))

    def test_polyline_slot_contact_rejects_hidden_component(self):
        plane = Feature.create('Plane', 'datum_plane', {})
        points = {'a': [0, 0], 'b': [4, 0], 'c': [4, 3],
                  's0': [2, 1], 's1': [3, 1], 's2': [4, 1], 's3': [5, 1]}
        sketch = Feature.create('Sketch', 'planar_sketch', {
            'points': points,
            'entities': [{'type': 'polyline_slot', 'points': ['a', 'b', 'c'], 'radius': 1},
                         {'type': 'spline', 'points': ['s0', 's1', 's2', 's3'], 'construction': True}],
            'constraints': [{'kind': 'fixed', 'entities': [name], 'value': value} for name, value in points.items()]}, plane.id)
        graph = FeatureGraph([plane, sketch])
        relation = {'kind': 'curvature_spline_slot',
                    'entities': [['s0', 's1', 's2', 's3'], 's0', ['@polyline_slot', 0], 'segment:0:left'], 'value': .5}
        updated = add_sketch_constraint(graph, sketch.id, relation)
        self.assertLess(updated.results[sketch.id].constraint_status.residual, 1e-7)
        points2 = dict(points, s0=[4, 1], s1=[5, 1], s2=[6, 1], s3=[7, 1])
        parameters = dict(sketch.parameters, points=points2,
                          constraints=[{'kind': 'fixed', 'entities': [name], 'value': value} for name, value in points2.items()])
        fixed_graph = FeatureGraph([plane, Feature(sketch.id, sketch.name, sketch.kind, parameters, plane.id)])
        relation['value'] = 1
        with self.assertRaisesRegex(RebuildError, 'hidden'):
            add_sketch_constraint(fixed_graph, sketch.id, relation)
    def test_nurbs_quarter_circle_matches_kernel_and_rebuilds(self):
        from UI.model.nurbs import nurbs_contact, nurbs_edge
        points = [[1., 0., 0.], [1., 1., 0.], [0., 1., 0.]]
        weights, knots = [1., 2**-.5, 1.], [0., 0., 0., 1., 1., 1.]
        position, tangent, curvature = nurbs_contact(points, 2, knots, weights, .5)
        np.testing.assert_allclose(position, [2**-.5, 2**-.5, 0], atol=1e-10)
        np.testing.assert_allclose(curvature, -position, atol=1e-10)
        edge = nurbs_edge(points, 2, knots, weights)
        np.testing.assert_allclose(tuple(edge.position_at(.5)), position, atol=1e-6)
        plane = Feature.create('Plane', 'datum_plane', {})
        sketch = Feature.create('Sketch', 'planar_sketch', {
            'points': {'a': [1, 0], 'b': [1, 1], 'c': [0, 1], 'o': [0, 0]},
            'entities': [{'type': 'nurbs', 'points': ['a', 'b', 'c'], 'degree': 2,
                          'weights': [1, 'weight', 1], 'knots': knots},
                         {'type': 'line', 'points': ['c', 'o', 'a']}]}, plane.id)
        solid = Feature.create('Extrude', 'extrude', {'distance': 2}, sketch.id)
        graph = FeatureGraph([plane, sketch, solid], {'weight': 2**-.5})
        self.assertAlmostEqual(graph.results[solid.id].volume, math.pi / 2, places=5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'nurbs.fcad'
            save_snapshot(path, graph.results[solid.id], [], [], feature_graph=graph)
            _, _, _, restored = load_snapshot(path, include_feature_graph=True)
            self.assertAlmostEqual(restored.results[solid.id].volume, math.pi / 2, places=5)
            restored.set_parameters({'weight': 1})
            self.assertNotAlmostEqual(restored.results[solid.id].volume, math.pi / 2, places=4)

    def test_nurbs_curvature_constraint_with_circle(self):
        plane = Feature.create('Plane', 'datum_plane', {})
        sketch = Feature.create('Sketch', 'planar_sketch', {
            'points': {'a': [1, 0], 'b': [1, 1], 'c': [0, 1], 'o': [0, 0]},
            'entities': [{'type': 'nurbs', 'points': ['a', 'b', 'c'], 'degree': 2,
                          'weights': [1, 2**-.5, 1], 'knots': [0, 0, 0, 1, 1, 1], 'construction': True},
                         {'type': 'circle', 'center': 'o', 'radius': 1}],
            'constraints': [{'kind': 'curvature_spline_circle', 'entities': [['@nurbs', 0], 'a', ['o', '@circle:1']], 'value': 0}]}, plane.id)
        graph = FeatureGraph([plane, sketch])
        self.assertLess(graph.results[sketch.id].constraint_status.residual, 1e-7)

    def test_nurbs_rejects_invalid_knots_and_weights(self):
        from UI.model.nurbs import nurbs_data
        points = [[0, 0], [1, 1], [2, 0]]
        with self.assertRaises(ValueError):
            nurbs_data(points, 2, [0, 0, 0, 1, 1, 1], [1, 0, 1])
        with self.assertRaises(ValueError):
            nurbs_data(points, 2, [0, 0, .2, 1, 1, 1], [1, 1, 1])
    def test_fit_spline_profile_drag_and_round_trip(self):
        from UI.model.constraints import fit_spline
        plane = Feature.create('Plane', 'datum_plane', {})
        sketch = Feature.create('Sketch', 'planar_sketch', {
            'points': {'a': [-2, 0], 'b': [-1, 2], 'c': [1, 2], 'd': [2, 0]},
            'entities': [{'type': 'fit_spline', 'points': ['a', 'b', 'c', 'd']},
                         {'type': 'line', 'points': ['d', 'a']}]}, plane.id)
        solid = Feature.create('Extrude', 'extrude', {'distance': 2}, sketch.id)
        graph = FeatureGraph([plane, sketch, solid])
        curve = fit_spline([graph.results[sketch.id].constraint_status.points[n] for n in ('a', 'b', 'c', 'd')])
        np.testing.assert_allclose(curve(1), [-1, 2], atol=1e-10)
        updated = move_sketch_point(graph, sketch.id, 'b', (-1, 3))
        self.assertNotAlmostEqual(graph.results[solid.id].volume, updated.results[solid.id].volume)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fit.fcad'
            save_snapshot(path, updated.results[solid.id], [], [], feature_graph=updated)
            _, _, _, restored = load_snapshot(path, include_feature_graph=True)
            self.assertAlmostEqual(restored.results[solid.id].volume, updated.results[solid.id].volume, places=5)

    def test_fit_spline_curvature_relation_on_slot(self):
        points = {'a': [0., 0.], 'b': [4., 0.], 'w': [0., 1.],
                  's0': [2., 1.], 's1': [3., 1.], 's2': [4., 1.], 's3': [5., 1.]}
        fixed = [Constraint('fixed', (name,), value) for name, value in points.items()]
        relation = Constraint('curvature_spline_slot',
            (('@fit', 's0', 's1', 's2', 's3'), 's0', ('a', 'b', 'w'), 'left'), .5)
        result = solve_constraints(points, {}, fixed + [relation])
        self.assertNotEqual(result.status, 'conflict')
        self.assertLess(result.residual, 1e-7)

    def test_point_on_segment_and_circle(self):
        points = {'a': [0., 0.], 'b': [4., 0.], 'p': [2., .3], 'c': [2., 1.]}
        fixed = [Constraint('fixed', (name,), points[name]) for name in ('a', 'b', 'c')]
        constraints = fixed + [Constraint('radius', (('c', 'r'),), 1),
                               Constraint('point_on_line', ('p', ('a', 'b'))),
                               Constraint('point_on_circle', ('p', ('c', 'r')))]
        result = solve_constraints(points, {'r': 1}, constraints)
        self.assertNotEqual(result.status, 'conflict')
        np.testing.assert_allclose(result.points['p'], [2, 0], atol=1e-6)
        bad = fixed + [Constraint('fixed', ('p',), [6, 0]), Constraint('point_on_line', ('p', ('a', 'b')))]
        self.assertEqual(solve_constraints(points, {'r': 1}, bad).status, 'conflict')
    def test_cubic_spline_closed_profile_rebuild_and_save(self):
        plane = Feature.create('Plane', 'datum_plane', {})
        sketch = Feature.create('Sketch', 'planar_sketch', {
            'points': {'a': [-2, 0], 'b': [-2, 2], 'c': [2, 2], 'd': [2, 0]},
            'entities': [{'type': 'spline', 'points': ['a', 'b', 'c', 'd']},
                         {'type': 'line', 'points': ['d', 'a']}]}, plane.id)
        solid = Feature.create('Extrude', 'extrude', {'distance': 2}, sketch.id)
        graph = FeatureGraph([plane, sketch, solid])
        changed = move_sketch_point(graph, sketch.id, 'b', (-2, 3))
        self.assertNotAlmostEqual(graph.results[solid.id].volume, changed.results[solid.id].volume)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'spline.fcad'
            save_snapshot(path, changed.results[solid.id], [], [], feature_graph=changed)
            _, _, _, restored = load_snapshot(path, include_feature_graph=True)
            self.assertAlmostEqual(restored.results[solid.id].volume, changed.results[solid.id].volume, places=5)

    def test_spline_curvature_on_slot_side_and_cap(self):
        from UI.model.constraints import spline_endpoint, slot_contact
        slot = [[0, 0], [4, 0], [0, 1]]
        point, tangent, curvature = slot_contact(slot, 'end_cap', 0.5)
        np.testing.assert_allclose(point, [5, 0], atol=1e-8)
        np.testing.assert_allclose(curvature, [-1, 0], atol=1e-8)
        plane = Feature.create('Plane', 'datum_plane', {})
        sketch = Feature.create('Sketch', 'planar_sketch', {
            'points': {'a': [0, 0], 'b': [4, 0], 'w': [0, 1],
                       's0': [2, 1], 's1': [3, 1], 's2': [4, 1], 's3': [5, 2]},
            'entities': [{'type': 'slot', 'points': ['a', 'b', 'w']},
                         {'type': 'spline', 'points': ['s0', 's1', 's2', 's3'], 'construction': True}],
            'constraints': [{'kind': 'curvature_spline_slot',
                             'entities': [['s0', 's1', 's2', 's3'], 's0', ['a', 'b', 'w'], 'left'],
                             'value': 'contact'}]}, plane.id)
        solid = Feature.create('Extrude', 'extrude', {'distance': 2}, sketch.id)
        graph = FeatureGraph([plane, sketch, solid], {'contact': .5})
        state = graph.results[sketch.id].constraint_status
        p, t, k = spline_endpoint([state.points[n] for n in ('s0', 's1', 's2', 's3')], 0)
        sp, st, sk = slot_contact([state.points[n] for n in ('a', 'b', 'w')], 'left', .5)
        np.testing.assert_allclose(p, sp, atol=1e-6)
        np.testing.assert_allclose(k, sk, atol=1e-6)
        graph.set_parameters({'contact': .25})
        self.assertNotEqual(graph.results[sketch.id].constraint_status.status, 'conflict')

    def test_spline_curvature_matches_arc_circle_and_ellipse(self):
        points = {'s0': [1., 0.], 's1': [1., 1/3], 's2': [5/6, 2/3], 's3': [0., 1.],
                  'a': [0., 1.], 'b': [1., 0.], 'm': [2**-.5, 2**-.5], 'c': [0., 0.]}
        fixed = [Constraint('fixed', (name,), value) for name, value in points.items()]
        fixed.append(Constraint('radius', (('c', 'r'),), 1))
        for relation in [Constraint('curvature_spline_arc', (('s0', 's1', 's2', 's3'), 's0', ('a', 'b', 'm'), 'b')),
                         Constraint('curvature_spline_circle', (('s0', 's1', 's2', 's3'), 's0', ('c', 'r')), 0),
                         Constraint('curvature_spline_ellipse', (('s0', 's1', 's2', 's3'), 's0', 'c'), (1, 1, 0, 0))]:
            result = solve_constraints(points, {'r': 1}, fixed + [relation])
            self.assertNotEqual(result.status, 'conflict', relation.kind)
            self.assertLess(result.residual, 1e-7)

    def test_arc_slot_curvature_uses_actual_component(self):
        points = {'a': [4., 0.], 'b': [5., 0.], 'w': [4., 1.],
                  'p': [5., 1.], 'q': [6., 0.], 'm': [5. + 2**-.5, 2**-.5]}
        fixed = [Constraint('fixed', (name,), value) for name, value in points.items()]
        relation = Constraint('curvature_arc_slot', (('p', 'q', 'm'), 'q', ('a', 'b', 'w'), 'end_cap'), .5)
        result = solve_constraints(points, {}, fixed + [relation])
        self.assertNotEqual(result.status, 'conflict')
        self.assertLess(result.residual, 1e-7)
        bad = Constraint('curvature_arc_slot', (('p', 'q', 'm'), 'q', ('a', 'b', 'w'), 'start_cap'), .5)
        self.assertEqual(solve_constraints(points, {}, fixed + [bad]).status, 'conflict')
    def test_arc_ellipse_curvature_tracks_parameters_and_save(self):
        from UI.model.constraints import arc_center_radius, ellipse_contact
        plane = Feature.create('Plane', 'datum_plane', {})
        sketch = Feature.create('Sketch', 'planar_sketch', {
            'points': {'a': [2, 0], 'b': [4, 0], 'm': [3, 1], 'center': [0, 0]},
            'entities': [{'type': 'arc', 'points': ['a', 'b', 'm'], 'construction': True},
                         {'type': 'ellipse', 'center': 'center', 'radius_x': 'width', 'radius_y': 2}],
            'constraints': [{'kind': 'fixed', 'entities': ['center'], 'value': [0, 0]},
                            {'kind': 'curvature_arc_ellipse', 'entities': [['a', 'b', 'm'], 'b', 1], 'value': 0}]}, plane.id)
        solid = Feature.create('Extrude', 'extrude', {'distance': 2}, sketch.id)
        graph = FeatureGraph([plane, sketch, solid], {'width': 4})
        graph.set_parameters({'width': 5})
        state = graph.results[sketch.id].constraint_status
        center, radius = arc_center_radius(*(state.points[n] for n in ('a', 'b', 'm')))
        contact, tangent, curvature = ellipse_contact(state.points['center'], 5, 2, 0, 0)
        np.testing.assert_allclose(state.points['b'], contact, atol=1e-6)
        np.testing.assert_allclose((center - contact) / radius**2, curvature, atol=1e-6)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ellipse.fcad'
            save_snapshot(path, graph.results[solid.id], [], [], feature_graph=graph)
            _, _, _, restored = load_snapshot(path, include_feature_graph=True)
            self.assertAlmostEqual(restored.results[solid.id].volume, graph.results[solid.id].volume, places=5)

    def test_ellipse_contact_curvature_on_rotated_ellipse(self):
        from UI.model.constraints import ellipse_contact
        point, tangent, curvature = ellipse_contact([1, 2], 4, 2, 90, 0)
        np.testing.assert_allclose(point, [1, 6], atol=1e-10)
        np.testing.assert_allclose(tangent, [-1, 0], atol=1e-10)
        np.testing.assert_allclose(curvature, [0, -1], atol=1e-10)

    def test_concentric_arc_circle_and_equal_radius(self):
        points = {'a': [-2., 0.], 'b': [2., 0.], 'm': [0., 2.], 'c': [1., 1.]}
        constraints = [Constraint('fixed', (name,), points[name]) for name in ('a', 'b', 'm')]
        constraints += [Constraint('concentric_arc_circle', (('a', 'b', 'm'), ('c', 'r'))),
                        Constraint('equal_radius_arc_circle', (('a', 'b', 'm'), ('c', 'r')))]
        result = solve_constraints(points, {'r': 1.}, constraints)
        self.assertNotEqual(result.status, 'conflict')
        np.testing.assert_allclose(result.points['c'], [0, 0], atol=1e-6)
        self.assertAlmostEqual(result.radii['r'], 2, places=6)

    def test_arc_radius_drives_extrusion_and_round_trips(self):
        plane = Feature.create('Plane', 'datum_plane', {})
        sketch = Feature.create('Sketch', 'planar_sketch', {
            'points': {'a': [-2, 0], 'b': [2, 0], 'm': [0, 2]},
            'entities': [{'type': 'arc', 'points': ['a', 'b', 'm']},
                         {'type': 'line', 'points': ['b', 'a']}]}, plane.id)
        solid = Feature.create('Extrude', 'extrude', {'distance': 2}, sketch.id)
        graph = FeatureGraph([plane, sketch, solid], {'arc_r': 3})
        updated = add_sketch_constraint(graph, sketch.id, {
            'kind': 'arc_radius', 'entities': [['a', 'b', 'm']], 'value': 'arc_r'})
        from UI.model.constraints import arc_center_radius
        state = updated.results[sketch.id].constraint_status
        self.assertAlmostEqual(arc_center_radius(*(state.points[n] for n in ('a', 'b', 'm')))[1], 3, places=6)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'arc.fcad'
            save_snapshot(path, updated.results[solid.id], [], [], feature_graph=updated)
            _, _, _, restored = load_snapshot(path, include_feature_graph=True)
            self.assertAlmostEqual(restored.results[solid.id].volume, updated.results[solid.id].volume, places=5)

    def test_arc_tangent_and_curvature_geometry(self):
        points = {'a': [-1., 0.], 'b': [0., 1.], 'm': [-2**-.5, 2**-.5],
                  'c': [0., 1.], 'd': [1., 0.], 'n': [2**-.5, 2**-.5],
                  'l1': [-2., 1.], 'l2': [2., 1.]}
        constraints = [Constraint('fixed', (name,), value) for name, value in points.items()]
        constraints += [Constraint('tangent_line_arc', (('l1', 'l2'), ('a', 'b', 'm'), 'b')),
                        Constraint('tangent_arc_arc', (('a', 'b', 'm'), 'b', ('c', 'd', 'n'), 'c')),
                        Constraint('curvature_arc_arc', (('a', 'b', 'm'), 'b', ('c', 'd', 'n'), 'c'))]
        solved = solve_constraints(points, {}, constraints)
        self.assertNotEqual(solved.status, 'conflict')
        self.assertLess(solved.residual, 1e-7)
        shifted = dict(points, n=[0.8, 0.8])
        conflicting = [Constraint('fixed', (name,), value) for name, value in shifted.items()]
        conflicting.append(Constraint('curvature_arc_arc', (('a', 'b', 'm'), 'b', ('c', 'd', 'n'), 'c')))
        self.assertEqual(solve_constraints(shifted, {}, conflicting).status, 'conflict')
        outside = dict(points, l1=[2., 1.], l2=[3., 1.])
        fixed_outside = [Constraint('fixed', (name,), value) for name, value in outside.items()]
        fixed_outside.append(Constraint('tangent_line_arc', (('l1', 'l2'), ('a', 'b', 'm'), 'b')))
        self.assertEqual(solve_constraints(outside, {}, fixed_outside).status, 'conflict')

    def test_arc_circle_curvature_and_reject_wrong_contact(self):
        points = {'a': [-1., 0.], 'b': [0., 1.], 'm': [-2**-.5, 2**-.5], 'center': [0., 0.]}
        fixed = [Constraint('fixed', (name,), value) for name, value in points.items()]
        fixed.append(Constraint('radius', (('center', 'r'),), 1))
        for kind in ('tangent_arc_circle', 'curvature_arc_circle'):
            relation = Constraint(kind, (('a', 'b', 'm'), 'b', ('center', 'r')))
            result = solve_constraints(points, {'r': 1}, fixed + [relation])
            self.assertNotEqual(result.status, 'conflict')
            self.assertLess(result.residual, 1e-7)
        bad = dict(points, center=[0., -1.])
        fixed_bad = [Constraint('fixed', (name,), value) for name, value in bad.items()]
        fixed_bad.extend([Constraint('radius', (('center', 'r'),), 1),
                          Constraint('tangent_arc_circle', (('a', 'b', 'm'), 'b', ('center', 'r')))])
        self.assertEqual(solve_constraints(bad, {'r': 1}, fixed_bad).status, 'conflict')

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

    def test_circle_dimension_rebuild_save_and_conflicting_drag(self):
        plane = Feature.create('Plane', 'datum_plane', {})
        sketch = Feature.create('Sketch', 'planar_sketch', {'points': {'c': [0, 0]},
            'entities': [{'type': 'circle', 'center': 'c', 'radius': 2}]}, plane.id)
        solid = Feature.create('Extrude', 'extrude', {'distance': 3}, sketch.id)
        graph = FeatureGraph([plane, sketch, solid], {'size': 6})
        dimension = {'kind': 'diameter', 'entities': [['c', '@circle:0']], 'value': 'size'}
        updated = add_sketch_constraint(graph, sketch.id, dimension)
        self.assertAlmostEqual(updated.results[solid.id].volume, 27 * math.pi, places=5)
        with self.assertRaises(RebuildError):
            set_circle_radius(updated, sketch.id, 0, 4)
        self.assertAlmostEqual(updated.results[sketch.id].constraint_status.radii['@circle:0'], 3, places=5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'circle.fcad'
            save_snapshot(path, updated.results[solid.id], [], [], feature_graph=updated)
            _, _, _, restored = load_snapshot(path, include_feature_graph=True)
            restored.set_parameters({'size': 8})
            self.assertAlmostEqual(restored.results[solid.id].volume, 48 * math.pi, places=5)

    def test_circle_line_tangency_and_equal_radius(self):
        plane = Feature.create('Plane', 'datum_plane', {})
        sketch = Feature.create('Sketch', 'planar_sketch', {
            'points': {'a': [-5, 0], 'b': [5, 0], 'c': [0, 2], 'd': [8, 2]},
            'entities': [{'type': 'line', 'points': ['a', 'b'], 'construction': True},
                         {'type': 'circle', 'center': 'c', 'radius': 1},
                         {'type': 'circle', 'center': 'd', 'radius': 2}],
            'constraints': [{'kind': 'fixed', 'entities': ['a'], 'value': [-5, 0]},
                            {'kind': 'fixed', 'entities': ['b'], 'value': [5, 0]},
                            {'kind': 'fixed', 'entities': ['c'], 'value': [0, 2]},
                            {'kind': 'tangent_line_circle', 'entities': [['a', 'b'], ['c', '@circle:1']]},
                            {'kind': 'equal_radius', 'entities': [['c', '@circle:1'], ['d', '@circle:2']]}]}, plane.id)
        graph = FeatureGraph([plane, sketch])
        radii = graph.results[sketch.id].constraint_status.radii
        self.assertAlmostEqual(radii['@circle:1'], 2, places=6)
        self.assertAlmostEqual(radii['@circle:2'], 2, places=6)

    def test_external_circle_tangency_and_symmetry(self):
        plane = Feature.create('Plane', 'datum_plane', {})
        sketch = Feature.create('Sketch', 'planar_sketch', {
            'points': {'a': [0, -5], 'b': [0, 5], 'c': [-3, 0], 'd': [3, 0]},
            'entities': [{'type': 'line', 'points': ['a', 'b'], 'construction': True},
                         {'type': 'circle', 'center': 'c', 'radius': 2},
                         {'type': 'circle', 'center': 'd', 'radius': 2}],
            'constraints': [{'kind': 'fixed', 'entities': ['a'], 'value': [0, -5]},
                            {'kind': 'fixed', 'entities': ['b'], 'value': [0, 5]},
                            {'kind': 'symmetric', 'entities': ['c', 'd', ['a', 'b']]},
                            {'kind': 'tangent_circles_external', 'entities': [['c', '@circle:1'], ['d', '@circle:2']]}]}, plane.id)
        graph = FeatureGraph([plane, sketch])
        state = graph.results[sketch.id].constraint_status
        self.assertAlmostEqual(np.linalg.norm(state.points['c'] - state.points['d']),
                               state.radii['@circle:1'] + state.radii['@circle:2'], places=6)
        self.assertAlmostEqual(state.points['c'][0], -state.points['d'][0], places=6)

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

    def test_line_constraints_and_removal_rebuild_transactionally(self):
        graph, sketch, solid = self.graph()
        parallel = {'kind': 'parallel', 'entities': [['a', 'b'], ['d', 'c']]}
        constrained = add_sketch_constraint(graph, sketch.id, parallel)
        angle = {'kind': 'angle', 'entities': [['a', 'b'], ['b', 'c']], 'value': '90'}
        constrained = add_sketch_constraint(constrained, sketch.id, angle)
        self.assertAlmostEqual(constrained.results[solid.id].volume, 32, places=5)
        self.assertEqual(len(constrained.features[1].parameters['constraints']), 2)
        removed = remove_sketch_constraint(constrained, sketch.id, 0)
        self.assertEqual(removed.features[1].parameters['constraints'], [angle])
        self.assertEqual(len(constrained.features[1].parameters['constraints']), 2)
        with self.assertRaises(ValueError):
            remove_sketch_constraint(constrained, sketch.id, 4)

    def test_midpoint_constraint_moves_point_and_rebuilds(self):
        graph, sketch, solid = self.graph()
        constrained = add_sketch_constraint(graph, sketch.id, {
            'kind': 'midpoint', 'entities': ['b', ['a', 'c']]})
        points = constrained.results[sketch.id].constraint_status.points
        np.testing.assert_allclose(points['b'], (points['a'] + points['c']) / 2, atol=1e-6)

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
