import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import build123d as cad

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))

from UI.model.ActionManager import ActionManager
from UI.model.feature_graph import Feature, FeatureGraph, RebuildError
from UI.model.profiles import build_profiles
from UI.model.sketch_links import make_sketch_link
from UI.model.sketch_projection_curves import projectable_sketch_curves
from UI.model.sketch_silhouette import project_body_silhouette
from UI.model.snapshot import load_snapshot, save_snapshot


class SketchLinkTest(unittest.TestCase):
    def test_include_solid_edge_tracks_height_and_world_translation_after_reopening(self):
        solid = Feature.create('Box', 'primitive', {
            'type': 'box', 'dimensions': [4, 6, 'height'], 'position': [0, 0, 'lift'],
        })
        plane = Feature.create('Sketch plane', 'datum_plane', {})
        graph = FeatureGraph([solid, plane], {'height': 10, 'lift': 4})
        body = graph.results[solid.id]
        index = next(i for i, edge in enumerate(body.edges()) if abs(edge.tangent_at(0).Z) > .99)
        link, preview = self.selection(graph, solid, plane, 'include', 'edge', index)
        self.assertEqual(tuple(preview.edges[0][1].position_at(.5)), tuple(body.edges()[index].position_at(.5)))
        sketch = Feature.create('Spatial reference', 'planar_sketch', {'entities': [], 'links': [link]}, plane.id)
        graph = FeatureGraph([solid, plane, sketch], graph.parameters)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'included-edge.fcad'
            save_snapshot(path, graph.latest_solid, [], [], feature_graph=graph)
            _, _, _, graph = load_snapshot(path, include_feature_graph=True)
        graph.set_parameters({'height': 6, 'lift': 10})
        included = graph.results[sketch.id]
        name, edge = included.spatial_edges[0]
        self.assertEqual(sorted(edge.position_at(t).Z for t in (0., 1.)), [7, 13])
        self.assertIn(name, included.off_plane_edge_names)
        self.assertFalse(included.profiles)
        self.assertFalse(graph.link_issues)

    def test_include_tilted_curve_uses_source_plane_and_survives_break_link(self):
        origin = Feature.create('Upper', 'datum_plane', {'origin': [2, 0, 'level'], 'normal': [0, 1, 1]})
        source = Feature.create('Source circle', 'planar_sketch', {
            'points': {'C': [0, 0]},
            'entities': [{'type': 'circle', 'center': 'C', 'radius': 'radius'}],
        }, origin.id)
        plane = Feature.create('Destination plane', 'datum_plane', {'origin': [0, 0, 'destination']})
        parameters = {'level': 8, 'radius': 2, 'destination': 0}
        graph = FeatureGraph([origin, source, plane], parameters)
        link, _ = self.selection(graph, source, plane, 'include', 'sketch_curve', 0)
        sketch = Feature.create('Included circle', 'planar_sketch', {'entities': [], 'links': [link]}, plane.id)
        graph = FeatureGraph([*graph.features, sketch], parameters)
        graph.set_parameters({'level': 12, 'radius': 3, 'destination': 20})
        included = graph.results[sketch.id]
        edge = included.spatial_edges[0][1]
        self.assertAlmostEqual(edge.arc_center.Z, 12)
        self.assertAlmostEqual(edge.radius, 3)
        self.assertAlmostEqual(edge.bounding_box().max.Z - edge.bounding_box().min.Z, 3 * math.sqrt(2))
        self.assertFalse(included.profiles)

        graph.edit(source.id, suppressed=True)
        graph.set_parameters({'level': 17, 'radius': 4, 'destination': -2})
        self.assertIn(link['id'], graph.link_issues[sketch.id])
        self.assertAlmostEqual(graph.results[sketch.id].spatial_edges[0][1].arc_center.Z, 12)
        manager = ActionManager(None)
        manager.sketch_editing = True
        manager.history_edit_id = sketch.id
        manager.history_edit_graph = graph
        manager.render_history_sketch = Mock()
        manager.break_sketch_link(link['id'])
        graph = manager.history_edit_graph
        graph.edit(source.id, suppressed=False)
        detached = graph.results[sketch.id]
        self.assertFalse(detached.linked_edges)
        self.assertAlmostEqual(detached.spatial_edges[0][1].arc_center.Z, 12)
        self.assertAlmostEqual(detached.spatial_edges[0][1].radius, 3)
        self.assertFalse(graph.link_issues)

    def test_coplanar_include_curve_drives_extrusion_without_using_off_plane_edges(self):
        plane = Feature.create('Sketch plane', 'datum_plane', {})
        upper = Feature.create('Upper plane', 'datum_plane', {'origin': [0, 0, 4]})
        entities = [{'type': 'circle', 'center': 'C', 'radius': 'radius'}]
        source = Feature.create('Source circle', 'planar_sketch', {'points': {'C': [0, 0]}, 'entities': entities}, plane.id)
        other = Feature.create('Raised circle', 'planar_sketch', {'points': {'C': [0, 0]}, 'entities': entities}, upper.id)
        graph = FeatureGraph([plane, upper, source, other], {'radius': 2})
        links = [self.selection(graph, feature, plane, 'include', 'sketch_curve', 0)[0] for feature in (source, other)]
        sketch = Feature.create('Mixed sketch', 'planar_sketch', {'entities': [], 'links': links}, plane.id)
        graph = FeatureGraph([*graph.features, sketch], graph.parameters)
        key, = graph.results[sketch.id].profile_keys
        self.assertIsNotNone(key)
        feature = Feature.create('Extrude', 'extrude', {'distance': 2, 'profile_key': key}, sketch.id)
        graph = FeatureGraph([*graph.features, feature], graph.parameters)
        graph.set_parameters({'radius': 3})
        self.assertEqual(len(graph.results[sketch.id].spatial_edges), 2)
        self.assertEqual(len(graph.results[sketch.id].off_plane_edge_names), 1)
        self.assertAlmostEqual(graph.results[feature.id].volume, 18 * math.pi)

    def test_project_silhouette_removes_inner_edges_and_preserves_through_hole(self):
        stepped = cad.Box(8, 6, 2).fuse(cad.Box(2, 2, 4)).cut(cad.Cylinder(1, 10))
        plane = cad.Plane(origin=(2, 4, 5))
        outline = project_body_silhouette(stepped, plane)
        profiles = build_profiles([], (0, 0, 1), linked_edges=[edge for _, edge in outline])
        self.assertEqual(len(profiles), 2)
        self.assertEqual({name for name, _ in outline}, {'exterior-1', 'hole-1'})
        ring, hole = sorted(profiles, key=lambda profile: profile.area, reverse=True)
        self.assertAlmostEqual(ring.area, 48 - math.pi, places=5)
        self.assertAlmostEqual(hole.area, math.pi, places=5)
        self.assertTrue(all(abs(edge.position_at(.4).Z - 5) < 1e-7 for _, edge in outline))

    def test_sphere_silhouette_rebuilds_dependent_profile_on_inclined_plane(self):
        source = Feature.create('Sphere', 'primitive', {'type': 'sphere', 'dimensions': ['radius']})
        plane = Feature.create('Inclined', 'datum_plane', {'origin': [0, 0, 5], 'normal': [0, 1, 1]})
        graph = FeatureGraph([source, plane], {'radius': 2})
        link, preview = self.selection(graph, source, plane, 'project', 'body', 0)
        self.assertEqual(len(preview.edges), 1)
        sketch, extrusion, graph = self.sketch_and_extrude(source, plane, graph, link)
        key = graph.results[sketch.id].profile_keys[0]
        self.assertIsNotNone(key)
        graph.set_parameters({'radius': 3})
        self.assertEqual(graph.results[sketch.id].profile_keys, (key,))
        self.assertAlmostEqual(graph.results[sketch.id].profiles[0].area, 9 * math.pi)
        self.assertAlmostEqual(graph.results[extrusion.id].volume, 27 * math.pi)
        self.assertFalse(graph.link_issues)

    def test_project_earlier_sketch_curves_rebuilds_extrusion_after_roundtrip(self):
        source_plane = Feature.create('Upper', 'datum_plane', {'origin': [0, 0, 8]})
        source = Feature.create('Source sketch', 'planar_sketch', {
            'points': {'A': [0, 0], 'B': ['width', 0], 'C': ['width', 3], 'D': [0, 3]},
            'entities': [{'type': 'line', 'points': ['A', 'B', 'C', 'D', 'A']}],
        }, source_plane.id)
        plane = Feature.create('Lower', 'datum_plane', {})
        graph = FeatureGraph([source_plane, source, plane], {'width': 4})
        links = [self.selection(graph, source, plane, 'project', 'sketch_curve', index)[0] for index in range(4)]
        sketch = Feature.create('Projected sketch', 'planar_sketch', {'entities': [], 'links': links}, plane.id)
        graph = FeatureGraph([*graph.features, sketch], graph.parameters)
        key = graph.results[sketch.id].profile_keys[0]
        self.assertIsNotNone(key)
        extrusion = Feature.create('Extrude', 'extrude', {'distance': 2, 'profile_key': key}, sketch.id)
        graph = FeatureGraph([*graph.features, extrusion], graph.parameters)
        self.assertAlmostEqual(graph.results[extrusion.id].volume, 24)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sketch-curves.fcad'
            save_snapshot(path, graph.latest_solid, [], [], feature_graph=graph)
            _, _, _, graph = load_snapshot(path, include_feature_graph=True)
        graph.set_parameters({'width': 7})
        self.assertEqual(graph.results[sketch.id].profile_keys, (key,))
        self.assertAlmostEqual(graph.results[sketch.id].profiles[0].area, 21)
        self.assertAlmostEqual(graph.results[extrusion.id].volume, 42)
        self.assertFalse(graph.link_issues)

    def test_sketch_curve_scene_picking_without_solid_and_link_reuse(self):
        plane = Feature.create('Plane', 'datum_plane', {})
        source = Feature.create('Source rectangle', 'rectangle_sketch', {'width': 5, 'height': 3}, plane.id)
        sketch = Feature.create('Destination', 'planar_sketch', {'entities': []}, plane.id)
        later = Feature.create('Later rectangle', 'rectangle_sketch', {'width': 2, 'height': 2}, plane.id)
        graph = FeatureGraph([plane, source, sketch, later])
        manager = ActionManager(SimpleNamespace(add_mesh=Mock(side_effect=lambda *args, **kwargs: object()),
                                                remove_actor=Mock(), render=Mock()))
        manager.sketch_editing = True
        manager.history_edit_id = sketch.id
        manager.history_edit_graph = graph
        manager.flush_history_tools = Mock()
        manager.render_history_sketch = Mock()
        manager.start_sketch_link('project', 'sketch_curve')
        session = manager.sketch_modifier
        self.assertEqual(session.source_id, source.id)
        self.assertEqual(len(projectable_sketch_curves(session.shape)), 4)
        for actor in session.source_curves:
            manager._pick_actor = Mock(return_value=SimpleNamespace(GetActor=lambda: actor))
            manager._on_hover()
            self.assertEqual(len(session.preview_actors), 1)
            manager._on_click()
        self.assertAlmostEqual(manager.history_edit_graph.results[sketch.id].profiles[0].area, 15)
        self.assertFalse(session.apply())
        self.assertTrue(manager.stop_sketch_modifier())

    def test_project_face_rebuilds_profile_and_extrusion_after_save_and_open(self):
        source, plane, graph = self.model()
        shape = graph.results[source.id]
        face_index = max(enumerate(shape.faces()), key=lambda pair: pair[1].center().Z)[0]
        link, preview = self.selection(graph, source, plane, 'project', 'face', face_index)
        self.assertEqual(len(preview.edges), 4)
        sketch, extrusion, graph = self.sketch_and_extrude(source, plane, graph, link)
        key = graph.results[sketch.id].profile_keys[0]
        self.assertIsNotNone(key)
        self.assertAlmostEqual(graph.results[sketch.id].profiles[0].area, 24)
        self.assertAlmostEqual(graph.results[extrusion.id].volume, 72)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'projection.fcad'
            save_snapshot(path, graph.latest_solid, [], [], feature_graph=graph)
            _, _, _, restored = load_snapshot(path, include_feature_graph=True)
            restored.set_parameters({'width': 6, 'level': 0})
        self.assertEqual(restored.results[sketch.id].profile_keys, (key,))
        self.assertAlmostEqual(restored.results[sketch.id].profiles[0].area, 36)
        self.assertAlmostEqual(restored.results[extrusion.id].volume, 108)

    def test_body_section_follows_model_and_plane_and_rebuilds_extrusion(self):
        source, plane, graph = self.model()
        link, preview = self.selection(graph, source, plane, 'intersect', 'body', 0)
        self.assertEqual(len(preview.edges), 4)
        sketch, extrusion, graph = self.sketch_and_extrude(source, plane, graph, link)
        key = graph.results[sketch.id].profile_keys[0]
        graph.set_parameters({'width': 7, 'level': 1})
        updated = graph.results[sketch.id]
        self.assertEqual(updated.profile_keys, (key,))
        self.assertTrue(all(abs(edge.position_at(.5).Z - 1) < 1e-7 for _, edge in updated.linked_edges))
        self.assertAlmostEqual(updated.profiles[0].area, 42)
        self.assertAlmostEqual(graph.results[extrusion.id].volume, 126)
        graph.set_parameters({'width': 7, 'level': 9})
        self.assertIn('does not intersect', graph.link_issues[sketch.id][link['id']])
        self.assertAlmostEqual(graph.results[extrusion.id].volume, 126)
        graph.set_parameters({'width': 7, 'level': 1})
        self.assertNotIn(sketch.id, graph.link_issues)
        graph.edit(source.id, suppressed=True)
        self.assertIn(link['id'], graph.link_issues[sketch.id])
        self.assertAlmostEqual(graph.results[extrusion.id].volume, 126)
        graph.edit(source.id, suppressed=False)
        self.assertNotIn(sketch.id, graph.link_issues)
        self.assertAlmostEqual(graph.results[extrusion.id].volume, 126)

    def test_intersect_cylinder_rebuilds_round_profile_when_radius_changes(self):
        source = Feature.create('Cylinder', 'primitive', {'type': 'cylinder', 'dimensions': ['width', 8]})
        plane = Feature.create('Plane', 'datum_plane', {'origin': [0, 0, 'level']})
        graph = FeatureGraph([source, plane], {'width': 2, 'level': 0})
        link, preview = self.selection(graph, source, plane, 'intersect', 'body', 0)
        self.assertEqual(len(preview.edges), 1)
        sketch, extrusion, graph = self.sketch_and_extrude(source, plane, graph, link)
        graph.set_parameters({'width': 3, 'level': 1})
        self.assertAlmostEqual(graph.results[sketch.id].profiles[0].area, 9 * math.pi)
        self.assertAlmostEqual(graph.results[extrusion.id].volume, 27 * math.pi)

    def test_project_and_intersect_perpendicular_edge_yield_linked_point(self):
        source, plane, graph = self.model()
        edges = graph.results[source.id].edges()
        index = next(i for i, edge in enumerate(edges) if abs(edge.tangent_at(0).Z) > .99)
        for operation in ('project', 'intersect'):
            with self.subTest(operation=operation):
                link, geometry = self.selection(graph, source, plane, operation, 'edge', index)
                self.assertEqual(len(geometry.points), 1)
                self.assertFalse(geometry.edges)
                self.assertAlmostEqual(geometry.points[0].Z, 0)
                sketch = Feature.create('Points', 'planar_sketch', {'entities': [], 'links': [link]}, plane.id)
                updated = FeatureGraph([source, plane, sketch], graph.parameters)
                updated.set_parameters({'width': 6, 'level': 1})
                point, = updated.results[sketch.id].linked_points
                self.assertAlmostEqual(abs(point.X), 3)
                self.assertAlmostEqual(point.Z, 1)

    def test_sketch_cannot_link_forward_in_timeline(self):
        source, plane, graph = self.model()
        link, _ = self.selection(graph, source, plane, 'intersect', 'body', 0)
        sketch = Feature.create('Sketch', 'planar_sketch', {'entities': [], 'links': [link]}, plane.id)
        with self.assertRaisesRegex(RebuildError, 'later feature'):
            FeatureGraph([plane, sketch, source], graph.parameters)

    def test_lost_projection_can_be_relinked_in_scene_after_reopening(self):
        source, plane, graph = self.model()
        face_index = max(enumerate(graph.results[source.id].faces()), key=lambda pair: pair[1].center().Z)[0]
        link, _ = self.selection(graph, source, plane, 'project', 'face', face_index)
        sketch, extrusion, graph = self.sketch_and_extrude(source, plane, graph, link)
        replacement = Feature.create('Replacement', 'primitive', {
            'type': 'box', 'dimensions': ['width', 6, 8], 'position': [12, 0, 0],
        })
        graph = FeatureGraph([source, replacement, *graph.features[1:]], graph.parameters)
        graph.edit(source.id, parameters={'type': 'cylinder', 'dimensions': [3, 8]})
        self.assertIn(link['id'], graph.link_issues[sketch.id])
        self.assertAlmostEqual(graph.results[extrusion.id].volume, 72)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'lost-project.fcad'
            save_snapshot(path, graph.latest_solid, [], [], feature_graph=graph)
            _, _, _, graph = load_snapshot(path, include_feature_graph=True)
        self.assertIn(link['id'], graph.link_issues[sketch.id])
        plotter = SimpleNamespace(add_mesh=Mock(side_effect=lambda *args, **kwargs: object()),
                                  remove_actor=Mock(), render=Mock())
        manager = ActionManager(plotter)
        manager.sketch_editing = True
        manager.history_edit_id = sketch.id
        manager.history_edit_graph = graph
        manager.flush_history_tools = Mock()
        manager.render_history_sketch = Mock()
        manager.start_sketch_link('project', 'face', replacement.id, link['id'])
        session = manager.sketch_modifier
        top = max(enumerate(session.shape.faces()), key=lambda pair: pair[1].center().Z)[0]
        cell = next(i for i, face in enumerate(session.source_mesh.cell_data['face_id']) if face == top)
        manager._pick_actor = Mock(return_value=SimpleNamespace(GetCellId=lambda: cell))
        manager._on_hover()
        self.assertTrue(session.preview_actors)
        self.assertIn(link['id'], manager.history_edit_graph.link_issues[sketch.id])
        manager._on_click()
        self.assertIsNone(manager.sketch_modifier)
        repaired = manager.history_edit_graph
        self.assertNotIn(sketch.id, repaired.link_issues)
        self.assertAlmostEqual(repaired.results[extrusion.id].center().X, 12)
        repaired.set_parameters({'width': 6, 'level': 0})
        self.assertAlmostEqual(repaired.results[extrusion.id].volume, 108)

    def test_break_link_preserves_circle_profile_and_stops_following_source(self):
        source = Feature.create('Cylinder', 'primitive', {'type': 'cylinder', 'dimensions': ['width', 8]})
        plane = Feature.create('Plane', 'datum_plane', {'origin': [0, 0, 'level']})
        graph = FeatureGraph([source, plane], {'width': 2, 'level': 0})
        link, _ = self.selection(graph, source, plane, 'intersect', 'body', 0)
        sketch, extrusion, graph = self.sketch_and_extrude(source, plane, graph, link)
        graph.set_parameters({'width': 3, 'level': 0})
        key = graph.results[sketch.id].profile_keys[0]
        manager = ActionManager(None)
        manager.sketch_editing = True
        manager.history_edit_id = sketch.id
        manager.history_edit_graph = graph
        manager.render_history_sketch = Mock()
        manager.break_sketch_link(link['id'])
        result = manager.history_edit_graph
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'detached.fcad'
            save_snapshot(path, result.latest_solid, [], [], feature_graph=result)
            _, _, _, result = load_snapshot(path, include_feature_graph=True)
        result.set_parameters({'width': 5, 'level': 9})
        self.assertNotIn(sketch.id, result.link_issues)
        rebuilt = result.results[sketch.id]
        self.assertFalse(rebuilt.linked_edges)
        self.assertEqual(rebuilt.profile_keys, (key,))
        self.assertAlmostEqual(rebuilt.detached_edges[0][1].radius, 3)
        self.assertAlmostEqual(rebuilt.detached_edges[0][1].center().Z, 9)
        self.assertAlmostEqual(result.results[extrusion.id].volume, 27 * math.pi)

    def test_scene_previews_geometry_before_click_and_keeps_source_earlier(self):
        source, plane, graph = self.model()
        sketch = Feature.create('Sketch', 'planar_sketch', {'entities': []}, plane.id)
        later = Feature.create('Later solid', 'primitive', {'type': 'sphere', 'dimensions': [1]})
        graph = FeatureGraph([source, plane, sketch, later], graph.parameters)
        plotter = SimpleNamespace(add_mesh=Mock(side_effect=lambda *args, **kwargs: object()),
                                  remove_actor=Mock(), render=Mock())
        manager = ActionManager(plotter)
        manager.sketch_editing = True
        manager.history_edit_id = sketch.id
        manager.history_edit_graph = manager.parametric_graph = graph
        manager.flush_history_tools = Mock()
        manager.render_history_sketch = Mock()
        manager._pick_actor = Mock(return_value=SimpleNamespace(GetCellId=lambda: 0))
        manager.start_sketch_link('intersect', 'body')
        session = manager.sketch_modifier
        self.assertEqual(session.source_id, source.id)
        manager._on_hover()
        self.assertEqual(len(session.preview_actors), 4)
        self.assertIs(manager.history_edit_graph, graph)
        manager._on_click()
        self.assertEqual(len(manager.history_edit_graph.results[sketch.id].profiles), 1)
        self.assertIs(manager.parametric_graph, graph)
        self.assertFalse(session.apply())
        self.assertTrue(manager.stop_sketch_modifier())
        plotter.remove_actor.assert_any_call(session.source_actor)

    @staticmethod
    def model():
        source = Feature.create('Box', 'primitive', {'type': 'box', 'dimensions': ['width', 6, 8]})
        plane = Feature.create('Plane', 'datum_plane', {'origin': [0, 0, 'level']})
        return source, plane, FeatureGraph([source, plane], {'width': 4, 'level': 0})

    @staticmethod
    def selection(graph, source, plane, operation, kind, index):
        return make_sketch_link(operation, kind, source.id, graph.results[source.id], index,
                                graph.body_names[source.id], graph.results[plane.id])

    @staticmethod
    def sketch_and_extrude(source, plane, graph, link):
        sketch = Feature.create('Linked sketch', 'planar_sketch', {'entities': [], 'links': [link]}, plane.id)
        with_sketch = FeatureGraph([source, plane, sketch], graph.parameters)
        profile_key = with_sketch.results[sketch.id].profile_keys[0]
        extrusion = Feature.create('Extrude', 'extrude', {'distance': 3, 'profile_key': profile_key}, sketch.id)
        return sketch, extrusion, FeatureGraph([source, plane, sketch, extrusion], graph.parameters)
