import math
import sys
import unittest
from pathlib import Path

import build123d as cad
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from UI.model.ActionManager import ActionManager
from UI.model.feature_graph import Feature, FeatureGraph
from UI.model.sketch_corners import build_corner, common_corner
from UI.model.sketch_offset import build_offset_chain, order_offset_chain
from UI.model.sketch_projection_curves import sketch_curve_name


class SketchCornersTest(unittest.TestCase):
    def viewport(self, graph, sketch):
        plotter = pv.Plotter(off_screen=True, window_size=(620, 480))
        self.addCleanup(plotter.close)
        manager = ActionManager(plotter)
        manager.history_edit_graph = manager.parametric_graph = graph
        manager.history_edit_id = sketch.id
        manager.activated_sketch_id = 1
        manager.sketchs_data[1] = {'sketch_func': []}
        manager.sketch_editing = True
        manager.render_history_sketch()
        plotter.view_xy()
        plotter.enable_parallel_projection()
        plotter.camera.parallel_scale = 9
        plotter.show(auto_close=False)

        def hover(x, y):
            plotter.renderer.SetWorldPoint(x, y, 0., 1.)
            plotter.renderer.WorldToDisplay()
            px, py, _ = plotter.renderer.GetDisplayPoint()
            plotter.iren.interactor.SetEventPosition(round(px), round(py))
            manager._on_hover()

        return manager, hover

    def rectangle(self):
        plane = Feature.create('Plane', 'datum_plane', {})
        outline = {'type': 'line', 'points': ['a', 'b', 'c', 'd', 'a']}
        sketch = Feature.create('Sketch', 'planar_sketch', {
            'points': {'a': [0, 0], 'b': ['width', 0], 'c': ['width', 6], 'd': [0, 6]},
            'entities': [outline],
        }, plane.id)
        return plane, sketch, outline

    def test_scene_fillet_and_chamfer_rebuild_preexisting_extrude(self):
        plane, sketch, _ = self.rectangle()
        parameters = {'width': 8}
        key, = FeatureGraph([plane, sketch], parameters).results[sketch.id].profile_keys
        solid = Feature.create('Extrude', 'extrude', {'distance': 3, 'profile_key': key}, sketch.id)
        for command, removed in [('fillet', 4 - math.pi), ('chamfer', 2)]:
            with self.subTest(command=command):
                graph = FeatureGraph([plane, sketch, solid], parameters)
                manager, hover = self.viewport(graph, sketch)
                size = [1.]
                manager.start_sketch_corner(command, lambda: size[0])
                session = manager.sketch_modifier
                hover(4, 0)
                manager._on_click()
                hover(4, 6)
                manager._on_click()
                self.assertEqual(len(session.selected), 1)
                hover(8, 3)
                self.assertEqual(len(session.preview), 5)
                manager._on_click()
                old = session.preview[-1]
                size[0] = 2.
                session.refresh_preview()
                self.assertIsNot(session.preview[-1], old)
                size[0] = 20.
                session.refresh_preview()
                self.assertFalse(session.confirm())
                self.assertFalse(manager.history_edit_graph.results[sketch.id].corner_edges)
                self.assertIs(manager.parametric_graph, graph)
                size[0] = 2.
                session.refresh_preview()
                self.assertTrue(session.confirm())
                self.assertAlmostEqual(manager.history_edit_graph.results[solid.id].volume, 3 * (48 - removed))
                manager.finish_history_sketch()
                reopened = FeatureGraph.from_data(manager.parametric_graph.to_data())
                self.assertEqual(reopened.features[1].parameters['corners'][0]['size'], 2.)
                reopened.set_parameters({'width': 10})
                rebuilt = reopened.results[sketch.id]
                self.assertEqual(rebuilt.profile_keys, (key,))
                self.assertAlmostEqual(reopened.results[solid.id].volume, 3 * (60 - removed))
                name, connector = rebuilt.corner_edges[0]
                self.assertTrue(name.startswith('corner:'))
                if command == 'fillet':
                    self.assertAlmostEqual(connector.radius, 2.)
                else:
                    self.assertAlmostEqual(connector.length, 2. * math.sqrt(2))

    def test_neighbouring_corners_that_consume_an_edge_are_rejected(self):
        plane, sketch, outline = self.rectangle()
        graph = FeatureGraph([plane, sketch], {'width': 8})
        parameters = graph.features[1].parameters
        refs = [sketch_curve_name(outline, i) for i in range(3)]
        parameters['corners'] = [
            {'id': 'one', 'kind': 'fillet', 'size': 4, 'sources': refs[:2], 'ends': [1, 0]},
            {'id': 'two', 'kind': 'chamfer', 'size': 4, 'sources': refs[1:], 'ends': [1, 0]},
        ]
        with self.assertRaisesRegex(ValueError, 'consume the same segment'):
            graph.edit(sketch.id, parameters=parameters)
        self.assertAlmostEqual(graph.results[sketch.id].profiles[0].area, 48)

    def test_curve_corner_is_tangent_and_uses_the_sketch_plane(self):
        plane = cad.Plane((3, 2, -4), x_dir=(1, 0, 0), z_dir=(0, 1, 1))
        a = cad.Edge.make_line((0, 0), (8, 0))
        b = cad.Edge.make_bezier((8, 0), (8, 2), (7, 4), (6, 6))
        world_edges = [plane.from_local_coords(edge) for edge in (a, b)]
        fillet, _ = build_corner(plane, world_edges, common_corner(*world_edges), 'fillet', 1)
        self.assertAlmostEqual(fillet.radius, 1)
        for t, source in enumerate(world_edges):
            self.assertLess(source.distance_to(fillet.position_at(t)), 1e-6)
            self.assertAlmostEqual(plane.to_local_coords(fillet.position_at(t)).Z, 0)

    def test_offset_uses_the_trimmed_segments_and_corner_connector(self):
        plane, sketch, outline = self.rectangle()
        refs = [sketch_curve_name(outline, i) for i in range(4)]
        parameters = sketch.parameters
        parameters['corners'] = [
            {'id': 'first', 'kind': 'fillet', 'size': 2, 'sources': refs[:2], 'ends': [1, 0]},
        ]
        graph = FeatureGraph([plane, sketch], {'width': 8})
        result = graph.results[sketch.id]
        edges = dict((*result.projectable_curves, *result.corner_edges))
        chain = order_offset_chain(edges, refs[0], automatic=True)
        self.assertEqual(len(chain), 5)
        groups = build_offset_chain(result.plane, edges, chain, 1, expected_closed=True)
        expected = cad.Face(cad.Wire([edge for group in groups for edge in group])).area
        updated = graph.features[1].parameters
        updated['offsets'] = [{'id': 'outline', 'sources': chain, 'distance': 1, 'closed': True}]
        graph.edit(sketch.id, parameters=updated)
        self.assertAlmostEqual(sum(profile.area for profile in graph.results[sketch.id].profiles), expected)
