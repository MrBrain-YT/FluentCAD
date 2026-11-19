import math
import sys
import unittest
from pathlib import Path

import build123d as cad
import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from UI.model.ActionManager import ActionManager
from UI.model.feature_graph import Feature, FeatureGraph
from UI.model.sketch_mirror import reflect_sketch
from UI.model.sketch_projection_curves import sketch_curve_name


class SketchMirrorTest(unittest.TestCase):
    def test_viewport_mirror_rebuilds_reflected_profile_and_solid(self):
        plane = Feature.create('Plane', 'datum_plane', {})
        sketch = Feature.create('Sketch', 'planar_sketch', {
            'points': {'o': [5, 0], 'a': ['axis', -8], 'b': ['axis', 8], 'p': [6, 4]},
            'entities': [{'type': 'circle', 'center': 'o', 'radius': 'radius'},
                         {'type': 'line', 'points': ['a', 'b'], 'construction': True}],
        }, plane.id)
        graph = FeatureGraph([plane, sketch], {'radius': 2, 'axis': 0})
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
        plotter.camera.focal_point = (0, 0, 0)
        camera = plotter.camera
        camera.position = (0, 0, camera.position[2])
        plotter.enable_parallel_projection()
        camera.parallel_scale = 10
        plotter.show(auto_close=False)

        def hover(x, y):
            plotter.renderer.SetWorldPoint(x, y, 0., 1.)
            plotter.renderer.WorldToDisplay()
            px, py, _ = plotter.renderer.GetDisplayPoint()
            plotter.iren.interactor.SetEventPosition(round(px), round(py))
            manager._on_hover()

        manager.start_sketch_mirror()
        session = manager.sketch_modifier
        unmodified = manager.history_edit_graph
        self.assertFalse(session.choose_axis())
        hover(5, 2)
        manager._on_click()
        hover(6, 4)
        manager._on_click()
        self.assertEqual([kind for kind, _ in session.selected], ['curve', 'point'])
        self.assertTrue(session.confirm())
        hover(5, 2)
        manager._on_click()
        self.assertIsNone(session.axis)
        hover(0, 0)
        self.assertEqual(len(session.preview), 5)
        manager._on_click()
        self.assertIs(manager.history_edit_graph, unmodified)
        self.assertIsNotNone(session.axis)
        self.assertTrue(session.confirm())
        manager.finish_history_sketch()
        committed = manager.parametric_graph
        mirrored = committed.results[sketch.id]
        self.assertEqual(len(mirrored.mirror_edges), 1)
        self.assertEqual(len(mirrored.mirror_points), 1)
        np.testing.assert_allclose(tuple(mirrored.mirror_points[0][1]), (-6, 4, 0))
        index = next(i for i, face in enumerate(mirrored.profiles) if face.center().X < 0)
        key = mirrored.profile_keys[index]
        self.assertIsNotNone(key)
        extrusion = Feature.create('Extrude', 'extrude', {'distance': 3, 'profile_key': key}, sketch.id)
        saved = FeatureGraph([*committed.features, extrusion], committed.parameters)
        reopened = FeatureGraph.from_data(saved.to_data())
        reopened.set_parameters({'radius': 3, 'axis': 1})
        after = reopened.results[sketch.id]
        self.assertEqual(after.mirror_edges[0][0], mirrored.mirror_edges[0][0])
        self.assertIn(key, after.profile_keys)
        np.testing.assert_allclose(tuple(after.mirror_points[0][1]), (-4, 4, 0))
        self.assertAlmostEqual(reopened.results[extrusion.id].volume, 27 * math.pi)
        self.assertAlmostEqual(reopened.results[extrusion.id].center().X, -3)
        self.assertAlmostEqual(saved.results[extrusion.id].volume, 12 * math.pi)

    def test_reflected_half_outline_closes_and_rebuilds_as_one_profile(self):
        plane = Feature.create('Plane', 'datum_plane', {})
        outline = {'type': 'line', 'points': ['a', 'b', 'c', 'd']}
        axis = {'type': 'line', 'points': ['a', 'd'], 'construction': True}
        guide = {'type': 'line', 'points': ['u', 'v'], 'construction': True}
        sketch = Feature.create('Sketch', 'planar_sketch', {
            'points': {'a': [0, -2], 'b': ['width', -2], 'c': ['width', 2], 'd': [0, 2],
                       'u': [1, -1], 'v': ['width + 1', 1]},
            'entities': [outline, axis, guide],
            'mirrors': [{'id': 'half', 'axis': sketch_curve_name(axis, 0),
                         'curves': [*(sketch_curve_name(outline, i) for i in range(3)), sketch_curve_name(guide, 0)]}],
        }, plane.id)
        graph = FeatureGraph([plane, sketch], {'width': 4})
        original = graph.results[sketch.id]
        key, = original.profile_keys
        self.assertIsNotNone(key)
        self.assertAlmostEqual(original.profiles[0].area, 32)
        self.assertEqual(original.mirror_construction_edges, {'mirror:half:edge:3'})
        extrusion = Feature.create('Extrude', 'extrude', {'distance': 2, 'profile_key': key}, sketch.id)
        graph = FeatureGraph([plane, sketch, extrusion], graph.parameters)
        graph.set_parameters({'width': 6})
        self.assertEqual(graph.results[sketch.id].profile_keys, (key,))
        self.assertAlmostEqual(graph.results[extrusion.id].volume, 96)
        broken = graph.features[1].parameters
        broken['mirrors'][0]['axis'] = 'missing'
        with self.assertRaisesRegex(ValueError, 'mirror axis is missing'):
            graph.edit(sketch.id, parameters=broken)
        self.assertAlmostEqual(graph.results[extrusion.id].volume, 96)

    def test_tilted_plane_reflects_curve_exactly_and_skips_invariant_sources(self):
        plane = cad.Plane((3, 4, 5), x_dir=(1, 0, 0), z_dir=(0, 1, 1))
        axis = plane.from_local_coords(cad.Edge.make_line((1, -6), (1, 6)))
        spline = plane.from_local_coords(cad.Edge.make_bezier((3, 0), (4, 1), (5, 2), (6, 0)))
        invariant = plane.from_local_coords(cad.Edge.make_ellipse(4, 3, cad.Plane((1, 0, 0))))
        point = plane.from_local_coords((1, 2, 0))
        result = reflect_sketch(plane, axis, [(spline, False), (invariant, True)], [point], 'test')
        self.assertEqual(len(result.edges), 1)
        self.assertFalse(result.points)
        mirrored = result.edges[0][1]
        self.assertEqual(mirrored.geom_type, cad.GeomType.BEZIER)
        for t in (0., .23, .72, 1.):
            u, v, _ = tuple(plane.to_local_coords(spline.position_at(t)))
            expected = plane.from_local_coords((2 - u, v, 0))
            self.assertLess(cad.Vertex(*tuple(expected)).distance_to(mirrored), 1e-6)
