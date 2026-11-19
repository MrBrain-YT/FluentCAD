import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from UI.model.ActionManager import ActionManager
from UI.model.feature_graph import Feature, FeatureGraph
from UI.model.sketch_scale import scale_sketch_selection


class SketchScaleTest(unittest.TestCase):
    def test_scene_center_factor_preview_and_existing_extrude(self):
        plane = Feature.create('Plane', 'datum_plane', {})
        sketch = Feature.create('Sketch', 'planar_sketch', {
            'points': {'a': [0, 0], 'b': [4, 0], 'c': [4, 3], 'd': [0, 3], 'center': [1, 1]},
            'entities': [{'type': 'line', 'points': list('abcda')}],
            'constraints': [{'kind': 'horizontal', 'entities': [['a', 'b']]},
                            {'kind': 'distance', 'entities': ['a', 'b'], 'value': 'width'}],
        }, plane.id)
        key, = FeatureGraph([plane, sketch], {'width': 4}).results[sketch.id].profile_keys
        solid = Feature.create('Extrude', 'extrude', {'distance': 2, 'profile_key': key}, sketch.id)
        graph = FeatureGraph([plane, sketch, solid], {'width': 4})
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
        plotter.camera.parallel_scale = 10
        plotter.show(auto_close=False)
        factor = [1.5]
        manager.start_sketch_scale(lambda: factor[0])
        session = manager.sketch_modifier

        def click(x, y):
            renderer = plotter.renderer
            renderer.SetWorldPoint(x, y, 0., 1.)
            renderer.WorldToDisplay()
            px, py, _ = renderer.GetDisplayPoint()
            plotter.iren.interactor.SetEventPosition(round(px), round(py))
            manager._on_hover()
            manager._on_click()

        click(2, 0)
        self.assertTrue(session.choose_center())
        manager.snap_to_grid = manager.snap_to_points = False
        click(-2, -2)
        np.testing.assert_allclose(session.center, (-2., -2.), atol=.05)
        self.assertIsNotNone(session.candidate)
        self.assertTrue(session.choose_center())
        click(1, 1)
        self.assertEqual(session.center, (1., 1.))
        self.assertAlmostEqual(session.candidate.results[solid.id].volume, 54)
        factor[0] = 2
        session.refresh_preview()
        self.assertAlmostEqual(session.candidate.results[solid.id].volume, 96)
        self.assertAlmostEqual(manager.history_edit_graph.results[solid.id].volume, 24)
        self.assertTrue(session.confirm())
        manager.finish_history_sketch()
        rebuilt = FeatureGraph.from_data(manager.parametric_graph.to_data())
        result = rebuilt.results[sketch.id]
        self.assertEqual(result.profile_keys, (key,))
        np.testing.assert_allclose(result.constraint_status.points['a'], (-1, -1), atol=1e-6)
        np.testing.assert_allclose(result.constraint_status.points['b'], (7, -1), atol=1e-6)
        self.assertEqual(rebuilt.features[1].parameters['constraints'][0]['kind'], 'horizontal')
        rebuilt.set_parameters({'width': 5})
        points = rebuilt.results[sketch.id].constraint_status.points
        self.assertAlmostEqual(np.linalg.norm(points['b'] - points['a']), 10)

    def test_circle_scale_preserves_expression_and_conflict_rejects_move(self):
        plane = Feature.create('Plane', 'datum_plane', {})
        sketch = Feature.create('Sketch', 'planar_sketch', {
            'points': {'o': [2, 0]},
            'entities': [{'type': 'circle', 'center': 'o', 'radius': 'radius'}],
            'constraints': [{'kind': 'fixed', 'entities': ['o'], 'value': [2, 0]}],
        }, plane.id)
        graph = FeatureGraph([plane, sketch], {'radius': 2})
        scaled = scale_sketch_selection(graph, sketch.id, [('entity', 0)], (2, 0), .5)
        self.assertAlmostEqual(scaled.results[sketch.id].profiles[0].area, math.pi)
        scaled.set_parameters({'radius': 3})
        self.assertAlmostEqual(scaled.results[sketch.id].profiles[0].area, 2.25 * math.pi)
        with self.assertRaisesRegex(ValueError, 'conflict'):
            scale_sketch_selection(graph, sketch.id, [('entity', 0)], (0, 0), .5)
        self.assertAlmostEqual(graph.results[sketch.id].profiles[0].area, 4 * math.pi)
