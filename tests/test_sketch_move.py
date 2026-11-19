import sys
import unittest
from pathlib import Path

import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from UI.model.ActionManager import ActionManager
from UI.model.feature_graph import Feature, FeatureGraph
from UI.model.sketch_move import transform_sketch_selection


class SketchMoveTest(unittest.TestCase):
    def test_scene_move_and_copy_keep_constraints_and_rebuild_solids(self):
        plane = Feature.create('Plane', 'datum_plane', {})
        constraint = {'kind': 'distance', 'entities': ['a', 'b'], 'value': 4}
        sketch = Feature.create('Sketch', 'planar_sketch', {
            'points': {'a': [0, 0], 'b': [4, 0], 'c': [4, 3], 'd': [0, 3]},
            'entities': [{'type': 'line', 'points': list('abcda')}], 'constraints': [constraint],
        }, plane.id)
        key, = FeatureGraph([plane, sketch]).results[sketch.id].profile_keys
        solid = Feature.create('Extrude', 'extrude', {'distance': 2, 'profile_key': key}, sketch.id)
        graph = FeatureGraph([plane, sketch, solid])
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
        settings = [(6., 0.), 90., False]
        manager.start_sketch_move(lambda: tuple(settings))
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
        self.assertIsNotNone(session.candidate)
        self.assertTrue(session.preview)
        self.assertAlmostEqual(graph.results[solid.id].center().X, 2)
        self.assertTrue(session.confirm())
        moved = manager.history_edit_graph
        self.assertEqual(moved.results[sketch.id].profile_keys, (key,))
        self.assertAlmostEqual(moved.results[solid.id].center().X, 4.5)
        self.assertAlmostEqual(moved.results[solid.id].volume, 24)
        settings[:] = [(-8., 0.), 0., True]
        click(6, 2)
        self.assertTrue(session.confirm())
        manager.finish_history_sketch()
        saved = FeatureGraph.from_data(manager.parametric_graph.to_data())
        self.assertEqual(len(saved.results[sketch.id].profiles), 2)
        original, copied = saved.features[1].parameters['constraints']
        self.assertEqual(original, constraint)
        self.assertEqual(copied['value'], 4)
        self.assertNotEqual(copied['entities'], original['entities'])
        self.assertAlmostEqual(saved.results[solid.id].center().X, 4.5)
        copy_key = next(profile_key for profile_key in saved.results[sketch.id].profile_keys if profile_key != key)
        self.assertIsNotNone(copy_key)
        copy_solid = Feature.create('Copy Extrude', 'extrude', {'distance': 2, 'profile_key': copy_key}, sketch.id)
        both = FeatureGraph([*saved.features, copy_solid])
        self.assertAlmostEqual(both.results[copy_solid.id].center().X, -3.5)
        self.assertAlmostEqual(both.results[copy_solid.id].volume, 24)

    def test_rotation_conflicting_with_horizontal_constraint_is_transactional(self):
        plane = Feature.create('Plane', 'datum_plane', {})
        sketch = Feature.create('Sketch', 'planar_sketch', {
            'points': {'a': [0, 0], 'b': [4, 0]},
            'entities': [{'type': 'line', 'points': ['a', 'b']}],
            'constraints': [{'kind': 'horizontal', 'entities': [['a', 'b']]}],
        }, plane.id)
        graph = FeatureGraph([plane, sketch])
        before = graph.to_data()
        for copy in (False, True):
            with self.assertRaises(ValueError):
                transform_sketch_selection(graph, sketch.id, [('entity', 0)], (3, 5), 45, copy)
        self.assertEqual(graph.to_data(), before)
