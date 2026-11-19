import math
import sys
import unittest
from pathlib import Path

import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from UI.model.ActionManager import ActionManager
from UI.model.feature_graph import Feature, FeatureGraph


class SketchPatternTest(unittest.TestCase):
    def test_scene_patterns_rebuild_existing_instance_profiles(self):
        for kind in ('rectangular', 'circular'):
            with self.subTest(kind=kind):
                plane = Feature.create('Plane', 'datum_plane', {})
                sketch = Feature.create('Sketch', 'planar_sketch', {
                    'points': {'o': [5, 0], 'center': ['cx', 0], 'a': [-5, -4], 'b': ['bx', 'by']},
                    'entities': [{'type': 'circle', 'center': 'o', 'radius': 'radius'},
                                 {'type': 'line', 'points': ['a', 'b'], 'construction': True}],
                }, plane.id)
                graph = FeatureGraph([plane, sketch], {'radius': 1, 'cx': 0, 'bx': 0, 'by': -4})
                plotter = pv.Plotter(off_screen=True, window_size=(640, 480))
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
                plotter.camera.parallel_scale = 12
                plotter.show(auto_close=False)
                settings = [2, 2, 8., 7.] if kind == 'rectangular' else [4, 360.]
                manager.start_sketch_pattern(kind, lambda: tuple(settings))
                session = manager.sketch_modifier

                def click(x, y):
                    plotter.renderer.SetWorldPoint(x, y, 0., 1.)
                    plotter.renderer.WorldToDisplay()
                    px, py, _ = plotter.renderer.GetDisplayPoint()
                    plotter.iren.interactor.SetEventPosition(round(px), round(py))
                    manager._on_hover()
                    manager._on_click()

                click(5, 1)
                self.assertTrue(session.choose_reference())
                click(*((-2.5, -4) if kind == 'rectangular' else (0, 0)))
                self.assertGreaterEqual(len(session.preview), 5)
                old_preview = session.preview[-1]
                settings[2 if kind == 'rectangular' else 1] = 9. if kind == 'rectangular' else 180.
                session.refresh_preview()
                self.assertIsNot(session.preview[-1], old_preview)
                settings[2 if kind == 'rectangular' else 1] = 8. if kind == 'rectangular' else 360.
                self.assertFalse(manager.history_edit_graph.results[sketch.id].pattern_edges)
                self.assertTrue(session.confirm())
                manager.finish_history_sketch()
                committed = manager.parametric_graph
                before = committed.results[sketch.id]
                self.assertEqual(len(before.pattern_edges), 3)
                self.assertEqual(len(before.profiles), 4)
                initial_x, initial_y = (13, 7) if kind == 'rectangular' else (-5, 0)
                index = next(i for i, face in enumerate(before.profiles)
                             if abs(face.center().X - initial_x) < 1e-6 and abs(face.center().Y - initial_y) < 1e-6)
                key = before.profile_keys[index]
                self.assertIsNotNone(key)
                solid = Feature.create('Extrude', 'extrude', {'distance': 2, 'profile_key': key}, sketch.id)
                built = FeatureGraph([*committed.features, solid], committed.parameters)
                reopened = FeatureGraph.from_data(built.to_data())
                reopened.set_parameters({'radius': 1.5, 'cx': 1, 'bx': -5, 'by': 1})
                after = reopened.results[sketch.id]
                self.assertEqual([name for name, _ in before.pattern_edges], [name for name, _ in after.pattern_edges])
                self.assertIn(key, after.profile_keys)
                result = reopened.results[solid.id]
                self.assertAlmostEqual(result.volume, 4.5 * math.pi)
                self.assertAlmostEqual(result.center().X, -2 if kind == 'rectangular' else -3)
                self.assertAlmostEqual(result.center().Y, 8 if kind == 'rectangular' else 0)
                invalid = reopened.features[1].parameters
                invalid['patterns'][0]['spacing_x' if kind == 'rectangular' else 'angle'] = 0
                with self.assertRaises(ValueError):
                    reopened.edit(sketch.id, parameters=invalid)
                self.assertAlmostEqual(reopened.results[solid.id].volume, 4.5 * math.pi)
