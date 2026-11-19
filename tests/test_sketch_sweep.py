import math
import os
import sys
import unittest
from pathlib import Path

import pyvista as pv
from PySide6 import QtWidgets

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from UI.main.sweep_dialog import SweepDialog
from UI.model.ActionManager import ActionManager
from UI.model.feature_graph import Feature, FeatureGraph


class SketchSweepTest(unittest.TestCase):
    def test_scene_profile_and_reversed_path_selection_rebuilds_solid_after_edit(self):
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        plane = Feature.create('Profile plane', 'datum_plane', {})
        profile = Feature.create('Section', 'planar_sketch', {
            'points': {'o': [0, 0]}, 'entities': [{'type': 'circle', 'center': 'o', 'radius': 'radius'}],
        }, plane.id)
        path_plane = Feature.create('Path plane', 'datum_plane', {'normal': [0, -1, 0], 'x_dir': [1, 0, 0]})
        path = Feature.create('Rail', 'planar_sketch', {
            'points': {'a': [0, 0], 'b': [0, 'height'], 'c': ['width', 'height']},
            'entities': [{'type': 'line', 'points': ['a', 'b', 'c']}],
        }, path_plane.id)
        graph = FeatureGraph([plane, profile, path_plane, path], {'radius': 1, 'height': 5, 'width': 7})
        plotter = pv.Plotter(off_screen=True, window_size=(720, 560))
        self.addCleanup(plotter.close)
        manager = ActionManager(plotter)
        manager.parametric_graph = graph
        dialog = SweepDialog(manager)
        self.addCleanup(dialog.deleteLater)
        plotter.view_isometric()
        plotter.camera.position = (20, 20, 20)
        plotter.camera.focal_point = (0, 0, 0)
        plotter.enable_parallel_projection()
        plotter.camera.parallel_scale = 8
        plotter.show(auto_close=False)

        def click(world):
            renderer = plotter.renderer
            renderer.SetWorldPoint(*world, 1.)
            renderer.WorldToDisplay()
            x, y, _ = renderer.GetDisplayPoint()
            plotter.iren.interactor.SetEventPosition(round(x), round(y))
            dialog.scene_click()

        click((0, 0, 0))
        self.assertEqual(dialog.profile, (profile.id, 0))
        click((3, 0, 5))
        self.assertFalse(dialog.ok.isEnabled())
        self.assertEqual(len(dialog.path), 1)
        click((0, 0, 2.5))
        self.assertTrue(dialog.ok.isEnabled(), dialog.message.text())
        self.assertIsNotNone(dialog.preview)
        self.assertIs(manager.parametric_graph, graph)
        self.assertAlmostEqual(dialog.result_shape.volume, 12 * math.pi)
        click((3, 0, 5))
        self.assertAlmostEqual(dialog.result_shape.volume, 5 * math.pi)
        click((3, 0, 5))
        dialog.ok.click()
        self.assertFalse(dialog.actors)
        self.assertIsNone(dialog.preview)
        committed = dialog.candidate
        sweep = committed.features[-1]
        scale = Feature.create('Dependent scale', 'scale', {'factor': 2}, sweep.id)
        saved = FeatureGraph([*committed.features, scale], graph.parameters)
        reopened = FeatureGraph.from_data(saved.to_data())
        reopened.set_parameters({'radius': 1.5, 'height': 5, 'width': 7})
        self.assertAlmostEqual(reopened.results[sweep.id].volume, 27 * math.pi)
        reopened.set_parameters({'radius': 1.5, 'height': 8, 'width': 10})
        self.assertAlmostEqual(reopened.results[sweep.id].volume, 40.5 * math.pi)
        self.assertAlmostEqual(reopened.results[scale.id].volume, 324 * math.pi, places=6)
        reopened.edit(path.id, suppressed=True)
        self.assertNotIn(sweep.id, reopened.results)
        self.assertIn(scale.id, reopened.blocked)
        reopened.edit(path.id, suppressed=False)
        self.assertAlmostEqual(reopened.results[scale.id].volume, 324 * math.pi, places=6)
        changed_path = next(item.parameters for item in reopened.features if item.id == path.id)
        changed_path['entities'][0]['points'] = ['a', 'c']
        with self.assertRaisesRegex(ValueError, 'path segment is missing'):
            reopened.edit(path.id, parameters=changed_path)
        self.assertAlmostEqual(reopened.results[sweep.id].volume, 40.5 * math.pi)
        app.processEvents()
