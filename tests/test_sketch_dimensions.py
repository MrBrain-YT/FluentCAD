import math
import os
import sys
import unittest
from pathlib import Path

import pyvista as pv

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from PySide6 import QtCore, QtWidgets
from UI.main.sketch_dimension_dialog import edit_sketch_dimension
from UI.model.ActionManager import ActionManager
from UI.model.feature_graph import Feature, FeatureGraph
from UI.model.sketch_dimensions import dimension_annotation, dimension_from_selection, update_sketch_dimension


class SketchDimensionsTest(unittest.TestCase):
    def test_viewport_creates_edits_and_cancels_dimension_on_extruded_profile(self):
        plane = Feature.create('Plane', 'datum_plane', {})
        sketch = Feature.create('Sketch', 'planar_sketch', {
            'points': {'c': [0, 0]}, 'entities': [{'type': 'circle', 'center': 'c', 'radius': 2}],
        }, plane.id)
        initial = FeatureGraph([plane, sketch], {'width': 3})
        key, = initial.results[sketch.id].profile_keys
        extrude = Feature.create('Extrude', 'extrude', {'distance': 2, 'profile_key': key}, sketch.id)
        scale = Feature.create('Scale', 'scale', {'factor': 2}, extrude.id)
        initial = FeatureGraph([plane, sketch, extrude, scale], initial.parameters)
        plotter = pv.Plotter(off_screen=True, window_size=(500, 400))
        self.addCleanup(plotter.close)
        manager = ActionManager(plotter)
        manager.history_edit_id, manager.history_edit_graph = sketch.id, initial
        manager.sketch_editing = True
        manager.activated_sketch_id = 1
        manager.sketchs_data[1] = {'sketch_func': []}
        manager.parametric_graph = initial
        manager.render_history_sketch()
        plotter.view_xy()
        plotter.enable_parallel_projection()
        plotter.camera.parallel_scale = 8
        plotter.show(auto_close=False)

        def point(world):
            renderer = plotter.renderer
            renderer.SetWorldPoint(*world, 1.)
            renderer.WorldToDisplay()
            x, y, _ = renderer.GetDisplayPoint()
            plotter.iren.interactor.SetEventPosition(round(x), round(y))
            manager._on_hover()

        calls = []

        def edit(dimension, preview):
            calls.append(dimension['value'])
            text, radius = [('2 * width', 3), ('8', 4), ('10', 5)][len(calls) - 1]
            candidate = preview(text)
            self.assertAlmostEqual(candidate.results[sketch.id].constraint_status.radii['@circle:0'], radius)
            self.assertAlmostEqual(candidate.results[extrude.id].volume, 2 * math.pi * radius**2)
            self.assertAlmostEqual(candidate.results[scale.id].volume, 16 * math.pi * radius**2)
            self.assertEqual(candidate.results[sketch.id].profile_keys, (key,))
            self.assertAlmostEqual(initial.results[extrude.id].volume, 8 * math.pi)
            return candidate if len(calls) < 3 else None

        manager.start_sketch_dimension('auto', edit)
        point((0, 2, 0))
        self.assertTrue(manager.sketch_modifier.preview)
        manager._on_click()
        self.assertEqual(manager.sketch_modifier.pending['kind'], 'diameter')
        point((4, 3, 0))
        self.assertEqual(len(manager.sketch_modifier.preview), 2)
        self.assertEqual(calls, [])
        manager._on_click()
        self.assertAlmostEqual(manager.history_edit_graph.results[extrude.id].volume, 18 * math.pi)
        for _ in range(2):
            point(manager.history_dimension_labels[0][0])
            manager._on_click()
        self.assertEqual(calls, [4, '2 * width', 8])
        self.assertAlmostEqual(manager.history_edit_graph.results[extrude.id].volume, 32 * math.pi)
        self.assertAlmostEqual(manager.parametric_graph.results[extrude.id].volume, 8 * math.pi)
        manager.finish_history_sketch()
        self.assertIsNone(manager.sketch_modifier)
        self.assertAlmostEqual(manager.parametric_graph.results[extrude.id].volume, 32 * math.pi)
        self.assertAlmostEqual(manager.parametric_graph.results[scale.id].volume, 256 * math.pi)
        restored = FeatureGraph.from_data(manager.parametric_graph.to_data())
        self.assertAlmostEqual(restored.results[extrude.id].volume, 32 * math.pi)
        self.assertIn('placement', restored.features[1].parameters['constraints'][0])

    def test_line_point_and_arc_dimension_types(self):
        plane = Feature.create('Plane', 'datum_plane', {})
        sketch = Feature.create('Sketch', 'planar_sketch', {
            'points': {'a': [0, 0], 'b': [4, 0], 'c': [4, 3], 'm': [6, 1.5]},
            'entities': [{'type': 'line', 'points': ['a', 'b', 'c']},
                         {'type': 'arc', 'points': ['b', 'c', 'm']}],
        }, plane.id)
        graph = FeatureGraph([plane, sketch])
        state = graph.results[sketch.id].constraint_status
        lines = [('line', ('a', 'b')), ('line', ('b', 'c'))]
        points = [('point', ('a',)), ('point', ('c',))]
        cases = [('auto', lines[:1], 'distance', 4, '6'),
                 ('auto', points, 'distance', 5, '7'),
                 ('horizontal_distance', points, 'horizontal_distance', 4, '6'),
                 ('vertical_distance', points, 'vertical_distance', 3, '4'),
                 ('angle', lines, 'angle', 90, '65'),
                 ('diameter', [('arc', ('b', 'c', 'm'))], 'arc_diameter', 3.125, '4')]
        for mode, selected, kind, measured, value in cases:
            with self.subTest(kind=kind, selected=selected):
                dimension = dimension_from_selection(mode, selected, state)
                self.assertEqual(dimension['kind'], kind)
                self.assertAlmostEqual(dimension['value'], measured)
                rebuilt = update_sketch_dimension(graph, sketch.id, dimension, value)
                new_state = rebuilt.results[sketch.id].constraint_status
                self.assertAlmostEqual(dimension_from_selection(mode, selected, new_state)['value'], float(value), places=5)
                annotation = dimension_annotation(rebuilt.features[1].parameters['constraints'][0], new_state, {})
                self.assertIn(value, annotation[2])
        with self.assertRaisesRegex(ValueError, 'positive'):
            update_sketch_dimension(graph, sketch.id, dimension_from_selection('auto', lines[:1], state), '0')
        self.assertNotIn('constraints', graph.features[1].parameters)

    def test_dialog_shows_validation_and_commits_latest_preview(self):
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        values = []
        state = []

        def preview(text):
            values.append(text)
            if text == 'bad':
                raise ValueError('Invalid dimension.')
            return text

        def change():
            dialog = app.activeModalWidget()
            field = dialog.findChild(QtWidgets.QLineEdit)
            buttons = dialog.findChild(QtWidgets.QDialogButtonBox)
            ok = buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
            timer = dialog.findChild(QtCore.QTimer)
            field.setText('bad')
            timer.timeout.emit()
            state.append(ok.isEnabled())
            field.setText('9')
            timer.timeout.emit()
            state.append(ok.isEnabled())
            field.setText('10')
            ok.click()

        QtCore.QTimer.singleShot(0, change)
        result = edit_sketch_dimension(None, {'kind': 'radius', 'value': 2.0}, preview)
        self.assertEqual(state, [False, True])
        self.assertEqual(values, ['2.0', 'bad', '9', '10'])
        self.assertEqual(result, '10')
