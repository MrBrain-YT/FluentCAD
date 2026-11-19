import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from UI.model.ActionManager import ActionManager
from UI.model.constraints import Constraint, solve_constraints
from UI.model.feature_graph import Feature, FeatureGraph
from UI.model.scene_constraint_rules import ConstraintTarget, constraint_from_targets


class SketchSceneConstraintsTest(unittest.TestCase):
    def viewport(self, sketch, graph):
        plotter = pv.Plotter(off_screen=True, window_size=(560, 440))
        self.addCleanup(plotter.close)
        manager = ActionManager(plotter)
        manager.parametric_graph = manager.history_edit_graph = graph
        manager.history_edit_id = sketch.id
        manager.sketch_editing = True
        manager.activated_sketch_id = 1
        manager.sketchs_data[1] = {'sketch_func': []}
        manager.render_history_sketch()
        plotter.view_xy()
        plotter.enable_parallel_projection()
        plotter.camera.parallel_scale = 8
        plotter.show(auto_close=False)

        def hover(x, y):
            plotter.renderer.SetWorldPoint(x, y, 0., 1.)
            plotter.renderer.WorldToDisplay()
            px, py, _ = plotter.renderer.GetDisplayPoint()
            plotter.iren.interactor.SetEventPosition(round(px), round(py))
            manager._on_hover()

        return manager, hover

    def test_tangency_preview_and_dimension_edit_rebuild_dependent_operations(self):
        plane = Feature.create('Plane', 'datum_plane', {})
        sketch = Feature.create('Sketch', 'planar_sketch', {
            'points': {'o': [0, 0], 'a': [-6, 3], 'b': [6, 3]},
            'entities': [{'type': 'circle', 'center': 'o', 'radius': 2},
                         {'type': 'line', 'points': ['a', 'b'], 'construction': True}],
            'constraints': [{'kind': 'fixed', 'entities': ['o'], 'value': [0, 0]},
                            {'kind': 'horizontal', 'entities': [['a', 'b']]},
                            {'kind': 'diameter', 'entities': [['o', '@circle:0']],
                             'value': 4, 'placement': [5, -4]}],
        }, plane.id)
        key, = FeatureGraph([plane, sketch]).results[sketch.id].profile_keys
        extrude = Feature.create('Extrude', 'extrude', {'distance': 3, 'profile_key': key}, sketch.id)
        scale = Feature.create('Scale', 'scale', {'factor': 2}, extrude.id)
        initial = FeatureGraph([plane, sketch, extrude, scale])
        manager, hover = self.viewport(sketch, initial)

        def select_line_and_hover_circle():
            manager.start_sketch_constraint('tangent')
            hover(-4, 3)
            manager._on_click()
            hover(0, -2)
            session = manager.sketch_modifier
            self.assertIsNotNone(session.candidate)
            self.assertGreaterEqual(len(session.preview), 3)
            self.assertAlmostEqual(session.candidate.results[sketch.id].constraint_status.points['a'][1], 2, places=6)
            self.assertEqual(len(manager.history_edit_graph.features[1].parameters['constraints']), 3)
            return session

        session = select_line_and_hover_circle()
        manager.stop_sketch_modifier()
        self.assertEqual(session.preview, [])
        select_line_and_hover_circle()
        manager._on_click()
        self.assertEqual(manager.history_edit_graph.features[1].parameters['constraints'][-1]['kind'], 'tangent_line_circle')
        manager.stop_sketch_modifier()
        manager.start_sketch_dimension('auto', lambda dimension, preview: preview('8'))
        anchor, _ = manager.history_dimension_labels[2]
        hover(*anchor[:2])
        manager._on_click()
        manager.finish_history_sketch()
        rebuilt = FeatureGraph.from_data(manager.parametric_graph.to_data())
        state = rebuilt.results[sketch.id].constraint_status
        self.assertAlmostEqual(state.radii['@circle:0'], 4)
        self.assertAlmostEqual(state.points['a'][1], 4)
        self.assertAlmostEqual(state.points['b'][1], 4)
        self.assertEqual(rebuilt.results[sketch.id].profile_keys, (key,))
        self.assertEqual(rebuilt.features[1].parameters['constraints'][-1],
                         {'kind': 'tangent_line_circle', 'entities': [['a', 'b'], ['o', '@circle:0']]})
        self.assertAlmostEqual(rebuilt.results[extrude.id].volume, 48 * math.pi)
        self.assertAlmostEqual(rebuilt.results[scale.id].volume, 384 * math.pi)
        self.assertAlmostEqual(initial.results[extrude.id].volume, 12 * math.pi)

    def test_conflicting_point_selection_preserves_graph_and_fix_can_be_removed(self):
        plane = Feature.create('Plane', 'datum_plane', {})
        sketch = Feature.create('Sketch', 'planar_sketch', {
            'points': {'a': [-3, 0], 'b': [3, 2]},
            'entities': [{'type': 'line', 'points': ['a', 'b'], 'construction': True}],
            'constraints': [{'kind': 'fixed', 'entities': ['a'], 'value': [-3, 0]},
                            {'kind': 'fixed', 'entities': ['b'], 'value': [3, 2]}],
        }, plane.id)
        manager, hover = self.viewport(sketch, FeatureGraph([plane, sketch]))
        messages = []
        manager.status_callback = messages.append
        manager.start_sketch_constraint('coincident')
        current = manager.history_edit_graph
        hover(-3, 0)
        manager._on_click()
        hover(3, 2)
        self.assertIn('conflict', messages[-1])
        self.assertIsNone(manager.sketch_modifier.candidate)
        manager._on_click()
        self.assertIs(manager.history_edit_graph, current)
        manager.stop_sketch_modifier()
        manager.start_sketch_constraint('fixed')
        hover(3, 2)
        self.assertIn('Unfix point preview', messages[-1])
        manager._on_click()
        self.assertEqual(len(manager.history_edit_graph.features[1].parameters['constraints']), 1)
        manager.stop_sketch_modifier()
        manager.start_sketch_constraint('horizontal')
        hover(-3, 0)
        manager._on_click()
        hover(3, 2)
        self.assertIsNotNone(manager.sketch_modifier.candidate)
        manager._on_click()
        state = manager.history_edit_graph.results[sketch.id].constraint_status
        self.assertAlmostEqual(state.points['b'][1], 0, places=6)

    def test_curvature_picks_spline_endpoint_and_circle_contact(self):
        plane = Feature.create('Plane', 'datum_plane', {})
        sketch = Feature.create('Sketch', 'planar_sketch', {
            'points': {'o': [0, 0], 'a': [0, 2], 'b': [1, 2], 'c': [2, 1.25], 'd': [4, 1]},
            'entities': [{'type': 'circle', 'center': 'o', 'radius': 2},
                         {'type': 'spline', 'points': ['a', 'b', 'c', 'd'], 'construction': True}],
        }, plane.id)
        manager, hover = self.viewport(sketch, FeatureGraph([plane, sketch]))
        manager.start_sketch_constraint('curvature')
        edge = next(edge for index, _, edge, _ in manager.history_segment_actors.values() if index == 1)
        hover(*tuple(edge.position_at(.5))[:2])
        self.assertEqual(manager.sketch_modifier.preview_key[1].kind, 'spline')
        manager._on_click()
        hover(0, 2)
        self.assertIsNotNone(manager.sketch_modifier.candidate)
        manager._on_click()
        entry, = manager.history_edit_graph.features[1].parameters['constraints']
        self.assertEqual(entry['kind'], 'curvature_spline_circle')
        self.assertEqual(entry['entities'], [['a', 'b', 'c', 'd'], 'a', ['o', '@circle:0']])
        self.assertAlmostEqual(entry['value'], 90, delta=2)
        state = manager.history_edit_graph.results[sketch.id].constraint_status
        angle = math.radians(entry['value'])
        contact = state.points['o'] + state.radii['@circle:0'] * np.array([math.cos(angle), math.sin(angle)])
        np.testing.assert_allclose(state.points['a'], contact, atol=1e-6)

    def test_coincidence_with_arc_is_bounded_by_the_arc_endpoints(self):
        points = {'a': [-2, 0], 'b': [2, 0], 'm': [0, 2], 'p': [0, 3]}
        fixed_arc = [Constraint('fixed', (name,), points[name]) for name in ('a', 'b', 'm')]
        on_arc = Constraint('point_on_arc', ('p', ('a', 'b', 'm')))
        solved = solve_constraints(points, {}, [*fixed_arc, on_arc])
        np.testing.assert_allclose(solved.points['p'], [0, 2], atol=1e-6)
        opposite_side = Constraint('fixed', ('p',), [0, -2])
        self.assertEqual(solve_constraints(points, {}, [*fixed_arc, on_arc, opposite_side]).status, 'conflict')

    def test_curve_rules_encode_contact_and_normalize_selection_order(self):
        circle = ConstraintTarget('circle', ('o', '@circle:2'), angle=90)
        line = ConstraintTarget('line', ('a', 'b'))
        arc = ConstraintTarget('arc', ('e', 'f', 'm'), endpoint='f')
        spline = ConstraintTarget('spline', ('@nurbs', 3), endpoint='s')
        point = ConstraintTarget('point', ('p',))
        cases = [
            ('midpoint', [line, point], 'midpoint', ['p', ['a', 'b']]),
            ('coincident', [circle, point], 'point_on_circle', ['p', ['o', '@circle:2']]),
            ('coincident', [point, arc], 'point_on_arc', ['p', ['e', 'f', 'm']]),
            ('equal', [circle, arc], 'equal_radius_arc_circle', [['e', 'f', 'm'], ['o', '@circle:2']]),
            ('tangent', [arc, line], 'tangent_line_arc', [['a', 'b'], ['e', 'f', 'm'], 'f']),
            ('curvature', [arc, spline], 'curvature_spline_arc', [['@nurbs', 3], 's', ['e', 'f', 'm'], 'f']),
            ('tangent', [circle, spline], 'tangent_spline_circle', [['@nurbs', 3], 's', ['o', '@circle:2']]),
        ]
        for mode, targets, kind, entities in cases:
            with self.subTest(kind=kind):
                entry = constraint_from_targets(mode, targets, {})
                self.assertEqual((entry['kind'], entry['entities']), (kind, entities))
        self.assertEqual(entry['value'], 90)
        self.assertEqual(constraint_from_targets('fixed', [point], {'p': np.array([2, 1])})['value'], [2, 1])
