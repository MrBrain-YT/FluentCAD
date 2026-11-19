import math
import sys
import unittest
from pathlib import Path

import build123d as cad
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from UI.model.ActionManager import ActionManager
from UI.model.feature_graph import Feature, FeatureGraph
from UI.model.nurbs import periodic_nurbs_edge
from UI.model.profiles import indexed_sketch_edges
from UI.model.sketch_modify import extend_curve_span, trim_at, trim_sections
from UI.model.sketch_offset import build_offset_chain, order_offset_chain


class FreeformSketchModifyTest(unittest.TestCase):
    def scene(self, graph, sketch):
        plotter = pv.Plotter(off_screen=True, window_size=(740, 560))
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

        def hover(point):
            plotter.renderer.SetWorldPoint(*point, 1.)
            plotter.renderer.WorldToDisplay()
            x, y, _ = plotter.renderer.GetDisplayPoint()
            plotter.iren.interactor.SetEventPosition(round(x), round(y))
            manager._on_hover()

        return manager, hover

    def test_ellipse_trim_and_extend_preserve_exact_rebuildable_profiles(self):
        cases = [('trim', 180, .75), ('extend', 60, .8)]
        for command, arc_end, pick in cases:
            with self.subTest(command=command):
                plane = Feature.create('Plane', 'datum_plane', {})
                sketch = Feature.create('Sketch', 'planar_sketch', {
                    'points': {'center': [0, 0], 'a': ['major', 0], 'c': [0, 'minor']},
                    'entities': [{'type': 'ellipse', 'center': 'center', 'radius_x': 'major', 'radius_y': 'minor',
                                  'is_closed': False, 'start_angle': 0, 'end_angle': arc_end},
                                 {'type': 'line', 'points': ['a', 'center', 'c']}],
                }, plane.id)
                original = FeatureGraph([plane, sketch], {'major': 8, 'minor': 4})
                manager, hover = self.scene(original, sketch)
                manager.start_sketch_modifier(command, lambda: 1)
                _, edge = indexed_sketch_edges(original.results[sketch.id].tools[0], (0, 0, 1))[0]
                hover(tuple(edge.position_at(pick)))
                self.assertIsNotNone(manager.sketch_modifier.actor)
                self.assertIs(manager.parametric_graph, original)
                manager._on_click()
                self.assertIsNot(manager.history_edit_graph, original)
                manager.finish_history_sketch()
                edited = manager.parametric_graph
                result = edited.results[sketch.id]
                self.assertAlmostEqual(result.profiles[0].area, 8 * math.pi, places=5)
                self.assertEqual(result.entities[0]['radius_x'], 'major')
                self.assertTrue(all(segment.geom_type == cad.GeomType.ELLIPSE
                                    for _, segment in indexed_sketch_edges(result.tools[0], (0, 0, 1))))
                key, = result.profile_keys
                self.assertIsNotNone(key)
                extrusion = Feature.create('Extrude', 'extrude', {'distance': 2, 'profile_key': key}, sketch.id)
                committed = FeatureGraph([*edited.features, extrusion], edited.parameters)
                reopened = FeatureGraph.from_data(committed.to_data())
                reopened.set_parameters({'major': 10, 'minor': 5})
                self.assertEqual(reopened.results[sketch.id].profile_keys, (key,))
                self.assertAlmostEqual(reopened.results[extrusion.id].volume, 25 * math.pi, places=4)

    def test_spline_extend_from_scene_rebuilds_on_control_point_change(self):
        plane = Feature.create('Plane', 'datum_plane', {})
        sketch = Feature.create('Sketch', 'planar_sketch', {
            'points': {'a': [0, 0], 'b': [1, 0], 'c': [2, 'height / 3'], 'd': [3, 'height'],
                       'e': [4, 0], 'f': [4, '16 * height / 9']},
            'entities': [{'type': 'spline', 'points': ['a', 'b', 'c', 'd']},
                         {'type': 'line', 'points': ['a', 'e', 'f']}],
        }, plane.id)
        original = FeatureGraph([plane, sketch], {'height': 2})
        manager, hover = self.scene(original, sketch)
        manager.start_sketch_modifier('extend', lambda: 1)
        spline, = indexed_sketch_edges(original.results[sketch.id].tools[0], (0, 0, 1))
        hover(tuple(spline[1].position_at(.85)))
        self.assertIsNotNone(manager.sketch_modifier.actor)
        manager._on_click()
        self.assertIsNot(manager.history_edit_graph, original)
        candidate = manager.history_edit_graph
        key, = candidate.results[sketch.id].profile_keys
        self.assertIsNotNone(key)
        extrusion = Feature.create('Extrude', 'extrude', {'distance': 3, 'profile_key': key}, sketch.id)
        model = FeatureGraph([*candidate.features, extrusion], candidate.parameters)
        volume = model.results[extrusion.id].volume
        model.set_parameters({'height': 3})
        self.assertEqual(model.results[sketch.id].profile_keys, (key,))
        self.assertAlmostEqual(model.results[extrusion.id].volume, volume * 1.5, places=5)

    def test_scene_trim_of_spline_keeps_its_basis_and_profile(self):
        plane = Feature.create('Plane', 'datum_plane', {})
        sketch = Feature.create('Sketch', 'planar_sketch', {
            'points': {'a': [0, 0], 'b': [1, 0], 'c': [2, 'height / 3'], 'd': [3, 'height'],
                       'e': [2, 0], 'f': [2, '4 * height / 9']},
            'entities': [{'type': 'spline', 'points': ['a', 'b', 'c', 'd']},
                         {'type': 'line', 'points': ['a', 'e', 'f']}],
        }, plane.id)
        original = FeatureGraph([plane, sketch], {'height': 2})
        manager, hover = self.scene(original, sketch)
        manager.start_sketch_modifier('trim', lambda: 1)
        _, edge = indexed_sketch_edges(original.results[sketch.id].tools[0], (0, 0, 1))[0]
        hover(tuple(edge.position_at(.88)))
        self.assertIsNotNone(manager.sketch_modifier.actor)
        manager._on_click()
        edited = manager.history_edit_graph.results[sketch.id]
        self.assertEqual(edited.entities[0]['points'], ['a', 'b', 'c', 'd'])
        _, trimmed = indexed_sketch_edges(edited.tools[0], (0, 0, 1))[0]
        self.assertAlmostEqual(trimmed.position_at(1).X, 2)
        self.assertEqual(trimmed.geom_type, cad.GeomType.BEZIER)
        key, = edited.profile_keys
        self.assertEqual(original.results[sketch.id].profile_keys, (key,))
        manager.history_edit_graph.set_parameters({'height': 4})
        self.assertAlmostEqual(manager.history_edit_graph.results[sketch.id].profiles[0].area,
                               edited.profiles[0].area * 2)

    def test_scene_offset_of_spline_and_elliptical_arc_rebuilds_extrusion(self):
        cases = [
            ({'type': 'ellipse', 'center': 'o', 'radius_x': 'size', 'radius_y': 3,
              'is_closed': False, 'start_angle': 0, 'end_angle': 180}, ['a', 'b']),
            ({'type': 'spline', 'points': ['b', 'c', 'd', 'a']}, ['a', 'b']),
        ]
        for curve, line in cases:
            with self.subTest(kind=curve['type']):
                plane = Feature.create('Plane', 'datum_plane', {})
                sketch = Feature.create('Sketch', 'planar_sketch', {
                    'points': {'o': [0, 0], 'a': ['-size', 0], 'b': ['size', 0],
                               'c': ['size', 4], 'd': ['-size', 4]},
                    'entities': [{**curve, 'construction': True},
                                 {'type': 'line', 'points': line, 'construction': True}],
                }, plane.id)
                original = FeatureGraph([plane, sketch], {'size': 5})
                manager, hover = self.scene(original, sketch)
                manager.start_sketch_offset(lambda: .4, lambda: True, lambda: 'round')
                _, edge = indexed_sketch_edges(original.results[sketch.id].tools[0], (0, 0, 1))[0]
                hover(tuple(edge.position_at(.5)))
                session = manager.sketch_modifier
                self.assertEqual(len(session.preview), 2)
                manager._on_click()
                self.assertEqual(len(session.selected), 2)
                self.assertTrue(session.confirm())
                manager.finish_history_sketch()
                committed = manager.parametric_graph
                key, = committed.results[sketch.id].profile_keys
                self.assertIsNotNone(key)
                extrusion = Feature.create('Extrude', 'extrude', {'distance': 2, 'profile_key': key}, sketch.id)
                model = FeatureGraph([*committed.features, extrusion], committed.parameters)
                reopened = FeatureGraph.from_data(model.to_data())
                original_area = reopened.results[sketch.id].profiles[0].area
                reopened.set_parameters({'size': 7})
                updated = reopened.results[sketch.id]
                self.assertEqual(updated.profile_keys, (key,))
                self.assertGreater(updated.profiles[0].area, original_area)
                self.assertAlmostEqual(reopened.results[extrusion.id].volume, 2 * updated.profiles[0].area, places=4)

    def test_b_spline_continuation_and_trim_across_ellipse_seam(self):
        spline = cad.Edge.make_spline([(0, 0), (1, 1), (2, 4), (3, 9)], parameters=[0, 1, 2, 3])
        target = cad.Edge.make_line((-1, -3), (-1, 5))
        extension = extend_curve_span(spline, 0, 1, [target], True)
        self.assertAlmostEqual(extension.replacement[0].position_at(0).X, -1)
        self.assertAlmostEqual(extension.replacement[0].position_at(0).Y, 1)
        ellipse = cad.Edge.make_ellipse(6, 3)
        divider = cad.Edge.make_line((0, -5), (0, 5))
        edit = trim_at(trim_sections(ellipse, [divider]), .99)
        remaining, = edit.replacement
        self.assertEqual(remaining.geom_type, cad.GeomType.ELLIPSE)
        self.assertAlmostEqual(remaining.position_at(.5).X, -6)

    def test_curved_chain_inset_trims_a_corner_and_rejects_a_cusp(self):
        arc = cad.Edge.make_ellipse(5, 3, start_angle=0, end_angle=180)
        edges = {'arc': arc, 'line': cad.Edge.make_line((-5, 0), (5, 0))}
        refs = order_offset_chain(edges, 'arc')
        groups = build_offset_chain(cad.Plane.XY, edges, refs, -.2)
        self.assertEqual([len(group) for group in groups], [1, 1])
        with self.assertRaisesRegex(ValueError, 'cusp'):
            build_offset_chain(cad.Plane.XY, edges, refs, -3)

    def test_periodic_nurbs_trim_and_offset_cross_the_native_seam(self):
        spline = periodic_nurbs_edge([(-3, -3, 0), (3, -3, 0), (3, 3, 0), (-3, 3, 0)], 2, [1] * 4)
        divider = cad.Edge.make_line((-5, 0), (5, 0))
        edit = trim_at(trim_sections(spline, [divider]), .5)
        remaining, = edit.replacement
        self.assertAlmostEqual(remaining.position_at(.5).Y, -3)
        self.assertTrue(remaining.is_valid)
        groups = build_offset_chain(cad.Plane.XY, {'spline': spline}, [['spline', False]], .2)
        wire = cad.Wire(groups[0])
        self.assertTrue(wire.is_closed)
        self.assertGreater(cad.Face(wire).area, cad.Face(cad.Wire([spline])).area)
