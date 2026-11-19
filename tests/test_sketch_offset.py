import math
import sys
import unittest
from pathlib import Path

import build123d as cad
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from UI.model.ActionManager import ActionManager
from UI.model.feature_graph import Feature, FeatureGraph
from UI.model.sketch_dimensions import update_sketch_dimension
from UI.model.sketch_offset import build_offset_chain, order_offset_chain


class SketchOffsetTest(unittest.TestCase):
    def test_viewport_chain_offset_rebuilds_after_dimension_edit(self):
        plane = Feature.create('Plane', 'datum_plane', {})
        width = {'kind': 'horizontal_distance', 'entities': ['a', 'b'], 'value': 6}
        sketch = Feature.create('Sketch', 'planar_sketch', {
            'points': {'a': [0, 0], 'b': [6, 0], 'c': [6, 4], 'd': [0, 4]},
            'entities': [{'type': 'line', 'points': ['a', 'b', 'c', 'd', 'a'], 'construction': True}],
            'constraints': [{'kind': 'fixed', 'entities': ['a'], 'value': [0, 0]},
                            {'kind': 'fixed', 'entities': ['d'], 'value': [0, 4]},
                            {'kind': 'horizontal', 'entities': [['a', 'b']]},
                            {'kind': 'horizontal', 'entities': [['c', 'd']]},
                            {'kind': 'vertical', 'entities': [['b', 'c']]}, width],
        }, plane.id)
        graph = FeatureGraph([plane, sketch])
        plotter = pv.Plotter(off_screen=True, window_size=(500, 440))
        self.addCleanup(plotter.close)
        manager = ActionManager(plotter)
        manager.history_edit_graph = manager.parametric_graph = graph
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

        current_distance = [1.]
        manager.start_sketch_offset(lambda: current_distance[0], lambda: False)
        manual = manager.sketch_modifier
        hover(3, 0)
        manager._on_click()
        hover(3, 4)
        manager._on_click()
        self.assertEqual(len(manual.selected), 1)
        hover(6, 2)
        manager._on_click()
        self.assertEqual(len(manual.selected), 2)
        manager.stop_sketch_modifier()
        self.assertFalse(manual.preview)
        self.assertFalse(manager.history_edit_graph.results[sketch.id].offset_edges)

        join = ['auto']
        manager.start_sketch_offset(lambda: current_distance[0], lambda: True, lambda: join[0])
        session = manager.sketch_modifier
        hover(3, 0)
        self.assertEqual(len(session.preview), 2)
        self.assertEqual(session.selected, [])
        self.assertFalse(session.confirm())
        manager._on_click()
        self.assertEqual(len(session.selected), 4)
        self.assertEqual(len(manager.history_edit_graph.results[sketch.id].offset_edges), 0)
        current_distance[0] = 1.5
        session.refresh_preview()
        self.assertEqual(len(session.preview), 2)
        sharp_preview = session.preview[-1]
        join[0] = 'round'
        session.refresh_preview()
        self.assertIsNot(session.preview[-1], sharp_preview)
        self.assertTrue(session.confirm())
        manager.finish_history_sketch()
        committed = manager.parametric_graph
        offset = committed.results[sketch.id]
        self.assertEqual(len(offset.offset_edges), 8)
        self.assertAlmostEqual(offset.profiles[0].area, 54 + 2.25 * math.pi)
        self.assertEqual(len(committed.features[1].parameters['entities']), 1)
        key, = offset.profile_keys
        self.assertIsNotNone(key)
        solid = Feature.create('Extrude', 'extrude', {'distance': 2, 'profile_key': key}, sketch.id)
        with_solid = FeatureGraph([*committed.features, solid])
        rebuilt = update_sketch_dimension(with_solid, sketch.id, width, '10', index=5)
        reloaded = FeatureGraph.from_data(rebuilt.to_data())
        changed = reloaded.results[sketch.id]
        self.assertEqual([name for name, _ in changed.offset_edges], [name for name, _ in offset.offset_edges])
        self.assertEqual(changed.profile_keys, (key,))
        self.assertEqual(reloaded.features[1].parameters['offsets'][0]['join'], 'round')
        self.assertAlmostEqual(changed.profiles[0].area, 82 + 2.25 * math.pi)
        self.assertAlmostEqual(reloaded.results[solid.id].volume, 164 + 4.5 * math.pi)
        self.assertAlmostEqual(with_solid.results[solid.id].volume, 108 + 4.5 * math.pi)

    def test_offset_corner_modes_have_distinct_external_geometry_and_same_inset(self):
        points = [(0, 0), (6, 0), (6, 4), (0, 4), (0, 0)]
        edges = {str(i): cad.Edge.make_line(a, b) for i, (a, b) in enumerate(zip(points, points[1:]))}
        refs = order_offset_chain(edges, '0')
        for mode, area in [('auto', 48), ('bevel', 46), ('round', 44 + math.pi)]:
            with self.subTest(mode=mode):
                groups = build_offset_chain(cad.Plane.XY, edges, refs, 1, join=mode)
                self.assertAlmostEqual(cad.Face(cad.Wire([edge for g in groups for edge in g])).area, area)
                inset = build_offset_chain(cad.Plane.XY, edges, refs, -.5, join=mode)
                self.assertAlmostEqual(cad.Face(cad.Wire([edge for g in inset for edge in g])).area, 15)
                if mode == 'round':
                    for i, (segment, corner) in enumerate(groups):
                        self.assertAlmostEqual(corner.radius, 1)
                        self.assertAlmostEqual(segment.tangent_at(1).dot(corner.tangent_at(0)), 1)
                        following = groups[(i + 1) % len(groups)][0]
                        self.assertAlmostEqual(corner.tangent_at(1).dot(following.tangent_at(0)), 1)
        path = {name: edges[name] for name in ('0', '1')}
        for mode in ('round', 'bevel'):
            with self.assertRaisesRegex(ValueError, 'inside corner'):
                build_offset_chain(cad.Plane.XY, path, order_offset_chain(path, '0'), 5, join=mode)

    def test_connected_line_and_arc_on_tilted_plane(self):
        plane = cad.Plane((3, -1, 4), x_dir=(1, 0, 0), z_dir=(0, 1, 1))
        point = lambda u, v: plane.from_local_coords((u, v, 0))
        line = cad.Edge.make_line(point(0, 0), point(6, 0))
        arc = cad.Edge.make_three_point_arc(point(8, 2), point(6 + math.sqrt(2), 2 - math.sqrt(2)), point(6, 0))
        edges = {'line': line, 'arc': arc}
        refs = order_offset_chain(edges, 'line')
        self.assertEqual(refs, [['line', False], ['arc', True]])
        offset_line, offset_arc = build_offset_chain(plane, edges, refs, .5)
        line_boundary = cad.Wire(offset_line)
        arc_boundary = cad.Wire(offset_arc)
        self.assertAlmostEqual(plane.to_local_coords(line_boundary.position_at(.5)).Y, .5)
        for edge in offset_arc:
            self.assertAlmostEqual(edge.radius, 1.5)
        self.assertTrue(cad.Wire([*offset_line, *offset_arc], sequenced=True).is_valid)
        self.assertLess(line_boundary.distance_to(arc_boundary), 1e-6)

    def test_branch_requires_an_explicit_path_and_closed_inset_cannot_collapse(self):
        edges = {'first': cad.Edge.make_line((0, 0), (4, 0)),
                 'second': cad.Edge.make_line((4, 0), (4, 3)),
                 'branch': cad.Edge.make_line((4, 0), (7, 0))}
        with self.assertRaisesRegex(ValueError, 'branches'):
            order_offset_chain(edges, 'first', automatic=True)
        path = {name: edges[name] for name in ('first', 'second')}
        references = order_offset_chain(path, 'first')
        self.assertTrue(build_offset_chain(cad.Plane.XY, path, references, -1))
        path.update({'third': cad.Edge.make_line((4, 3), (0, 3)),
                     'fourth': cad.Edge.make_line((0, 3), (0, 0))})
        closed = order_offset_chain(path, 'first', automatic=True)
        with self.assertRaisesRegex(ValueError, 'Cannot offset'):
            build_offset_chain(cad.Plane.XY, path, closed, -2)
        path['fourth'] = cad.Edge.make_line((0, 3), (0, 1))
        with self.assertRaisesRegex(ValueError, 'changed between open and closed'):
            build_offset_chain(cad.Plane.XY, path, closed, 1, expected_closed=True)

    def test_open_offset_joins_native_edges_in_a_stable_dependent_profile(self):
        plane = Feature.create('Plane', 'datum_plane', {})
        sketch = Feature.create('Sketch', 'planar_sketch', {
            'points': {'a': [0, 0], 'b': ['width', 0], 'c': ['width', 4],
                       'd': ['width - 1', 4], 'e': [0, 4], 'f': [0, 1]},
            'entities': [{'type': 'line', 'points': ['a', 'b', 'c'], 'construction': True},
                         {'type': 'line', 'points': ['d', 'e', 'f']}],
            'offsets': [{'id': 'path', 'sources': [['["line",["a","b"]]', False], ['["line",["b","c"]]', False]],
                         'distance': 1, 'closed': False}],
        }, plane.id)
        graph = FeatureGraph([plane, sketch], {'width': 6})
        key, = graph.results[sketch.id].profile_keys
        self.assertIsNotNone(key)
        solid = Feature.create('Extrude', 'extrude', {'distance': 2, 'profile_key': key}, sketch.id)
        graph = FeatureGraph([*graph.features, solid], graph.parameters)
        self.assertAlmostEqual(graph.results[solid.id].volume, 30)
        graph.set_parameters({'width': 9})
        self.assertEqual(graph.results[sketch.id].profile_keys, (key,))
        self.assertAlmostEqual(graph.results[solid.id].volume, 48)
        invalid = graph.features[1].parameters
        invalid['entities'][0]['points'] = ['a', 'b']
        with self.assertRaisesRegex(ValueError, 'source segment is missing'):
            graph.edit(sketch.id, parameters=invalid)
        self.assertAlmostEqual(graph.results[solid.id].volume, 48)
