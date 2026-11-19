import math
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import build123d as cad

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))

from UI.model.ActionManager import ActionManager
from UI.model.feature_graph import Feature, FeatureGraph
from UI.model.profiles import sketch_edges
from UI.model.sketch_modify import extend_edge, modified_sketch_parameters, offset_edge, trim_at, trim_sections
from UI.model.sketch_modify_session import SketchModifySession


NORMAL = (0, 0, 1)


class SketchModifyTest(unittest.TestCase):
    def test_trim_removes_only_clicked_interval_of_polyline(self):
        graph, feature = self.sketch({
            'points': {'A': [0, 0], 'B': [10, 0], 'C': [10, 4]},
            'entities': [{'type': 'line', 'points': ['A', 'B', 'C']}],
        })
        sketch = graph.results[feature.id]
        source = sketch_edges(sketch.tools, NORMAL)[0]
        boundaries = [cad.Edge.make_line((x, -1), (x, 1)) for x in (2, 7)]
        modification = trim_at(trim_sections(source, boundaries), .5)
        self.assertAlmostEqual(modification.preview.length, 5)
        self.assertEqual(len(modification.replacement), 2)

        params = modified_sketch_parameters(feature.parameters, sketch, 0, 0, source, modification)
        graph.edit(feature.id, parameters=params)
        edges = sketch_edges(graph.results[feature.id].tools, NORMAL)
        self.assertEqual(sorted(round(item.length) for item in edges), [2, 3, 4])
        self.assertEqual(params['entities'][-1]['points'], ['B', 'C'])

    def test_trim_circle_respects_seam_and_serializes_remaining_arc(self):
        graph, feature = self.sketch({
            'points': {'C': [0, 0]},
            'entities': [{'type': 'circle', 'center': 'C', 'radius': 5}],
        })
        sketch = graph.results[feature.id]
        source = sketch_edges(sketch.tools, NORMAL)[0]
        boundary = cad.Edge.make_line((0, -8), (0, 8))
        modification = trim_at(trim_sections(source, [boundary]), .05)
        self.assertGreater(modification.preview.position_at(.5).X, 4.9)
        self.assertLess(modification.replacement[0].position_at(.5).X, -4.9)
        params = modified_sketch_parameters(feature.parameters, sketch, 0, 0, source, modification)
        graph.edit(feature.id, parameters=params)
        rebuilt = sketch_edges(graph.results[feature.id].tools, NORMAL)[0]
        self.assertAlmostEqual(rebuilt.length, 5 * math.pi)
        self.assertLess(rebuilt.position_at(.5).X, -4.9)

    def test_extend_uses_nearest_boundary_for_line_and_arc(self):
        line = cad.Edge.make_line((0, 0), (1, 0))
        crossing = cad.Edge.make_line((5, -1), (5, 1))
        collinear = cad.Edge.make_line((2, 0), (3, 0))
        result = extend_edge(line, [crossing, collinear], False, NORMAL)
        self.assertAlmostEqual(result.preview.length, 1)
        self.assertAlmostEqual(result.replacement[0].length, 2)
        with self.assertRaisesRegex(ValueError, 'no boundary'):
            extend_edge(line, [crossing], True, NORMAL)

        arc = cad.Edge.make_circle(5, end_angle=90)
        boundary = cad.Edge.make_line((-8, 0), (-2, 0))
        result = extend_edge(arc, [boundary], False, NORMAL)
        self.assertAlmostEqual(result.replacement[0].length, 5 * math.pi)
        self.assertAlmostEqual(result.preview.position_at(1).X, -5)

    def test_offset_copies_curve_without_removing_constraints(self):
        graph, feature = self.sketch({
            'points': {'C': [0, 0]},
            'entities': [{'type': 'circle', 'center': 'C', 'radius': 5}],
            'constraints': [{'kind': 'radius', 'entities': [['C', '@circle:0']], 'value': 5}],
        })
        sketch = graph.results[feature.id]
        source = sketch_edges(sketch.tools, NORMAL)[0]
        modification = offset_edge(source, -2, NORMAL)
        params = modified_sketch_parameters(feature.parameters, sketch, 0, 0, source, modification, copy_only=True)
        graph.edit(feature.id, parameters=params)
        edges = sketch_edges(graph.results[feature.id].tools, NORMAL)
        self.assertEqual([round(edge.radius) for edge in edges], [5, 3])
        self.assertEqual(params['constraints'], feature.parameters['constraints'])
        with self.assertRaisesRegex(ValueError, 'Remove constraints'):
            modified_sketch_parameters(feature.parameters, sketch, 0, 0, source, trim_at(trim_sections(source, []), .5))

    def test_removing_unconstrained_segment_keeps_other_geometry_references(self):
        graph, feature = self.sketch({
            'points': {'A': [-9, 0], 'B': [-7, 0], 'C': [0, 0]},
            'entities': [{'type': 'line', 'points': ['A', 'B']}, {'type': 'circle', 'center': 'C', 'radius': 3}],
            'constraints': [{'kind': 'radius', 'entities': [['C', '@circle:1']], 'value': 3}],
        })
        sketch = graph.results[feature.id]
        source = sketch_edges(sketch.tools, NORMAL)[0]
        params = modified_sketch_parameters(feature.parameters, sketch, 0, 0, source, trim_at(trim_sections(source, []), .5))
        graph.edit(feature.id, parameters=params)
        self.assertEqual(params['constraints'][0]['entities'], [['C', '@circle:0']])
        self.assertEqual(set(params['points']), {'C'})
        self.assertAlmostEqual(graph.results[feature.id].profiles[0].area, 9 * math.pi)

    def test_scene_hover_is_preview_only_and_click_edits_candidate(self):
        graph, feature = self.sketch({
            'points': {'A': [0, 0], 'B': [4, 0]},
            'entities': [{'type': 'line', 'points': ['A', 'B']}],
        })
        plotter = SimpleNamespace(render=Mock(), add_mesh=Mock(return_value=object()), remove_actor=Mock())
        manager = ActionManager(plotter)
        manager.sketch_editing = True
        manager.history_edit_graph = graph
        manager.history_edit_id = feature.id
        manager.parametric_graph = graph
        manager.render_history_sketch = Mock()
        actor = object()
        edge = sketch_edges(graph.results[feature.id].tools, NORMAL)[0]
        manager.history_segment_actors = {actor: (0, 0, edge, True)}
        manager._pick_actor = Mock(return_value=SimpleNamespace(GetActor=lambda: actor, GetPickPosition=lambda: (2, 0, 0)))
        manager.sketch_modifier = SketchModifySession(manager, 'offset', lambda: 2)

        manager._on_hover()
        self.assertIs(manager.history_edit_graph, graph)
        preview = plotter.add_mesh.call_args.args[0]
        self.assertAlmostEqual(preview.points[0][1], 2)
        self.assertFalse(plotter.add_mesh.call_args.kwargs['pickable'])
        manager._on_click()
        self.assertIs(manager.parametric_graph, graph)
        self.assertEqual(len(manager.history_edit_graph.results[feature.id].tools), 2)
        manager.render_history_sketch.assert_called_once()
        self.assertTrue(manager.stop_sketch_modifier())

    def test_trim_cannot_commit_a_broken_dependent_extrusion(self):
        graph, feature = self.sketch({
            'points': {'C': [0, 0]},
            'entities': [{'type': 'circle', 'center': 'C', 'radius': 3}],
        })
        extrusion = Feature.create('Extrude', 'extrude', {'distance': 2}, feature.id)
        graph = FeatureGraph([*graph.features, extrusion])
        manager = ActionManager(SimpleNamespace(render=Mock(), add_mesh=Mock(), remove_actor=Mock()))
        manager.history_edit_id = feature.id
        manager.history_edit_graph = graph
        manager.status_callback = Mock()
        manager.render_history_sketch = Mock()
        actor = object()
        edge = sketch_edges(graph.results[feature.id].tools, NORMAL)[0]
        manager.history_segment_actors = {actor: (0, 0, edge, True)}
        manager._pick_actor = Mock(return_value=SimpleNamespace(GetActor=lambda: actor, GetPickPosition=lambda: (3, 0, 0)))
        modifier = SketchModifySession(manager, 'trim', lambda: 0)
        self.assertFalse(modifier.apply())
        self.assertIs(manager.history_edit_graph, graph)
        self.assertAlmostEqual(graph.results[extrusion.id].volume, 18 * math.pi)
        manager.render_history_sketch.assert_not_called()
        self.assertIn('Extrude', manager.status_callback.call_args.args[0])

    @staticmethod
    def sketch(parameters):
        plane = Feature.create('Plane', 'datum_plane', {})
        feature = Feature.create('Sketch', 'planar_sketch', parameters, plane.id)
        return FeatureGraph([plane, feature]), feature


if __name__ == '__main__':
    unittest.main()
