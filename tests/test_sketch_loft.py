import math
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pyvista as pv
import build123d as cad
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.gp import gp_Pnt, gp_Vec
from PySide6 import QtCore, QtWidgets

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from UI.main.content import LayoutManager
from UI.main.loft_dialog import LoftDialog
from UI.model.ActionManager import ActionManager
from UI.model.feature_graph import Feature, FeatureGraph
from UI.model.geometry_references import make_reference
from UI.model.loft_rims import section_rims
from UI.model.sketch_loft import build_sketch_loft
from UI.model.curve_spans import native_curve


class SketchLoftTest(unittest.TestCase):
    def test_manual_hole_matches_change_the_scene_preview(self):
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        features = []
        for height, a, b in ((0, [-2, 0], [2, 0]), (10, [0, -2], [0, 2])):
            plane = Feature.create('Section plane', 'datum_plane', {'origin': [0, 0, height]})
            sketch = Feature.create(f'Section {height}', 'planar_sketch', {
                'points': {'o': [0, 0], 'a': a, 'b': b},
                'entities': [{'type': 'circle', 'center': name, 'radius': radius}
                             for name, radius in (('o', 5), ('a', .5), ('b', .5))],
            }, plane.id)
            features.extend((plane, sketch))
        graph = FeatureGraph(features)
        plotter = pv.Plotter(off_screen=True, window_size=(860, 660))
        self.addCleanup(plotter.close)
        manager = ActionManager(plotter)
        manager.install_parametric_graph(graph)
        dialog = LoftDialog(manager)
        self.addCleanup(dialog.deleteLater)
        plotter.view_isometric()
        plotter.camera.position = (20, 20, 17)
        plotter.camera.focal_point = (0, 0, 5)
        plotter.enable_parallel_projection()
        plotter.camera.parallel_scale = 10
        plotter.show(auto_close=False)

        def click(x, y, z):
            renderer = plotter.renderer
            renderer.SetWorldPoint(x, y, z, 1.)
            renderer.WorldToDisplay()
            x, y, _ = renderer.GetDisplayPoint()
            plotter.iren.interactor.SetEventPosition(round(x), round(y))
            dialog.scene_click()

        click(3, 0, 0)
        click(3, 0, 10)
        self.assertEqual([section[0] for section in dialog.sections], [features[1].id, features[3].id])
        dialog.match_holes.setChecked(True)
        click(-1.5, 0, 0)
        self.assertIsNotNone(dialog.pending_hole)
        self.assertFalse(dialog.ok.isEnabled())
        click(.5, 2, 10)
        self.assertTrue(dialog.ok.isEnabled(), dialog.message.text())
        self.assertIsNotNone(dialog.preview)
        solid = dialog.result_shape.solids()[-1]
        self.assertFalse(solid.is_inside((-1, 1, 5)))
        self.assertTrue(solid.is_inside((-1, -1, 5)))
        self.assertEqual(dialog.match_list.count(), 1)
        dialog.move_section(-1)
        self.assertEqual(dialog.match_list.count(), 1)
        reverse = dialog.result_shape.solids()[-1]
        self.assertFalse(reverse.is_inside((-1, 1, 5)))
        dialog.match_list.setCurrentRow(0)
        dialog.remove_match.click()
        self.assertEqual(dialog.match_list.count(), 0)
        click(.5, -2, 10)
        click(-1.5, 0, 0)
        self.assertTrue(dialog.ok.isEnabled(), dialog.message.text())
        dialog.ok.click()
        reopened = FeatureGraph.from_data(dialog.candidate.to_data())
        solid = reopened.results[reopened.latest_solid_id].solids()[-1]
        self.assertTrue(solid.is_inside((-1, 1, 5)))
        self.assertFalse(solid.is_inside((-1, -1, 5)))
        high = features[3]
        points = {**high.parameters['points'], 'a': [1, -2], 'b': [-1, 2]}
        reopened.edit(high.id, parameters={
            **high.parameters, 'points': points, 'entities': list(reversed(high.parameters['entities'])),
        })
        solid = reopened.results[reopened.latest_solid_id].solids()[-1]
        self.assertFalse(solid.is_inside((-.5, -1, 5)))
        self.assertTrue(solid.is_inside((-1.5, 1, 5)))
        app.processEvents()

    def test_two_hole_supports_follow_sketch_contours_after_source_edit(self):
        features, sections = [], []
        for height in (0, 5, 10):
            plane = Feature.create('Section plane', 'datum_plane', {'origin': [0, 0, height]})
            sketch = Feature.create('Two holes', 'planar_sketch', {
                'points': {'outer': [0, 0], 'left': [-2, 0], 'right': [2, 0]},
                'entities': [{'type': 'circle', 'center': name, 'radius': radius}
                             for name, radius in (('outer', 6), ('left', 'left_radius'), ('right', 1))],
            }, plane.id)
            features.extend((plane, sketch))
            sections.append(sketch)
        params = {'left_radius': .5}
        graph = FeatureGraph(features, params)
        selections = []
        for section in sections:
            sketch = graph.results[section.id]
            index = next(i for i, face in enumerate(sketch.profiles) if len(face.inner_wires()) == 2)
            selections.append({'sketch_id': section.id, 'profile_key': sketch.profile_keys[index]})
        end = sections[-1]
        support = Feature.create('Support', 'extrude', {
            'profile_key': selections[-1]['profile_key'], 'along_sketch_normal': True, 'distance': 2,
        }, end.id)
        graph = FeatureGraph([*features, support], params)
        sketch = graph.results[end.id]
        profile = sketch.profiles[sketch.profile_keys.index(selections[-1]['profile_key'])]
        rims = dict(section_rims(sketch, profile)[1:])
        self.assertEqual(len(rims), 2)
        self.assertNotIn(None, rims)
        shape = graph.results[support.id]
        conditions = {}
        for key, rim in rims.items():
            radius = (rim.area / math.pi) ** .5
            index = next(i for i, face in enumerate(shape.faces()) if face.geom_type == cad.GeomType.CYLINDER
                         and math.isclose(BRepAdaptor_Surface(face.wrapped).Cylinder().Radius(), radius))
            conditions[key] = {'mode': 'tangent' if radius < .75 else 'curvature',
                               'face': make_reference(support.id, shape, 'face', index, graph.body_names[support.id])}
        by_position = []
        for selection in selections:
            section = graph.results[selection['sketch_id']]
            profile = section.profiles[section.profile_keys.index(selection['profile_key'])]
            by_position.append({round(face.center().X): key for key, face in section_rims(section, profile)[1:]})
        matches = [[level, left[x], right[x]]
                   for level, (left, right) in enumerate(zip(by_position, by_position[1:])) for x in left]
        loft = Feature.create('Loft', 'loft', {'sections': selections, 'rim_ends': {'end': conditions},
                                             'hole_matches': matches})
        graph = FeatureGraph([*features, support, loft], params)
        volume = graph.results[loft.id].volume
        graph.edit(end.id, parameters={**end.parameters, 'entities': list(reversed(end.parameters['entities']))})
        graph.edit(support.id, parameters={**support.parameters, 'distance': 4})
        solid = graph.results[loft.id].solids()[0]
        self.assertAlmostEqual(solid.volume, volume, places=5)
        self.assertFalse(solid.is_inside((-2, 0, 5)))
        self.assertFalse(solid.is_inside((2, 0, 5)))
        graph.set_parameters({'left_radius': .8})
        solid = graph.results[loft.id].solids()[0]
        self.assertAlmostEqual(volume - solid.volume, math.pi * 10 * (.8**2 - .5**2), places=5)
        self.assertFalse(solid.is_inside((-2, .7, 5)))
        self.assertEqual(graph.features[-1].parameters['hole_matches'], matches)
        self.assertEqual(graph.features[-1].parameters['rim_ends']['end'], conditions)
        with self.assertRaisesRegex(ValueError, 'only once'):
            graph.edit(loft.id, parameters={**loft.parameters, 'hole_matches': [*matches, matches[0]]})

    def test_guided_loft_matches_multiple_holes_and_accepts_an_inner_support(self):
        def ring(radius, x, z):
            return cad.Wire([cad.Edge.make_circle(radius, cad.Plane((x, 0, z)))])

        low = cad.Face(ring(6, 0, 0), [ring(.5, -2, 0), ring(1, 2, 0)])
        high = cad.Face(ring(6, 0, 10), [ring(1, 2, 10), ring(.5, -2, 10)])
        start = cad.extrude(low, amount=-2, dir=(0, 0, 1))
        end = cad.extrude(high, amount=2, dir=(0, 0, 1))
        outside = next(face for face in start.faces() if face.geom_type == cad.GeomType.CYLINDER
                       and abs(face.area - 24 * math.pi) < 1e-6)
        inside = next(face for face in end.faces() if face.geom_type == cad.GeomType.CYLINDER
                      and abs(face.area - 4 * math.pi) < 1e-6)
        guides = [cad.Edge.make_line((x, 0, 0), (x, 0, 10)) for x in (6, -1.5)]
        ends = {'start': ('tangent', outside), 'end': ('curvature', inside)}
        loft = build_sketch_loft([low, high], guides=guides, ends=ends)
        self.assertAlmostEqual(loft.volume, 347.5 * math.pi, places=5)
        self.assertFalse(loft.solids()[0].is_inside((-2, 0, 5)))
        self.assertFalse(loft.solids()[0].is_inside((2, 0, 5)))
        unperforated = cad.Face(ring(6, 0, 10), [ring(1, 2, 10)])
        with self.assertRaisesRegex(ValueError, 'same number of holes'):
            build_sketch_loft([low, unperforated], guides=guides, ends=ends)
        other_inside = next(face for face in end.faces() if face.geom_type == cad.GeomType.CYLINDER
                            and abs(face.area - 2 * math.pi) < 1e-6)
        rim_ends = {'start': {0: ('tangent', outside)},
                    'end': {1: ('curvature', inside), 2: ('tangent', other_inside)}}
        explicit = build_sketch_loft([low, high], guides=guides, rim_ends=rim_ends)
        self.assertAlmostEqual(explicit.volume, loft.volume, places=5)
        with self.assertRaisesRegex(ValueError, 'selected rim'):
            build_sketch_loft([low, high], guides=guides, rim_ends={'end': {1: ('tangent', other_inside)}})
        with self.assertRaisesRegex(ValueError, 'respect manual hole matches'):
            build_sketch_loft([low, high], guides=guides, hole_matches={0: {1: 1}})

    def test_hollow_loft_rebuilds_inner_and_outer_rims_after_scene_selection(self):
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        features = []
        sections = []
        for height in (0, 10):
            plane = Feature.create('Section plane', 'datum_plane', {'origin': [0, 0, height]})
            sketch = Feature.create(f'Annulus {height}', 'planar_sketch', {
                'points': {'o': [0, 0], 'h': [0, 0]},
                'entities': [{'type': 'circle', 'center': 'o', 'radius': 5},
                             {'type': 'circle', 'center': 'h', 'radius': 'hole'}],
            }, plane.id)
            features.extend((plane, sketch))
            sections.append(sketch)
        params = {'hole': 1, 'depth': 2, 'end_depth': 2, 'bulge': 8}
        initial = FeatureGraph(features, params)
        supports = []
        for index, sketch in enumerate(sections):
            result = initial.results[sketch.id]
            annulus = next(i for i, face in enumerate(result.profiles) if face.inner_wires())
            args = {'profile_key': result.profile_keys[annulus], 'along_sketch_normal': True,
                    'distance': '-depth' if index == 0 else 'end_depth'}
            if supports:
                args['base_id'] = supports[-1].id
            supports.append(Feature.create('Pipe support', 'extrude', args, sketch.id))
        rail_plane = Feature.create('Rail plane', 'datum_plane', {'normal': [0, -1, 0], 'x_dir': [1, 0, 0]})
        points = {f'p{i}': [x, z] for i, (x, z) in
                  enumerate(zip((5, 5, 5, 'bulge', 5, 5, 5), (0, 1, 3, 5, 7, 9, 10)))}
        points.update({'a': ['hole', 0], 'b': ['hole', 10]})
        rails = Feature.create('Outer and inner guides', 'planar_sketch', {
            'points': points,
            'entities': [{'type': 'nurbs', 'points': [f'p{i}' for i in range(7)], 'degree': 5,
                          'knots': [0] * 6 + [.5] + [1] * 6, 'weights': [1] * 7},
                         {'type': 'line', 'points': ['a', 'b']}],
        }, rail_plane.id)
        graph = FeatureGraph([*features, *supports, rail_plane, rails], params)
        plotter = pv.Plotter(off_screen=True, window_size=(860, 660))
        self.addCleanup(plotter.close)
        manager = ActionManager(plotter)
        manager.install_parametric_graph(graph)
        dialog = LoftDialog(manager)
        self.addCleanup(dialog.deleteLater)
        plotter.view_isometric()
        plotter.camera.position = (20, 20, 19)
        plotter.camera.focal_point = (0, 0, 5)
        plotter.enable_parallel_projection()
        plotter.camera.parallel_scale = 14
        plotter.show(auto_close=False)

        def click(world):
            renderer = plotter.renderer
            renderer.SetWorldPoint(*world, 1.)
            renderer.WorldToDisplay()
            x, y, _ = renderer.GetDisplayPoint()
            plotter.iren.interactor.SetEventPosition(round(x), round(y))
            dialog.scene_click()

        for height in (0, 10):
            click((2, 0, height))
        for feature_id, index in dialog.sections:
            self.assertTrue(graph.results[feature_id].profiles[index].inner_wires())
        self.assertEqual(len(dialog.sections), 2)
        dialog.match_holes.setChecked(True)
        click((1, 0, 0))
        click((1, 0, 10))
        self.assertEqual(dialog.match_list.count(), 1)
        dialog.choose_guides.setChecked(True)
        for _, rail in graph.results[rails.id].projectable_curves:
            click(tuple(rail.position_at(.48)))
        self.assertEqual(len(dialog.guides), 2)
        for side, height in (('start', -1), ('end', 11)):
            dialog.end_modes[side].setCurrentIndex(2)
            dialog.choose_support(side)
            click((5 / math.sqrt(2), 5 / math.sqrt(2), height))
            self.assertTrue(dialog.ok.isEnabled(), dialog.message.text())
        for side, height, camera in (('start', -1, (-5, 0, -19)), ('end', 11, (-5, 0, 29))):
            plotter.camera.position = camera
            plotter.camera.focal_point = (0, 0, 5)
            plotter.camera.up = (0, 1, 0)
            plotter.camera.parallel_scale = 6
            plotter.reset_camera_clipping_range()
            self.assertEqual(dialog.rim_choices[side].itemText(1), 'Hole 1')
            dialog.rim_choices[side].setCurrentIndex(1)
            self.assertEqual(dialog.end_modes[side].currentIndex(), 0)
            dialog.end_modes[side].setCurrentIndex(1 if side == 'start' else 2)
            self.assertFalse(dialog.ok.isEnabled())
            dialog.choose_support(side)
            click((1, 0, height))
            self.assertTrue(dialog.ok.isEnabled(), dialog.message.text())
            dialog.rim_choices[side].setCurrentIndex(0)
            self.assertEqual(dialog.end_modes[side].currentIndex(), 2)
            self.assertEqual(dialog.support_buttons[side].text(), 'Change Face')
        self.assertTrue(dialog.ok.isEnabled(), dialog.message.text())
        self.assertIsNotNone(dialog.preview)
        self.assertIs(manager.parametric_graph, graph)
        dialog.ok.click()
        reopened = FeatureGraph.from_data(dialog.candidate.to_data())
        loft = reopened.features[-1]
        for side, hole_mode in (('start', 'tangent'), ('end', 'curvature')):
            conditions = loft.parameters['rim_ends'][side]
            self.assertEqual(conditions['outer']['mode'], 'curvature')
            hole = next(value for key, value in conditions.items() if key != 'outer')
            self.assertEqual(hole['mode'], hole_mode)
            self.assertNotEqual(hole['face'], conditions['outer']['face'])

        def verify():
            solid = reopened.results[loft.id].solids()[-1]
            self.assertTrue(solid.is_valid)
            self.assertFalse(solid.is_inside((0, 0, 5)))
            self.assertTrue(solid.is_inside((3, 0, 5)))
            self.assertEqual(sum(bool(face.inner_wires()) for face in solid.faces()
                                 if face.geom_type == cad.GeomType.PLANE), 2)
            seam = next(edge for edge in solid.edges() if edge.geom_type == cad.GeomType.BSPLINE
                        and abs(edge.position_at(0).X - 5) < 1e-6
                        and abs(edge.position_at(1).X - 5) < 1e-6)
            curve, start, end = native_curve(seam)
            for parameter in (start, end):
                tangent, curvature = gp_Vec(), gp_Vec()
                curve.D2(parameter, gp_Pnt(), tangent, curvature)
                self.assertAlmostEqual(tangent.X(), 0., places=6)
                self.assertAlmostEqual(curvature.X(), 0., places=6)
            return solid.volume

        before = verify()
        reopened.set_parameters({**reopened.parameters, 'hole': 1.5})
        hollow = verify()
        self.assertAlmostEqual(before - hollow, 12.5 * math.pi, places=5)
        reopened.set_parameters({**reopened.parameters, 'bulge': 9})
        guided = verify()
        self.assertGreater(guided, hollow)
        previous = reopened.results[loft.id]
        previous_support = reopened.results[supports[0].id]
        reopened.set_parameters({**reopened.parameters, 'depth': 3})
        self.assertNotEqual(previous_support.volume, reopened.results[supports[0].id].volume)
        self.assertIsNot(previous, reopened.results[loft.id])
        self.assertAlmostEqual(verify(), guided, places=5)
        previous = reopened.results[loft.id]
        previous_support = reopened.results[supports[1].id]
        reopened.set_parameters({**reopened.parameters, 'end_depth': 4})
        self.assertNotEqual(previous_support.volume, reopened.results[supports[1].id].volume)
        self.assertIsNot(previous, reopened.results[loft.id])
        self.assertAlmostEqual(verify(), guided, places=5)

        later_sketch = Feature.create('Later sketch', 'planar_sketch', {
            'points': {'o': [0, 0]}, 'entities': [{'type': 'circle', 'center': 'o', 'radius': 4}],
        }, sections[0].input_id)
        scale = Feature.create('Scale Loft', 'scale', {'factor': 2}, loft.id)
        history = FeatureGraph([*reopened.features, later_sketch, scale], reopened.parameters)
        manager.install_parametric_graph(history)
        final_actor = plotter.actors['main_part']
        owner = LayoutManager.__new__(LayoutManager)
        owner.action_manager = manager
        owner.window = QtWidgets.QMainWindow()
        owner.timeline_widget = QtWidgets.QListWidget()
        owner.tool_tabs = QtWidgets.QTabWidget()
        owner.shortcuts = []
        owner.operation_panel = QtWidgets.QWidget()
        owner.operation_content = QtWidgets.QWidget(owner.operation_panel)
        layout = QtWidgets.QVBoxLayout(owner.operation_content)
        owner.show_operation_panel = layout.addWidget

        class EditLoftMenu(QtWidgets.QMenu):
            def exec(self, _position):
                return next(action for action in self.actions() if action.text() == 'Edit Loft')

        def open_edit():
            with (patch.object(QtWidgets, 'QMenu', EditLoftMenu),
                  patch.object(QtWidgets, 'QMessageBox') as messages):
                owner.parametric_context_menu(len(reopened.features) - 1, QtCore.QPoint())
            messages.warning.assert_not_called()
            return owner.active_operation_dialog

        dialog = open_edit()
        self.assertTrue(dialog.ok.isEnabled(), dialog.message.text())
        self.assertIsNotNone(dialog.preview)
        self.assertFalse(final_actor.GetVisibility())
        self.assertNotIn((later_sketch.id, 0), dialog.profile_actors)
        self.assertEqual([key for key, _ in dialog.sections], [sketch.id for sketch in sections])
        self.assertEqual(dialog.guides, [tuple(ref) for ref in loft.parameters['guides']])
        dialog.match_holes.setChecked(True)
        self.assertEqual(dialog.match_list.count(), 1)
        dialog.remove_match.click()
        dialog.reject()
        self.assertIs(manager.parametric_graph, history)
        self.assertTrue(final_actor.GetVisibility())
        self.assertIsNone(dialog.source_actor)

        dialog = open_edit()
        dialog.rim_choices['start'].setCurrentIndex(1)
        self.assertEqual(dialog.end_modes['start'].currentData(), 'tangent')
        self.assertEqual(dialog.support_buttons['start'].text(), 'Change Face')
        plotter.camera.position = (-5, 0, -19)
        plotter.camera.focal_point = (0, 0, 5)
        plotter.camera.up = (0, 1, 0)
        plotter.camera.parallel_scale = 6
        plotter.reset_camera_clipping_range()
        dialog.choose_support('start')
        click((1.5, 0, -1))
        self.assertTrue(dialog.ok.isEnabled(), dialog.message.text())
        key = dialog._rim_setting_key('start')
        self.assertEqual(dialog.support_faces[key]['feature_id'], supports[-1].id)
        dialog.end_modes['start'].setCurrentIndex(2)
        plotter.camera.position = (20, 20, 19)
        plotter.camera.parallel_scale = 14
        plotter.reset_camera_clipping_range()
        dialog.choose_guides.setChecked(True)
        rail = reopened.results[rails.id].projectable_curves[0][1]
        click(tuple(rail.position_at(.48)))
        self.assertTrue(dialog.ok.isEnabled(), dialog.message.text())
        self.assertEqual(len(dialog.guides), 1)
        self.assertIs(manager.parametric_graph, history)
        dialog.ok.click()
        edited = manager.parametric_graph
        self.assertEqual([item.id for item in history.features], [item.id for item in edited.features])
        saved = next(item for item in edited.features if item.id == loft.id)
        self.assertEqual(saved.parameters['hole_matches'], loft.parameters['hole_matches'])
        expected_ends = {side: dict(conditions) for side, conditions in loft.parameters['rim_ends'].items()}
        expected_ends['start'][key[2]] = {'mode': 'curvature', 'face': dialog.support_faces[key]}
        self.assertEqual(saved.parameters['rim_ends'], expected_ends)
        self.assertEqual(saved.parameters['guides'], [list(dialog.guides[0])])
        self.assertLess(edited.results[loft.id].volume, history.results[loft.id].volume)
        self.assertAlmostEqual(edited.results[scale.id].volume, 8 * edited.results[loft.id].volume, places=5)
        self.assertTrue(manager.undo_solid())
        self.assertEqual(manager.parametric_graph.to_data(), history.to_data())

        legacy = {key: value for key, value in loft.parameters.items() if key != 'rim_ends'}
        legacy['ends'] = {side: conditions['outer'] for side, conditions in loft.parameters['rim_ends'].items()}
        legacy_graph = FeatureGraph.from_data(history.to_data())
        legacy_graph.edit(loft.id, parameters=legacy)
        manager.install_parametric_graph(legacy_graph)
        dialog = open_edit()
        self.assertTrue(dialog.ok.isEnabled(), dialog.message.text())
        self.assertEqual(dialog.end_modes['start'].currentData(), 'curvature')
        self.assertEqual(dialog.support_buttons['start'].text(), 'Change Face')
        dialog.reject()
        self.assertEqual(manager.parametric_graph.to_data(), legacy_graph.to_data())
        app.processEvents()

    def test_spherical_support_matches_nonzero_meridian_curvature(self):
        sphere = cad.Solid.make_sphere(3, angle1=-90, angle2=0)
        support = next(face for face in sphere.faces() if face.geom_type == cad.GeomType.SPHERE)
        profiles = [cad.Face(cad.Wire([cad.Edge.make_circle(radius, cad.Plane((0, 0, z)))]))
                    for radius, z in ((3, 0), (4, 6))]
        body = build_sketch_loft(profiles, ends={'start': ('curvature', support)})
        seam = next(edge for edge in body.edges() if edge.geom_type == cad.GeomType.BEZIER)
        curve, first, _ = native_curve(seam)
        derivative, second = gp_Vec(), gp_Vec()
        curve.D2(first, gp_Pnt(), derivative, second)
        self.assertAlmostEqual(derivative.X(), 0.)
        self.assertAlmostEqual(second.X() / derivative.Z()**2, -1 / 3)

    def test_scene_combined_loft_rebuilds_after_guide_support_and_section_changes(self):
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        start = Feature.create('Cone support', 'primitive', {
            'type': 'cone', 'dimensions': ['bottom', 2, 4], 'position': [0, 0, -2],
        })
        end = Feature.create('Cylinder support', 'primitive', {
            'type': 'cylinder', 'dimensions': [4, 4], 'position': [0, 0, 12],
        }, start.id)
        features = [start, end]
        for height, radius in ((0, 2), (10, 4)):
            plane = Feature.create('Plane', 'datum_plane', {'origin': [0, 0, height]})
            profile = Feature.create('Section', 'planar_sketch', {
                'points': {'o': [0, 0]}, 'entities': [{'type': 'circle', 'center': 'o', 'radius': radius}],
            }, plane.id)
            features.extend((plane, profile))
        guide_plane = Feature.create('Guide plane', 'datum_plane', {'normal': [0, -1, 0], 'x_dir': [1, 0, 0]})
        guide = Feature.create('Meridian', 'planar_sketch', {
            'points': {name: [radius, 2 * index] for index, (name, radius) in enumerate(
                zip('abcdef', (2, '2 + (2 - bottom) / 2', 'bulge', 4, 4, 4)))},
            'entities': [{'type': 'nurbs', 'points': list('abcdef'), 'degree': 5,
                          'knots': [0] * 6 + [1] * 6, 'weights': [1] * 6}],
        }, guide_plane.id)
        middle_plane = Feature.create('Middle plane', 'datum_plane', {'origin': [0, 0, 'station']})
        middle = Feature.create('Middle section', 'planar_sketch', {
            'points': {'o': [0, 0]},
            'entities': [{'type': 'circle', 'center': 'o', 'radius': 'mid_radius'}],
        }, middle_plane.id)
        graph = FeatureGraph([*features, guide_plane, guide, middle_plane, middle],
                             {'bottom': 1, 'bulge': 5, 'station': 5, 'mid_radius': 3})
        plotter = pv.Plotter(off_screen=True, window_size=(840, 640))
        self.addCleanup(plotter.close)
        manager = ActionManager(plotter)
        manager.install_parametric_graph(graph)
        dialog = LoftDialog(manager)
        self.addCleanup(dialog.deleteLater)
        plotter.view_isometric()
        plotter.camera.position = (20, 20, 19)
        plotter.camera.focal_point = (0, 0, 5)
        plotter.enable_parallel_projection()
        plotter.camera.parallel_scale = 13
        plotter.show(auto_close=False)

        def click(world):
            renderer = plotter.renderer
            renderer.SetWorldPoint(*world, 1.)
            renderer.WorldToDisplay()
            x, y, _ = renderer.GetDisplayPoint()
            plotter.iren.interactor.SetEventPosition(round(x), round(y))
            dialog.scene_click()

        click((0, 0, 0))
        click((0, 0, 10))
        self.assertTrue(dialog.ok.isEnabled(), dialog.message.text())
        dialog.end_modes['start'].setCurrentIndex(1)
        self.assertFalse(dialog.ok.isEnabled())
        dialog.choose_support('start')
        click((1.5 / math.sqrt(2), 1.5 / math.sqrt(2), -2))
        self.assertTrue(dialog.ok.isEnabled(), dialog.message.text())
        dialog.end_modes['end'].setCurrentIndex(2)
        dialog.choose_support('end')
        click((4 / math.sqrt(2), 4 / math.sqrt(2), 12))
        self.assertTrue(dialog.ok.isEnabled(), dialog.message.text())

        def derivatives(shape, at_end):
            edge = next(edge for edge in shape.solids()[-1].edges()
                        if edge.geom_type in (cad.GeomType.BEZIER, cad.GeomType.BSPLINE))
            curve, first, last = native_curve(edge)
            derivative, second = gp_Vec(), gp_Vec()
            curve.D2(last if at_end else first, gp_Pnt(), derivative, second)
            slope = derivative.X() / derivative.Z()
            curvature = (second.X() - slope * second.Z()) / derivative.Z()**2
            return slope, curvature

        slope, second = derivatives(dialog.result_shape, False)
        self.assertAlmostEqual(slope, .25)
        self.assertGreater(abs(second), 1e-4)
        self.assertAlmostEqual(derivatives(dialog.result_shape, True)[0], 0.)
        self.assertAlmostEqual(derivatives(dialog.result_shape, True)[1], 0.)
        dialog.end_modes['start'].setCurrentIndex(2)
        self.assertTrue(dialog.ok.isEnabled(), dialog.message.text())
        self.assertAlmostEqual(derivatives(dialog.result_shape, False)[1], 0.)
        dialog.choose_guides.setChecked(True)
        rail = graph.results[guide.id].projectable_curves[0][1]
        click(tuple(rail.position_at(.38)))
        self.assertEqual(len(dialog.guides), 1)
        self.assertFalse(dialog.ok.isEnabled())
        self.assertIn('guide curvature', dialog.message.text())
        dialog.end_modes['start'].setCurrentIndex(1)
        self.assertTrue(dialog.ok.isEnabled(), dialog.message.text())
        self.assertIsNotNone(dialog.preview)
        self.assertAlmostEqual(derivatives(dialog.result_shape, False)[0], .25)
        self.assertAlmostEqual(derivatives(dialog.result_shape, True)[1], 0.)
        initial_volume = dialog.result_shape.solids()[-1].volume
        dialog.ok.click()
        reopened = FeatureGraph.from_data(dialog.candidate.to_data())
        loft = reopened.features[-1]
        reopened.set_parameters({**reopened.parameters, 'bulge': 6})
        widened = reopened.results[loft.id].solids()[-1]
        self.assertGreater(widened.volume, initial_volume)
        reopened.set_parameters({**reopened.parameters, 'bottom': .5})
        rebuilt = reopened.results[loft.id]
        self.assertAlmostEqual(derivatives(rebuilt, False)[0], .375)
        self.assertAlmostEqual(derivatives(rebuilt, True)[1], 0.)
        rebuilt_rail = reopened.results[guide.id].projectable_curves[0][1]
        check = cad.Vertex(*tuple(rebuilt_rail.position_at(.44)))
        self.assertLess(min(check.distance_to(edge) for edge in rebuilt.solids()[-1].edges()), 1e-6)

        def radius_at(t):
            return sum(math.comb(5, i) * t**i * (1 - t)**(5 - i) * radius
                       for i, radius in enumerate((2, 2.75, 6, 4, 4, 4)))

        reopened.set_parameters({**reopened.parameters, 'mid_radius': radius_at(.5)})
        entry = {**loft.parameters, 'sections': loft.parameters['sections'].copy()}
        profile = reopened.results[middle.id]
        entry['sections'].insert(1, {'sketch_id': middle.id, 'profile_key': profile.profile_keys[0]})
        reopened.edit(loft.id, parameters=entry)
        before = reopened.results[loft.id]
        reopened.set_parameters({**reopened.parameters, 'station': 6, 'mid_radius': radius_at(.6)})
        self.assertIsNot(reopened.results[loft.id], before)
        self.assertAlmostEqual(reopened.results[loft.id].volume, before.volume, places=5)
        with self.assertRaisesRegex(ValueError, 'meet each section boundary'):
            reopened.set_parameters({**reopened.parameters, 'mid_radius': 1})
        app.processEvents()

    def test_combined_loft_accepts_consistent_guides_at_different_angles(self):
        profiles = [cad.Face(cad.Wire([cad.Edge.make_circle(radius, cad.Plane((0, 0, z)))]))
                    for radius, z in ((2, 0), (4, 10))]
        cylinder = cad.Cylinder(2, 4).translate((0, 0, -2))
        support = next(face for face in cylinder.faces() if face.geom_type == cad.GeomType.CYLINDER)
        rail = cad.Edge.make_bezier((2, 0, 0), (2, 0, 3), (3, 0, 6), (4, 0, 10))
        ends = {'start': ('tangent', support)}
        single = build_sketch_loft(profiles, ends=ends, guides=[rail])
        opposite = rail.rotate(cad.Axis.Z, 90).reversed()
        multiple = build_sketch_loft(profiles, ends=ends, guides=[rail, opposite])
        self.assertAlmostEqual(multiple.volume, single.volume)
        different = cad.Edge.make_bezier((0, 2, 0), (0, 3, 3), (0, 4, 6), (0, 4, 10))
        with self.assertRaisesRegex(ValueError, 'incompatible with tangent'):
            build_sketch_loft(profiles, ends=ends, guides=[rail, different])

    def test_noncoaxial_elliptical_loft_from_scene_keeps_independent_guides_and_g2(self):
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        lower_plane = Feature.create('Lower plane', 'datum_plane', {})
        upper_plane = Feature.create('Upper plane', 'datum_plane', {'origin': [1, 0, 10]})

        def section(name, plane, radius):
            return Feature.create(name, 'planar_sketch', {
                'points': {'o': [0, 0]},
                'entities': [{'type': 'ellipse', 'center': 'o', 'radius_x': radius, 'radius_y': 2}],
            }, plane.id)

        lower = section('Lower ellipse', lower_plane, 3)
        upper = section('Upper ellipse', upper_plane, 'width')
        first = Feature.create('Lower support', 'extrude', {'distance': '-depth', 'along_sketch_normal': True}, lower.id)
        last = Feature.create('Upper support', 'extrude', {'distance': 2, 'base_id': first.id,
                                                           'along_sketch_normal': True}, upper.id)
        guide_plane = Feature.create('Guide plane', 'datum_plane', {'normal': [0, -1, 0], 'x_dir': [1, 0, 0]})
        points, curves = {}, []
        for name, values in (('right', (3, 3, 3, 'bulge', '1 + width', '1 + width', '1 + width')),
                             ('left', (-3, -3, -3, -4, '1 - width', '1 - width', '1 - width'))):
            names = [f'{name}{i}' for i in range(7)]
            points.update((key, [radius, z]) for key, radius, z in zip(names, values, (0, 1, 3, 5, 7, 9, 10)))
            curves.append({'type': 'nurbs', 'points': names, 'degree': 5,
                           'knots': [0] * 6 + [.5] + [1] * 6, 'weights': [1] * 7})
        rails = Feature.create('Independent rails', 'planar_sketch', {'points': points, 'entities': curves}, guide_plane.id)
        graph = FeatureGraph([lower_plane, lower, upper_plane, upper, first, last, guide_plane, rails],
                             {'width': 4, 'depth': 2, 'bulge': 7})
        plotter = pv.Plotter(off_screen=True, window_size=(860, 660))
        self.addCleanup(plotter.close)
        manager = ActionManager(plotter)
        manager.install_parametric_graph(graph)
        dialog = LoftDialog(manager)
        self.addCleanup(dialog.deleteLater)
        plotter.view_isometric()
        plotter.camera.position = (20, 20, 19)
        plotter.camera.focal_point = (0, 0, 5)
        plotter.enable_parallel_projection()
        plotter.camera.parallel_scale = 12
        plotter.show(auto_close=False)

        def click(world):
            renderer = plotter.renderer
            renderer.SetWorldPoint(*world, 1.)
            renderer.WorldToDisplay()
            x, y, _ = renderer.GetDisplayPoint()
            plotter.iren.interactor.SetEventPosition(round(x), round(y))
            dialog.scene_click()

        click((0, 0, 0))
        click((1, 0, 10))
        dialog.choose_guides.setChecked(True)
        for _, rail in graph.results[rails.id].projectable_curves:
            click(tuple(rail.position_at(.47)))
        self.assertEqual(len(dialog.guides), 2)
        for side, point in (('start', (3 / math.sqrt(2), 2 / math.sqrt(2), -1)),
                            ('end', (1 + 4 / math.sqrt(2), 2 / math.sqrt(2), 11))):
            dialog.end_modes[side].setCurrentIndex(1 if side == 'start' else 2)
            dialog.choose_support(side)
            click(point)
        self.assertTrue(dialog.ok.isEnabled(), dialog.message.text())
        dialog.end_modes['start'].setCurrentIndex(2)
        self.assertTrue(dialog.ok.isEnabled(), dialog.message.text())
        self.assertIsNotNone(dialog.preview)
        dialog.ok.click()
        reopened = FeatureGraph.from_data(dialog.candidate.to_data())
        loft = reopened.features[-1]

        def verify(graph):
            solid = graph.results[loft.id].solids()[-1]
            for _, rail in graph.results[rails.id].projectable_curves:
                for t in (.13, .43, .79):
                    point = cad.Vertex(*tuple(rail.position_at(t)))
                    self.assertLess(min(point.distance_to(edge) for edge in solid.edges()), 1e-6)
            skins = [f for f in solid.faces() if f.geom_type == cad.GeomType.BSPLINE]
            checked = set()
            for face in skins:
                surface = BRepAdaptor_Surface(face.wrapped)
                first, last = surface.FirstUParameter(), surface.LastUParameter()
                for v in (surface.FirstVParameter(), surface.LastVParameter()):
                    for t in (.13, .37, .67, .87):
                        du, dv, duu, dvv, duv = (gp_Vec() for _ in range(5))
                        position = gp_Pnt()
                        surface.D2(first + t * (last - first), v, position, du, dv, duu, dvv, duv)
                        z = round(position.Z())
                        if z not in (0, 10) or abs(position.Z() - z) > 1e-6:
                            continue
                        checked.add(z)
                        self.assertAlmostEqual(math.hypot(dv.X(), dv.Y()), 0., places=6)
                        self.assertAlmostEqual(math.hypot(dvv.X(), dvv.Y()), 0., places=6)
            self.assertEqual(checked, {0, 10})
            return solid.volume

        volume = verify(reopened)
        reopened.set_parameters({**reopened.parameters, 'bulge': 8})
        changed = verify(reopened)
        self.assertGreater(changed, volume)
        reopened.set_parameters({**reopened.parameters, 'width': 4.5})
        wider = verify(reopened)
        self.assertGreater(wider, changed)
        before = reopened.results[loft.id]
        reopened.set_parameters({**reopened.parameters, 'depth': 3})
        self.assertIsNot(reopened.results[loft.id], before)
        self.assertAlmostEqual(verify(reopened), wider, places=6)
        app.processEvents()

    def test_guided_polygon_sections_keeps_sharp_corners(self):
        sections = [cad.Face.make_rect(4, 4, cad.Plane((0, 0, z))) for z in (0, 5, 10)]
        guides = [cad.Edge.make_line((x, 2, 0), (x, 2, 10)) for x in (2, -2)]
        loft = build_sketch_loft(sections, guides=guides)
        self.assertAlmostEqual(loft.volume, 160)

    def test_scene_guides_shape_solid_and_rebuild_with_profile_or_guide_changes(self):
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        features = []
        for height in (0, 5, 10):
            plane = Feature.create('Section plane', 'datum_plane', {'origin': [0, 0, height]})
            sketch = Feature.create(f'Section {height}', 'planar_sketch', {
                'points': {'o': [0, 0]},
                'entities': [{'type': 'ellipse', 'center': 'o', 'radius_x': 2,
                              'radius_y': 'width', 'is_closed': True}],
            }, plane.id)
            features.extend((plane, sketch))
        plane = Feature.create('Guide plane', 'datum_plane', {'normal': [0, -1, 0], 'x_dir': [1, 0, 0]})
        rail = Feature.create('Guides', 'planar_sketch', {
            'points': {'a': [2, 0], 'b': ['2 + bulge', 3], 'c': ['2 - bulge', 7], 'd': [2, 10],
                       'e': [-2, 10], 'f': [-2, 0]},
            'entities': [{'type': 'spline', 'points': list('abcd')},
                         {'type': 'line', 'points': ['e', 'f']}],
        }, plane.id)
        graph = FeatureGraph([*features, plane, rail], {'width': 1.5, 'bulge': 2})
        manager = ActionManager(pv.Plotter(off_screen=True, window_size=(820, 620)))
        plotter = manager.plotter
        self.addCleanup(plotter.close)
        manager.parametric_graph = graph
        dialog = LoftDialog(manager)
        self.addCleanup(dialog.deleteLater)
        plotter.view_isometric()
        plotter.camera.position = (20, 20, 25)
        plotter.camera.focal_point = (0, 0, 5)
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

        for height in (0, 5, 10):
            click((0, 0, height))
        plain = dialog.result_shape
        self.assertIsNotNone(plain)
        dialog.choose_guides.setChecked(True)
        curves = graph.results[rail.id].projectable_curves
        click(tuple(curves[0][1].position_at(.25)))
        self.assertEqual(len(dialog.guides), 1)
        self.assertTrue(dialog.ok.isEnabled(), dialog.message.text())
        self.assertGreater(dialog.result_shape.bounding_box().max.X, plain.bounding_box().max.X + .4)
        click((-2, 0, 7))
        self.assertEqual(len(dialog.guides), 2)
        self.assertTrue(dialog.ok.isEnabled(), dialog.message.text())
        self.assertIsNotNone(dialog.preview)
        dialog.ok.click()
        self.assertFalse(dialog.guide_actors)

        reopened = FeatureGraph.from_data(dialog.candidate.to_data())
        loft = reopened.features[-1]
        self.assertEqual(len(loft.parameters['guides']), 2)
        before = reopened.results[loft.id]
        reopened.set_parameters({'width': 1.7, 'bulge': 2})
        wider = reopened.results[loft.id]
        self.assertGreater(wider.volume, before.volume)
        reopened.set_parameters({'width': 1.7, 'bulge': 3})
        shaped = reopened.results[loft.id]
        self.assertGreater(shaped.bounding_box().max.X, wider.bounding_box().max.X + .25)
        guide = reopened.results[rail.id].projectable_curves[0][1]
        point = cad.Vertex(*tuple(guide.position_at(.22)))
        self.assertLess(min(point.distance_to(edge) for edge in shaped.edges()), 1e-6)
        missing = next(item.parameters for item in reopened.features if item.id == rail.id)
        missing['points']['a'] = [4, 0]
        with self.assertRaisesRegex(ValueError, 'meet each section boundary'):
            reopened.edit(rail.id, parameters=missing)
        self.assertIs(reopened.results[loft.id], shaped)
        reopened.edit(rail.id, suppressed=True)
        self.assertIn(loft.id, reopened.blocked)
        app.processEvents()

    def test_scene_order_and_commit_rebuild_after_profile_edit(self):
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        features, sketches = [], []
        for height, radius in ((0, 1), (5, 'middle'), (10, 1)):
            plane = Feature.create(f'Plane {height}', 'datum_plane', {'origin': [0, 0, height]})
            sketch = Feature.create(f'Section {height}', 'planar_sketch', {
                'points': {'o': [0, 0]},
                'entities': [{'type': 'circle', 'center': 'o', 'radius': radius}],
            }, plane.id)
            features.extend((plane, sketch))
            sketches.append(sketch)
        graph = FeatureGraph(features, {'middle': 1})
        plotter = pv.Plotter(off_screen=True, window_size=(720, 560))
        self.addCleanup(plotter.close)
        manager = ActionManager(plotter)
        manager.parametric_graph = graph
        owner = LayoutManager.__new__(LayoutManager)
        owner.action_manager = manager
        owner.window = QtWidgets.QMainWindow()
        owner.tool_tabs = QtWidgets.QTabWidget()
        owner.shortcuts = []
        owner.operation_panel = QtWidgets.QWidget()
        owner.operation_content = QtWidgets.QWidget(owner.operation_panel)
        layout = QtWidgets.QVBoxLayout(owner.operation_content)
        owner.show_operation_panel = layout.addWidget
        dialog = LoftDialog(manager, owner.window)
        owner.show_scene_operation(dialog, 'Loft')
        plotter.view_isometric()
        plotter.camera.position = (20, 20, 25)
        plotter.camera.focal_point = (0, 0, 5)
        plotter.enable_parallel_projection()
        plotter.camera.parallel_scale = 8
        plotter.show(auto_close=False)

        def click(height):
            renderer = plotter.renderer
            renderer.SetWorldPoint(0, 0, height, 1.)
            renderer.WorldToDisplay()
            x, y, _ = renderer.GetDisplayPoint()
            plotter.iren.interactor.SetEventPosition(round(x), round(y))
            manager._on_click()

        click(0)
        self.assertFalse(dialog.ok.isEnabled())
        click(10)
        self.assertIsNotNone(dialog.preview)
        click(5)
        self.assertEqual(dialog.sections, [(sketches[i].id, 0) for i in (0, 2, 1)])
        dialog.up.click()
        self.assertEqual(dialog.sections, [(sketch.id, 0) for sketch in sketches])
        self.assertTrue(dialog.ok.isEnabled(), dialog.message.text())
        self.assertAlmostEqual(dialog.result_shape.volume, 10 * math.pi, places=6)
        self.assertIs(manager.parametric_graph, graph)
        dialog.ok.click()
        self.assertFalse(manager.interaction_locked)
        self.assertIsNone(manager.operation_scene_click)
        self.assertFalse(dialog.actors)
        self.assertIsNone(dialog.preview)

        committed = manager.parametric_graph
        loft = committed.features[-1]
        self.assertEqual(loft.kind, 'loft')
        self.assertEqual([section['sketch_id'] for section in loft.parameters['sections']],
                         [sketch.id for sketch in sketches])
        downstream = Feature.create('Scale', 'scale', {'factor': 2}, loft.id)
        reopened = FeatureGraph([*committed.features, downstream], committed.parameters)
        reopened = FeatureGraph.from_data(reopened.to_data())
        reopened.set_parameters({'middle': 1.5})
        volume = reopened.results[loft.id].volume
        self.assertGreater(volume, 10 * math.pi)
        self.assertAlmostEqual(reopened.results[downstream.id].volume, volume * 8, places=5)
        reopened.edit(sketches[1].id, suppressed=True)
        self.assertIn(loft.id, reopened.blocked)
        self.assertIn(downstream.id, reopened.blocked)
        reopened.edit(sketches[1].id, suppressed=False)
        self.assertAlmostEqual(reopened.results[loft.id].volume, volume)
        bad = {**loft.parameters, 'sections': [loft.parameters['sections'][0]] * 2}
        with self.assertRaisesRegex(ValueError, 'selected more than once'):
            reopened.edit(loft.id, parameters=bad)
        self.assertAlmostEqual(reopened.results[loft.id].volume, volume)

        cut_plane = Feature.create('Cut plane', 'datum_plane', {'origin': [0, 0, 9]})
        cut_sketch = Feature.create('Cut sketch', 'planar_sketch', {
            'points': {'o': [0, 0]}, 'entities': [{'type': 'circle', 'center': 'o', 'radius': .2}],
        }, cut_plane.id)
        cut = Feature.create('Required cut', 'extrude', {
            'distance': .5, 'operation': 'cut', 'base_id': loft.id, 'along_sketch_normal': True,
        }, cut_sketch.id)
        history = FeatureGraph([*reopened.features[:-1], cut_plane, cut_sketch, cut], reopened.parameters)
        manager.install_parametric_graph(history)
        owner.edit_loft(loft.id)
        editor = owner.active_operation_dialog
        editor.remove_section(2)
        self.assertIsNotNone(editor.preview)
        self.assertFalse(editor.ok.isEnabled())
        self.assertIn('Required cut', editor.message.text())
        self.assertIs(manager.parametric_graph, history)
        editor.reject()
        self.assertIs(manager.parametric_graph, history)
        app.processEvents()
