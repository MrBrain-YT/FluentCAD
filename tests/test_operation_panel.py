import os
import sys
import unittest
import tempfile
import gc
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))

from PySide6 import QtCore, QtGui, QtWidgets
from UI.main.content import LayoutManager
from UI.main.panel_inputs import PanelInputs
from UI.main.operation_panel import OperationPanel
from UI.main.model_browser import ModelBrowser
from UI.main.navigation_cube import NavigationCube, REGIONS
from UI.main.camera_transition import CameraTransition
from UI.main.mouse_navigation import MouseNavigation
from UI.main.extrude_dialog import ExtrudeDialog, RevolveDialog
from UI.main.scene_value_editor import SceneValueEditor
from UI.main.arrow_hit_filter import ArrowHitFilter, segment_distance
from PySide6 import QtTest
from UI.main.extrude_handle import observe_arrow_camera, schedule_handle_sync, stop_handle_sync, finish_handle_drag
from UI.model.ActionManager import ActionManager
import vtk


class OperationPanelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cls.app.setQuitOnLastWindowClosed(False)

    def tearDown(self):
        self.app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
        self.app.processEvents()
        gc.collect()

    def test_dark_theme_keeps_operation_fields_and_status_text_readable(self):
        from PySide6 import QtGui
        from UI.main.theme import apply_theme

        def luminance(color):
            channels = [color.redF(), color.greenF(), color.blueF()]
            linear = [value / 12.92 if value <= .04045 else ((value + .055) / 1.055) ** 2.4
                      for value in channels]
            return sum(a * b for a, b in zip(linear, (.2126, .7152, .0722)))

        def check_contrast(widget, foreground, background):
            palette = widget.palette()
            a, b = sorted(luminance(palette.color(role)) for role in (foreground, background))
            self.assertGreaterEqual((b + .05) / (a + .05), 4.5)

        original = self.app.palette()
        window = QtWidgets.QMainWindow()
        panel = QtWidgets.QFrame(window)
        panel.setObjectName('operationPanel')
        window.setCentralWidget(panel)
        layout = QtWidgets.QVBoxLayout(panel)
        field = QtWidgets.QLineEdit('12.5', panel)
        warning = QtWidgets.QLabel('Invalid extrusion', panel)
        warning.setProperty('role', 'error')
        layout.addWidget(field)
        layout.addWidget(warning)
        window.statusBar().showMessage('Select a face')
        try:
            apply_theme(window, dark=True)
            window.show()
            self.app.processEvents()
            roles = QtGui.QPalette.ColorRole
            check_contrast(field, roles.Text, roles.Base)
            check_contrast(field, roles.HighlightedText, roles.Highlight)
            check_contrast(warning, roles.WindowText, roles.Window)
            check_contrast(window.statusBar(), roles.WindowText, roles.Window)
            self.assertLess(luminance(field.palette().color(roles.Base)), .1)
            apply_theme(window, dark=False)
            self.app.processEvents()
            check_contrast(field, roles.Text, roles.Base)
            self.assertGreater(luminance(field.palette().color(roles.Base)), .7)
        finally:
            window.close()
            self.app.setPalette(original)

    def test_embedded_dialog_can_collapse_and_accept(self):
        panel = QtWidgets.QFrame()
        layout = QtWidgets.QVBoxLayout(panel)
        title = QtWidgets.QToolButton()
        title.setCheckable(True)
        scroll = QtWidgets.QScrollArea()
        content = QtWidgets.QWidget()
        QtWidgets.QVBoxLayout(content)
        scroll.setWidget(content)
        layout.addWidget(title)
        layout.addWidget(scroll)
        title.toggled.connect(scroll.setVisible)
        owner = SimpleNamespace(operation_title=title, operation_content=content,
                                operation_panel=panel, operation_scroll=scroll)
        dialog = QtWidgets.QDialog()
        dialog.setWindowTitle('Extrude')
        finished = Mock()
        dialog.finished.connect(finished)
        LayoutManager.show_operation_panel(owner, dialog)
        self.app.processEvents()
        self.assertFalse(dialog.isWindow())
        self.assertEqual(dialog.parentWidget(), content)
        title.setChecked(False)
        self.assertTrue(scroll.isHidden())
        finished.assert_not_called()
        title.setChecked(True)
        dialog.accept()
        finished.assert_called_once_with(QtWidgets.QDialog.Accepted)
        panel.close()

    def test_clipping_keeps_nearby_geometry_and_far_bound(self):
        camera = SimpleNamespace(GetClippingRange=lambda: (1., 100.), SetClippingRange=Mock())
        renderer = SimpleNamespace(ComputeVisiblePropBounds=lambda: (-1, 1, -1, 1, -1, 1),
                                   ResetCameraClippingRange=Mock())
        manager = ActionManager(None)
        manager.plotter = SimpleNamespace(camera=camera, renderer=renderer)
        manager.refresh_camera_clipping()
        camera.SetClippingRange.assert_called_once_with(.001, 120.)

    def panel_owner(self):
        panel = QtWidgets.QFrame()
        panel_layout = QtWidgets.QVBoxLayout(panel)
        title = QtWidgets.QToolButton()
        title.setCheckable(True)
        scroll = QtWidgets.QScrollArea()
        content = QtWidgets.QWidget()
        QtWidgets.QVBoxLayout(content)
        scroll.setWidget(content)
        panel_layout.addWidget(title)
        panel_layout.addWidget(scroll)
        title.toggled.connect(scroll.setVisible)
        panel.hide()
        owner = SimpleNamespace(operation_title=title, operation_content=content,
                                operation_panel=panel, operation_scroll=scroll)
        owner.show_operation_panel = lambda dialog: LayoutManager.show_operation_panel(owner, dialog)
        return owner

    def test_input_selection_is_embedded_and_cancel_cleans_panel(self):
        owner = self.panel_owner()
        inputs = PanelInputs(owner)
        def accept():
            dialog = owner.operation_content.findChild(QtWidgets.QInputDialog)
            self.assertFalse(dialog.isWindow())
            dialog.setTextValue('second')
            dialog.accept()
        QtCore.QTimer.singleShot(0, accept)
        self.assertEqual(inputs.getItem(None, 'Curve', 'Select:', ['first', 'second'], 0, False), ('second', True))
        self.assertTrue(owner.operation_panel.isHidden())
        QtCore.QTimer.singleShot(0, lambda: owner.operation_content.findChild(QtWidgets.QInputDialog).reject())
        self.assertEqual(inputs.getInt(None, 'Polygon', 'Sides:', 6, 3, 128), (6, False))
        self.assertTrue(owner.operation_panel.isHidden())

    def test_nested_prompt_restores_previous_editor(self):
        owner = self.panel_owner()
        inputs = PanelInputs(owner)
        editor = QtWidgets.QDialog()
        editor.setWindowTitle('Sketch')
        owner.show_operation_panel(editor)
        def accept():
            prompt = owner.operation_content.findChild(QtWidgets.QInputDialog)
            self.assertTrue(editor.isHidden())
            prompt.accept()
        QtCore.QTimer.singleShot(0, accept)
        self.assertEqual(inputs.getText(None, 'NURBS', 'Weight:', text='2'), ('2', True))
        self.assertFalse(editor.isHidden())
        self.assertEqual(owner.operation_title.text(), 'Sketch')
        owner.operation_panel.close()

    def test_overlay_tracks_right_edge_and_collapses_to_header(self):
        viewport = QtWidgets.QWidget()
        viewport.resize(800, 600)
        panel = OperationPanel(viewport)
        layout = QtWidgets.QVBoxLayout(panel)
        title = QtWidgets.QToolButton()
        title.setCheckable(True)
        title.setChecked(True)
        content = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.addWidget(QtWidgets.QLabel('Parameters'))
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(content)
        layout.addWidget(title)
        layout.addWidget(scroll)
        panel.title, panel.content, panel.scroll = title, content, scroll
        panel.reposition()
        self.assertEqual(panel.geometry().right(), 787)
        self.assertLess(panel.height(), viewport.height())
        expanded_height = panel.height()
        title.setChecked(False)
        panel.reposition()
        self.assertLess(panel.height(), expanded_height)
        viewport.resize(600, 450)
        panel.reposition()
        self.assertEqual(panel.geometry().right(), 587)

    def test_navigation_widget_remains_uncovered_even_when_collapsed(self):
        viewport = QtWidgets.QWidget()
        viewport.resize(800, 600)
        panel = OperationPanel(viewport)
        layout = QtWidgets.QVBoxLayout(panel)
        title = QtWidgets.QToolButton()
        title.setCheckable(True)
        title.setChecked(True)
        content = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.addWidget(QtWidgets.QLabel('Parameters'))
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(content)
        layout.addWidget(title)
        layout.addWidget(scroll)
        panel.title, panel.content, panel.scroll = title, content, scroll
        panel.navigation_size = lambda: (140, 140)
        navigation = QtCore.QRect(660, 0, 140, 140)
        panel.reposition()
        self.assertFalse(panel.geometry().intersects(navigation))
        title.setChecked(False)
        panel.reposition()
        self.assertFalse(panel.geometry().intersects(navigation))
        self.assertLessEqual(panel.height(), max(24, title.sizeHint().height()) + 2)
        viewport.resize(800, 180)
        panel.reposition()
        self.assertFalse(panel.geometry().intersects(navigation))

    def test_overlay_reacts_to_content_size_changes(self):
        viewport = QtWidgets.QWidget()
        viewport.resize(800, 600)
        panel = OperationPanel(viewport)
        layout = QtWidgets.QVBoxLayout(panel)
        title = QtWidgets.QToolButton()
        title.setCheckable(True)
        title.setChecked(True)
        content = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content)
        label = QtWidgets.QLabel('Details')
        label.setFixedHeight(20)
        content_layout.addWidget(label)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        layout.addWidget(title)
        layout.addWidget(scroll)
        panel.title, panel.content, panel.scroll = title, content, scroll
        content.installEventFilter(panel)
        viewport.show()
        panel.show()
        panel.reposition()
        self.app.processEvents()
        before = panel.height()
        label.setFixedHeight(120)
        for _ in range(4):
            self.app.processEvents()
        self.assertGreater(panel.height(), before)
        viewport.close()

    def test_preview_draws_model_and_handles_in_one_render(self):
        dialog = ExtrudeDialog.__new__(ExtrudeDialog)
        QtWidgets.QDialog.__init__(dialog)
        dialog.handle = SimpleNamespace(sync=Mock())
        dialog.manager = SimpleNamespace(plotter=SimpleNamespace(render=Mock()))
        dialog._finish_preview()
        dialog.handle.sync.assert_called_once_with(render=False)
        dialog.manager.plotter.render.assert_called_once()

    def test_scene_value_editor_synchronizes_and_keeps_offset(self):
        viewport = QtWidgets.QWidget()
        viewport.resize(500, 400)
        viewport.renderer = SimpleNamespace(SetWorldPoint=Mock(), WorldToDisplay=Mock(),
                                            GetDisplayPoint=lambda: (200., 200., .5))
        field = ExtrudeDialog.number(5., -100., 100.)
        editor = SceneValueEditor(viewport, field)
        editor.set_point((1, 2, 3))
        self.assertEqual(editor.editor.value(), 5.)
        field.setValue(12.)
        self.assertEqual(editor.editor.value(), 12.)
        editor.editor.setValue(7.)
        self.assertEqual(field.value(), 7.)
        before = editor.pos()
        editor.offset += QtCore.QPoint(20, 10)
        editor.reposition()
        self.assertEqual(editor.pos() - before, QtCore.QPoint(20, 10))
        editor.dispose()
        viewport.close()

    def test_browser_hides_one_body_without_changing_model(self):
        from UI.model.feature_graph import Feature, FeatureGraph
        from UI.model.ModelConverter import b123d_to_pro_mesh
        box = Feature.create('Box', 'primitive', {'type': 'box', 'dimensions': [2, 2, 2]})
        pattern = Feature.create('Pattern', 'rectangular_pattern', {'count_x': 2, 'count_y': 1, 'spacing_x': 5, 'spacing_y': 5}, box.id)
        graph = FeatureGraph([box, pattern])
        shape = graph.results[pattern.id]
        mesh = b123d_to_pro_mesh(shape)
        mapper = SimpleNamespace(dataset=mesh, SetInputData=Mock())
        actor = SimpleNamespace(mapper=mapper, SetVisibility=Mock())
        manager = SimpleNamespace(current_part=shape, parametric_graph=graph, profiles=[], construction_planes=[],
            plotter=SimpleNamespace(actors={'main_part': actor}, render=Mock()),
            selected_face_id=-1, selected_face_ids=set(), _clear_highlight=Mock())
        viewport = QtWidgets.QWidget()
        browser = ModelBrowser(viewport)
        browser.refresh(manager)
        bodies = browser.topLevelItem(1)
        bodies.child(0).setCheckState(0, QtCore.Qt.Unchecked)
        displayed = mapper.SetInputData.call_args.args[0]
        self.assertLess(displayed.n_cells, mesh.n_cells)
        self.assertEqual(len(shape.solids()), 2)
        bodies.child(0).setCheckState(0, QtCore.Qt.Checked)
        self.assertIs(mapper.SetInputData.call_args.args[0], mesh)
        viewport.close()

    def test_body_visibility_tracks_reordering_and_new_bodies(self):
        import build123d as cad
        from UI.model.ModelConverter import b123d_to_pro_mesh
        bodies = [cad.Box(2, 2, 2).moved(cad.Pos(x, 0, 0)) for x in (0, 5, 10)]
        shape = cad.Part(bodies[:2])
        def actor_for(model):
            return SimpleNamespace(mapper=SimpleNamespace(dataset=b123d_to_pro_mesh(model), SetInputData=Mock()), SetVisibility=Mock())
        plotter = SimpleNamespace(actors={'main_part': actor_for(shape)}, render=Mock())
        manager = SimpleNamespace(current_part=shape, parametric_graph=None, sketchs_data={}, profiles=[], construction_planes=[],
            plotter=plotter, selected_face_id=-1, selected_face_ids=set(), _clear_highlight=Mock(), status_callback=Mock())
        viewport = QtWidgets.QWidget()
        browser = ModelBrowser(viewport)
        browser.refresh(manager)
        browser.topLevelItem(1).child(0).setCheckState(0, QtCore.Qt.Unchecked)
        settings = browser.export_visibility()
        manager.current_part = cad.Part([bodies[2], bodies[1], bodies[0]])
        plotter.actors['main_part'] = actor_for(manager.current_part)
        browser.refresh(manager)
        self.assertEqual(browser.hidden_bodies, {2})
        browser.reset_visibility()
        browser.restore_visibility(manager, settings)
        self.assertEqual(browser.hidden_bodies, {2})
        viewport.close()

    def test_parametric_body_visibility_survives_resize_and_pattern_growth(self):
        from UI.model.feature_graph import Feature, FeatureGraph
        from UI.model.ModelConverter import b123d_to_pro_mesh
        box = Feature.create('Box', 'primitive', {'type': 'box', 'dimensions': ['w', 2, 2]})
        pattern = Feature.create('Pattern', 'rectangular_pattern', {'count_x': 'n', 'count_y': 1, 'spacing_x': 8, 'spacing_y': 8}, box.id)
        graph = FeatureGraph([box, pattern], {'w': 2, 'n': 2})
        def actor_for(shape):
            return SimpleNamespace(mapper=SimpleNamespace(dataset=b123d_to_pro_mesh(shape), SetInputData=Mock()), SetVisibility=Mock())
        shape = graph.results[pattern.id]
        plotter = SimpleNamespace(actors={'main_part': actor_for(shape)}, render=Mock())
        manager = SimpleNamespace(current_part=shape, parametric_graph=graph, sketchs_data={}, profiles=[], construction_planes=[],
            plotter=plotter, selected_face_id=-1, selected_face_ids=set(), _clear_highlight=Mock(), status_callback=Mock())
        viewport = QtWidgets.QWidget()
        browser = ModelBrowser(viewport)
        browser.refresh(manager)
        browser.topLevelItem(1).child(1).setCheckState(0, QtCore.Qt.Unchecked)
        settings = browser.export_visibility()
        self.assertEqual(len(settings['hidden_body_ids']), 1)
        graph.set_parameters({'w': 3, 'n': 3})
        manager.current_part = graph.results[pattern.id]
        plotter.actors['main_part'] = actor_for(manager.current_part)
        browser.refresh(manager)
        self.assertEqual(len(browser.hidden_bodies), 1)
        index = next(iter(browser.hidden_bodies))
        self.assertAlmostEqual(list(manager.current_part.solids())[index].center().X, 8.)
        from UI.model.snapshot import save_snapshot, load_snapshot
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.fcad'
            save_snapshot(path, manager.current_part, [], [], feature_graph=graph, visibility=browser.export_visibility())
            _, _, _, restored_graph, visibility = load_snapshot(path, include_feature_graph=True, include_visibility=True)
            manager.parametric_graph = restored_graph
            manager.current_part = restored_graph.results[pattern.id]
            plotter.actors['main_part'] = actor_for(manager.current_part)
            browser.reset_visibility()
            browser.restore_visibility(manager, visibility)
            self.assertEqual(len(browser.hidden_bodies), 1)
            self.assertAlmostEqual(list(manager.current_part.solids())[next(iter(browser.hidden_bodies))].center().X, 8.)

    def test_browser_restores_visible_sketch_without_solid(self):
        from UI.model.feature_graph import Feature, FeatureGraph
        plane = Feature.create('Plane', 'datum_plane', {})
        sketch = Feature.create('Sketch', 'planar_sketch', {
            'entities': [{'type': 'circle', 'center': [0, 0], 'radius': 2}],
        }, plane.id)
        manager = SimpleNamespace(
            current_part=None, parametric_graph=FeatureGraph([plane, sketch]), profiles=[], construction_planes=[],
            plotter=SimpleNamespace(actors={}, render=Mock()), show_parametric_profiles=Mock(),
        )
        viewport = QtWidgets.QWidget()
        browser = ModelBrowser(viewport)
        browser.restore_visibility(manager, {
            'hidden_bodies': [], 'visible_planes': [], 'visible_sketch': sketch.id,
        })
        self.assertEqual(browser.topLevelItem(1).childCount(), 0)
        self.assertEqual(browser.topLevelItem(2).child(0).text(0), 'Sketch')
        manager.show_parametric_profiles.assert_called_once_with(sketch.id)
        viewport.close()

    def test_browser_toggles_and_saves_spatial_only_sketch_visibility(self):
        from UI.model.feature_graph import Feature, FeatureGraph
        plane = Feature.create('Plane', 'datum_plane', {})
        sketch = Feature.create('Spatial sketch', 'planar_sketch', {'entities': []}, plane.id)
        actor = vtk.vtkActor()
        manager = SimpleNamespace(
            current_part=None, parametric_graph=FeatureGraph([plane, sketch]), profiles=[], construction_planes=[],
            spatial_sketch_id=sketch.id, sketch_spatial_actors=[actor], sketch_editing=False,
            plotter=SimpleNamespace(actors={}, render=Mock()),
        )
        viewport = QtWidgets.QWidget()
        browser = ModelBrowser(viewport)
        browser.refresh(manager)
        item = browser.topLevelItem(2).child(0)
        self.assertEqual(item.checkState(0), QtCore.Qt.Checked)
        self.assertEqual(browser.export_visibility()['visible_sketch'], sketch.id)
        item.setCheckState(0, QtCore.Qt.Unchecked)
        self.assertFalse(actor.GetVisibility())
        self.assertIsNone(browser.export_visibility()['visible_sketch'])
        viewport.close()

    def test_arrow_shaft_can_be_dragged(self):
        self.assertAlmostEqual(segment_distance((50, 4), (0, 0), (100, 0)), 4.)
        viewport = QtWidgets.QWidget()
        viewport.resize(200, 200)
        renderer = SimpleNamespace(SetWorldPoint=Mock(), WorldToDisplay=Mock())
        renderer.GetDisplayPoint = lambda: renderer.SetWorldPoint.call_args.args[:3]
        viewport.renderer = renderer
        handle = SimpleNamespace(plotter=viewport, start_drag=Mock(), drag=Mock(), end_drag=Mock())
        hit = ArrowHitFilter(handle)
        hit.segments['primary'] = ((20, 100, 0), (180, 100, 0))
        QtTest.QTest.mousePress(viewport, QtCore.Qt.LeftButton, pos=QtCore.QPoint(80, 100))
        QtTest.QTest.mouseMove(viewport, QtCore.QPoint(100, 100))
        QtTest.QTest.mouseRelease(viewport, QtCore.Qt.LeftButton, pos=QtCore.QPoint(100, 100))
        handle.start_drag.assert_called_once_with('primary')
        self.assertGreaterEqual(handle.drag.call_count, 1)
        handle.end_drag.assert_called_once()
        hit.dispose()

    def test_navigation_cube_selects_face_edge_corner_and_home(self):
        import math
        import numpy as np
        import pyvista as pv

        viewport = QtWidgets.QWidget()
        viewport.resize(650, 460)
        self.addCleanup(viewport.close)
        plotter = pv.Plotter(off_screen=True)
        self.addCleanup(plotter.close)
        plotter.add_mesh(pv.Cube())
        plotter.enable_parallel_projection()
        plotter.view_isometric()
        camera = plotter.camera
        home_direction = tuple(camera.GetDirectionOfProjection())
        owner = LayoutManager.__new__(LayoutManager)
        owner.plotter = plotter
        owner.action_manager = ActionManager(None)
        owner.action_manager.plotter = plotter
        transition = CameraTransition(viewport, plotter, owner.action_manager)
        cube = NavigationCube(viewport, camera, transition.home)
        cube.viewRequested.connect(transition.orient)
        cube.dragRequested.connect(transition.drag)
        cube.gestureStarted.connect(transition.stop)
        viewport.show()
        self.app.processEvents()
        self.assertEqual(cube.geometry().right(), viewport.width() - 13)
        mesh = cube.scene.actor.GetMapper().GetInput()
        self.assertEqual((mesh.GetNumberOfPoints(), mesh.GetNumberOfPolys()), (24, 26))
        open_edges = vtk.vtkFeatureEdges()
        open_edges.SetInputData(mesh)
        open_edges.FeatureEdgesOff()
        open_edges.ManifoldEdgesOff()
        open_edges.Update()
        self.assertEqual(open_edges.GetOutput().GetNumberOfCells(), 0)
        self.assertEqual(cube._image().pixelColor(0, 0).alpha(), 0)
        self.assertFalse(cube.mask().contains(QtCore.QPoint(1, 1)))
        self.assertTrue(cube.testAttribute(QtCore.Qt.WA_NativeWindow))
        self.assertFalse(cube.testAttribute(QtCore.Qt.WA_TranslucentBackground))
        self.assertTrue(cube.mask().contains(cube.home.geometry().center()))
        self.assertFalse(cube.mask().contains(cube.home.geometry().topLeft()))

        def set_front():
            camera.SetFocalPoint(2, 3, 4)
            camera.SetPosition(2, -7, 4)
            camera.SetViewUp(0, 0, 1)
            camera.SetParallelScale(6)

        def click(direction):
            image = cube._image()
            vertices = next(vertices for target, vertices, _name in REGIONS if target == direction)
            renderer = cube.scene.renderer
            renderer.SetWorldPoint(*vertices.mean(axis=0), 1.)
            renderer.WorldToDisplay()
            x, y, _ = renderer.GetDisplayPoint()
            point = QtCore.QPoint(
                round(cube.canvas.x() + x * cube.canvas.width() / image.width()),
                round(cube.canvas.y() + (image.height() - 1 - y) * cube.canvas.height() / image.height()),
            )
            self.assertEqual(cube._target_at(QtCore.QPointF(point))[0], direction)
            start = camera.GetDirectionOfProjection()
            scale = math.sqrt(sum(value * value for value in direction))
            final = -np.asarray(direction) / scale
            QtTest.QTest.mouseClick(cube, QtCore.Qt.LeftButton, pos=point)
            self.assertEqual(camera.GetDirectionOfProjection(), start)
            self.assertEqual(transition.animation.state(), QtCore.QAbstractAnimation.Running)
            transition.animation.setCurrentTime(transition.animation.duration() // 2)
            if not np.allclose(start, final):
                self.assertFalse(np.allclose(camera.GetDirectionOfProjection(), start))
                self.assertFalse(np.allclose(camera.GetDirectionOfProjection(), final))
            transition.animation.setCurrentTime(transition.animation.duration())
            for actual, expected in zip(camera.GetDirectionOfProjection(), direction):
                self.assertAlmostEqual(actual, -expected / scale)
            self.assertEqual(camera.GetFocalPoint(), (2., 3., 4.))
            self.assertAlmostEqual(camera.GetDistance(), 10.)
            self.assertAlmostEqual(camera.GetParallelScale(), 6.)

        set_front()
        click((0, -1, 1))
        set_front()
        click((1, -1, 1))
        click((0, -1, 0))
        # An external camera orbit changes the targets shown by the cube.
        camera.SetPosition(12, 3, 4)
        camera.SetViewUp(0, 0, 1)
        right_face = QtCore.QPointF(71, 85)
        self.assertEqual(cube._target_at(right_face)[0], (1, 0, 0))
        click((1, 0, 0))

        def drag():
            start, end = QtCore.QPointF(71, 85), QtCore.QPointF(110, 54)
            for kind, point, changed, held in (
                    (QtCore.QEvent.MouseButtonPress, start, QtCore.Qt.LeftButton, QtCore.Qt.LeftButton),
                    (QtCore.QEvent.MouseMove, end, QtCore.Qt.NoButton, QtCore.Qt.LeftButton),
                    (QtCore.QEvent.MouseButtonRelease, end, QtCore.Qt.LeftButton, QtCore.Qt.NoButton)):
                self.app.sendEvent(cube, QtGui.QMouseEvent(kind, point, point, changed, held, QtCore.Qt.NoModifier))

        drag()
        dragged = camera.GetDirectionOfProjection()
        self.assertFalse(np.allclose(dragged, (-1., 0., 0.)))
        self.assertEqual(transition.animation.state(), QtCore.QAbstractAnimation.Stopped)
        self.assertEqual(camera.GetFocalPoint(), (2., 3., 4.))
        self.assertAlmostEqual(camera.GetDistance(), 10.)
        self.assertAlmostEqual(camera.GetParallelScale(), 6.)

        transition.orient((0, 1, 0))
        transition.animation.setCurrentTime(70)
        camera.SetParallelScale(7)
        self.assertEqual(transition.animation.state(), QtCore.QAbstractAnimation.Stopped)
        set_front()
        owner.action_manager.sketch_editing = True
        drag()
        QtTest.QTest.mouseClick(cube, QtCore.Qt.LeftButton, pos=QtCore.QPoint(71, 52))
        cube.home.click()
        self.assertEqual(camera.GetDirectionOfProjection(), (0., 1., 0.))
        self.assertEqual(transition.animation.state(), QtCore.QAbstractAnimation.Stopped)
        owner.action_manager.sketch_editing = False
        cube.home.click()
        self.assertEqual(camera.GetDirectionOfProjection(), (0., 1., 0.))
        transition.animation.setCurrentTime(transition.animation.duration())
        np.testing.assert_allclose(camera.GetDirectionOfProjection(), home_direction)
        viewport.resize(500, 380)
        self.app.processEvents()
        self.assertEqual(cube.geometry().right(), viewport.width() - 13)

    def test_extrude_panel_fits_without_scrollbars(self):
        viewport = QtWidgets.QFrame()
        viewport.resize(900, 700)
        owner = LayoutManager.__new__(LayoutManager)
        owner.render_container = viewport
        owner.show_scene_context_menu = Mock()
        owner.browser_toggle = QtWidgets.QToolButton(viewport)
        owner.view_menu = QtWidgets.QMenu(viewport)
        owner.command_actions = {key: owner.view_menu.addAction(key) for key in ('fit', 'projection')}
        with patch('UI.main.content.QtInteractor', side_effect=lambda parent: QtWidgets.QWidget(parent)):
            LayoutManager.add_3d_viewport(owner)
        plotter = SimpleNamespace(actors={}, remove_actor=Mock(), render=Mock())
        manager = SimpleNamespace(plotter=plotter, current_part=None,
                                  build_extrusion=Mock(side_effect=ValueError('preview unavailable')))
        dialog = ExtrudeDialog(manager, 0)
        viewport.show()
        LayoutManager.show_operation_panel(owner, dialog)
        self.app.processEvents()
        owner.operation_panel.reposition()
        self.app.processEvents()
        self.assertFalse(owner.operation_scroll.horizontalScrollBar().isVisible())
        self.assertFalse(owner.operation_scroll.verticalScrollBar().isVisible())
        self.assertEqual(dialog.distance.buttonSymbols(), QtWidgets.QAbstractSpinBox.NoButtons)
        dialog.reject()
        viewport.close()

    def test_viewport_mouse_pan_orbit_and_cursor_zoom(self):
        import numpy as np
        import pyvista as pv

        plotter = pv.Plotter(off_screen=True, window_size=(600, 420))
        self.addCleanup(plotter.close)
        plotter.add_mesh(pv.Cube(), name='main_part')
        viewport = QtWidgets.QWidget()
        self.addCleanup(viewport.close)
        viewport.resize(600, 420)
        viewport.camera, viewport.renderer = plotter.camera, plotter.renderer
        viewport.ren_win, viewport.actors, viewport.render = plotter.ren_win, plotter.actors, plotter.render
        manager = SimpleNamespace(sketch_editing=False, update_grid_lod=Mock())
        navigation = MouseNavigation(viewport, manager)
        camera = plotter.camera
        renderer = plotter.renderer
        camera.SetPosition(0, 0, 8)
        camera.SetFocalPoint(0, 0, 0)
        camera.SetViewUp(0, 1, 0)
        camera.ParallelProjectionOn()

        def drag(button, modifier=QtCore.Qt.NoModifier):
            start, end = QtCore.QPointF(240, 180), QtCore.QPointF(295, 155)
            for kind, point, changed, held in (
                    (QtCore.QEvent.MouseButtonPress, start, button, button),
                    (QtCore.QEvent.MouseMove, end, QtCore.Qt.NoButton, button),
                    (QtCore.QEvent.MouseButtonRelease, end, button, QtCore.Qt.NoButton)):
                event = QtGui.QMouseEvent(kind, point, point, changed, held, modifier)
                self.app.sendEvent(viewport, event)

        def display(world):
            renderer.SetWorldPoint(*world, 1.)
            renderer.WorldToDisplay()
            return np.asarray(renderer.GetDisplayPoint()[:2])

        def wheel(pixel):
            point = QtCore.QPointF(pixel[0], viewport.height() - 1 - pixel[1])
            event = QtGui.QWheelEvent(point, point, QtCore.QPoint(), QtCore.QPoint(0, 120), QtCore.Qt.NoButton,
                                      QtCore.Qt.NoModifier, QtCore.Qt.NoScrollPhase, False)
            self.app.sendEvent(viewport, event)

        old = display((0, 0, 0))
        drag(QtCore.Qt.MiddleButton)
        np.testing.assert_allclose(display((0, 0, 0)) - old, (55, 25), atol=1e-7)
        direction = camera.GetDirectionOfProjection()
        drag(QtCore.Qt.MiddleButton, QtCore.Qt.ShiftModifier)
        self.assertFalse(np.allclose(camera.GetDirectionOfProjection(), direction))
        self.assertIsNone(navigation.previous)

        camera.ParallelProjectionOff()
        camera.SetFocalPoint(0, 0, 0)
        camera.SetPosition(0, 0, 8)
        camera.SetViewUp(0, 1, 0)
        surface_point = (.25, .2, .5)
        anchor = display(surface_point)
        distance = camera.GetDistance()
        wheel(anchor)
        np.testing.assert_allclose(display(surface_point), anchor, atol=1e-7)
        self.assertLess(camera.GetDistance(), distance)

        manager.sketch_editing = True
        normal = np.array([1., 2., 3.]) / np.sqrt(14)
        focus = np.array([2., 3., 4.])
        camera.SetFocalPoint(*focus)
        camera.SetPosition(*(focus + 8 * normal))
        camera.SetViewUp(0, 0, 1)
        camera.OrthogonalizeViewUp()
        camera.ParallelProjectionOn()
        old_up = camera.GetViewUp()
        drag(QtCore.Qt.MiddleButton, QtCore.Qt.ShiftModifier)
        self.assertFalse(np.allclose(camera.GetFocalPoint(), focus))
        sketch_point = focus + .75 * np.cross(normal, old_up)
        anchor = display(sketch_point)
        scale = camera.GetParallelScale()
        wheel(anchor)
        np.testing.assert_allclose(display(sketch_point), anchor, atol=1e-7)
        np.testing.assert_allclose(camera.GetDirectionOfProjection(), -normal, atol=1e-7)
        np.testing.assert_allclose(camera.GetViewUp(), old_up, atol=1e-7)
        self.assertAlmostEqual(np.dot(np.asarray(camera.GetFocalPoint()) - focus, normal), 0.)
        self.assertTrue(camera.GetParallelProjection())
        self.assertLess(camera.GetParallelScale(), scale)
        self.assertEqual(manager.update_grid_lod.call_count, 2)

    def test_extrude_panel_passes_multiple_profiles_and_scene_target_to_preview(self):
        plotter = SimpleNamespace(actors={}, remove_actor=Mock(), render=Mock())
        target = {'kind': 'face', 'index': 2}
        manager = SimpleNamespace(
            plotter=plotter, current_part=None, build_extrusion=Mock(side_effect=ValueError('test preview')),
            _select_profile=Mock(return_value=True), extrusion_profile_indices=Mock(return_value=(0, 1)),
            pick_extrusion_target=Mock(return_value=(target, 2)),
        )
        dialog = ExtrudeDialog(manager, 0)
        dialog.scene_click()
        dialog.extent.setCurrentIndex(dialog.extent.findData('to_object'))
        dialog.pick_target.click()
        self.assertFalse(dialog.buttons.button(QtWidgets.QDialogButtonBox.Ok).isEnabled())
        dialog.scene_click()
        dialog.reverse.setChecked(True)
        dialog.flush_preview()
        manager._select_profile.assert_called_once_with(toggle=True)
        manager.pick_extrusion_target.assert_called_once_with('face')
        manager.build_extrusion.assert_called_with((0, 1), 0, 'new_body', 0, 0, False, None, 'to_object', target, True)
        self.assertEqual(dialog.profile_count.text(), '2 selected')
        dialog.reject()

    def test_revolve_panel_passes_scene_axis_to_preview(self):
        plotter = SimpleNamespace(actors={}, remove_actor=Mock(), render=Mock())
        selection = {'kind': 'edge', 'index': 5}
        manager = SimpleNamespace(
            plotter=plotter, build_revolution=Mock(side_effect=ValueError('test preview')),
            pick_revolution_axis=Mock(return_value=(selection, 'Edge 6')),
        )
        dialog = RevolveDialog(manager, 0)
        dialog.axis.setCurrentIndex(dialog.axis.findData('model_geometry'))
        dialog.pick_axis.click()
        dialog.scene_click()
        dialog.flush_preview()
        manager.pick_revolution_axis.assert_called_once_with(0, False)
        manager.build_revolution.assert_called_with(0, selection, 360, 'new_body')
        self.assertIn('Edge 6', dialog.pick_axis.text())
        dialog.reject()

    def test_preview_coalesces_updates_and_accept_flushes_latest(self):
        dialog = ExtrudeDialog.__new__(ExtrudeDialog)
        QtWidgets.QDialog.__init__(dialog)
        dialog.buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok)
        dialog.result_shape = object()
        dialog.update_preview = Mock(side_effect=lambda: setattr(dialog, 'result_shape', 'latest'))
        dialog.request_preview()
        dialog.request_preview()
        self.assertFalse(dialog.buttons.button(QtWidgets.QDialogButtonBox.Ok).isEnabled())
        dialog.update_preview.assert_not_called()
        dialog.accept()
        dialog.update_preview.assert_called_once()
        self.assertEqual(dialog.result_shape, 'latest')
        self.assertEqual(dialog.result(), QtWidgets.QDialog.Accepted)

    def test_accept_applies_uncommitted_numeric_expression(self):
        dialog = ExtrudeDialog.__new__(ExtrudeDialog)
        QtWidgets.QDialog.__init__(dialog)
        dialog.buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok)
        field = ExtrudeDialog.number(1., -100., 100.)
        field.setParent(dialog)
        dialog.result_shape = object()
        field.valueChanged.connect(dialog.request_preview)
        dialog.update_preview = Mock(side_effect=lambda: setattr(dialog, 'result_shape', field.value()))
        field.lineEdit().setText('10 / 2')
        dialog.accept()
        self.assertEqual(dialog.result_shape, 5.)
        self.assertEqual(dialog.result(), QtWidgets.QDialog.Accepted)

    def test_clipping_changes_do_not_trigger_arrow_rebuild(self):
        camera = vtk.vtkCamera()
        handle = SimpleNamespace(plotter=SimpleNamespace(camera=camera,
            ren_win=SimpleNamespace(GetSize=lambda: (800, 600))), sync=Mock())
        observe_arrow_camera(handle)
        camera.SetClippingRange(.01, 500.)
        self.assertFalse(handle._camera_timer.isActive())
        camera.SetPosition(4, 3, 2)
        self.assertTrue(handle._camera_timer.isActive())
        handle._camera_timer.stop()
        camera.RemoveObserver(handle._camera_observer)

    def test_drag_updates_are_coalesced_and_can_be_cancelled(self):
        handle = SimpleNamespace(dialog=SimpleNamespace(request_preview=Mock()), sync=Mock())
        schedule_handle_sync(handle)
        schedule_handle_sync(handle)
        self.assertTrue(handle._drag_sync_timer.isActive())
        handle.sync.assert_not_called()
        stop_handle_sync(handle)
        self.assertFalse(handle._drag_sync_timer.isActive())

    def test_drag_end_flushes_pending_preview_without_duplicate_sync(self):
        handle = SimpleNamespace(dialog=SimpleNamespace(flush_preview=Mock(return_value=True)), sync=Mock())
        finish_handle_drag(handle)
        handle.dialog.flush_preview.assert_called_once()
        handle.sync.assert_not_called()
        handle.dialog.flush_preview.return_value = False
        finish_handle_drag(handle)
        handle.sync.assert_called_once()


if __name__ == '__main__':
    unittest.main()
