import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))

from PySide6 import QtCore, QtWidgets
from UI.main.content import LayoutManager
from UI.main.panel_inputs import PanelInputs
from UI.main.operation_panel import OperationPanel
from UI.main.extrude_dialog import ExtrudeDialog
from UI.main.extrude_handle import observe_arrow_camera, schedule_handle_sync, stop_handle_sync
from UI.model.ActionManager import ActionManager
import vtk


class OperationPanelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

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
        navigation = QtCore.QRect(660, 0, 140, 140)
        panel.reposition()
        self.assertFalse(panel.geometry().intersects(navigation))

    def test_extrude_panel_fits_without_scrollbars(self):
        viewport = QtWidgets.QFrame()
        viewport.resize(900, 700)
        owner = SimpleNamespace(render_container=viewport)
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


if __name__ == '__main__':
    unittest.main()
