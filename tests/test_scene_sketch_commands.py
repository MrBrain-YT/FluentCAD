import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from PySide6.QtWidgets import QApplication, QInputDialog, QDialog, QLineEdit
from UI.model.ActionManager import ActionManager
from UI.model.feature_graph import Feature, FeatureGraph
from UI.main.scene_sketch_commands import SceneSketchCommands


class SceneCommandsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def commands(self):
        plane = Feature.create('Plane','datum_plane',{})
        sketch = Feature.create('Sketch','planar_sketch', {
            'points':{'c':[0,0], 'a':[4,0], 'b':[5,1], 'd':[6,0]},
            'entities':[{'type':'circle','center':'c','radius':2},
                        {'type':'nurbs','points':['a','b','d'],'degree':2,'weights':[1,1,1],
                         'knots':[0,0,0,1,1,1],'construction':True}]},plane.id)
        graph = FeatureGraph([plane,sketch], {'w':.7})
        manager = ActionManager(None)
        manager.history_edit_id, manager.history_edit_graph = sketch.id, graph
        manager.render_history_sketch, manager.status_callback = Mock(), Mock()
        return SceneSketchCommands(manager), manager, graph, sketch

    def test_radius_constraint_updates_candidate_only(self):
        commands, manager, original, sketch = self.commands()
        with patch.object(QInputDialog,'getItem',side_effect=[('Radius',True),('Circle 1 (center c)',True)]), \
             patch.object(QInputDialog,'getText',return_value=('3',True)):
            commands.create_constraint()
        self.assertAlmostEqual(manager.history_edit_graph.results[sketch.id].constraint_status.radii['@circle:0'],3)
        self.assertNotIn('constraints', original.features[1].parameters)

    def test_nurbs_parameter_dialog_keeps_expression(self):
        commands, manager, original, _ = self.commands()
        def accept(dialog):
            field = next(item for item in dialog.findChildren(QLineEdit) if item.text() == '1, 1, 1')
            field.setText('1, w, 1')
            return QDialog.Accepted
        with patch.object(QInputDialog,'getItem',return_value=('nurbs 2',True)), patch.object(QDialog,'exec',accept):
            commands.edit_nurbs_parameters()
        self.assertEqual(manager.history_edit_graph.features[1].parameters['entities'][1]['weights'],[1,'w',1])
        self.assertEqual(original.features[1].parameters['entities'][1]['weights'],[1,1,1])

    def test_invalid_nurbs_parameters_do_not_change_candidate(self):
        commands, manager, original, _ = self.commands()
        def accept(dialog):
            field = next(item for item in dialog.findChildren(QLineEdit) if item.text() == '1, 1, 1')
            field.setText('1, -1, 1')
            return QDialog.Accepted
        with patch.object(QInputDialog,'getItem',return_value=('nurbs 2',True)), patch.object(QDialog,'exec',accept):
            commands.edit_nurbs_parameters()
        self.assertIs(manager.history_edit_graph, original)
        self.assertEqual(original.features[1].parameters['entities'][1]['weights'],[1,1,1])
        self.assertIn('positive', manager.status_callback.call_args.args[0])

    def test_slot_command_reuses_scene_candidate(self):
        commands, manager, original, _ = self.commands()
        with patch.object(QInputDialog,'getItem',return_value=('Spline 2: a → d',True)), \
             patch.object(QInputDialog,'getText',return_value=('.2',True)):
            commands.create_spline_slot()
        self.assertEqual(manager.history_edit_graph.features[1].parameters['entities'][-1]['type'],'spline_slot')
        self.assertEqual(len(original.features[1].parameters['entities']),2)


if __name__ == '__main__':
    unittest.main()
