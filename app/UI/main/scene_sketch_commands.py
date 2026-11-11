"""Constraint and NURBS commands operating on the 3-D scene sketch candidate."""
from PySide6 import QtCore, QtWidgets
from UI.main.parametric_sketch_dialog import ParametricSketchDialog
from UI.model.parametric_sketch import circle_radius_name


class _Point:
    def __init__(self, position): self.position = position
    def pos(self): return QtCore.QPointF(*map(float, self.position))


class _Status:
    def __init__(self, manager): self.manager = manager
    def setText(self, text): self.manager.status_callback(text)


class _ScenePlaceholder:
    def addPath(self, *_args): return None


class SceneSketchCommands(ParametricSketchDialog):
    """Reuse parameter dialogs without constructing or displaying a second canvas."""
    def __init__(self, manager, parent=None, panel_inputs=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.panel_inputs = panel_inputs or QtWidgets.QInputDialog
        self.manager = manager
        self.graph = manager.history_edit_graph
        self.feature_id = manager.history_edit_id
        self.message = _Status(manager)
        self.scene = _ScenePlaceholder()
        self.refresh_geometry()

    def radius(self, index):
        return self.graph.results[self.feature_id].constraint_status.radii[circle_radius_name(index)]

    def refresh_geometry(self):
        self.manager.history_edit_graph = self.graph
        feature = self.feature()
        state = self.graph.results[self.feature_id].constraint_status
        self.handles = {name:_Point(point) for name, point in state.points.items()}
        self.lines, self.circles, self.arcs, self.arc_entities = [], [], [], []
        self.ellipses, self.splines, self.periodic_splines = [], [], []
        self.slots, self.polyline_slots, self.arc_slots, self.spline_slots, self.composite_slots = [], [], [], [], []
        entities = feature.parameters['entities']
        for index, entity in enumerate(entities):
            kind = entity.get('type')
            if kind == 'line':
                self.lines.extend((None, a, b) for a, b in zip(entity['points'], entity['points'][1:]))
            elif kind == 'circle': self.circles.append((None, entity['center'], None, index))
            elif kind == 'arc': self.arc_entities.append((index, entity['points']))
            elif kind == 'ellipse': self.ellipses.append((None, entity['center'], index))
            elif kind in ('spline','fit_spline','nurbs'): self.splines.append((None, entity['points'], index))
            elif kind == 'periodic_nurbs': self.periodic_splines.append((None, entity['points'], index))
            elif kind == 'slot': self.slots.append((None, entity['points'], index))
            elif kind == 'polyline_slot': self.polyline_slots.append((None, entity['points'], index))
            elif kind == 'arc_slot': self.arc_slots.append((None, entity['points'], index))
            elif kind == 'spline_slot': self.spline_slots.append((None, entities[entity['curve']]['points'], index))
            elif kind == 'composite_path_slot': self.composite_slots.append((None, entity['curves'], index))
        self.manager.render_history_sketch()
