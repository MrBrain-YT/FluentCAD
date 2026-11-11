"""Camera navigation for the sketch viewport."""
import vtk


class SketchCameraStyle(vtk.vtkInteractorStyleTrackballCamera):
    """Keep left drags in the sketch plane; Ctrl + left drag rotates."""
    def __init__(self, manager):
        super().__init__()
        self.manager = manager
        self.rotating = False
        self.AddObserver('LeftButtonPressEvent', self._press)
        self.AddObserver('LeftButtonReleaseEvent', self._release)

    def _press(self, *_):
        interactor = self.GetInteractor()
        if interactor is not None and interactor.GetControlKey():
            x, y = interactor.GetEventPosition()
            self.FindPokedRenderer(x, y)
            if self.GetCurrentRenderer() is not None:
                self.rotating = True
                self.StartRotate()

    def _release(self, *_):
        if self.rotating:
            self.EndRotate()
            self.rotating = False
        self.manager._finish_history_drag()
