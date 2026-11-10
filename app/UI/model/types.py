from dataclasses import dataclass, field
from typing import Union, TYPE_CHECKING

from pyvistaqt import QtInteractor
import pyvista as pv

if TYPE_CHECKING:
    from UI.model.ActionManager import ActionManager

@dataclass
class SkecthObj():
    plotter:Union[QtInteractor, pv.Plotter]
    sketch_data:list
    is_drawing = False
    sketch_instance:'ActionManager'
    render_obj:dict[str, list[pv.Actor]]
    points:list = field(default_factory=list)