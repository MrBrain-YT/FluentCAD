from PySide6 import QtWidgets, QtCore, QtGui

class ActionManagerUI:
    
    def __init__(self, tool_tabs_widgets:dict) -> None:
        self.tool_tabs_widgets = tool_tabs_widgets
    
    def activate_scetch_mode(self, tool_tabs:QtWidgets.QTabWidget, tab_name:str) -> None:
        widget = self.tool_tabs_widgets[tab_name]
        tool_tabs.setTabVisible(tool_tabs.indexOf(widget), True)
        tool_tabs.setCurrentWidget(widget)
