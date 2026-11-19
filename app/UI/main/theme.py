"""Shared palette and styles for CAD controls and viewport overlays."""
import os
from pathlib import Path

from PySide6 import QtGui, QtWidgets


LIGHT = {
    'window': '#eef2f5', 'panel': '#f7f9fb', 'well': '#e9eef3', 'field': '#ffffff',
    'text': '#1d2b39', 'muted': '#536473', 'border': '#bec9d3', 'accent': '#8b4418',
    'selection': '#245980', 'hover': '#e1edf6', 'error': '#962d29',
    'finish_text': '#246447', 'finish_background': '#e6f4ec',
    'canvas': '#c4cdd9', 'canvas_top': '#e4ebf1',
}
DARK = {
    'window': '#1a2330', 'panel': '#232e3c', 'well': '#303e4e', 'field': '#151e2a',
    'text': '#f1f4f7', 'muted': '#b9c8d7', 'border': '#546881', 'accent': '#ffc789',
    'selection': '#2b5c82', 'hover': '#3a4c62', 'error': '#ffaaa1',
    'finish_text': '#b1e7cd', 'finish_background': '#274338',
    'canvas': '#253345', 'canvas_top': '#42566b',
}


def theme_palette(colors):
    palette = QtGui.QPalette()
    roles = QtGui.QPalette.ColorRole
    mapping = {
        roles.Window: 'window', roles.WindowText: 'text', roles.Base: 'field',
        roles.AlternateBase: 'well', roles.Text: 'text', roles.Button: 'well',
        roles.ButtonText: 'text', roles.ToolTipBase: 'well', roles.ToolTipText: 'text',
        roles.Highlight: 'selection', roles.HighlightedText: 'text', roles.Link: 'accent',
        roles.PlaceholderText: 'muted', roles.Mid: 'border', roles.Dark: 'border',
        roles.Light: 'muted', roles.BrightText: 'error',
    }
    for role, name in mapping.items():
        palette.setColor(role, QtGui.QColor(colors[name]))
    palette.setColor(roles.HighlightedText, QtGui.QColor('#ffffff'))
    for role in (roles.Text, roles.WindowText, roles.ButtonText):
        palette.setColor(QtGui.QPalette.Disabled, role, QtGui.QColor(colors['muted']))
    return palette


def apply_theme(window, dark):
    if os.name == 'nt' and not QtGui.QFontDatabase.families():
        font_file = Path(os.environ.get('SystemRoot', r'C:\Windows')) / 'Fonts' / 'segoeui.ttf'
        font_id = QtGui.QFontDatabase.addApplicationFont(str(font_file))
        families = QtGui.QFontDatabase.applicationFontFamilies(font_id)
        if families:
            font = QtWidgets.QApplication.font()
            font.setFamily(families[0])
            QtWidgets.QApplication.setFont(font)
    c = DARK if dark else LIGHT
    palette = theme_palette(c)
    QtWidgets.QApplication.setPalette(palette)
    window.setPalette(palette)
    window.setStyleSheet(f'''
        #applicationBar {{ background: {c['window']}; border-bottom: 1px solid {c['border']}; }}
        QLabel#documentLabel {{ color: {c['text']}; font-weight: 600; padding: 1px 8px; }}
        QLabel[role="sectionTitle"] {{ color: {c['muted']}; font-weight: 600; padding: 0 5px; }}
        QToolButton[chrome="true"] {{ color: {c['text']}; background: transparent; border: 1px solid transparent;
            border-radius: 5px; min-width: 26px; min-height: 24px; padding: 2px 4px; }}
        QToolButton[chrome="true"]:hover, QToolButton[chrome="true"]:checked {{
            background: {c['hover']}; border-color: {c['border']};
        }}
        QToolButton[chrome="true"]:focus, QToolButton[ribbon="true"]:focus {{ border-color: {c['accent']}; }}
        QTabWidget, QTabBar {{ background: {c['window']}; }}
        QTabWidget::pane {{ border: 0; background: {c['window']}; }}
        QTabBar::tab {{ color: {c['muted']}; background: {c['window']}; padding: 6px 18px; border: 0; }}
        QTabBar::tab:selected {{ color: {c['accent']}; background: {c['panel']}; border-bottom: 2px solid {c['accent']};
            font-weight: 600; }}
        QScrollArea#ribbonScroll {{ background: {c['panel']}; }}
        QFrame#ribbon {{ background: {c['panel']}; border-bottom: 1px solid {c['border']}; }}
        QToolButton[ribbon="true"] {{ border: 1px solid transparent; border-radius: 3px; padding: 3px; background: transparent; }}
        QToolButton[ribbon="true"]:hover {{ border-color: {c['border']}; background: {c['hover']}; }}
        QToolButton[groupLabel="true"] {{ color: {c['muted']}; border: 0; background: transparent; padding: 2px 7px; font-size: 12px; }}
        QToolButton[groupLabel="true"]::menu-indicator {{ image: none; }}
        QToolButton[groupLabel="true"]:hover {{ color: {c['accent']}; background: {c['hover']}; }}
        QToolButton#finishSketch {{ color: {c['finish_text']}; background: {c['finish_background']};
            border: 1px solid {c['border']}; padding: 3px; border-radius: 3px; }}
        QToolButton#finishSketch:hover {{ background: {c['hover']}; }}
        QMenu, QComboBox QAbstractItemView {{ background: {c['field']}; color: {c['text']}; border: 1px solid {c['border']}; }}
        QMenu {{ padding: 4px; }}
        QMenu::item {{ padding: 6px 24px 6px 10px; }}
        QMenu::item:selected, QListWidget::item:selected, QTreeWidget::item:selected {{
            background: {c['selection']}; color: #ffffff;
        }}
        QFrame#operationPanel, QFrame#sceneValueEditor, QFrame#viewportNavigation {{
            background: {c['panel']}; border: 1px solid {c['border']}; border-radius: 6px;
        }}
        QToolButton#operationTitle {{ background: {c['well']}; color: {c['text']}; border: 0;
            border-bottom: 1px solid {c['border']}; text-align: left; padding: 3px 5px; font-weight: bold; }}
        QWidget#operationContent, #operationPanel QScrollArea, #operationPanel QDialog {{ background: {c['panel']}; }}
        QLineEdit, QAbstractSpinBox, QComboBox, QPlainTextEdit {{
            background: {c['field']}; color: {c['text']}; border: 1px solid {c['border']}; padding: 2px;
            selection-background-color: {c['selection']}; selection-color: #ffffff;
        }}
        QLineEdit:focus, QAbstractSpinBox:focus, QComboBox:focus, QPlainTextEdit:focus {{ border-color: {c['accent']}; }}
        QLineEdit:disabled, QAbstractSpinBox:disabled, QComboBox:disabled, QMenu::item:disabled {{
            color: {c['muted']}; background: {c['well']};
        }}
        QPushButton {{ background: {c['well']}; color: {c['text']}; border: 1px solid {c['border']}; padding: 3px 10px; }}
        QPushButton:hover {{ background: {c['hover']}; border-color: {c['accent']}; }}
        QPushButton:disabled {{ color: {c['muted']}; }}
        QLabel[role="error"] {{ color: {c['error']}; }}
        QLabel#sceneValueGrip {{ color: {c['muted']}; }}
        QTreeWidget#modelBrowser {{ background: {c['panel']}; color: {c['text']}; border: 1px solid {c['border']}; }}
        QTreeWidget#modelBrowser::item {{ min-height: 23px; }}
        QTreeWidget#modelBrowser::indicator:unchecked {{ border: 1px solid {c['border']}; background: {c['field']}; }}
        QHeaderView::section {{ background: {c['well']}; color: {c['text']}; border: 0; padding: 8px; font-weight: 600; }}
        QFrame#timelineBar {{ background: {c['panel']}; border-top: 1px solid {c['border']}; }}
        QListWidget#solidTimeline {{ background: {c['window']}; color: {c['text']}; border: 0; }}
        QListWidget#solidTimeline::item {{ padding: 3px 8px; border-right: 1px solid {c['border']}; }}
        QListWidget#solidTimeline::item:hover:!selected {{ background: {c['hover']}; }}
        QSplitter::handle {{ background: {c['border']}; width: 3px; }}
        QStatusBar {{ background: {c['window']}; color: {c['text']}; }}
        QToolTip {{ background: {c['well']}; color: {c['text']}; border: 1px solid {c['border']}; }}
        QFrame[role="ribbonSeparator"] {{ color: {c['border']}; }}
    ''')
