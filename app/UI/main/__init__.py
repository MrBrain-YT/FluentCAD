import sys
from pathlib import Path

from PySide6 import QtWidgets, QtGui, QtCore



def open_window():
    app = QtWidgets.QApplication(sys.argv)
    splash = show_splash_screen()
    window = QtWidgets.QMainWindow()
    window.setWindowTitle("Fluent CAD")
    # Create central widget
    central_widget = QtWidgets.QWidget()
    # load app libs
    from UI.main.content import LayoutManager
    central_widget = LayoutManager(window, central_widget).central_widget
    window.setCentralWidget(central_widget)
    window.showMaximized()
    close_splash_screen(splash)
    sys.exit(app.exec())
    
def show_splash_screen():
    splash_pix = QtGui.QPixmap(str(Path(__file__).resolve().parent.parent / 'images' / 'splash.png'))
    screen = QtWidgets.QApplication.primaryScreen()
    screen_geometry = screen.geometry()
    target_width = screen_geometry.width() // 3
    scaled_pix = splash_pix.scaledToWidth(
        target_width, 
        QtCore.Qt.TransformationMode.SmoothTransformation
    )
    
    splash = QtWidgets.QSplashScreen(scaled_pix)
    # find center in sceen
    splash_x = screen_geometry.center().x() - scaled_pix.width() // 2
    splash_y = screen_geometry.center().y() - scaled_pix.height() // 2
    splash.move(splash_x, splash_y)
    
    splash.show()
    return splash

def close_splash_screen(splash):
    splash.close()
