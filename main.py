
import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from gui.main_window import MainWindow
from utils.logger import app_logger


def main():
    app = QApplication(sys.argv)
    app.setApplicationName('Video to 3D Visualization &amp; Editing System')
    app.setOrganizationName('3D Vision')
    
    window = MainWindow()
    window.show()
    
    app_logger.info('Application started')
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

