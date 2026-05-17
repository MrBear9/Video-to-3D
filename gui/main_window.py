
from PyQt5.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, 
                             QSplitter, QStatusBar, QMenuBar, QAction, QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
import sys
from pathlib import Path

from gui.video_panel import VideoPanel
from gui.viewport_3d import Viewport3D
from gui.control_panel import ControlPanel, SceneTree
from core.model_io import ModelIO
from core.material_editor import MaterialEditor
from core.lighting_system import LightingSystem
from utils.logger import app_logger


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.video_processor = None
        self.current_model = None
        self.material_editor = MaterialEditor()
        self.lighting_system = LightingSystem()
        
        self.init_ui()
        self.init_menu()
        self.connect_signals()
        app_logger.info('Application initialized')
    
    def init_ui(self):
        self.setWindowTitle('Video to 3D Visualization &amp; Editing System')
        self.setMinimumSize(1200, 800)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        
        left_splitter = QSplitter(Qt.Vertical)
        self.video_panel = VideoPanel()
        self.scene_tree = SceneTree()
        left_splitter.addWidget(self.video_panel)
        left_splitter.addWidget(self.scene_tree)
        
        center_splitter = QSplitter(Qt.Vertical)
        self.viewport = Viewport3D()
        center_splitter.addWidget(self.viewport)
        
        right_splitter = QSplitter(Qt.Vertical)
        self.control_panel = ControlPanel()
        right_splitter.addWidget(self.control_panel)
        
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(left_splitter)
        main_splitter.addWidget(center_splitter)
        main_splitter.addWidget(right_splitter)
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 3)
        main_splitter.setStretchFactor(2, 1)
        
        main_layout.addWidget(main_splitter)
        central_widget.setLayout(main_layout)
        
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage('Ready')
    
    def init_menu(self):
        menubar = self.menuBar()
        
        file_menu = menubar.addMenu('&amp;File')
        
        open_video_action = QAction('&amp;Open Video', self)
        open_video_action.setShortcut('Ctrl+O')
        open_video_action.triggered.connect(self.video_panel.load_video)
        file_menu.addAction(open_video_action)
        
        import_model_action = QAction('&amp;Import Model', self)
        import_model_action.triggered.connect(self.import_model)
        file_menu.addAction(import_model_action)
        
        export_model_action = QAction('&amp;Export Model', self)
        export_model_action.triggered.connect(self.export_model)
        file_menu.addAction(export_model_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('E&amp;xit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        help_menu = menubar.addMenu('&amp;Help')
        about_action = QAction('&amp;About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def connect_signals(self):
        self.video_panel.video_loaded.connect(self.on_video_loaded)
        self.viewport.model_loaded.connect(self.on_model_loaded)
        self.control_panel.import_model_clicked.connect(self.import_model)
        self.control_panel.export_model_clicked.connect(self.export_model)
        self.control_panel.material_changed.connect(self.on_material_changed)
        self.control_panel.light_changed.connect(self.on_light_changed)
    
    def on_video_loaded(self, processor):
        self.video_processor = processor
        self.status_bar.showMessage(f'Video loaded: {processor.video_path}')
        app_logger.info(f'Video loaded: {processor.video_path}')
    
    def on_model_loaded(self, model):
        info = ModelIO.get_model_info(model)
        self.scene_tree.add_item(f'Model {len(self.scene_tree.tree.findItems("", Qt.MatchContains)) + 1}', model)
        self.status_bar.showMessage(f'Model loaded: {info["type"]}')
        app_logger.info(f'Model loaded: {info}')
    
    def import_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 'Import 3D Model', '', 
            '3D Models (*.obj *.ply *.stl *.glb *.gltf);;All Files (*)'
        )
        if file_path:
            try:
                model = ModelIO.import_model(file_path)
                self.current_model = model
                self.viewport.add_model(model)
                app_logger.info(f'Model imported: {file_path}')
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Failed to import model: {str(e)}')
                app_logger.error(f'Failed to import model: {e}')
    
    def export_model(self):
        if not self.current_model:
            QMessageBox.warning(self, 'Warning', 'No model to export')
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, 'Export 3D Model', '', 
            'OBJ (*.obj);;PLY (*.ply);;STL (*.stl);;GLB (*.glb)'
        )
        if file_path:
            try:
                success = ModelIO.export_model(self.current_model, file_path)
                if success:
                    QMessageBox.information(self, 'Success', 'Model exported successfully')
                    app_logger.info(f'Model exported: {file_path}')
                else:
                    QMessageBox.critical(self, 'Error', 'Failed to export model')
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Failed to export model: {str(e)}')
                app_logger.error(f'Failed to export model: {e}')
    
    def on_material_changed(self, material):
        if self.current_model and hasattr(self.current_model, 'vertices'):
            try:
                self.current_model = self.material_editor.apply_material(self.current_model, material)
                self.viewport.update()
                self.status_bar.showMessage(f'Material applied: {material}')
                app_logger.info(f'Material applied: {material}')
            except Exception as e:
                app_logger.error(f'Failed to apply material: {e}')
    
    def on_light_changed(self):
        intensity = self.control_panel.get_ambient_intensity()
        self.lighting_system.set_ambient_light((0.2, 0.2, 0.2), intensity)
        self.viewport.update()
    
    def show_about(self):
        QMessageBox.about(
            self, 
            'About', 
            'Video to 3D Visualization &amp; Editing System\n'
            'A system for video-driven 3D reconstruction and intelligent editing.'
        )
    
    def closeEvent(self, event):
        app_logger.info('Application closing')
        event.accept()

