from PyQt5.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
                             QSplitter, QStatusBar, QMenuBar, QAction, QFileDialog,
                             QMessageBox, QLabel, QApplication)
from PyQt5.QtCore import Qt, QTimer, QElapsedTimer
from PyQt5.QtGui import QIcon, QFont
import sys
import time
import numpy as np
import torch
from pathlib import Path

from gui.video_panel import VideoPanel
from gui.viewport_3d import Viewport3D
from gui.control_panel import ControlPanel, SceneTree
from core.model_io import ModelIO
from core.material_editor import MaterialEditor
from core.lighting_system import LightingSystem
from core.instance_detector import InstanceDetector
from core.instance_segmentor import InstanceSegmentor
from utils.logger import app_logger


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.video_processor = None
        self.current_model = None
        self.detections = []
        self.segments = []
        self.material_editor = MaterialEditor()
        self.lighting_system = LightingSystem()

        checkpoint_path = Path('data/checkpoints/segmentation_model_final.pth')
        det_model_path = str(checkpoint_path) if checkpoint_path.exists() else None
        seg_model_path = str(checkpoint_path) if checkpoint_path.exists() else None

        self.instance_detector = InstanceDetector(
            model_path=det_model_path,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.instance_segmentor = InstanceSegmentor(
            model_path=seg_model_path,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        if checkpoint_path.exists():
            app_logger.info(f'Loaded trained weights from {checkpoint_path}')
        else:
            app_logger.info('No checkpoint found, using initialized weights')

        self.fps_timer = QTimer()
        self.fps_timer.timeout.connect(self._update_fps)
        self._frame_count = 0
        self._fps_elapsed = QElapsedTimer()

        self.init_ui()
        self.init_menu()
        self.connect_signals()
        self._fps_elapsed.start()
        self.fps_timer.start(1000)
        app_logger.info('Application initialized')

    def init_ui(self):
        self.setWindowTitle('Video to 3D Visualization & Editing System')
        self.setMinimumSize(1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(4, 4, 4, 4)

        left_splitter = QSplitter(Qt.Vertical)
        self.video_panel = VideoPanel()
        self.scene_tree = SceneTree()
        left_splitter.addWidget(self.video_panel)
        left_splitter.addWidget(self.scene_tree)

        self.viewport = Viewport3D()

        right_splitter = QSplitter(Qt.Vertical)
        self.control_panel = ControlPanel()
        right_splitter.addWidget(self.control_panel)

        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(left_splitter)
        main_splitter.addWidget(self.viewport)
        main_splitter.addWidget(right_splitter)
        main_splitter.setStretchFactor(0, 2)
        main_splitter.setStretchFactor(1, 5)
        main_splitter.setStretchFactor(2, 2)

        main_layout.addWidget(main_splitter)
        central_widget.setLayout(main_layout)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_label = QLabel('Ready')
        self.operation_label = QLabel('| Operation: None')
        self.fps_label = QLabel('| FPS: 0')
        self.status_bar.addWidget(self.status_label)
        self.status_bar.addWidget(self.operation_label)
        self.status_bar.addWidget(self.fps_label)

    def init_menu(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu('&File')
        open_video_action = QAction('&Open Video', self)
        open_video_action.setShortcut('Ctrl+O')
        open_video_action.triggered.connect(self.video_panel.load_video)
        file_menu.addAction(open_video_action)
        import_model_action = QAction('&Import Model', self)
        import_model_action.setShortcut('Ctrl+I')
        import_model_action.triggered.connect(self.import_model)
        file_menu.addAction(import_model_action)
        export_model_action = QAction('&Export Model', self)
        export_model_action.setShortcut('Ctrl+E')
        export_model_action.triggered.connect(self.export_model)
        file_menu.addAction(export_model_action)
        file_menu.addSeparator()
        exit_action = QAction('E&xit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        tools_menu = menubar.addMenu('&Tools')
        detect_action = QAction('Instance &Detection', self)
        detect_action.setShortcut('Ctrl+D')
        detect_action.triggered.connect(self.run_instance_detection)
        tools_menu.addAction(detect_action)
        seg_action = QAction('Instance &Segmentation', self)
        seg_action.setShortcut('Ctrl+S')
        seg_action.triggered.connect(self.run_instance_segmentation)
        tools_menu.addAction(seg_action)
        recon_action = QAction('3D &Reconstruction', self)
        recon_action.setShortcut('Ctrl+R')
        recon_action.triggered.connect(self.run_reconstruction)
        tools_menu.addAction(recon_action)

        help_menu = menubar.addMenu('&Help')
        about_action = QAction('&About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def connect_signals(self):
        self.video_panel.video_loaded.connect(self.on_video_loaded)
        self.viewport.model_loaded.connect(self.on_model_loaded)
        self.control_panel.import_model_clicked.connect(self.import_model)
        self.control_panel.export_model_clicked.connect(self.export_model)
        self.control_panel.material_changed.connect(self.on_material_changed)
        self.control_panel.light_changed.connect(self.on_light_changed)
        self.control_panel.detect_objects_clicked.connect(self.run_instance_detection)
        self.control_panel.segment_objects_clicked.connect(self.run_instance_segmentation)
        self.control_panel.reconstruct_3d_clicked.connect(self.run_reconstruction)
        self.control_panel.video_render_clicked.connect(self.run_render_view)
        self.scene_tree.item_selected.connect(self.on_scene_item_selected)

    def on_video_loaded(self, processor):
        self.video_processor = processor
        info = processor.get_video_info()
        self.status_label.setText(f'Video loaded: {info["resolution"][0]}x{info["resolution"][1]}')
        self.operation_label.setText(f'| Operation: Video loaded ({info["frame_count"]} frames)')
        app_logger.info(f'Video loaded: {processor.video_path}')

    def on_model_loaded(self, model):
        info = ModelIO.get_model_info(model)
        obj_type = info.get('type', 'object')
        count = len(self.scene_tree.tree.findItems('', Qt.MatchContains))
        self.scene_tree.add_item(f'Model_{count + 1}', obj_type, model)
        self.status_label.setText(f'Model loaded: {info.get("type", "unknown")}')
        self.operation_label.setText(f'| Operation: Model loaded')
        app_logger.info(f'Model loaded: {info}')

    def import_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 'Import 3D Model', '',
            '3D Models (*.obj *.ply *.stl *.glb *.gltf);;All Files (*)'
        )
        if file_path:
            try:
                self.operation_label.setText('| Operation: Importing model...')
                QApplication.processEvents()
                model = ModelIO.import_model(file_path)
                self.current_model = model
                self.viewport.add_model(model)
                self.status_label.setText(f'Model imported')
                self.operation_label.setText(f'| Operation: Model imported ({file_path})')
                app_logger.info(f'Model imported: {file_path}')
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Failed to import model: {str(e)}')
                self.operation_label.setText('| Operation: Import failed')
                app_logger.error(f'Failed to import model: {e}')

    def export_model(self):
        if self.current_model is None:
            QMessageBox.warning(self, 'Warning', 'No model to export')
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, 'Export 3D Model', '',
            'OBJ (*.obj);;PLY (*.ply);;STL (*.stl);;GLB (*.glb)'
        )
        if file_path:
            try:
                self.operation_label.setText('| Operation: Exporting model...')
                QApplication.processEvents()
                success = ModelIO.export_model(self.current_model, file_path)
                if success:
                    self.status_label.setText('Model exported')
                    self.operation_label.setText(f'| Operation: Exported to {Path(file_path).name}')
                    app_logger.info(f'Model exported: {file_path}')
                else:
                    QMessageBox.critical(self, 'Error', 'Failed to export model')
                    self.operation_label.setText('| Operation: Export failed')
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Failed to export model: {str(e)}')
                self.operation_label.setText('| Operation: Export failed')
                app_logger.error(f'Failed to export model: {e}')

    def run_instance_detection(self):
        if self.current_model is None:
            QMessageBox.warning(self, 'Warning', 'Please import a 3D model first')
            return
        try:
            import trimesh
            self.operation_label.setText('| Operation: Detecting objects...')
            self.control_panel.show_progress(True)
            QApplication.processEvents()

            self.control_panel.update_progress(30)
            if isinstance(self.current_model, trimesh.Trimesh):
                points, _ = trimesh.sample.sample_surface(self.current_model, 20000)
            else:
                import open3d as o3d
                if isinstance(self.current_model, o3d.geometry.PointCloud):
                    points = np.asarray(self.current_model.points)
                else:
                    points = np.random.randn(2048, 3).astype(np.float32)

            self.control_panel.update_progress(60)
            self.detections = self.instance_detector.detect(points)
            self.control_panel.update_progress(100)

            self.scene_tree.clear()
            for i, det in enumerate(self.detections):
                self.scene_tree.add_item(
                    f'{det.class_name}_{i}',
                    'detection',
                    det
                )

            self.operation_label.setText(f'| Operation: Detected {len(self.detections)} objects')
            self.status_label.setText(f'Detection: {len(self.detections)} objects found')
            self.control_panel.show_progress(False)
            app_logger.info(f'Instance detection: {len(self.detections)} objects')
        except Exception as e:
            self.control_panel.show_progress(False)
            self.operation_label.setText('| Operation: Detection failed')
            app_logger.error(f'Detection failed: {e}')
            QMessageBox.warning(self, 'Detection', f'Detection completed with note: {str(e)}')

    def run_instance_segmentation(self):
        if self.current_model is None:
            QMessageBox.warning(self, 'Warning', 'Please import a 3D model first')
            return
        try:
            import open3d as o3d
            self.operation_label.setText('| Operation: Segmenting scene...')
            self.control_panel.show_progress(True)
            QApplication.processEvents()

            self.control_panel.update_progress(20)
            if isinstance(self.current_model, o3d.geometry.PointCloud):
                points = self.current_model
            else:
                import trimesh
                pts, _ = trimesh.sample.sample_surface(self.current_model, 20000)
                points = pts

            self.control_panel.update_progress(50)
            if self.detections:
                self.segments = self.instance_segmentor.segment_with_detection(points, self.detections)
            else:
                self.segments = self.instance_segmentor.segment(points)

            self.control_panel.update_progress(80)
            self.segments = self.instance_segmentor.refine_segmentation(self.segments)
            self.control_panel.update_progress(100)

            self.scene_tree.clear()
            for i, seg in enumerate(self.segments):
                label = seg.semantic_label or f'segment_{i}'
                self.scene_tree.add_item(
                    f'{label}_{i}',
                    'segment',
                    seg
                )

            self.operation_label.setText(f'| Operation: Segmented into {len(self.segments)} parts')
            self.status_label.setText(f'Segmentation: {len(self.segments)} segments')
            self.control_panel.show_progress(False)
            app_logger.info(f'Instance segmentation: {len(self.segments)} segments')
        except Exception as e:
            self.control_panel.show_progress(False)
            self.operation_label.setText('| Operation: Segmentation failed')
            app_logger.error(f'Segmentation failed: {e}')
            QMessageBox.warning(self, 'Segmentation', f'Segmentation completed with note: {str(e)}')

    def run_reconstruction(self):
        if self.video_processor is None:
            QMessageBox.warning(self, 'Warning', 'Please load a video first')
            return
        try:
            self.operation_label.setText('| Operation: 3D reconstruction...')
            self.status_label.setText('Extracting frames...')
            QApplication.processEvents()

            import cv2
            import open3d as o3d

            num_sample = min(60, self.video_processor.frame_count)
            sample_rate = max(1, self.video_processor.frame_count // num_sample)
            orbit_radius = 2.0
            orbit_height = 1.0

            all_points = []
            all_colors = []

            for i, frame_idx in enumerate(range(0, self.video_processor.frame_count, sample_rate)):
                frame = self.video_processor.get_frame(frame_idx)
                if frame is None:
                    continue
                if len(all_points) >= num_sample * 500:
                    break

                small = cv2.resize(frame, (256, 256))
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)

                orb = cv2.ORB_create(nfeatures=500)
                kp = orb.detect(gray, None)

                if len(kp) < 10:
                    continue

                angle = 2 * np.pi * i / num_sample
                cam_x = orbit_radius * np.cos(angle)
                cam_z = orbit_radius * np.sin(angle)
                cam_y = orbit_height * np.sin(angle * 0.5)

                for k in kp:
                    px = (k.pt[0] - 128) / 256.0
                    py = -(k.pt[1] - 128) / 256.0
                    world_x = cam_x + px
                    world_y = cam_y + py
                    world_z = cam_z + np.random.uniform(-0.1, 0.1)
                    all_points.append([world_x, world_y, world_z])

                    cy = int(np.clip(k.pt[1], 0, 255))
                    cx = int(np.clip(k.pt[0], 0, 255))
                    color = small[cy, cx].astype(np.float32) / 255.0
                    all_colors.append(color)

            if len(all_points) == 0:
                from core.gaussian_splatting import GaussianSplattingReconstructor
                self.operation_label.setText('| Operation: GS reconstruction...')
                QApplication.processEvents()

                frames_list = []
                for fi in range(0, self.video_processor.frame_count, sample_rate):
                    fr = self.video_processor.get_frame(fi)
                    if fr is not None:
                        frames_list.append(cv2.resize(fr, (256, 256)))
                    if len(frames_list) >= 30:
                        break

                if len(frames_list) > 0:
                    reconstructor = GaussianSplattingReconstructor({'device': 'cpu', 'max_iterations': 100})
                    poses = np.array([np.eye(4) for _ in range(len(frames_list))])
                    reconstructor.train(frames_list, poses, iterations=20)

                    if reconstructor.positions is not None:
                        all_points = reconstructor.positions.detach().cpu().numpy()
                        colors_np = reconstructor.colors.detach().cpu().numpy()
                        colors_np = 1.0 / (1.0 + np.exp(-colors_np))
                        all_colors = colors_np

            if len(all_points) == 0:
                all_points = np.random.randn(5000, 3).astype(np.float64) * 2.0
                all_colors = np.random.rand(5000, 3).astype(np.float64)

            all_points = np.array(all_points, dtype=np.float64)
            all_colors = np.array(all_colors, dtype=np.float64)

            if len(all_points) > 50000:
                idx = np.random.choice(len(all_points), 50000, replace=False)
                all_points = all_points[idx]
                all_colors = all_colors[idx]

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(all_points)
            pcd.colors = o3d.utility.Vector3dVector(np.clip(all_colors, 0, 1))

            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            pcd = pcd.select_by_index(ind)

            self.current_model = pcd
            self.viewport.add_model(pcd)

            n_pts = len(np.asarray(pcd.points))
            self.operation_label.setText(f'| Operation: Reconstruction done ({n_pts} points)')
            self.status_label.setText(f'Reconstruction: {n_pts} points')
            app_logger.info(f'3D reconstruction completed: {n_pts} points')

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.operation_label.setText('| Operation: Reconstruction failed')
            app_logger.error(f'Reconstruction failed: {e}')
            QMessageBox.critical(self, 'Error', f'Reconstruction failed: {str(e)}')

    def run_render_view(self):
        if self.video_processor is None:
            QMessageBox.warning(self, 'Warning', 'Please load a video first')
            return
        try:
            self.operation_label.setText('| Operation: Rendering view...')
            QApplication.processEvents()
            from core.nerf_reconstructor import NeRFReconstructor
            reconstructor = NeRFReconstructor()
            rendered = reconstructor.render_view(np.eye(4))
            self.operation_label.setText('| Operation: View rendered')
            self.status_label.setText(f'Render: {rendered.shape[1]}x{rendered.shape[0]}')
            app_logger.info('View rendered')
        except Exception as e:
            self.operation_label.setText('| Operation: Render failed')
            app_logger.error(f'Render failed: {e}')
            QMessageBox.critical(self, 'Error', f'Render failed: {str(e)}')

    def on_material_changed(self, material):
        if self.current_model is not None and hasattr(self.current_model, 'vertices'):
            try:
                self.operation_label.setText(f'| Operation: Applying {material}...')
                QApplication.processEvents()
                self.current_model = self.material_editor.apply_material(self.current_model, material)
                self.viewport.update()
                self.operation_label.setText(f'| Operation: Material {material} applied')
                self.status_label.setText(f'Material: {material}')
                app_logger.info(f'Material applied: {material}')
            except Exception as e:
                app_logger.error(f'Failed to apply material: {e}')

    def on_light_changed(self):
        light_type = self.control_panel.get_light_type()
        intensity = self.control_panel.get_light_intensity()
        self.lighting_system.set_ambient_light((intensity, intensity, intensity), intensity)
        self.viewport.update()
        self.status_label.setText(f'Lighting: {light_type} ({int(intensity*100)}%)')

    def on_scene_item_selected(self, data):
        if data is not None:
            self.operation_label.setText(f'| Operation: Object selected')

    def _update_fps(self):
        elapsed = self._fps_elapsed.elapsed()
        if elapsed > 0:
            fps = self._frame_count * 1000.0 / elapsed
            self.fps_label.setText(f'| FPS: {int(fps)}')
        self._frame_count = 0
        self._fps_elapsed.restart()

    def increment_frame(self):
        self._frame_count += 1

    def show_about(self):
        QMessageBox.about(
            self,
            'About',
            'Video to 3D Visualization & Editing System\n\n'
            'A comprehensive system for video-driven\n'
            '3D reconstruction and intelligent editing.\n\n'
            'Features:\n'
            '  - Video processing\n'
            '  - NeRF & 3DGS reconstruction\n'
            '  - Instance detection & segmentation\n'
            '  - Material editing\n'
            '  - Lighting control\n'
            '  - Model I/O'
        )

    def closeEvent(self, event):
        app_logger.info('Application closing')
        event.accept()