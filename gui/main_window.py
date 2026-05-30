from PyQt5.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
                             QSplitter, QStatusBar, QMenuBar, QAction, QFileDialog,
                             QMessageBox, QLabel, QApplication)
from PyQt5.QtCore import Qt, QTimer, QElapsedTimer
from PyQt5.QtGui import QIcon, QFont
import json
import sys
import time
import numpy as np
import torch
from pathlib import Path
from types import SimpleNamespace

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
        self._original_model = None
        self.selected_scene_object = None
        self.current_video_path = None
        self.detections = []
        self.segments = []
        self.material_editor = MaterialEditor()
        self.lighting_system = LightingSystem()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.instance_detector = InstanceDetector(device=device)
        self.instance_segmentor = InstanceSegmentor(device=device)

        checkpoint_path = Path('data/checkpoints/segmentation_model_final.pth')
        if checkpoint_path.exists():
            self._load_segmentation_weights(checkpoint_path)
        else:
            app_logger.info('No segmentation checkpoint found')

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
        self.video_panel.setMinimumHeight(180)
        self.scene_tree = SceneTree()
        self.scene_tree.setMinimumHeight(80)
        left_splitter.addWidget(self.video_panel)
        left_splitter.addWidget(self.scene_tree)
        left_splitter.setCollapsible(0, False)
        left_splitter.setCollapsible(1, False)

        self.viewport = Viewport3D()

        right_splitter = QSplitter(Qt.Vertical)
        self.control_panel = ControlPanel()
        self.control_panel.setMinimumWidth(200)
        right_splitter.addWidget(self.control_panel)
        right_splitter.setCollapsible(0, False)

        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(left_splitter)
        main_splitter.addWidget(self.viewport)
        main_splitter.addWidget(right_splitter)
        main_splitter.setStretchFactor(0, 2)
        main_splitter.setStretchFactor(1, 5)
        main_splitter.setStretchFactor(2, 2)
        main_splitter.setCollapsible(0, False)
        main_splitter.setCollapsible(1, False)
        main_splitter.setCollapsible(2, False)
        main_splitter.setSizes([260, 600, 260])

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

        edit_menu = menubar.addMenu('&Edit')
        clear_action = QAction('&Clear Scene', self)
        clear_action.setShortcut('Ctrl+Shift+C')
        clear_action.triggered.connect(self.clear_all_models)
        edit_menu.addAction(clear_action)

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
        self.control_panel.clear_models_clicked.connect(self.clear_all_models)
        self.scene_tree.item_selected.connect(self.on_scene_item_selected)
        self.scene_tree.item_delete_requested.connect(self.on_scene_item_delete_requested)

    def on_video_loaded(self, processor):
        self.video_processor = processor
        self.current_video_path = Path(processor.video_path)
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
                self._original_model = model
                self.selected_scene_object = None
                self.viewport.select_object(None)
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
            self.operation_label.setText('| Operation: Detecting objects...')
            self.control_panel.show_progress(True)
            QApplication.processEvents()

            self.control_panel.update_progress(30)
            points = self._points_for_analysis(self.current_model)

            self.control_panel.update_progress(60)
            self.detections = self.instance_detector.detect(points)
            self.control_panel.update_progress(100)

            self.scene_tree.clear()
            self.selected_scene_object = None
            self.viewport.select_object(None)
            self.scene_tree.add_item('Reconstructed_Scene', 'mesh', self.current_model)
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
            s3dis_segments = self._load_s3dis_segments_for_current_video()
            if s3dis_segments:
                self.segments = s3dis_segments
                self.control_panel.update_progress(100)
            else:
                if isinstance(self.current_model, o3d.geometry.PointCloud):
                    points = self.current_model
                else:
                    points = self._points_for_analysis(self.current_model)

                self.control_panel.update_progress(50)
                if self.detections:
                    self.segments = self.instance_segmentor.segment_with_detection(points, self.detections)
                else:
                    self.segments = self.instance_segmentor.segment(points)

                self.control_panel.update_progress(80)
                self.segments = self.instance_segmentor.refine_segmentation(self.segments)
                self.control_panel.update_progress(100)

            self.scene_tree.clear()
            self.selected_scene_object = None
            self.viewport.select_object(None)
            self.scene_tree.add_item('Reconstructed_Scene', 'mesh', self.current_model)
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
            self.status_label.setText('Running RGB-D TSDF fusion...')
            QApplication.processEvents()

            from core.rgbd_tsdf_reconstructor import RGBDTSDFReconstructor

            project_dir = RGBDTSDFReconstructor.project_dir_for_video(
                str(self.video_processor.video_path)
            )
            if project_dir is None:
                raise ValueError(
                    'No RGB-D project found for this video. '
                    'Use tools/generate_s3dis_scene_video.py to create '
                    'an S3DIS walkthrough video with color/depth/pose data first.'
                )

            reconstructor = RGBDTSDFReconstructor()
            result = reconstructor.reconstruct_from_project(str(project_dir))

            self.viewport.clear_models()
            self.scene_tree.clear()
            self.detections = []
            self.segments = []
            self.current_model = result.mesh
            self._original_model = result.mesh
            self.selected_scene_object = None
            self.viewport.select_object(None)
            self.viewport.add_model(result.mesh)

            self.operation_label.setText(
                f'| Operation: Reconstruction done '
                f'({len(result.mesh.vertices)} vertices, {len(result.mesh.triangles)} faces)'
            )
            self.status_label.setText(
                f'TSDF Fusion: {result.frames} RGB-D frames'
            )
            app_logger.info(
                'TSDF reconstruction completed: '
                f'{len(result.mesh.vertices)} vertices, '
                f'{len(result.mesh.triangles)} faces, '
                f'{result.frames} frames'
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.operation_label.setText('| Operation: Reconstruction failed')
            app_logger.error(f'Reconstruction failed: {e}')
            QMessageBox.critical(self, 'Error', f'Reconstruction failed: {str(e)}')

    def run_render_view(self):
        if self.current_model is None:
            QMessageBox.warning(self, 'Warning', 'Please reconstruct or import a 3D model first')
            return
        self.selected_scene_object = None
        self.viewport.select_object(None)
        self.scene_tree.clear_selection()
        self.viewport.fit_view()
        self.operation_label.setText('| Operation: View reset')
        self.status_label.setText('View reset to full scene')

    def on_material_changed(self, material):
        if self.current_model is None:
            return
        try:
            import trimesh
            self.operation_label.setText(f'| Operation: Applying {material}...')
            QApplication.processEvents()

            source = self.current_model
            if self.selected_scene_object is not None and self._mesh_vertices(source) is not None:
                new_model = self._apply_material_to_selected_object(source, self.selected_scene_object, material)
            elif isinstance(source, trimesh.Trimesh):
                new_model = self.material_editor.apply_material(source, material)
            elif self._is_open3d_mesh(source):
                new_model = self._apply_material_to_entire_open3d_mesh(source, material)
            else:
                self.operation_label.setText('| Operation: Material only for mesh models')
                return

            self.current_model = new_model
            if self._original_model is None:
                self._original_model = new_model
            self.viewport.models = [new_model]
            self.viewport.select_object(self.selected_scene_object)
            self.viewport.update()
            self.operation_label.setText(f'| Operation: Material {material} applied')
            self.status_label.setText(f'Material: {material}')
            app_logger.info(f'Material applied: {material}')
        except Exception as e:
            app_logger.error(f'Failed to apply material: {e}')
            QMessageBox.warning(self, 'Material', f'Failed: {str(e)}')

    def on_light_changed(self):
        light_type = self.control_panel.get_light_type()
        intensity = self.control_panel.get_light_intensity()
        self.lighting_system.set_ambient_light((intensity, intensity, intensity), intensity)
        self.viewport.apply_lighting(light_type, intensity)
        self.viewport.update()
        self.status_label.setText(f'Lighting: {light_type} ({int(intensity*100)}%)')

    def on_scene_item_selected(self, data):
        if data is self.selected_scene_object:
            self.selected_scene_object = None
            self.viewport.select_object(None)
            self.scene_tree.clear_selection()
            self.operation_label.setText('| Operation: Selection cleared')
            return
        self.selected_scene_object = data
        self.viewport.select_object(data)
        if data is not None:
            label = getattr(data, 'semantic_label', None) or getattr(data, 'class_name', 'Object')
            self.operation_label.setText(f'| Operation: Selected {label}')

    def on_scene_item_delete_requested(self, data):
        if data is None:
            return
        if data is self.current_model:
            self.current_model = None
            self._original_model = None
            self.viewport.remove_model(data)
        else:
            if self.current_model is not None and self._selection_points(data) is not None:
                updated = self._remove_selection_from_model(self.current_model, data)
                if updated is not self.current_model:
                    self.current_model = updated
                    self._original_model = updated
                    self.viewport.models = [updated]
                    self.viewport.update()
            self.viewport.select_object(None)
        if data is self.selected_scene_object:
            self.selected_scene_object = None
        self.detections = [det for det in self.detections if det is not data]
        self.segments = [seg for seg in self.segments if seg is not data]
        self.status_label.setText('Object deleted')
        self.operation_label.setText('| Operation: Object deleted')

    def clear_all_models(self):
        self.current_model = None
        self._original_model = None
        self.selected_scene_object = None
        self.detections = []
        self.segments = []
        self.viewport.clear_models()
        self.scene_tree.clear()
        self.status_label.setText('Scene cleared')
        self.operation_label.setText('| Operation: All models removed')
        app_logger.info('Scene cleared')

    def _load_segmentation_weights(self, checkpoint_path):
        try:
            from models.segmentation_model import SegmentationModel
            self.seg_model = SegmentationModel(
                num_classes=self.instance_detector.num_classes,
                in_channels=6
            ).to(self.instance_detector.device)
            self.seg_model.eval()
            checkpoint = torch.load(checkpoint_path, map_location=self.instance_detector.device)
            self.seg_model.load_state_dict(checkpoint['model_state_dict'])
            app_logger.info(f'Segmentation model loaded from {checkpoint_path} '
                            f'(epoch {checkpoint.get("epoch", "?")})')
        except Exception as e:
            app_logger.warning(f'Could not load segmentation weights: {e}')
            self.seg_model = None

    def _load_s3dis_segments_for_current_video(self):
        if self.current_video_path is None:
            return []
        summary_path = Path('data/output_models') / f'{self.current_video_path.stem}_summary.json'
        if not summary_path.exists():
            return []
        try:
            summary = json.loads(summary_path.read_text(encoding='utf-8'))
            segments = []
            for inst in summary.get('instances', []):
                ply_path = Path(inst.get('path', ''))
                if not ply_path.is_absolute():
                    ply_path = Path.cwd() / ply_path
                if not ply_path.exists():
                    continue
                import open3d as o3d
                pcd = o3d.io.read_point_cloud(str(ply_path))
                points = np.asarray(pcd.points)
                if len(points) == 0:
                    continue
                obj = SimpleNamespace(
                    semantic_label=inst.get('name', 'object'),
                    class_name=inst.get('name', 'object'),
                    points=points,
                    source_path=str(ply_path),
                    label_id=inst.get('label_id'),
                )
                segments.append(obj)
            return segments
        except Exception as e:
            app_logger.warning(f'Could not load S3DIS scene objects: {e}')
            return []

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
            '  - S3DIS RGB-D TSDF reconstruction\n'
            '  - Instance detection & segmentation\n'
            '  - Material editing\n'
            '  - Lighting control\n'
            '  - Model I/O'
        )

    def closeEvent(self, event):
        app_logger.info('Application closing')
        event.accept()

    def _apply_material_to_selected_object(self, mesh, selection, material):
        mat = self.material_editor.MATERIALS[material]
        new_mesh = mesh.copy() if hasattr(mesh, 'copy') else mesh

        vertices = self._mesh_vertices(new_mesh)
        if len(vertices) == 0:
            return new_mesh

        selected_points = self._selection_points(selection)
        if selected_points is None or len(selected_points) == 0:
            return self.material_editor.apply_material(new_mesh, material)

        if hasattr(selection, 'bbox'):
            bbox = np.asarray(selection.bbox)
            min_coords = bbox.min(axis=0)
            max_coords = bbox.max(axis=0)
            mask = np.all((vertices >= min_coords) & (vertices <= max_coords), axis=1)
        else:
            min_coords = selected_points.min(axis=0)
            max_coords = selected_points.max(axis=0)
            padding = max(np.linalg.norm(max_coords - min_coords) * 0.03, 1e-3)
            mask = np.all((vertices >= min_coords - padding) & (vertices <= max_coords + padding), axis=1)

        if not mask.any():
            return self.material_editor.apply_material(new_mesh, material)

        colors = self._mesh_vertex_colors(new_mesh, len(vertices))
        colors[mask, :3] = np.array(mat.albedo) * 255
        colors[mask, 3] = int((1.0 - mat.transparency) * 255)
        self._set_mesh_vertex_colors(new_mesh, colors)
        return new_mesh

    def _apply_material_to_entire_open3d_mesh(self, mesh, material):
        mat = self.material_editor.MATERIALS[material]
        new_mesh = self._copy_open3d_mesh(mesh)
        vertices = self._mesh_vertices(new_mesh)
        if vertices is None or len(vertices) == 0:
            return new_mesh
        colors = np.ones((len(vertices), 4), dtype=np.uint8) * 255
        colors[:, :3] = (np.array(mat.albedo) * 255).astype(np.uint8)
        colors[:, 3] = int((1.0 - mat.transparency) * 255)
        self._set_mesh_vertex_colors(new_mesh, colors)
        return new_mesh

    def _copy_open3d_mesh(self, mesh):
        import open3d as o3d
        new_mesh = o3d.geometry.TriangleMesh()
        new_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
        new_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles))
        if mesh.has_vertex_colors():
            new_mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_colors))
        if mesh.has_vertex_normals():
            new_mesh.vertex_normals = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_normals))
        return new_mesh

    def _selection_points(self, selection):
        if hasattr(selection, 'points'):
            return np.asarray(selection.points)
        if hasattr(selection, 'bbox'):
            return np.asarray(selection.bbox)
        if hasattr(selection, 'vertices'):
            return np.asarray(selection.vertices)
        return None

    def _mesh_vertex_colors(self, mesh, count):
        if self._is_open3d_mesh(mesh):
            colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None
            if colors is None or len(colors) != count:
                colors = np.ones((count, 4), dtype=np.uint8) * 180
                colors[:, 3] = 255
            else:
                colors = np.clip(colors * 255, 0, 255).astype(np.uint8)
                if colors.shape[1] == 3:
                    alpha = np.full((count, 1), 255, dtype=np.uint8)
                    colors = np.hstack([colors, alpha])
            return colors.copy()

        colors = None
        if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
            colors = np.asarray(mesh.visual.vertex_colors)
        if colors is None or len(colors) != count:
            colors = np.ones((count, 4), dtype=np.uint8) * 180
            colors[:, 3] = 255
        if colors.shape[1] == 3:
            alpha = np.full((count, 1), 255, dtype=colors.dtype)
            colors = np.hstack([colors, alpha])
        return colors.copy()

    def _set_mesh_vertex_colors(self, mesh, colors):
        if self._is_open3d_mesh(mesh):
            import open3d as o3d
            mesh.vertex_colors = o3d.utility.Vector3dVector(colors[:, :3].astype(np.float64) / 255.0)
        elif hasattr(mesh, 'visual'):
            mesh.visual.vertex_colors = colors.astype(np.uint8)

    def _is_open3d_mesh(self, model):
        try:
            import open3d as o3d
            return isinstance(model, o3d.geometry.TriangleMesh)
        except Exception:
            return False

    def _mesh_vertices(self, model):
        if self._is_open3d_mesh(model):
            return np.asarray(model.vertices)
        if hasattr(model, 'vertices'):
            return np.asarray(model.vertices)
        return None

    def _points_for_analysis(self, model, sample_count=20000):
        try:
            import open3d as o3d
            if isinstance(model, o3d.geometry.PointCloud):
                return np.asarray(model.points)
            if isinstance(model, o3d.geometry.TriangleMesh):
                pcd = model.sample_points_uniformly(number_of_points=sample_count)
                return np.asarray(pcd.points)
        except Exception:
            pass
        try:
            import trimesh
            if isinstance(model, trimesh.Trimesh):
                points, _ = trimesh.sample.sample_surface(model, sample_count)
                return points
        except Exception:
            pass
        if hasattr(model, 'points'):
            return np.asarray(model.points)
        if hasattr(model, 'vertices'):
            return np.asarray(model.vertices)
        return np.random.randn(2048, 3).astype(np.float32)

    def _remove_selection_from_model(self, model, selection):
        points = self._selection_points(selection)
        if points is None or len(points) == 0:
            return model

        min_coords = points.min(axis=0)
        max_coords = points.max(axis=0)
        padding = max(np.linalg.norm(max_coords - min_coords) * 0.02, 1e-3)
        min_coords -= padding
        max_coords += padding

        try:
            import open3d as o3d
            if isinstance(model, o3d.geometry.TriangleMesh):
                vertices = np.asarray(model.vertices)
                triangles = np.asarray(model.triangles)
                if len(vertices) == 0 or len(triangles) == 0:
                    return model
                inside = np.all((vertices >= min_coords) & (vertices <= max_coords), axis=1)
                keep_triangles = ~inside[triangles].any(axis=1)
                if keep_triangles.all():
                    return model
                new_model = o3d.geometry.TriangleMesh()
                new_model.vertices = o3d.utility.Vector3dVector(vertices)
                new_model.triangles = o3d.utility.Vector3iVector(triangles[keep_triangles])
                if model.has_vertex_colors():
                    new_model.vertex_colors = o3d.utility.Vector3dVector(np.asarray(model.vertex_colors))
                if model.has_vertex_normals():
                    new_model.vertex_normals = o3d.utility.Vector3dVector(np.asarray(model.vertex_normals))
                new_model.remove_unreferenced_vertices()
                new_model.compute_vertex_normals()
                return new_model
        except Exception as e:
            app_logger.warning(f'Open3D object removal failed: {e}')

        try:
            import trimesh
            if isinstance(model, trimesh.Trimesh):
                new_model = model.copy()
                vertices = np.asarray(new_model.vertices)
                inside = np.all((vertices >= min_coords) & (vertices <= max_coords), axis=1)
                keep_faces = ~inside[np.asarray(new_model.faces)].any(axis=1)
                if keep_faces.all():
                    return model
                new_model.update_faces(keep_faces)
                new_model.remove_unreferenced_vertices()
                return new_model
        except Exception as e:
            app_logger.warning(f'Trimesh object removal failed: {e}')

        return model
