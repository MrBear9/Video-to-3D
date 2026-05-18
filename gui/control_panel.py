from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QGroupBox,
                             QComboBox, QLabel, QSlider, QFileDialog, QTreeWidget,
                             QTreeWidgetItem, QSplitter, QProgressBar, QScrollArea)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
import numpy as np


class ControlPanel(QWidget):
    import_model_clicked = pyqtSignal()
    export_model_clicked = pyqtSignal()
    material_changed = pyqtSignal(str)
    light_changed = pyqtSignal()
    detect_objects_clicked = pyqtSignal()
    segment_objects_clicked = pyqtSignal()
    reconstruct_3d_clicked = pyqtSignal()
    video_render_clicked = pyqtSignal()
    clear_models_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        container = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(8)

        video_group = QGroupBox('Video Tools')
        video_layout = QVBoxLayout()
        self.reconstruct_btn = QPushButton('3D Reconstruction')
        self.reconstruct_btn.setToolTip('Generate 3D model from video frames')
        self.reconstruct_btn.clicked.connect(self.reconstruct_3d_clicked.emit)
        video_layout.addWidget(self.reconstruct_btn)
        self.render_view_btn = QPushButton('Render View')
        self.render_view_btn.setToolTip('Render a novel view from reconstructed model')
        self.render_view_btn.clicked.connect(self.video_render_clicked.emit)
        video_layout.addWidget(self.render_view_btn)
        video_group.setLayout(video_layout)
        layout.addWidget(video_group)

        io_group = QGroupBox('Model I/O')
        io_layout = QVBoxLayout()
        self.import_btn = QPushButton('Import Model')
        self.import_btn.setToolTip('Load existing 3D model (OBJ/PLY/STL/GLB)')
        self.import_btn.clicked.connect(self.import_model_clicked.emit)
        io_layout.addWidget(self.import_btn)
        self.export_btn = QPushButton('Export Model')
        self.export_btn.setToolTip('Export current model to file')
        self.export_btn.clicked.connect(self.export_model_clicked.emit)
        io_layout.addWidget(self.export_btn)
        self.clear_models_btn = QPushButton('Clear All Models')
        self.clear_models_btn.setToolTip('Remove all loaded models from the scene')
        self.clear_models_btn.setStyleSheet('QPushButton { color: #cc4444; }')
        self.clear_models_btn.clicked.connect(self.clear_models_clicked.emit)
        io_layout.addWidget(self.clear_models_btn)
        io_group.setLayout(io_layout)
        layout.addWidget(io_group)

        analysis_group = QGroupBox('3D Analysis')
        analysis_layout = QVBoxLayout()
        self.detect_btn = QPushButton('Instance Detection')
        self.detect_btn.setToolTip('Detect individual objects in the scene')
        self.detect_btn.clicked.connect(self.detect_objects_clicked.emit)
        analysis_layout.addWidget(self.detect_btn)
        self.segment_btn = QPushButton('Instance Segmentation')
        self.segment_btn.setToolTip('Segment scene into individual object instances')
        self.segment_btn.clicked.connect(self.segment_objects_clicked.emit)
        analysis_layout.addWidget(self.segment_btn)
        self.progress_seg = QProgressBar()
        self.progress_seg.setMaximum(100)
        self.progress_seg.setValue(0)
        self.progress_seg.setVisible(False)
        analysis_layout.addWidget(self.progress_seg)
        analysis_group.setLayout(analysis_layout)
        layout.addWidget(analysis_group)

        material_group = QGroupBox('Material Editor')
        material_layout = QVBoxLayout()
        material_layout.addWidget(QLabel('Select Material:'))
        self.material_combo = QComboBox()
        self.material_combo.addItems([
            'metal', 'wood', 'plastic', 'glass', 'stone', 'ceramic', 'fabric'
        ])
        self.material_combo.currentTextChanged.connect(self.on_material_changed)
        material_layout.addWidget(self.material_combo)
        material_group.setLayout(material_layout)
        layout.addWidget(material_group)

        light_group = QGroupBox('Lighting')
        light_layout = QVBoxLayout()
        light_layout.addWidget(QLabel('Light Type:'))
        self.light_combo = QComboBox()
        self.light_combo.addItems(['ambient', 'directional', 'point', 'spot'])
        self.light_combo.currentTextChanged.connect(self.on_light_changed)
        light_layout.addWidget(self.light_combo)
        light_layout.addWidget(QLabel('Intensity:'))
        self.light_slider = QSlider(Qt.Horizontal)
        self.light_slider.setRange(0, 100)
        self.light_slider.setValue(50)
        self.light_slider.valueChanged.connect(self.on_light_changed)
        light_layout.addWidget(self.light_slider)
        self.light_intensity_label = QLabel('50%')
        light_layout.addWidget(self.light_intensity_label)
        light_group.setLayout(light_layout)
        layout.addWidget(light_group)

        layout.addStretch()
        container.setLayout(layout)
        scroll.setWidget(container)

        outer_layout = QVBoxLayout()
        outer_layout.addWidget(scroll)
        self.setLayout(outer_layout)

    def on_material_changed(self, material):
        self.material_changed.emit(material)

    def on_light_changed(self):
        intensity = self.light_slider.value()
        self.light_intensity_label.setText(f'{intensity}%')
        self.light_changed.emit()

    def get_light_type(self):
        return self.light_combo.currentText()

    def get_light_intensity(self):
        return self.light_slider.value() / 100.0

    def show_progress(self, visible=True):
        self.progress_seg.setVisible(visible)
        if not visible:
            self.progress_seg.setValue(0)

    def update_progress(self, value):
        self.progress_seg.setValue(value)


class SceneTree(QWidget):
    item_selected = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        header = QLabel('Scene Objects')
        header.setStyleSheet('font-weight: bold; padding: 4px;')
        layout.addWidget(header)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(['Type', 'Name'])
        self.tree.setColumnWidth(0, 60)
        self.tree.itemClicked.connect(self.on_item_clicked)
        layout.addWidget(self.tree)
        self.setLayout(layout)

    def add_item(self, name, obj_type='Object', data=None):
        item = QTreeWidgetItem(self.tree)
        item.setText(0, obj_type)
        item.setText(1, name)
        item.setData(0, Qt.UserRole, data)
        self.tree.addTopLevelItem(item)
        self.tree.expandAll()

    def clear(self):
        self.tree.clear()

    def on_item_clicked(self, item, column):
        data = item.data(0, Qt.UserRole)
        self.item_selected.emit(data)