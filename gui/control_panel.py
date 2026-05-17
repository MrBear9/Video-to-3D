
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QGroupBox, 
                             QComboBox, QLabel, QSlider, QFileDialog, QTreeWidget,
                             QTreeWidgetItem, QSplitter)
from PyQt5.QtCore import Qt, pyqtSignal
import numpy as np


class ControlPanel(QWidget):
    import_model_clicked = pyqtSignal()
    export_model_clicked = pyqtSignal()
    material_changed = pyqtSignal(str)
    light_changed = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        io_group = QGroupBox('Model I/O')
        io_layout = QVBoxLayout()
        self.import_btn = QPushButton('Import Model')
        self.import_btn.clicked.connect(self.import_model_clicked.emit)
        self.export_btn = QPushButton('Export Model')
        self.export_btn.clicked.connect(self.export_model_clicked.emit)
        io_layout.addWidget(self.import_btn)
        io_layout.addWidget(self.export_btn)
        io_group.setLayout(io_layout)
        layout.addWidget(io_group)
        
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
        light_layout.addWidget(QLabel('Ambient Intensity:'))
        self.ambient_slider = QSlider(Qt.Horizontal)
        self.ambient_slider.setRange(0, 100)
        self.ambient_slider.setValue(30)
        self.ambient_slider.valueChanged.connect(self.on_light_changed)
        light_layout.addWidget(self.ambient_slider)
        light_group.setLayout(light_layout)
        layout.addWidget(light_group)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def on_material_changed(self, material):
        self.material_changed.emit(material)
    
    def on_light_changed(self):
        self.light_changed.emit()
    
    def get_ambient_intensity(self):
        return self.ambient_slider.value() / 100.0


class SceneTree(QWidget):
    item_selected = pyqtSignal(object)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        self.tree = QTreeWidget()
        self.tree.setHeaderLabel('Scene Objects')
        self.tree.itemClicked.connect(self.on_item_clicked)
        layout.addWidget(self.tree)
        self.setLayout(layout)
    
    def add_item(self, name, data=None):
        item = QTreeWidgetItem(self.tree)
        item.setText(0, name)
        item.setData(0, Qt.UserRole, data)
        self.tree.addTopLevelItem(item)
    
    def clear(self):
        self.tree.clear()
    
    def on_item_clicked(self, item, column):
        data = item.data(0, Qt.UserRole)
        self.item_selected.emit(data)

