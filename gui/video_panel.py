
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, 
                             QSlider, QFileDialog, QGroupBox, QFormLayout)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np
from core.video_processor import VideoProcessor


class VideoPanel(QWidget):
    video_loaded = pyqtSignal(object)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.video_processor = None
        self.current_frame = None
        self.is_playing = False
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self.next_frame)
        
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        self.video_label = QLabel('No video loaded')
        self.video_label.setMinimumSize(300, 200)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet('background-color: #222; border: 1px solid #444;')
        layout.addWidget(self.video_label)
        
        controls_group = QGroupBox('Controls')
        controls_layout = QVBoxLayout()
        
        self.load_btn = QPushButton('Load Video')
        self.load_btn.clicked.connect(self.load_video)
        controls_layout.addWidget(self.load_btn)
        
        self.play_btn = QPushButton('Play')
        self.play_btn.clicked.connect(self.toggle_play)
        self.play_btn.setEnabled(False)
        controls_layout.addWidget(self.play_btn)
        
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(100)
        self.frame_slider.valueChanged.connect(self.seek_frame)
        self.frame_slider.setEnabled(False)
        controls_layout.addWidget(self.frame_slider)
        
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        info_group = QGroupBox('Video Info')
        info_layout = QFormLayout()
        self.resolution_label = QLabel('-')
        self.fps_label = QLabel('-')
        self.duration_label = QLabel('-')
        info_layout.addRow('Resolution:', self.resolution_label)
        info_layout.addRow('FPS:', self.fps_label)
        info_layout.addRow('Duration:', self.duration_label)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        self.setLayout(layout)
    
    def load_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 'Open Video File', '', 
            'Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)'
        )
        
        if file_path:
            try:
                self.video_processor = VideoProcessor(file_path)
                info = self.video_processor.get_video_info()
                
                self.resolution_label.setText(f"{info['resolution'][0]}x{info['resolution'][1]}")
                self.fps_label.setText(f"{info['fps']:.2f}")
                self.duration_label.setText(f"{info['duration']:.2f}s")
                
                self.frame_slider.setMaximum(info['frame_count'] - 1)
                self.frame_slider.setEnabled(True)
                self.play_btn.setEnabled(True)
                
                self.show_frame(0)
                self.video_loaded.emit(self.video_processor)
            except Exception as e:
                self.video_label.setText(f"Error: {str(e)}")
    
    def show_frame(self, frame_idx):
        if self.video_processor:
            frame = self.video_processor.get_frame(frame_idx)
            if frame is not None:
                self.current_frame = frame
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                scaled_pixmap = pixmap.scaled(
                    self.video_label.size(), 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
                self.video_label.setPixmap(scaled_pixmap)
    
    def toggle_play(self):
        if not self.video_processor:
            return
        
        self.is_playing = not self.is_playing
        
        if self.is_playing:
            self.play_btn.setText('Pause')
            fps = self.video_processor.fps or 30
            interval = int(1000 / fps)
            self.play_timer.start(interval)
        else:
            self.play_btn.setText('Play')
            self.play_timer.stop()
    
    def next_frame(self):
        if self.video_processor:
            info = self.video_processor.get_video_info()
            next_idx = (self.frame_slider.value() + 1) % info['frame_count']
            self.frame_slider.setValue(next_idx)
            self.show_frame(next_idx)
    
    def seek_frame(self, value):
        self.show_frame(value)

