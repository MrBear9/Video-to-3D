
from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QMouseEvent, QWheelEvent
import OpenGL.GL as gl
import numpy as np
import open3d as o3d


class Viewport3D(QOpenGLWidget):
    model_loaded = pyqtSignal(object)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.models = []
        self.camera_pos = np.array([0.0, 0.0, 5.0])
        self.camera_target = np.array([0.0, 0.0, 0.0])
        self.camera_up = np.array([0.0, 1.0, 0.0])
        self.rotation_x = 0.0
        self.rotation_y = 0.0
        self.zoom = 5.0
        self.last_mouse_pos = None
        self.setMinimumSize(400, 400)
        
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(visible=False)
    
    def initializeGL(self):
        gl.glClearColor(0.1, 0.1, 0.15, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_LIGHT0)
    
    def resizeGL(self, w, h):
        gl.glViewport(0, 0, w, h)
    
    def paintGL(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        self.setup_camera()
        self.draw_models()
    
    def setup_camera(self):
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        
        aspect = self.width() / max(self.height(), 1)
        gl.glFrustum(-aspect*0.1, aspect*0.1, -0.1, 0.1, 0.1, 1000.0)
        
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        
        gl.glTranslatef(0, 0, -self.zoom)
        gl.glRotatef(self.rotation_x, 1, 0, 0)
        gl.glRotatef(self.rotation_y, 0, 1, 0)
    
    def draw_models(self):
        for model in self.models:
            self.draw_model(model)
    
    def draw_model(self, model):
        if isinstance(model, o3d.geometry.PointCloud):
            self.draw_point_cloud(model)
        elif hasattr(model, 'vertices') and hasattr(model, 'faces'):
            self.draw_mesh(model)
    
    def draw_point_cloud(self, pcd):
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else np.ones_like(points) * 0.7
        
        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        gl.glEnableClientState(gl.GL_COLOR_ARRAY)
        gl.glVertexPointer(3, gl.GL_DOUBLE, 0, points)
        gl.glColorPointer(3, gl.GL_DOUBLE, 0, colors)
        gl.glPointSize(2.0)
        gl.glDrawArrays(gl.GL_POINTS, 0, len(points))
        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
        gl.glDisableClientState(gl.GL_COLOR_ARRAY)
    
    def draw_mesh(self, mesh):
        pass
    
    def add_model(self, model):
        self.models.append(model)
        self.model_loaded.emit(model)
        self.update()
    
    def clear_models(self):
        self.models = []
        self.update()
    
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.last_mouse_pos = event.pos()
    
    def mouseMoveEvent(self, event: QMouseEvent):
        if self.last_mouse_pos is not None:
            dx = event.x() - self.last_mouse_pos.x()
            dy = event.y() - self.last_mouse_pos.y()
            
            self.rotation_y += dx * 0.5
            self.rotation_x += dy * 0.5
            
            self.last_mouse_pos = event.pos()
            self.update()
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.last_mouse_pos = None
    
    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y() / 120
        self.zoom -= delta * 0.5
        self.zoom = max(1.0, min(self.zoom, 50.0))
        self.update()

