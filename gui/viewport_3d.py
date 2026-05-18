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
        self.rotation_x = 0.0
        self.rotation_y = 0.0
        self.zoom = 5.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.last_mouse_pos = None
        self._mouse_action = None
        self.setMinimumSize(400, 400)
        self.setFocusPolicy(Qt.StrongFocus)

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(visible=False)

    def initializeGL(self):
        gl.glClearColor(0.12, 0.12, 0.16, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_LIGHT0)
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, [5.0, 10.0, 5.0, 1.0])
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, [0.9, 0.9, 0.9, 1.0])
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
        gl.glEnable(gl.GL_COLOR_MATERIAL)
        gl.glColorMaterial(gl.GL_FRONT_AND_BACK, gl.GL_DIFFUSE)

    def resizeGL(self, w, h):
        gl.glViewport(0, 0, w, h)

    def paintGL(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        self.setup_camera()
        self.draw_models()
        self.draw_axes()

    def setup_camera(self):
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        aspect = self.width() / max(self.height(), 1)
        gl.glFrustum(-aspect * 0.1, aspect * 0.1, -0.1, 0.1, 0.1, 1000.0)

        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        gl.glTranslatef(0, 0, -self.zoom)
        gl.glRotatef(self.rotation_x, 1, 0, 0)
        gl.glRotatef(self.rotation_y, 0, 1, 0)
        gl.glTranslatef(self.pan_x, self.pan_y, 0)

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
        import trimesh
        if isinstance(mesh, trimesh.Trimesh):
            vertices = mesh.vertices
            faces = mesh.faces

            if hasattr(mesh.visual, 'vertex_colors'):
                colors = mesh.visual.vertex_colors[:, :3] / 255.0
            else:
                colors = np.ones_like(vertices) * 0.7

            vertices = vertices.astype(np.float32)
            colors = colors.astype(np.float32)

            vertex_array = vertices[faces].reshape(-1, 3)
            color_array = colors[faces].reshape(-1, 3)

            gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
            gl.glEnableClientState(gl.GL_COLOR_ARRAY)
            gl.glVertexPointer(3, gl.GL_FLOAT, 0, vertex_array)
            gl.glColorPointer(3, gl.GL_FLOAT, 0, color_array)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, vertex_array.shape[0])
            gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
            gl.glDisableClientState(gl.GL_COLOR_ARRAY)
        elif isinstance(mesh, o3d.geometry.TriangleMesh):
            vertices = np.asarray(mesh.vertices).astype(np.float32)
            triangles = np.asarray(mesh.triangles)

            if mesh.has_vertex_colors():
                colors = np.asarray(mesh.vertex_colors).astype(np.float32)
            else:
                colors = np.ones_like(vertices) * 0.7

            vertex_array = vertices[triangles].reshape(-1, 3)
            color_array = colors[triangles].reshape(-1, 3)

            gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
            gl.glEnableClientState(gl.GL_COLOR_ARRAY)
            gl.glVertexPointer(3, gl.GL_FLOAT, 0, vertex_array)
            gl.glColorPointer(3, gl.GL_FLOAT, 0, color_array)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, vertex_array.shape[0])
            gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
            gl.glDisableClientState(gl.GL_COLOR_ARRAY)

    def draw_axes(self):
        size = min(self.width(), self.height()) * 0.12
        margin = 30

        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPushMatrix()
        gl.glLoadIdentity()
        gl.glOrtho(0, self.width(), 0, self.height(), -1000, 1000)

        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glPushMatrix()
        gl.glLoadIdentity()

        ox, oy = margin, margin

        gl.glDisable(gl.GL_LIGHTING)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glLineWidth(2.5)

        origin = np.array([ox, oy, 0.0])
        scale = size / np.sqrt(3)

        axes = {
            'X': (np.array([1, 0, 0]), np.array([1.0, 0.2, 0.2])),
            'Y': (np.array([0, 1, 0]), np.array([0.2, 1.0, 0.2])),
            'Z': (np.array([0, 0, 1]), np.array([0.2, 0.4, 1.0])),
        }

        for label, (direction, color) in axes.items():
            gl.glColor3f(*color)
            gl.glBegin(gl.GL_LINES)
            gl.glVertex3f(origin[0], origin[1], origin[2])
            end = origin + direction * scale
            gl.glVertex3f(end[0], end[1], end[2])
            gl.glEnd()

        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_LIGHTING)
        gl.glLineWidth(1.0)

        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPopMatrix()
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glPopMatrix()

    def add_model(self, model):
        self.models.append(model)
        self._fit_view_to_model(model)
        self.model_loaded.emit(model)
        self.update()

    def _fit_view_to_model(self, model):
        import trimesh
        if isinstance(model, trimesh.Trimesh):
            pts = model.vertices
        elif isinstance(model, o3d.geometry.PointCloud):
            pts = np.asarray(model.points)
        elif isinstance(model, o3d.geometry.TriangleMesh):
            pts = np.asarray(model.vertices)
        else:
            return

        if len(pts) == 0:
            return

        center = np.mean(pts, axis=0)
        extents = np.max(pts, axis=0) - np.min(pts, axis=0)
        max_extent = max(extents)

        if max_extent > 0.01:
            self.pan_x = -center[0] * 0.5 / max(max_extent, 1.0)
            self.pan_y = -center[1] * 0.5 / max(max_extent, 1.0)
            self.zoom = max(3.0, max_extent * 1.2)

    def clear_models(self):
        self.models = []
        self.rotation_x = 0.0
        self.rotation_y = 0.0
        self.zoom = 5.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.update()

    def apply_lighting(self, light_type='ambient', intensity=0.5):
        self.makeCurrent()
        if light_type == 'ambient':
            gl.glDisable(gl.GL_LIGHT0)
            gl.glDisable(gl.GL_LIGHT1)
            ambient = [intensity * 0.6, intensity * 0.6, intensity * 0.6, 1.0]
            diffuse = [intensity * 0.8, intensity * 0.8, intensity * 0.8, 1.0]
            gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT, ambient)
            gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, diffuse)
            gl.glEnable(gl.GL_LIGHT0)
        elif light_type == 'directional':
            gl.glEnable(gl.GL_LIGHT0)
            gl.glEnable(gl.GL_LIGHT1)
            gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, [1.0, 2.0, 1.0, 1.0])
            gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, [intensity, intensity, intensity, 1.0])
            gl.glLightfv(gl.GL_LIGHT1, gl.GL_POSITION, [-1.0, -1.0, -0.5, 1.0])
            gl.glLightfv(gl.GL_LIGHT1, gl.GL_DIFFUSE, [intensity*0.3, intensity*0.3, intensity*0.3, 1.0])
        elif light_type == 'point':
            gl.glEnable(gl.GL_LIGHT0)
            gl.glDisable(gl.GL_LIGHT1)
            gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, [0.0, 3.0, 2.0, 1.0])
            gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, [intensity, intensity, intensity, 1.0])
        elif light_type == 'spot':
            gl.glEnable(gl.GL_LIGHT0)
            gl.glDisable(gl.GL_LIGHT1)
            gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, [0.0, 2.0, 5.0, 1.0])
            gl.glLightfv(gl.GL_LIGHT0, gl.GL_SPOT_DIRECTION, [0.0, -0.3, -1.0])
            gl.glLightf(gl.GL_LIGHT0, gl.GL_SPOT_CUTOFF, 30.0)
            gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, [intensity, intensity, intensity, 1.0])
        self.doneCurrent()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self._mouse_action = 'pan'
            self.last_mouse_pos = event.pos()
        elif event.button() == Qt.MiddleButton:
            self._mouse_action = 'pan'
            self.last_mouse_pos = event.pos()
        elif event.button() == Qt.RightButton:
            self._mouse_action = 'rotate'
            self.last_mouse_pos = event.pos()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.last_mouse_pos is None or self._mouse_action is None:
            return

        dx = event.x() - self.last_mouse_pos.x()
        dy = event.y() - self.last_mouse_pos.y()

        if dx == 0 and dy == 0:
            return

        if self._mouse_action == 'rotate':
            self.rotation_y += dx * 0.5
            self.rotation_x += dy * 0.5
        elif self._mouse_action == 'pan':
            self.pan_x += dx * 0.005 * (self.zoom / 5.0)
            self.pan_y -= dy * 0.005 * (self.zoom / 5.0)

        self.last_mouse_pos = event.pos()
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        btn = event.button()
        if btn == Qt.LeftButton and self._mouse_action == 'pan':
            self._mouse_action = None
            self.last_mouse_pos = None
        elif btn == Qt.MiddleButton and self._mouse_action == 'pan':
            self._mouse_action = None
            self.last_mouse_pos = None
        elif btn == Qt.RightButton and self._mouse_action == 'rotate':
            self._mouse_action = None
            self.last_mouse_pos = None

    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y() / 120
        self.zoom -= delta * 0.5
        self.zoom = max(1.5, min(self.zoom, 100.0))
        self.update()

    def cleanup(self):
        if hasattr(self, 'vis') and self.vis is not None:
            try:
                self.vis.destroy_window()
            except Exception:
                pass
            self.vis = None

    def closeEvent(self, event):
        self.cleanup()
        super().closeEvent(event)

    def __del__(self):
        self.cleanup()