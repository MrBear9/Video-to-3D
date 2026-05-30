from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QMouseEvent, QWheelEvent
import OpenGL.GL as gl
import numpy as np
import open3d as o3d
from config import Config


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
        self.selected_object = None
        self.lighting_mode = Config.LIGHTING_MODE
        self.lighting_intensity = Config.LIGHTING_INTENSITY
        self.setMinimumSize(400, 400)
        self.setFocusPolicy(Qt.StrongFocus)

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(visible=False)

    def initializeGL(self):
        gl.glClearColor(*Config.VIEWPORT_BACKGROUND)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_LIGHT0)
        gl.glEnable(gl.GL_LIGHT1)
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, [0.0, 0.0, 1.0, 0.0])
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, [0.75, 0.75, 0.75, 1.0])
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT, [0.45, 0.45, 0.45, 1.0])
        gl.glLightfv(gl.GL_LIGHT1, gl.GL_POSITION, [0.0, 0.0, -1.0, 0.0])
        gl.glLightfv(gl.GL_LIGHT1, gl.GL_DIFFUSE, [0.35, 0.35, 0.35, 1.0])
        gl.glLightModelfv(gl.GL_LIGHT_MODEL_AMBIENT, [0.55, 0.55, 0.55, 1.0])
        gl.glLightModeli(gl.GL_LIGHT_MODEL_TWO_SIDE, gl.GL_TRUE)
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
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, [0.0, 0.0, 1.0, 0.0])
        gl.glLightfv(gl.GL_LIGHT1, gl.GL_POSITION, [0.0, 0.0, -1.0, 0.0])

    def draw_models(self):
        for model in self.models:
            self.draw_model(model)

    def draw_model(self, model):
        if isinstance(model, o3d.geometry.PointCloud):
            self.draw_point_cloud(model)
        elif isinstance(model, o3d.geometry.TriangleMesh):
            self.draw_mesh(model)
        elif hasattr(model, 'points') and not hasattr(model, 'faces'):
            self.draw_simple_point_cloud(model)
        elif hasattr(model, 'vertices') and hasattr(model, 'faces'):
            self.draw_mesh(model)
        self.draw_selection_overlay(model)

    def draw_point_cloud(self, pcd):
        points = np.asarray(pcd.points, dtype=np.float32)
        colors = self._normalize_colors(
            np.asarray(pcd.colors) if pcd.has_colors() else None,
            len(points),
            Config.DEFAULT_POINT_COLOR,
        )
        colors = self._apply_display_lighting(colors)

        gl.glDisable(gl.GL_LIGHTING)
        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        gl.glEnableClientState(gl.GL_COLOR_ARRAY)
        gl.glVertexPointer(3, gl.GL_FLOAT, 0, points)
        gl.glColorPointer(3, gl.GL_FLOAT, 0, colors)
        gl.glPointSize(float(Config.POINT_SIZE))
        gl.glDrawArrays(gl.GL_POINTS, 0, len(points))
        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
        gl.glDisableClientState(gl.GL_COLOR_ARRAY)
        gl.glEnable(gl.GL_LIGHTING)

    def draw_simple_point_cloud(self, pcd):
        points = np.asarray(pcd.points, dtype=np.float32)
        colors = self._normalize_colors(
            np.asarray(pcd.colors) if hasattr(pcd, 'colors') else None,
            len(points),
            Config.DEFAULT_POINT_COLOR,
        )
        colors = self._apply_display_lighting(colors)

        gl.glDisable(gl.GL_LIGHTING)
        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        gl.glEnableClientState(gl.GL_COLOR_ARRAY)
        gl.glVertexPointer(3, gl.GL_FLOAT, 0, points)
        gl.glColorPointer(3, gl.GL_FLOAT, 0, colors)
        gl.glPointSize(float(Config.SIMPLE_POINT_SIZE))
        gl.glDrawArrays(gl.GL_POINTS, 0, len(points))
        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
        gl.glDisableClientState(gl.GL_COLOR_ARRAY)
        gl.glEnable(gl.GL_LIGHTING)

    def draw_mesh(self, mesh):
        try:
            import trimesh
            is_trimesh = isinstance(mesh, trimesh.Trimesh)
        except ImportError:
            is_trimesh = False

        if is_trimesh or (hasattr(mesh, 'vertices') and hasattr(mesh, 'faces') and not isinstance(mesh, o3d.geometry.TriangleMesh)):
            vertices = mesh.vertices
            faces = mesh.faces

            if hasattr(mesh.visual, 'vertex_colors'):
                colors = self._normalize_colors(mesh.visual.vertex_colors[:, :3], len(vertices), Config.DEFAULT_MESH_COLOR)
            else:
                colors = self._normalize_colors(None, len(vertices), Config.DEFAULT_MESH_COLOR)

            vertices = vertices.astype(np.float32)
            colors = colors.astype(np.float32)

            vertex_array = vertices[faces].reshape(-1, 3)
            color_array = self._apply_display_lighting(colors)[faces].reshape(-1, 3)

            gl.glDisable(gl.GL_LIGHTING)
            gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
            gl.glEnableClientState(gl.GL_COLOR_ARRAY)
            gl.glVertexPointer(3, gl.GL_FLOAT, 0, vertex_array)
            gl.glColorPointer(3, gl.GL_FLOAT, 0, color_array)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, vertex_array.shape[0])
            gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
            gl.glDisableClientState(gl.GL_COLOR_ARRAY)
            if Config.MESH_SHOW_WIREFRAME_OVERLAY:
                self._draw_wireframe(vertex_array)
            gl.glEnable(gl.GL_LIGHTING)
        elif isinstance(mesh, o3d.geometry.TriangleMesh):
            vertices = np.asarray(mesh.vertices).astype(np.float32)
            triangles = np.asarray(mesh.triangles)

            if mesh.has_vertex_colors():
                colors = self._normalize_colors(np.asarray(mesh.vertex_colors), len(vertices), Config.DEFAULT_MESH_COLOR)
            else:
                colors = self._normalize_colors(None, len(vertices), Config.DEFAULT_MESH_COLOR)

            vertex_array = vertices[triangles].reshape(-1, 3)
            color_array = self._apply_display_lighting(colors)[triangles].reshape(-1, 3)

            gl.glDisable(gl.GL_LIGHTING)
            gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
            gl.glEnableClientState(gl.GL_COLOR_ARRAY)
            gl.glVertexPointer(3, gl.GL_FLOAT, 0, vertex_array)
            gl.glColorPointer(3, gl.GL_FLOAT, 0, color_array)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, vertex_array.shape[0])
            gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
            gl.glDisableClientState(gl.GL_COLOR_ARRAY)
            if Config.MESH_SHOW_WIREFRAME_OVERLAY:
                self._draw_wireframe(vertex_array)
            gl.glEnable(gl.GL_LIGHTING)

    def _normalize_colors(self, colors, count, fallback):
        if colors is None or len(colors) != count:
            arr = np.tile(np.asarray(fallback, dtype=np.float32), (count, 1))
        else:
            arr = np.asarray(colors, dtype=np.float32)
            if arr.ndim == 1:
                arr = np.tile(arr[:3], (count, 1))
            arr = arr[:, :3]
            if arr.size and arr.max() > 1.0:
                arr = arr / 255.0
            arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
            arr = np.clip(arr, 0.0, 1.0)
            brightness = arr.mean(axis=1)
            dark = brightness < Config.MIN_COLOR_BRIGHTNESS
            if np.any(dark):
                arr[dark] = np.asarray(fallback, dtype=np.float32)
        return np.ascontiguousarray(arr.astype(np.float32))

    def _apply_display_lighting(self, colors):
        tint = np.asarray(Config.LIGHTING_TINTS.get(self.lighting_mode, (1.0, 1.0, 1.0)), dtype=np.float32)
        intensity = float(np.clip(self.lighting_intensity, 0.15, 1.6))
        if self.lighting_mode == 'ambient':
            factor = 0.75 + intensity * 0.35
        elif self.lighting_mode == 'directional':
            factor = 0.55 + intensity * 0.65
        elif self.lighting_mode == 'point':
            factor = 0.45 + intensity * 0.85
        elif self.lighting_mode == 'spot':
            factor = 0.35 + intensity * 1.05
        else:
            factor = 1.0
        return np.ascontiguousarray(np.clip(colors * tint * factor, 0.0, 1.0).astype(np.float32))

    def _draw_wireframe(self, vertex_array):
        if len(vertex_array) == 0:
            return
        gl.glDisable(gl.GL_LIGHTING)
        gl.glDisableClientState(gl.GL_COLOR_ARRAY)
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        gl.glColor3f(*Config.MESH_WIREFRAME_COLOR)
        gl.glLineWidth(float(Config.MESH_WIREFRAME_WIDTH))
        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        gl.glVertexPointer(3, gl.GL_FLOAT, 0, vertex_array)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, vertex_array.shape[0])
        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        gl.glLineWidth(1.0)

    def draw_selection_overlay(self, model):
        if self.selected_object is None:
            return

        points = self._selection_points(self.selected_object)
        if points is None or len(points) == 0:
            return

        bbox = self._bbox_from_points(points)
        self._draw_bbox(bbox, color=(1.0, 0.85, 0.15), line_width=3.0)

        if points.shape[0] > 8000:
            idx = np.linspace(0, points.shape[0] - 1, 8000).astype(np.int64)
            points = points[idx]

        gl.glDisable(gl.GL_LIGHTING)
        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        gl.glVertexPointer(3, gl.GL_FLOAT, 0, points.astype(np.float32))
        gl.glColor3f(1.0, 0.92, 0.05)
        gl.glPointSize(float(Config.SELECTION_POINT_SIZE))
        gl.glDrawArrays(gl.GL_POINTS, 0, len(points))
        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
        gl.glPointSize(float(Config.POINT_SIZE))
        gl.glEnable(gl.GL_LIGHTING)

    def _selection_points(self, data):
        if hasattr(data, 'points'):
            return np.asarray(data.points, dtype=np.float32)
        if hasattr(data, 'bbox'):
            return np.asarray(data.bbox, dtype=np.float32)
        if hasattr(data, 'vertices'):
            return np.asarray(data.vertices, dtype=np.float32)
        return None

    def _bbox_from_points(self, points):
        min_coords = points.min(axis=0)
        max_coords = points.max(axis=0)
        return np.array([
            [min_coords[0], min_coords[1], min_coords[2]],
            [max_coords[0], min_coords[1], min_coords[2]],
            [max_coords[0], max_coords[1], min_coords[2]],
            [min_coords[0], max_coords[1], min_coords[2]],
            [min_coords[0], min_coords[1], max_coords[2]],
            [max_coords[0], min_coords[1], max_coords[2]],
            [max_coords[0], max_coords[1], max_coords[2]],
            [min_coords[0], max_coords[1], max_coords[2]],
        ], dtype=np.float32)

    def _draw_bbox(self, bbox, color=(1.0, 0.85, 0.15), line_width=2.5):
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        gl.glDisable(gl.GL_LIGHTING)
        gl.glColor3f(*color)
        gl.glLineWidth(line_width)
        gl.glBegin(gl.GL_LINES)
        for a, b in edges:
            gl.glVertex3f(*bbox[a])
            gl.glVertex3f(*bbox[b])
        gl.glEnd()
        gl.glLineWidth(1.0)
        gl.glEnable(gl.GL_LIGHTING)

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

    def select_object(self, data):
        self.selected_object = data
        self.update()

    def remove_model(self, model):
        self.models = [m for m in self.models if m is not model]
        if self.selected_object is model:
            self.selected_object = None
        self.update()

    def fit_view(self):
        if not self.models:
            return
        self.selected_object = None
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.rotation_x = 0.0
        self.rotation_y = 0.0
        self._fit_view_to_model(self.models[-1])
        self.update()

    def _fit_view_to_model(self, model):
        try:
            import trimesh
            is_trimesh = isinstance(model, trimesh.Trimesh)
        except ImportError:
            is_trimesh = False

        if is_trimesh or (hasattr(model, 'vertices') and hasattr(model, 'faces') and not isinstance(model, o3d.geometry.TriangleMesh)):
            pts = model.vertices
        elif isinstance(model, o3d.geometry.PointCloud):
            pts = np.asarray(model.points)
        elif hasattr(model, 'points'):
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
        self.selected_object = None
        self.rotation_x = 0.0
        self.rotation_y = 0.0
        self.zoom = 5.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.update()

    def apply_lighting(self, light_type='ambient', intensity=0.5):
        self.lighting_mode = light_type
        self.lighting_intensity = intensity
        self.makeCurrent()
        if light_type == 'ambient':
            gl.glEnable(gl.GL_LIGHT0)
            gl.glEnable(gl.GL_LIGHT1)
            ambient = [max(intensity, 0.35)] * 3 + [1.0]
            diffuse = [max(intensity * 0.5, 0.25)] * 3 + [1.0]
            gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT, ambient)
            gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, diffuse)
        elif light_type == 'directional':
            gl.glEnable(gl.GL_LIGHT0)
            gl.glEnable(gl.GL_LIGHT1)
            gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, [0.0, 0.0, 1.0, 0.0])
            gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, [max(intensity, 0.35)] * 3 + [1.0])
            gl.glLightfv(gl.GL_LIGHT1, gl.GL_POSITION, [0.0, 0.0, -1.0, 0.0])
            gl.glLightfv(gl.GL_LIGHT1, gl.GL_DIFFUSE, [max(intensity * 0.45, 0.2)] * 3 + [1.0])
        elif light_type == 'point':
            gl.glEnable(gl.GL_LIGHT0)
            gl.glEnable(gl.GL_LIGHT1)
            gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, [0.0, 0.0, 1.0, 0.0])
            gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, [max(intensity, 0.35)] * 3 + [1.0])
        elif light_type == 'spot':
            gl.glEnable(gl.GL_LIGHT0)
            gl.glEnable(gl.GL_LIGHT1)
            gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, [0.0, 0.0, 1.0, 0.0])
            gl.glLightfv(gl.GL_LIGHT0, gl.GL_SPOT_DIRECTION, [0.0, -0.3, -1.0])
            gl.glLightf(gl.GL_LIGHT0, gl.GL_SPOT_CUTOFF, 30.0)
            gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, [max(intensity, 0.35)] * 3 + [1.0])
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
