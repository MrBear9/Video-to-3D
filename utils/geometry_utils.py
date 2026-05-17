
import numpy as np


def compute_bounding_box(points: np.ndarray) -&gt; np.ndarray:
    if points.ndim == 1:
        points = points.reshape(1, -1)
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    return np.array([min_coords, max_coords])


def compute_center(points: np.ndarray) -&gt; np.ndarray:
    return np.mean(points, axis=0)


def normalize_points(points: np.ndarray) -&gt; np.ndarray:
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    max_dist = np.max(np.linalg.norm(points_centered, axis=1))
    if max_dist &gt; 0:
        points_normalized = points_centered / max_dist
        return points_normalized, centroid, max_dist
    return points, centroid, 1.0


def rotation_matrix_x(angle: float) -&gt; np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rotation_matrix_y(angle: float) -&gt; np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotation_matrix_z(angle: float) -&gt; np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

