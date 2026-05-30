from pathlib import Path


class Config:
    ROOT_DIR = Path(__file__).resolve().parent

    SUPPORTED_VIDEO_FORMATS = [".mp4", ".avi", ".mov", ".mkv"]
    SUPPORTED_MODEL_IMPORT_FORMATS = [".obj", ".ply", ".stl", ".glb", ".gltf"]
    SUPPORTED_MODEL_EXPORT_FORMATS = [".obj", ".ply", ".stl", ".glb", ".gltf"]

    # 3D viewport rendering. Increase point sizes if the scene still looks sparse.
    VIEWPORT_BACKGROUND = (0.10, 0.11, 0.13, 1.0)
    POINT_SIZE = 4.0
    SIMPLE_POINT_SIZE = 4.5
    SELECTION_POINT_SIZE = 7.0
    MESH_SHOW_WIREFRAME_OVERLAY = True
    MESH_WIREFRAME_COLOR = (0.08, 0.08, 0.08)
    MESH_WIREFRAME_WIDTH = 0.6
    DEFAULT_MESH_COLOR = (0.72, 0.78, 0.86)
    DEFAULT_POINT_COLOR = (0.78, 0.82, 0.90)
    MIN_COLOR_BRIGHTNESS = 0.18
    LIGHTING_MODE = "ambient"
    LIGHTING_INTENSITY = 0.75
    LIGHTING_TINTS = {
        "ambient": (1.0, 1.0, 1.0),
        "directional": (1.0, 0.94, 0.82),
        "point": (0.82, 0.92, 1.0),
        "spot": (1.0, 0.82, 0.72),
    }

    # Reconstruction quality. Smaller voxel length gives denser meshes but costs time/memory.
    TSDF_VOXEL_LENGTH = 0.025
    TSDF_SDF_TRUNC = 0.10
    TSDF_DEPTH_SCALE = 1000.0
    TSDF_DEPTH_TRUNC = 8.0

    # Sampling used by detection/segmentation and generated scene display.
    ANALYSIS_SAMPLE_POINTS = 50000
    GENERATED_SCENE_MAX_POINTS = 300000
    GENERATED_VIDEO_FRAMES = 144
    GENERATED_VIDEO_SIZE = 720
    GENERATED_CAMERA_PATH_MODE = "coverage"

    # Point-cloud to mesh export parameters.
    EXPORT_NORMAL_RADIUS = 0.12
    EXPORT_NORMAL_MAX_NN = 30
    EXPORT_POISSON_DEPTH = 8
