from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import open3d as o3d
from config import Config


@dataclass
class TSDFReconstructionResult:
    mesh: o3d.geometry.TriangleMesh
    frames: int
    volume_type: str = "Open3D Scalable TSDF Fusion"


class RGBDTSDFReconstructor:
    """Mature RGB-D reconstruction based on Open3D TSDF fusion.

    This follows the mature RGB-D fusion family used by KinectFusion/Open3D:
    integrate RGB frames, depth frames, intrinsics and camera poses into a TSDF
    volume, then extract a triangle mesh. For the S3DIS-style point-cloud files
    in this project, tools/generate_s3dis_scene_video.py first renders a
    deterministic RGB-D walkthrough sequence from the labeled point cloud.
    """

    def __init__(
        self,
        voxel_length: float = None,
        sdf_trunc: float = None,
        depth_scale: float = None,
        depth_trunc: float = None,
    ):
        self.voxel_length = Config.TSDF_VOXEL_LENGTH if voxel_length is None else voxel_length
        self.sdf_trunc = Config.TSDF_SDF_TRUNC if sdf_trunc is None else sdf_trunc
        self.depth_scale = Config.TSDF_DEPTH_SCALE if depth_scale is None else depth_scale
        self.depth_trunc = Config.TSDF_DEPTH_TRUNC if depth_trunc is None else depth_trunc

    def reconstruct_from_project(self, project_dir: str) -> TSDFReconstructionResult:
        project = Path(project_dir)
        meta = self._load_meta(project)
        color_paths = sorted((project / "color").glob("*.png"))
        depth_paths = sorted((project / "depth").glob("*.png"))
        pose_paths = sorted((project / "pose").glob("*.txt"))
        if not color_paths or not depth_paths or not pose_paths:
            raise ValueError(f"RGB-D project is incomplete: {project}")

        count = min(len(color_paths), len(depth_paths), len(pose_paths))
        intrinsic = self._intrinsic(meta)
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self.voxel_length,
            sdf_trunc=self.sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

        used = 0
        for color_path, depth_path, pose_path in zip(color_paths[:count], depth_paths[:count], pose_paths[:count]):
            color = o3d.io.read_image(str(color_path))
            depth = o3d.io.read_image(str(depth_path))
            pose = np.loadtxt(pose_path).astype(np.float64)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color,
                depth,
                depth_scale=float(meta.get("depth_scale", self.depth_scale)),
                depth_trunc=float(meta.get("depth_trunc", self.depth_trunc)),
                convert_rgb_to_intensity=False,
            )
            volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))
            used += 1

        mesh = volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        try:
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_duplicated_vertices()
            mesh.remove_non_manifold_edges()
        except Exception:
            pass
        return TSDFReconstructionResult(mesh=mesh, frames=used)

    @staticmethod
    def project_dir_for_video(video_path: str) -> Optional[Path]:
        path = Path(video_path)
        candidates = [
            path.with_suffix(""),
            path.parent / f"{path.stem}_rgbd",
            Path("data/rgbd_sequences") / path.stem,
        ]
        for candidate in candidates:
            if (candidate / "meta.json").exists():
                return candidate
        return None

    def _load_meta(self, project: Path) -> Dict:
        import json

        meta_path = project / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing RGB-D metadata: {meta_path}")
        return json.loads(meta_path.read_text(encoding="utf-8"))

    def _intrinsic(self, meta: Dict):
        width = int(meta["width"])
        height = int(meta["height"])
        fx = float(meta["fx"])
        fy = float(meta["fy"])
        cx = float(meta["cx"])
        cy = float(meta["cy"])
        return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)


def write_depth_png(path: Path, depth_m: np.ndarray, depth_scale: float = 1000.0):
    depth_mm = np.clip(depth_m * depth_scale, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    cv2.imwrite(str(path), depth_mm)
