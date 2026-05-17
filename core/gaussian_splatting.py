import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import open3d as o3d
from pathlib import Path
import plyfile


@dataclass
class Gaussian3D:
    position: np.ndarray
    color: np.ndarray
    scale: np.ndarray
    rotation: np.ndarray
    opacity: float = 1.0
    
    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=np.float32)
        self.color = np.asarray(self.color, dtype=np.float32)
        self.scale = np.asarray(self.scale, dtype=np.float32)
        self.rotation = np.asarray(self.rotation, dtype=np.float32)
        self.opacity = float(self.opacity)


class GaussianSplattingReconstructor:
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.max_iterations = self.config.get('max_iterations', 30000)
        
        self.gaussians: List[Gaussian3D] = []
        self.positions: Optional[torch.Tensor] = None
        self.colors: Optional[torch.Tensor] = None
        self.scales: Optional[torch.Tensor] = None
        self.rotations: Optional[torch.Tensor] = None
        self.opacities: Optional[torch.Tensor] = None
        
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.background_color = torch.tensor([0.0, 0.0, 0.0], device=self.device)
    
    def initialize_gaussians(self, point_cloud: np.ndarray, colors: Optional[np.ndarray] = None):
        num_points = len(point_cloud)
        
        if colors is None:
            colors = np.ones((num_points, 3)) * 0.7
        
        positions = torch.from_numpy(point_cloud).float().to(self.device)
        colors_tensor = torch.from_numpy(colors).float().to(self.device)
        
        scales = torch.ones((num_points, 3), device=self.device) * 0.01
        rotations = torch.zeros((num_points, 4), device=self.device)
        rotations[:, 0] = 1.0
        opacities = torch.ones(num_points, device=self.device) * 0.5
        
        self.positions = nn.Parameter(positions, requires_grad=True)
        self.colors = nn.Parameter(colors_tensor, requires_grad=True)
        self.scales = nn.Parameter(torch.log(scales), requires_grad=True)
        self.rotations = nn.Parameter(rotations, requires_grad=True)
        self.opacities = nn.Parameter(torch.sigmoid(opacities), requires_grad=True)
        
        for i in range(num_points):
            self.gaussians.append(Gaussian3D(
                position=point_cloud[i],
                color=colors[i],
                scale=np.ones(3) * 0.01,
                rotation=self._quaternion_identity(),
                opacity=0.5
            ))
    
    def _quaternion_identity(self) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0, 0.0])
    
    def train(self, frames: List[np.ndarray], poses: np.ndarray, iterations: int = None):
        if iterations is None:
            iterations = self.max_iterations
        
        if self.positions is None:
            self._initialize_from_frames(frames, poses)
        
        self.optimizer = torch.optim.Adam([
            {'params': [self.positions], 'lr': 0.00016, 'name': 'positions'},
            {'params': [self.colors], 'lr': 0.0025, 'name': 'colors'},
            {'params': [self.scales], 'lr': 0.005, 'name': 'scales'},
            {'params': [self.rotations], 'lr': 0.001, 'name': 'rotations'},
            {'params': [self.opacities], 'lr': 0.05, 'name': 'opacities'},
        ])
        
        for iteration in range(iterations):
            loss = self._compute_loss(frames, poses)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration}/{iterations}, Loss: {loss.item():.6f}")
    
    def _initialize_from_frames(self, frames: List[np.ndarray], poses: np.ndarray):
        all_points = []
        all_colors = []
        
        for frame, pose in zip(frames, poses):
            points = self._depth_to_points(frame, pose)
            all_points.append(points)
            all_colors.append(self._image_to_colors(frame))
        
        point_cloud = np.concatenate(all_points, axis=0)
        colors = np.concatenate(all_colors, axis=0)
        
        if len(point_cloud) > 100000:
            indices = np.random.choice(len(point_cloud), 100000, replace=False)
            point_cloud = point_cloud[indices]
            colors = colors[indices]
        
        self.initialize_gaussians(point_cloud, colors)
    
    def _depth_to_points(self, frame: np.ndarray, pose: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        fx = fy = 500
        cx, cy = w / 2, h / 2
        
        if len(frame.shape) == 3:
            depth = frame[:, :, 0].astype(np.float32)
        else:
            depth = frame.astype(np.float32)
        
        y, x = np.where(depth > 0)
        z = depth[y, x]
        
        cam_coords = np.stack([
            (x - cx) * z / fx,
            (y - cy) * z / fy,
            z,
            np.ones_like(z)
        ], axis=0)
        
        world_coords = pose @ cam_coords
        
        return world_coords[:3].T
    
    def _image_to_colors(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        y, x = np.where(np.any(frame > 0, axis=-1))
        
        if len(y) == 0:
            return np.array([])
        
        colors = frame[y, x]
        
        if colors.dtype != np.float32:
            colors = colors.astype(np.float32) / 255.0
        
        return colors
    
    def _compute_loss(self, frames: List[np.ndarray], poses: np.ndarray) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=self.device)
        
        for _ in range(4):
            idx = np.random.randint(0, len(frames))
            frame = frames[idx]
            pose = poses[idx]
            
            rendered_np = self.render(viewpoint_camera=pose)
            rendered = torch.from_numpy(rendered_np).float().to(self.device)
            
            target = torch.from_numpy(frame[:, :, :3].astype(np.float32)).to(self.device) / 255.0
            if rendered.shape != target.shape:
                rendered = F.interpolate(
                    rendered.permute(2, 0, 1).unsqueeze(0),
                    size=(target.shape[0], target.shape[1]),
                    mode='bilinear'
                ).squeeze(0).permute(1, 2, 0)
            
            loss = F.mse_loss(rendered, target)
            total_loss = total_loss + loss
        
        return total_loss / 4.0
    
    def _densify_and_prune(self):
        if self.opacities is None:
            return
        
        pruned_mask = self.opacities.detach() > 0.005
        
        for param_name in ['positions', 'colors', 'scales', 'rotations', 'opacities']:
            param = getattr(self, param_name)
            if param is not None and param.shape[0] > pruned_mask.sum():
                setattr(self, param_name, nn.Parameter(param[pruned_mask].detach().clone(), requires_grad=True))
        
        if len(self.gaussians) > 0 and len(pruned_mask) == len(self.gaussians):
            self.gaussians = [g for g, m in zip(self.gaussians, pruned_mask) if m]
    
    def render(self, viewpoint_camera) -> np.ndarray:
        h, w = 256, 256
        
        if isinstance(viewpoint_camera, np.ndarray):
            pose = viewpoint_camera
        else:
            pose = self._quaternion_to_pose(viewpoint_camera)
        
        rendered_image = torch.zeros((h, w, 3), device=self.device)
        alpha_accum = torch.zeros((h, w, 1), device=self.device)
        
        if self.positions is None or len(self.positions) == 0:
            return rendered_image.cpu().numpy()
        
        view_matrix = torch.from_numpy(
            self._compute_view_matrix(pose).astype(np.float32)
        ).float().to(self.device)
        
        gaussian_positions = self.positions @ view_matrix[:3, :3].T + view_matrix[:3, 3]
        
        pos_np = gaussian_positions.detach().cpu().numpy()
        scale_np = torch.exp(self.scales).detach().cpu().numpy()
        opacity_np = self.opacities.detach().cpu().numpy()
        color_np = torch.sigmoid(self.colors).detach().cpu().numpy()
        
        for idx in range(len(self.positions)):
            p = pos_np[idx]
            depth = float(p[2]) + 1e-8
            u = float(p[0]) / depth * w / 2.0 + w / 2.0
            v = float(-p[1]) / depth * h / 2.0 + h / 2.0
            x, y = int(round(u)), int(round(v))
            
            if 0 <= x < w and 0 <= y < h:
                sigma = 1.0 / (float(np.prod(scale_np[idx])) + 1e-8)
                alpha = float(opacity_np[idx]) * sigma
                color = color_np[idx]
                
                rendered_image[y, x] += alpha * torch.from_numpy(color).float().to(self.device)
                alpha_accum[y, x] += alpha
        
        valid_mask = alpha_accum.squeeze(-1) > 0
        if valid_mask.any():
            rendered_image[valid_mask] = rendered_image[valid_mask] / (alpha_accum[valid_mask] + 1e-8)
        
        return rendered_image.cpu().numpy()
    
    def _compute_view_matrix(self, pose: np.ndarray) -> np.ndarray:
        view_matrix = np.eye(4)
        view_matrix[:3, :3] = pose[:3, :3].T
        view_matrix[:3, 3] = -pose[:3, :3].T @ pose[:3, 3]
        return view_matrix
    
    def _quaternion_to_pose(self, viewpoint_camera) -> np.ndarray:
        if hasattr(viewpoint_camera, 'rotation'):
            return viewpoint_camera.rotation
        elif hasattr(viewpoint_camera, 'pose'):
            return viewpoint_camera.pose
        else:
            return np.eye(4)
    
    def export_ply(self, path: str) -> bool:
        try:
            if self.positions is None or len(self.positions) == 0:
                return False
            
            positions = self.positions.detach().cpu().numpy()
            colors = (torch.sigmoid(self.colors).detach().cpu().numpy() * 255).astype(np.uint8)
            scales = torch.exp(self.scales).detach().cpu().numpy()
            opacities = self.opacities.detach().cpu().numpy()
            
            vertices = np.empty(len(positions), dtype=[(
                'x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
            ])
            
            vertices['x'] = positions[:, 0]
            vertices['y'] = positions[:, 1]
            vertices['z'] = positions[:, 2]
            vertices['nx'] = 0
            vertices['ny'] = 0
            vertices['nz'] = 0
            vertices['red'] = colors[:, 0]
            vertices['green'] = colors[:, 1]
            vertices['blue'] = colors[:, 2]
            
            el = plyfile.PlyElement.describe(vertices, 'vertex')
            plyfile.PlyData([el]).write(path)
            
            return True
        except Exception as e:
            print(f"Export failed: {e}")
            return False
    
    def export_point_cloud(self, path: str) -> bool:
        try:
            if self.positions is None or len(self.positions) == 0:
                return False
            
            positions = self.positions.detach().cpu().numpy()
            colors = (torch.sigmoid(self.colors).detach().cpu().numpy() * 255).astype(np.uint8)
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(positions)
            pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
            
            o3d.io.write_point_cloud(str(Path(path)), pcd)
            
            return True
        except Exception as e:
            print(f"Export failed: {e}")
            return False
    
    def save_checkpoint(self, path: str):
        checkpoint = {
            'positions': self.positions.detach().cpu() if self.positions is not None else None,
            'colors': self.colors.detach().cpu() if self.colors is not None else None,
            'scales': self.scales.detach().cpu() if self.scales is not None else None,
            'rotations': self.rotations.detach().cpu() if self.rotations is not None else None,
            'opacities': self.opacities.detach().cpu() if self.opacities is not None else None,
            'config': self.config
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        
        if checkpoint['positions'] is not None:
            self.positions = nn.Parameter(checkpoint['positions'].to(self.device), requires_grad=True)
        if checkpoint['colors'] is not None:
            self.colors = nn.Parameter(checkpoint['colors'].to(self.device), requires_grad=True)
        if checkpoint['scales'] is not None:
            self.scales = nn.Parameter(checkpoint['scales'].to(self.device), requires_grad=True)
        if checkpoint['rotations'] is not None:
            self.rotations = nn.Parameter(checkpoint['rotations'].to(self.device), requires_grad=True)
        if checkpoint['opacities'] is not None:
            self.opacities = nn.Parameter(checkpoint['opacities'].to(self.device), requires_grad=True)
        
        self.config = checkpoint.get('config', self.config)
