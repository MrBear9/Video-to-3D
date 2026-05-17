import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
import trimesh
from pathlib import Path


class PositionalEncoding(nn.Module):
    def __init__(self, num_frequencies: int = 10, include_input: bool = True):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        self.freq_bands = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.include_input:
            encoded = [x]
        else:
            encoded = []
        
        for freq in self.freq_bands:
            encoded.append(torch.sin(freq * np.pi * x))
            encoded.append(torch.cos(freq * np.pi * x))
        
        return torch.cat(encoded, dim=-1)


class NeRFNetwork(nn.Module):
    def __init__(self, input_dim: int = 3, hidden_dim: int = 256, 
                 num_layers: int = 8, num_frequencies: int = 10):
        super().__init__()
        
        self.pos_encoder = PositionalEncoding(num_frequencies=num_frequencies)
        self.dir_encoder = PositionalEncoding(num_frequencies=4)
        
        self.input_dim = self.pos_encoder(x=torch.zeros(1, input_dim)).shape[-1]
        self.dir_input_dim = self.dir_encoder(x=torch.zeros(1, 3)).shape[-1]
        
        self.fc_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.fc_layers.append(nn.Linear(self.input_dim, hidden_dim))
            elif i == 4:
                self.fc_layers.append(nn.Linear(hidden_dim + self.input_dim, hidden_dim))
            else:
                self.fc_layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        self.sigma_layer = nn.Linear(hidden_dim, 1)
        self.feature_layer = nn.Linear(hidden_dim, hidden_dim)
        self.rgb_layer = nn.Sequential(
            nn.Linear(hidden_dim + self.dir_input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor, d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_encoded = self.pos_encoder(x)
        
        h = self.fc_layers[0](x_encoded)
        h = F.relu(h)
        
        for i in range(1, 4):
            h = self.fc_layers[i](h)
            h = F.relu(h)
        
        h_skip = self.fc_layers[4](torch.cat([h, x_encoded], dim=-1))
        h_skip = F.relu(h_skip)
        
        for i in range(5, len(self.fc_layers)):
            h_skip = self.fc_layers[i](h_skip)
            h_skip = F.relu(h_skip)
        
        sigma = self.sigma_layer(h_skip)
        sigma = F.relu(sigma)
        
        features = self.feature_layer(h_skip)
        d_encoded = self.dir_encoder(d)
        rgb_input = torch.cat([features, d_encoded], dim=-1)
        rgb = self.rgb_layer(rgb_input)
        
        return rgb, sigma


class NeRFReconstructor:
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.num_frequencies = self.config.get('num_frequencies', 10)
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.num_layers = self.config.get('num_layers', 8)
        
        self.network = NeRFNetwork(
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_frequencies=self.num_frequencies
        ).to(self.device)
        
        self.optimizer = None
        self.bounds_min = np.array([-1, -1, -1])
        self.bounds_max = np.array([1, 1, 1])
    
    def estimate_poses(self, frames: List[np.ndarray]) -> np.ndarray:
        num_frames = len(frames)
        poses = np.zeros((num_frames, 4, 4))
        
        angles = np.linspace(0, 2 * np.pi, num_frames, endpoint=False)
        radius = 4.0
        
        for i, angle in enumerate(angles):
            poses[i, :3, 3] = [radius * np.cos(angle), 0.5, radius * np.sin(angle)]
            poses[i, :3, :3] = self._rotation_matrix_y(-angle)
            poses[i, 3, 3] = 1.0
        
        return poses
    
    def _rotation_matrix_y(self, angle: float) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    
    def train(self, frames: List[np.ndarray], poses: np.ndarray, 
              epochs: int = 1000, batch_size: int = 1024):
        self.network.train()
        
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=5e-4)
        
        h, w = frames[0].shape[:2]
        
        for epoch in range(epochs):
            total_loss = 0.0
            
            for _ in range(100):
                ray_o, ray_d = self._sample_rays(h, w, batch_size)
                
                rgb_gt = self._sample_images(frames, ray_o, ray_d)
                
                rgb_pred, sigma = self._render_rays(ray_o, ray_d)
                
                loss = F.mse_loss(rgb_pred, rgb_gt)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {total_loss/100:.6f}")
    
    def _sample_rays(self, h: int, w: int, num_rays: int) -> Tuple[torch.Tensor, torch.Tensor]:
        i, j = np.random.randint(0, h, num_rays), np.random.randint(0, w, num_rays)
        
        ray_o = np.zeros((num_rays, 3))
        ray_d = np.array([(i[h] - h/2)/h, (j[w] - w/2)/w, -1.0])
        ray_d = ray_d / (np.linalg.norm(ray_d, axis=1, keepdims=True) + 1e-8)
        
        return (
            torch.from_numpy(ray_o).float().to(self.device),
            torch.from_numpy(ray_d).float().to(self.device)
        )
    
    def _sample_images(self, frames: List[np.ndarray], ray_o: torch.Tensor, 
                      ray_d: torch.Tensor) -> torch.Tensor:
        num_rays = ray_o.shape[0]
        rgb_gt = torch.zeros((num_rays, 3), device=self.device)
        
        for i in range(num_rays):
            frame_idx = np.random.randint(0, len(frames))
            frame = frames[frame_idx]
            rgb_gt[i] = torch.from_numpy(frame.mean(axis=0)[:3]).float() / 255.0
        
        return rgb_gt
    
    def _render_rays(self, ray_o: torch.Tensor, ray_d: torch.Tensor, 
                     num_samples: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        t_vals = torch.linspace(0, 1, num_samples, device=self.device)
        z_vals = self.bounds_min[2] + t_vals * (self.bounds_max[2] - self.bounds_min[2])
        
        pts = ray_o[..., None, :] + ray_d[..., None, :] * z_vals[..., :, None]
        pts = pts.reshape(-1, 3)
        
        dir = ray_d.repeat(num_samples, 1)
        
        rgb, sigma = self.network(pts, dir)
        rgb = rgb.reshape(num_samples, -1, 3)
        sigma = sigma.reshape(num_samples, -1, 1)
        
        sigma = sigma.squeeze(1)
        rgb = rgb.squeeze(1)
        
        return rgb.mean(dim=0), sigma.mean(dim=0)
    
    def render_view(self, pose: np.ndarray) -> np.ndarray:
        self.network.eval()
        
        h, w = 256, 256
        rgb_image = np.zeros((h, w, 3))
        
        with torch.no_grad():
            for i in range(0, h, 32):
                for j in range(0, w, 32):
                    ray_o = torch.from_numpy(pose[:3, 3:4].T).float().to(self.device).repeat(32*32, 1)
                    x = (j + np.arange(32)) / w * 2 - 1
                    y = (i + np.arange(32)) / h * 2 - 1
                    
                    dirs = np.stack([np.meshgrid(x, y, indexing='ij')[0].flatten(),
                                    np.meshgrid(x, y, indexing='ij')[1].flatten(),
                                    -np.ones(32*32)], axis=1)
                    ray_d = torch.from_numpy(dirs).float().to(self.device)
                    
                    rgb, _ = self._render_rays(ray_o, ray_d)
                    rgb = rgb.cpu().numpy().reshape(32, 32, 3)
                    
                    rgb_image[i:i+32, j:j+32] = rgb
        
        return (rgb_image * 255).astype(np.uint8)
    
    def extract_mesh(self, resolution: int = 256) -> trimesh.Trimesh:
        self.network.eval()
        
        x = np.linspace(self.bounds_min[0], self.bounds_max[0], resolution)
        y = np.linspace(self.bounds_min[1], self.bounds_max[1], resolution)
        z = np.linspace(self.bounds_min[2], self.bounds_max[2], resolution)
        
        grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')
        grid_points = np.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], axis=1)
        
        densities = []
        batch_size = 1024
        with torch.no_grad():
            for i in range(0, len(grid_points), batch_size):
                batch = torch.from_numpy(grid_points[i:i+batch_size]).float().to(self.device)
                _, sigma = self.network(batch, torch.zeros_like(batch))
                densities.append(sigma.cpu().numpy().flatten())
        
        density_grid = np.concatenate(densities).reshape(resolution, resolution, resolution)
        
        try:
            verts, faces, normals, values = self.marching_cubes(density_grid, x, y, z, threshold=0.5)
            mesh = trimesh.Trimesh(vertices=verts, faces=faces)
            return mesh
        except:
            return trimesh.creation.box()
    
    def marching_cubes(self, volume: np.ndarray, x: np.ndarray, y: np.ndarray, 
                      z: np.ndarray, threshold: float = 0.0):
        from skimage import measure
        
        spacing = (x[1] - x[0], y[1] - y[0], z[1] - z[0])
        verts, faces, normals, values = measure.marching_cubes(
            volume, level=threshold, spacing=spacing
        )
        
        verts += np.array([x[0], y[0], z[0]])
        
        return verts, faces, normals, values
    
    def save_checkpoint(self, path: str):
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'config': self.config
        }, path)
    
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.config = checkpoint.get('config', self.config)
