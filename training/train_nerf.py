import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import open3d as o3d
import trimesh
import os
import sys

sys.path.append(str(Path(__file__).parent.parent))
from core.video_processor import VideoProcessor
from models.nerf_model import NeRFModel


class NeRFDataset(Dataset):
    def __init__(self, data_path: str, num_samples_per_image: int = 4096):
        self.data_path = Path(data_path)
        self.num_samples_per_image = num_samples_per_image
        self.images = []
        self.poses = []
        
        self._load_data()
    
    def _load_data(self):
        if self.data_path.exists():
            image_dir = self.data_path / 'images'
            if image_dir.exists():
                for img_file in sorted(image_dir.glob('*.png')):
                    img = o3d.io.read_image(str(img_file))
                    self.images.append(img)
        
        num_images = len(self.images)
        if num_images > 0:
            angles = np.linspace(0, 2 * np.pi, num_images, endpoint=False)
            radius = 4.0
            for angle in angles:
                pose = np.eye(4)
                pose[:3, 3] = [radius * np.cos(angle), 0.5, radius * np.sin(angle)]
                pose[:3, :3] = self._rotation_matrix_y(-angle)
                self.poses.append(pose)
    
    def _rotation_matrix_y(self, angle: float) -> np.ndarray:
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    
    def __len__(self) -> int:
        return len(self.images) * 100
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_idx = idx % len(self.images)
        
        img = self.images[img_idx]
        pose = self.poses[img_idx]
        
        h, w = 256, 256
        
        u = np.random.randint(0, w, self.num_samples_per_image)
        v = np.random.randint(0, h, self.num_samples_per_image)
        
        rays_o = torch.from_numpy(pose[:3, 3]).float().unsqueeze(0).repeat(self.num_samples_per_image, 1)
        
        x = (u - w / 2) / w
        y = (v - h / 2) / h
        dirs = np.stack([x, y, -np.ones(self.num_samples_per_image)], axis=1)
        dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8)
        
        rotation = pose[:3, :3]
        dirs_world = dirs @ rotation.T
        rays_d = torch.from_numpy(dirs_world).float()
        
        t_vals = np.random.uniform(0, 1, self.num_samples_per_image)
        t_vals = np.sort(t_vals)
        
        return {
            'rays_o': rays_o,
            'rays_d': rays_d,
            't_vals': torch.from_numpy(t_vals).float(),
            'pose': torch.from_numpy(pose).float()
        }


class NeRFTrainer:
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        self.device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.num_frequencies = self.config.get('num_frequencies', 10)
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.num_layers = self.config.get('num_layers', 8)
        self.learning_rate = self.config.get('learning_rate', 5e-4)
        self.batch_size = self.config.get('batch_size', 1024)
        self.num_epochs = self.config.get('num_epochs', 1000)
        
        self.model = NeRFModel(
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_frequencies=self.num_frequencies
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.5)
        
        self.train_losses = []
        self.val_losses = []
    
    def train(self, train_dataset: Dataset, val_dataset: Optional[Dataset] = None):
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True if self.device == 'cuda' else False
        )
        
        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0.0
            
            for batch in train_loader:
                rays_o = batch['rays_o'].to(self.device)
                rays_d = batch['rays_d'].to(self.device)
                t_vals = batch['t_vals'].to(self.device)
                
                rgb_pred, depth_pred = self._render_rays(rays_o, rays_d, t_vals)
                
                rgb_gt = torch.rand_like(rgb_pred)
                loss = nn.functional.mse_loss(rgb_pred, rgb_gt)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            self.scheduler.step()
            
            avg_loss = epoch_loss / len(train_loader)
            self.train_losses.append(avg_loss)
            
            if epoch % 100 == 0:
                print(f"Epoch [{epoch}/{self.num_epochs}], Loss: {avg_loss:.6f}")
            
            if epoch % 500 == 0 and epoch > 0:
                self.save_checkpoint(f'nerf_checkpoint_epoch_{epoch}.pth')
    
    def _render_rays(self, rays_o: torch.Tensor, rays_d: torch.Tensor,
                    t_vals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        num_rays = rays_o.shape[0]
        num_samples = t_vals.shape[0]
        
        bounds_min = torch.tensor([-1, -1, -1], device=self.device)
        bounds_max = torch.tensor([1, 1, 1], device=self.device)
        
        z_vals = bounds_min[2] + t_vals.unsqueeze(0) * (bounds_max[2] - bounds_min[2])
        
        pts = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(-1)
        pts = pts.reshape(-1, 3)
        
        dirs = rays_d.unsqueeze(1).repeat(1, num_samples, 1).reshape(-1, 3)
        
        rgb, sigma = self.model(pts, dirs)
        
        rgb = rgb.reshape(num_rays, num_samples, 3)
        sigma = sigma.reshape(num_rays, num_samples)
        
        sigma = torch.relu(sigma)
        
        alpha = 1.0 - torch.exp(-sigma * (z_vals[1] - z_vals[0]))
        
        weights = alpha * torch.cumprod(torch.cat([torch.ones((num_rays, 1), device=self.device), 
                                                    1.0 - alpha + 1e-10], dim=1), dim=1)[:, :-1]
        
        rgb_map = torch.sum(weights.unsqueeze(-1) * rgb, dim=1)
        depth_map = torch.sum(weights * z_vals.T, dim=1)
        
        return rgb_map, depth_map
    
    def validate(self, val_dataset: Dataset) -> float:
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for i in range(min(100, len(val_dataset))):
                batch = val_dataset[i]
                rays_o = batch['rays_o'].to(self.device)
                rays_d = batch['rays_d'].to(self.device)
                t_vals = batch['t_vals'].to(self.device)
                
                rgb_pred, _ = self._render_rays(rays_o, rays_d, t_vals)
                
                rgb_gt = torch.rand_like(rgb_pred)
                loss = nn.functional.mse_loss(rgb_pred, rgb_gt)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / 100
        self.val_losses.append(avg_val_loss)
        
        return avg_val_loss
    
    def save_checkpoint(self, path: str):
        checkpoint = {
            'epoch': len(self.train_losses),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint.get('scheduler_state_dict', {}))
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.config = checkpoint.get('config', self.config)
    
    def get_training_history(self) -> Dict[str, List[float]]:
        return {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses
        }
