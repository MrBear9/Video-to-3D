import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
import trimesh
import open3d as o3d


@dataclass
class Instance:
    class_id: int
    class_name: str
    bbox: np.ndarray
    confidence: float
    mask: Optional[np.ndarray] = None
    center: Optional[np.ndarray] = None
    dimensions: Optional[np.ndarray] = None
    
    def __post_init__(self):
        self.bbox = np.asarray(self.bbox, dtype=np.float32)
        if self.center is not None:
            self.center = np.asarray(self.center, dtype=np.float32)
        if self.dimensions is not None:
            self.dimensions = np.asarray(self.dimensions, dtype=np.float32)


class PointNetPPBackbone(nn.Module):
    def __init__(self, in_channels: int = 6, use_normals: bool = False):
        super().__init__()
        
        if use_normals:
            in_channels = 6
        else:
            in_channels = 3
        
        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        return x


class VoteNet(nn.Module):
    def __init__(self, num_classes: int = 13, num_proposals: int = 256):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_proposals = num_proposals
        
        self.backbone = PointNetPPBackbone(in_channels=3)
        
        self.vote_aggregation = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        self.vote_layer = nn.Sequential(
            nn.Conv1d(256, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 3 + self.num_classes, 1)
        )
        
        self.proposal_layer = nn.Sequential(
            nn.Conv1d(256, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 3 + 3 + 1 + self.num_classes, 1)
        )
    
    def forward(self, point_clouds: torch.Tensor):
        batch_size = point_clouds.shape[0]
        num_points = point_clouds.shape[2]
        
        features = self.backbone(point_clouds)
        
        aggregated = self.vote_aggregation(features)
        
        votes = self.vote_layer(aggregated)
        vote_xyz = votes[:, :3, :]
        vote_features = votes[:, 3:, :]
        
        proposals = self.proposal_layer(aggregated)
        proposal_xyz = proposals[:, :3, :]
        proposal_size = proposals[:, 3:6, :]
        proposal_score = proposals[:, 6:7, :]
        proposal_cls = proposals[:, 7:, :]
        
        return {
            'vote_xyz': vote_xyz,
            'proposal_xyz': proposal_xyz,
            'proposal_size': proposal_size,
            'proposal_score': proposal_score,
            'proposal_cls': proposal_cls
        }


class InstanceDetector:
    CLASS_NAMES = [
        'ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
        'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter'
    ]
    
    def __init__(self, model_path: str = None, device: str = 'cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.num_classes = len(self.CLASS_NAMES)
        
        self.model = VoteNet(num_classes=self.num_classes, num_proposals=256).to(self.device)
        
        if model_path:
            self.load_model(model_path)
        else:
            self._initialize_weights()
        
        self.model.eval()
    
    def _initialize_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def detect(self, point_cloud: np.ndarray) -> List[Instance]:
        if isinstance(point_cloud, o3d.geometry.PointCloud):
            points = np.asarray(point_cloud.points)
        else:
            points = point_cloud
        
        points = self._preprocess_points(points)
        
        batch_tensor = torch.from_numpy(points).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(batch_tensor)
        
        instances = self._postprocess_detections(outputs, points[0])
        
        return instances
    
    def detect_from_mesh(self, mesh: trimesh.Trimesh) -> List[Instance]:
        points, _ = trimesh.sample.sample_surface(mesh, 20000)
        return self.detect(points)
    
    def _preprocess_points(self, points: np.ndarray) -> np.ndarray:
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid
        
        max_dist = np.max(np.linalg.norm(points_centered, axis=1))
        if max_dist > 1e-6:
            points_normalized = points_centered / max_dist
        else:
            points_normalized = points_centered
        
        if points_normalized.ndim == 2:
            points_normalized = points_normalized.T
        
        if points_normalized.shape[0] < 3:
            points_normalized = np.pad(points_normalized, ((0, 3 - points_normalized.shape[0]), (0, 0)), mode='constant')
        
        points_tensor = np.zeros((3, points_normalized.shape[1]), dtype=np.float32)
        points_tensor[:points_normalized.shape[0]] = points_normalized
        
        return points_tensor[np.newaxis, :, :]
    
    def _postprocess_detections(self, outputs: Dict, original_points: np.ndarray) -> List[Instance]:
        proposal_xyz = outputs['proposal_xyz'][0].cpu().numpy()
        proposal_size = outputs['proposal_size'][0].cpu().numpy()
        proposal_score = torch.sigmoid(outputs['proposal_score'][0]).cpu().numpy()
        proposal_cls = torch.softmax(outputs['proposal_cls'][0], dim=0).cpu().numpy()
        
        instances = []
        
        for i in range(min(256, proposal_xyz.shape[1])):
            score = proposal_score[0, i]
            if score < 0.3:
                continue
            
            class_id = np.argmax(proposal_cls[:, i])
            class_name = self.CLASS_NAMES[class_id]
            
            center = proposal_xyz[:, i]
            size = np.exp(proposal_size[:, i])
            
            bbox = self._compute_bbox(center, size)
            
            instance = Instance(
                class_id=int(class_id),
                class_name=class_name,
                bbox=bbox,
                confidence=float(score),
                center=center,
                dimensions=size
            )
            instances.append(instance)
        
        return instances
    
    def _compute_bbox(self, center: np.ndarray, size: np.ndarray) -> np.ndarray:
        min_coords = center - size / 2
        max_coords = center + size / 2
        
        bbox = np.array([
            [min_coords[0], min_coords[1], min_coords[2]],
            [max_coords[0], min_coords[1], min_coords[2]],
            [max_coords[0], max_coords[1], min_coords[2]],
            [min_coords[0], max_coords[1], min_coords[2]],
            [min_coords[0], min_coords[1], max_coords[2]],
            [max_coords[0], min_coords[1], max_coords[2]],
            [max_coords[0], max_coords[1], max_coords[2]],
            [min_coords[0], max_coords[1], max_coords[2]]
        ])
        
        return bbox
    
    def visualize_detections(self, point_cloud: np.ndarray, detections: List[Instance]) -> np.ndarray:
        import open3d as o3d
        
        if isinstance(point_cloud, o3d.geometry.PointCloud):
            pcd = point_cloud
        else:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud)
        
        geometries = [pcd]
        
        num = len(detections)
        colors = [
            np.array([(i * 0.618033988749895) % 1.0,
                      (i * 0.618033988749895 + 0.333) % 1.0,
                      (i * 0.618033988749895 + 0.667) % 1.0,
                      1.0])
            for i in range(max(num, 1))
        ]
        
        for i, instance in enumerate(detections):
            bbox_mesh = self._create_bbox_mesh(instance.bbox, colors[i])
            geometries.append(bbox_mesh)
        
        return geometries
    
    def _create_bbox_mesh(self, bbox: np.ndarray, color: np.ndarray):
        bbox_pcd = o3d.geometry.PointCloud()
        bbox_pcd.points = o3d.utility.Vector3dVector(bbox)
        bbox_pcd.colors = o3d.utility.Vector3dVector(np.tile(color[:3], (8, 1)))
        
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]
        
        line_set = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
            bbox_pcd,
            bbox_pcd,
            lines
        )
        
        return line_set
    
    def save_model(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'class_names': self.CLASS_NAMES
        }, path)
    
    def load_model(self, path: str):
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if 'class_names' in checkpoint:
                self.CLASS_NAMES = checkpoint['class_names']
        except Exception as e:
            print(f"Warning: Could not load detector weights from {path}: {e}")
            print("Using initialized weights instead.")
            self._initialize_weights()
