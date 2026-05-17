import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
import trimesh
import open3d as o3d
from sklearn.cluster import DBSCAN


@dataclass
class Segment:
    segment_id: int
    points: np.ndarray
    color: Optional[np.ndarray] = None
    instance_id: Optional[int] = None
    semantic_label: Optional[str] = None
    
    def __post_init__(self):
        self.points = np.asarray(self.points, dtype=np.float32)
        if self.color is not None:
            self.color = np.asarray(self.color, dtype=np.float32)


class PointGroupBackbone(nn.Module):
    def __init__(self, in_channels: int = 6, out_channels: int = 128):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, out_channels, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(out_channels)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        return x


class PointGroupClustering(nn.Module):
    def __init__(self, radius: float = 0.3, nsample: int = 32):
        super().__init__()
        self.radius = radius
        self.nsample = nsample
    
    def forward(self, points, features, scores):
        return self._cluster_points(points, scores)
    
    def _cluster_points(self, points, scores):
        mask = scores > 0.5
        cluster_points = points[mask]
        
        if len(cluster_points) < 10:
            return []
        
        clustering = DBSCAN(eps=self.radius, min_samples=5).fit(cluster_points)
        labels = clustering.labels_
        
        clusters = []
        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1:
                continue
            cluster_mask = labels == label
            cluster_points_subset = cluster_points[cluster_mask]
            clusters.append(cluster_points_subset)
        
        return clusters


class PointGroupSegmentation(nn.Module):
    def __init__(self, num_classes: int = 13, feat_channels: int = 128):
        super().__init__()
        
        self.num_classes = num_classes
        
        self.backbone = PointGroupBackbone(in_channels=6, out_channels=feat_channels)
        
        self.semantic_layer = nn.Sequential(
            nn.Conv1d(feat_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, num_classes, 1)
        )
        
        self.offset_layer = nn.Sequential(
            nn.Conv1d(feat_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )
        
        self.score_layer = nn.Sequential(
            nn.Conv1d(feat_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, points, features):
        feat = self.backbone(features)
        
        semantic_scores = self.semantic_layer(feat)
        offset = self.offset_layer(feat)
        pixel_scores = self.score_layer(feat)
        
        return {
            'semantic': semantic_scores,
            'offset': offset,
            'score': pixel_scores
        }


class InstanceSegmentor:
    SEMANTIC_LABELS = [
        'ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
        'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter'
    ]
    
    def __init__(self, model_path: str = None, method: str = 'pointgroup', device: str = 'cuda'):
        self.method = method
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.num_classes = len(self.SEMANTIC_LABELS)
        
        self.model = PointGroupSegmentation(num_classes=self.num_classes).to(self.device)
        
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
    
    def segment(self, point_cloud: np.ndarray) -> List[Segment]:
        if isinstance(point_cloud, o3d.geometry.PointCloud):
            points = np.asarray(point_cloud.points)
            if point_cloud.has_colors():
                colors = np.asarray(point_cloud.colors)
            else:
                colors = np.ones((len(points), 3)) * 0.5
        else:
            points = point_cloud
            colors = np.ones((len(points), 3)) * 0.5
        
        features = self._compute_features(points, colors)
        
        points_tensor = torch.from_numpy(points.T).float().unsqueeze(0).to(self.device)
        features_tensor = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(points_tensor, features_tensor)
        
        segments = self._cluster_and_create_segments(
            points, colors, outputs, None
        )
        
        return segments
    
    def segment_with_detection(self, point_cloud: np.ndarray, 
                               detections: List) -> List[Segment]:
        if isinstance(point_cloud, o3d.geometry.PointCloud):
            points = np.asarray(point_cloud.points)
            if point_cloud.has_colors():
                colors = np.asarray(point_cloud.colors)
            else:
                colors = np.ones((len(points), 3)) * 0.5
        else:
            points = point_cloud
            colors = np.ones((len(points), 3)) * 0.5
        
        features = self._compute_features(points, colors)
        
        points_tensor = torch.from_numpy(points.T).float().unsqueeze(0).to(self.device)
        features_tensor = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(points_tensor, features_tensor)
        
        segments = self._cluster_and_create_segments(
            points, colors, outputs, detections
        )
        
        return segments
    
    def refine_segmentation(self, segments: List[Segment]) -> List[Segment]:
        refined_segments = []
        
        for i, segment in enumerate(segments):
            points = segment.points
            
            if len(points) < 50:
                continue
            
            centroid = np.mean(points, axis=0)
            distances = np.linalg.norm(points - centroid, axis=1)
            threshold = np.percentile(distances, 95)
            
            refined_mask = distances <= threshold
            refined_points = points[refined_mask]
            
            if len(refined_points) >= 10:
                refined_segment = Segment(
                    segment_id=segment.segment_id,
                    points=refined_points,
                    color=segment.color,
                    instance_id=segment.instance_id,
                    semantic_label=segment.semantic_label
                )
                refined_segments.append(refined_segment)
        
        return refined_segments
    
    def _compute_features(self, points: np.ndarray, 
                         colors: np.ndarray) -> np.ndarray:
        if colors.shape[1] == 4:
            colors = colors[:, :3]
        
        features = np.concatenate([colors.T, points.T], axis=0)
        
        return features.astype(np.float32)
    
    def _cluster_and_create_segments(self, points: np.ndarray, colors: np.ndarray,
                                    outputs: Dict, detections: List) -> List[Segment]:
        semantic_scores = torch.softmax(outputs['semantic'], dim=1)
        semantic_labels = torch.argmax(semantic_scores, dim=1).cpu().numpy()[0]
        
        pixel_scores = outputs['score'].cpu().numpy()[0, 0, :]
        
        clustering = DBSCAN(eps=0.3, min_samples=5)
        labels = clustering.fit_predict(points)
        
        segments = []
        segment_id = 0
        unique_labels = set(labels)
        
        for label in unique_labels:
            if label == -1:
                continue
            
            mask = labels == label
            segment_points = points[mask]
            segment_colors = colors[mask]
            
            semantic_idx = np.bincount(semantic_labels[mask]).argmax()
            semantic_label = self.SEMANTIC_LABELS[semantic_idx]
            
            instance_id = label
            
            segment = Segment(
                segment_id=segment_id,
                points=segment_points,
                color=np.mean(segment_colors, axis=0) if len(segment_colors) > 0 else np.array([0.5, 0.5, 0.5]),
                instance_id=instance_id,
                semantic_label=semantic_label
            )
            segments.append(segment)
            segment_id += 1
        
        return segments
    
    def get_segment_point_cloud(self, segment_id: int, 
                               segments: List[Segment]) -> np.ndarray:
        for segment in segments:
            if segment.segment_id == segment_id:
                return segment.points
        return np.array([])
    
    def save_model(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'semantic_labels': self.SEMANTIC_LABELS,
            'method': self.method
        }, path)
    
    def load_model(self, path: str):
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if 'semantic_labels' in checkpoint:
                self.SEMANTIC_LABELS = checkpoint['semantic_labels']
            if 'method' in checkpoint:
                self.method = checkpoint['method']
        except Exception as e:
            print(f"Warning: Could not load segmentor weights from {path}: {e}")
            print("Using initialized weights instead.")
            self._initialize_weights()
