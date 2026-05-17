import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict
import numpy as np


class VoteNet3DDetection(nn.Module):
    def __init__(self, num_classes: int = 13, num_proposals: int = 256,
                 vote_percentage: float = 0.25):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_proposals = num_proposals
        self.vote_percentage = vote_percentage
        
        self.backbone = BackbonePointNet(in_channels=3, output_dim=256)
        
        self.vote_stage = VoteStage(in_dim=256, mid_dim=256, out_dim=256)
        
        self.proposal_stage = ProposalLayer(
            num_classes=num_classes,
            num_proposals=num_proposals
        )
    
    def forward(self, point_clouds: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(point_clouds)
        
        votes, vote_features = self.vote_stage(features, point_clouds)
        
        proposals = self.proposal_stage(votes, vote_features)
        
        return {
            'votes': votes,
            'vote_features': vote_features,
            'proposals': proposals
        }


class BackbonePointNet(nn.Module):
    def __init__(self, in_channels: int = 3, output_dim: int = 256):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, output_dim, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(output_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xyz = x[:, :3, :].permute(0, 2, 1)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        features = x
        
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.repeat(1, 1, x.size(2))
        
        return features, xyz


class VoteStage(nn.Module):
    def __init__(self, in_dim: int = 256, mid_dim: int = 256, out_dim: int = 256):
        super().__init__()
        
        self.vote_mlp = nn.Sequential(
            nn.Conv1d(in_dim, mid_dim, 1),
            nn.BatchNorm1d(mid_dim),
            nn.ReLU(),
            nn.Conv1d(mid_dim, mid_dim, 1),
            nn.BatchNorm1d(mid_dim),
            nn.ReLU(),
            nn.Conv1d(mid_dim, out_dim + 3, 1)
        )
    
    def forward(self, features: torch.Tensor, 
                points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        votes_delta = self.vote_mlp(features)
        
        vote_xyz = votes_delta[:, :3, :]
        vote_features = votes_delta[:, 3:, :]
        
        votes = points + vote_xyz
        
        return votes, vote_features


class ProposalLayer(nn.Module):
    def __init__(self, num_classes: int = 13, num_proposals: int = 256):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_proposals = num_proposals
        
        self.score_mlp = nn.Sequential(
            nn.Conv1d(259, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 2 + num_classes, 1)
        )
        
        self.size_mlp = nn.Sequential(
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 6, 1)
        )
    
    def forward(self, votes: torch.Tensor, 
                vote_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = votes.size(0)
        
        xyz = votes.mean(dim=2, keepdim=True).repeat(1, 1, votes.size(2), 1)
        
        proposal_input = torch.cat([vote_features, xyz, votes - xyz], dim=1)
        
        scores = self.score_mlp(proposal_input)
        box_scores = scores[:, :2, :]
        cls_scores = scores[:, 2:, :]
        
        box_sizes = torch.exp(self.size_mlp(vote_features))
        
        top_k_indices = torch.topk(box_scores[:, 1, :], 
                                   k=min(self.num_proposals, votes.size(2)))[1]
        
        proposals = {
            'scores': box_scores,
            'cls_scores': cls_scores,
            'boxes': box_sizes,
            'top_k_indices': top_k_indices
        }
        
        return proposals


class DetectionModel(nn.Module):
    def __init__(self, num_classes: int = 13, num_proposals: int = 256):
        super().__init__()
        
        self.votenet = VoteNet3DDetection(
            num_classes=num_classes,
            num_proposals=num_proposals
        )
    
    def forward(self, point_clouds: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.votenet(point_clouds)
    
    def nms(self, boxes: torch.Tensor, scores: torch.Tensor, 
            iou_threshold: float = 0.5) -> torch.Tensor:
        if boxes.size(0) == 0:
            return torch.tensor([], dtype=torch.long, device=boxes.device)
        
        x1 = boxes[:, 0] - boxes[:, 3] / 2
        y1 = boxes[:, 1] - boxes[:, 4] / 2
        z1 = boxes[:, 2] - boxes[:, 5] / 2
        x2 = boxes[:, 0] + boxes[:, 3] / 2
        y2 = boxes[:, 1] + boxes[:, 4] / 2
        z2 = boxes[:, 2] + boxes[:, 5] / 2
        
        areas = (x2 - x1) * (y2 - y1) * (z2 - z1)
        
        _, order = scores.sort(0, descending=True)
        
        keep = []
        while order.numel() > 0:
            if order.numel() == 1:
                keep.append(order.item())
                break
            
            i = order[0].item()
            keep.append(i)
            
            xx1 = torch.maximum(x1[i], x1[order[1:]])
            yy1 = torch.maximum(y1[i], y1[order[1:]])
            zz1 = torch.maximum(z1[i], z1[order[1:]])
            xx2 = torch.minimum(x2[i], x2[order[1:]])
            yy2 = torch.minimum(y2[i], y2[order[1:]])
            zz2 = torch.minimum(z2[i], z2[order[1:]])
            
            inter = torch.clamp(xx2 - xx1, min=0) * \
                    torch.clamp(yy2 - yy1, min=0) * \
                    torch.clamp(zz2 - zz1, min=0)
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = (iou <= iou_threshold).nonzero(as_tuple=False).squeeze()
            
            if inds.numel() == 0:
                break
            order = order[inds + 1]
        
        return torch.tensor(keep, dtype=torch.long, device=boxes.device)
