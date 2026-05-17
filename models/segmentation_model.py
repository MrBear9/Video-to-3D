import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import numpy as np


class PointNetPlus(nn.Module):
    def __init__(self, in_channels: int = 6, num_classes: int = 20,
                 use_normals: bool = False):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32,
                                          in_channel=in_channels + 3, mlp=[64, 64, 128])
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64,
                                          in_channel=128 + 3, mlp=[128, 128, 256])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None,
                                          in_channel=256 + 3, mlp=[256, 512, 1024],
                                          group_all=True)

        self.fc_layer = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, xyz: torch.Tensor):
        B = xyz.size(0)
        coords = xyz[:, :3, :]
        features = xyz

        l1_xyz, l1_points = self.sa1(coords, features)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        x = l3_points.view(B, -1)
        x = self.fc_layer(x)

        return x, l2_xyz, l2_points, l1_xyz, l1_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint: int, radius: float, nsample: int,
                 in_channel: int, mlp: List[int], group_all: bool = False):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, coords: torch.Tensor, features: torch.Tensor):
        B, _, N = coords.shape
        coords_t = coords.permute(0, 2, 1)

        if self.group_all:
            new_xyz = torch.zeros(B, 1, 3, device=coords.device)
            grouped = coords_t.view(B, 1, N, 3)
            if features is not None:
                feats_t = features.permute(0, 2, 1)
                grouped_feats = feats_t.unsqueeze(1).repeat(1, 1, 1, 1)
        else:
            fps_idx = farthest_point_sample(coords_t, self.npoint)
            new_xyz = index_points(coords_t, fps_idx)
            idx = query_ball_point(self.radius, self.nsample, coords_t, new_xyz)
            grouped = index_points(coords_t, idx)
            grouped = grouped - new_xyz.view(B, self.npoint, 1, 3)

            if features is not None:
                feats_t = features.permute(0, 2, 1)
                grouped_feats = index_points(feats_t, idx)
            else:
                grouped_feats = None

        if features is not None:
            grouped = torch.cat([grouped, grouped_feats], dim=-1)

        grouped = grouped.permute(0, 3, 2, 1)

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            grouped = F.relu(bn(conv(grouped)))

        grouped = torch.max(grouped, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)

        return new_xyz, grouped


class SegmentationModel(nn.Module):
    def __init__(self, num_classes: int = 20, in_channels: int = 6):
        super().__init__()

        self.num_classes = num_classes

        self.encoder = PointNetPlus(in_channels=in_channels, num_classes=num_classes)

        fp1_dim = 256 + 128

        self.fp1 = nn.Sequential(
            nn.Conv1d(fp1_dim, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Conv1d(256, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Conv1d(128 + in_channels, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(128, num_classes, 1)
        )

    def forward(self, xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, N = xyz.shape

        global_feat, l2_xyz, l2_points, l1_xyz, l1_points = self.encoder(xyz)

        l2_points_up = F.interpolate(l2_points, size=l1_points.shape[-1],
                                     mode='nearest') if l1_points.shape[-1] > 0 else l2_points

        concat = torch.cat([l2_points_up, l1_points], dim=1)
        concat = self.fp1(concat)

        concat_up = F.interpolate(concat, size=N, mode='nearest')
        xyz_feat = torch.cat([concat_up, xyz], dim=1)
        x = self.classifier(xyz_feat)

        return x, l2_xyz


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    B, N, C = xyz.shape
    xyz_coords = xyz[:, :, :3]

    centroids = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
    distance = torch.ones(B, N, device=xyz.device) * 1e10

    farthest = torch.randint(0, N, (B,), device=xyz.device)

    batch_indices = torch.arange(B, device=xyz.device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz_coords[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz_coords - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return centroids


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def query_ball_point(radius: float, nsample: int,
                     xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape

    group_idx = torch.arange(N, dtype=torch.long, device=xyz.device).view(1, 1, N).repeat([B, S, 1])

    sqrdists = square_distance(new_xyz, xyz)

    group_idx[sqrdists > radius ** 2] = N

    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]

    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]

    return group_idx


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist