import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from models.segmentation_model import SegmentationModel


class SegmentationDataset(Dataset):
    def __init__(self, data_path: str, split: str = 'train',
                 num_points: int = 2048, transforms=None):
        self.data_path = Path(data_path)
        self.split = split
        self.num_points = num_points
        self.transforms = transforms

        self.point_clouds = []
        self.labels = []

        self._load_data()

    def _load_data(self):
        split_dir = self.data_path / self.split
        if not split_dir.exists():
            print(f"Warning: {split_dir} does not exist. Loading sample data.")
            self._generate_sample_data()
            return

        for room_file in sorted(split_dir.glob('*.npy')):
            try:
                data = np.load(room_file, allow_pickle=True).item()
                points = data.get('points', np.random.randn(2048, 6))
                labels = data.get('labels', np.random.randint(0, 13, 2048))

                if len(points) >= self.num_points:
                    idx = np.random.choice(len(points), self.num_points, replace=False)
                    self.point_clouds.append(torch.from_numpy(points[idx]).float())
                    self.labels.append(torch.from_numpy(labels[idx]).long())
            except Exception as e:
                pass

    def _generate_sample_data(self):
        for i in range(100):
            points = np.random.randn(self.num_points, 6).astype(np.float32)
            points[:, :3] = points[:, :3] * 2.0
            labels = np.random.randint(0, 13, self.num_points).astype(np.int64)

            self.point_clouds.append(torch.from_numpy(points))
            self.labels.append(torch.from_numpy(labels))

    def __len__(self) -> int:
        return len(self.point_clouds)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        points = self.point_clouds[idx]
        labels = self.labels[idx]

        if self.transforms:
            points = self.transforms(points)

        points = points.T.contiguous()

        return {
            'points': points,
            'labels': labels
        }


class SegmentationTrainer:
    def __init__(self, config: Dict = None):
        self.config = config or {}

        self.device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = self.config.get('num_classes', 13)
        self.in_channels = self.config.get('in_channels', 6)
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.batch_size = self.config.get('batch_size', 8)
        self.num_epochs = self.config.get('num_epochs', 100)

        self.model = SegmentationModel(
            num_classes=self.num_classes,
            in_channels=self.in_channels
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)
        self.criterion = nn.CrossEntropyLoss()

        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def prepare_dataset(self, dataset_name: str = 'scannet', data_path: str = None):
        if data_path is None:
            data_path = Path(__file__).parent.parent / 'data' / 'datasets' / dataset_name

        if not data_path.exists():
            print(f"Dataset path {data_path} does not exist.")
            print("Will use generated sample data for training.")

        return data_path

    def build_model(self, backbone: str = 'pointnet++'):
        print(f"Using {backbone} backbone")
        return self.model

    def train_epoch(self, epoch: int, train_loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            points = batch['points'].to(self.device)
            labels = batch['labels'].to(self.device)

            self.optimizer.zero_grad()

            outputs, aux_xyz = self.model(points)

            outputs_flat = outputs.permute(0, 2, 1).contiguous().view(-1, self.num_classes)
            labels_flat = labels.view(-1)

            loss = self.criterion(outputs_flat, labels_flat)

            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

            _, predicted = torch.max(outputs_flat.data, 1)
            total += labels_flat.size(0)
            correct += (predicted == labels_flat.data).sum().item()

        avg_loss = epoch_loss / max(len(train_loader), 1)
        accuracy = 100.0 * correct / max(total, 1)

        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        batch_count = 0

        with torch.no_grad():
            for batch in val_loader:
                points = batch['points'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs, aux_xyz = self.model(points)

                outputs_flat = outputs.permute(0, 2, 1).contiguous().view(-1, self.num_classes)
                labels_flat = labels.view(-1)

                loss = self.criterion(outputs_flat, labels_flat)
                val_loss += loss.item()

                _, predicted = torch.max(outputs_flat.data, 1)
                total += labels_flat.size(0)
                correct += (predicted == labels_flat.data).sum().item()
                batch_count += 1

        avg_val_loss = val_loss / max(batch_count, 1)
        val_accuracy = 100.0 * correct / max(total, 1)

        return {
            'loss': avg_val_loss,
            'accuracy': val_accuracy
        }

    def run(self, train_dataset: Dataset, val_dataset: Optional[Dataset] = None,
            epochs: int = None):
        if epochs is None:
            epochs = self.num_epochs

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device == 'cuda' else False,
            drop_last=True
        )

        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True if self.device == 'cuda' else False,
                drop_last=True
            )

        print(f"Starting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Number of parameters: {self._count_parameters():,}")

        for epoch in range(epochs):
            train_metrics = self.train_epoch(epoch, train_loader)

            self.train_losses.append(train_metrics['loss'])
            self.train_accuracies.append(train_metrics['accuracy'])

            val_metrics = None
            if val_loader:
                val_metrics = self.validate(val_loader)
                self.val_losses.append(val_metrics['loss'])
                self.val_accuracies.append(val_metrics['accuracy'])

            self.scheduler.step()

            if (epoch + 1) % 5 == 0:
                msg = f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_metrics['loss']:.4f}, "
                msg += f"Train Acc: {train_metrics['accuracy']:.2f}%"
                if val_metrics:
                    msg += f", Val Loss: {val_metrics['loss']:.4f}, "
                    msg += f"Val Acc: {val_metrics['accuracy']:.2f}%"
                print(msg)

            if (epoch + 1) % 50 == 0:
                self.save_checkpoint(f'segmentation_checkpoint_epoch_{epoch+1}.pth')

        print("Training completed!")

    def _count_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def save_checkpoint(self, path: str):
        checkpoint = {
            'epoch': len(self.train_losses),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
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
        self.train_accuracies = checkpoint.get('train_accuracies', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])

    def get_metrics(self) -> Dict[str, List[float]]:
        return {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'train_accuracy': self.train_accuracies,
            'val_accuracy': self.val_accuracies
        }