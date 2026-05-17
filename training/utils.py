import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from pathlib import Path
import json


class MetricsCalculator:
    def __init__(self, num_classes: int = 13):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        self.total_loss = 0.0
        self.num_samples = 0
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.class_correct = np.zeros(self.num_classes, dtype=np.int64)
        self.class_total = np.zeros(self.num_classes, dtype=np.int64)
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, loss: float):
        pred_flat = predictions.view(-1)
        target_flat = targets.view(-1)
        
        for t, p in zip(target_flat.cpu().numpy(), pred_flat.cpu().numpy()):
            if 0 <= t < self.num_classes and 0 <= p < self.num_classes:
                self.confusion_matrix[t, p] += 1
        
        for c in range(self.num_classes):
            mask = (targets == c)
            self.class_total[c] += mask.sum().item()
            self.class_correct[c] += ((predictions == c) & mask).sum().item()
        
        self.total_loss += loss
        self.num_samples += 1
    
    def get_metrics(self) -> Dict[str, float]:
        if self.num_samples == 0:
            return {}
        
        avg_loss = self.total_loss / self.num_samples
        
        class_accuracies = []
        for c in range(self.num_classes):
            if self.class_total[c] > 0:
                acc = 100.0 * self.class_correct[c] / self.class_total[c]
            else:
                acc = 0.0
            class_accuracies.append(acc)
        
        overall_acc = 100.0 * np.sum(np.diag(self.confusion_matrix)) / np.sum(self.confusion_matrix)
        
        iou_per_class = []
        for c in range(self.num_classes):
            tp = self.confusion_matrix[c, c]
            fp = np.sum(self.confusion_matrix[:, c]) - tp
            fn = np.sum(self.confusion_matrix[c, :]) - tp
            
            if tp + fp + fn > 0:
                iou = tp / (tp + fp + fn)
            else:
                iou = 0.0
            iou_per_class.append(iou)
        
        mean_iou = np.mean(iou_per_class)
        
        return {
            'loss': avg_loss,
            'overall_accuracy': overall_acc,
            'mean_iou': mean_iou,
            'class_accuracies': class_accuracies,
            'class_ious': iou_per_class
        }


class TrainingVisualizer:
    def __init__(self, save_dir: str = 'logs'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'learning_rate': []
        }
    
    def log_epoch(self, epoch: int, metrics: Dict):
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if key not in self.history:
                    self.history[key] = []
                self.history[key].append(value)
        
        self._save_history()
    
    def plot_training_curves(self):
        if len(self.history['train_loss']) == 0:
            print("No training history to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        if self.history.get('val_loss'):
            axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        if self.history.get('train_accuracy'):
            axes[0, 1].plot(self.history['train_accuracy'], label='Train Acc')
            if self.history.get('val_accuracy'):
                axes[0, 1].plot(self.history['val_accuracy'], label='Val Acc')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy (%)')
            axes[0, 1].set_title('Training and Validation Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        if self.history.get('learning_rate'):
            axes[1, 0].plot(self.history['learning_rate'])
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].grid(True)
        
        if self.history.get('mean_iou'):
            axes[1, 1].plot(self.history['mean_iou'], label='mIoU')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('mIoU')
            axes[1, 1].set_title('Mean Intersection over Union')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_curves.png', dpi=150)
        plt.close()
        
        print(f"Training curves saved to {self.save_dir / 'training_curves.png'}")
    
    def plot_confusion_matrix(self, confusion_matrix: np.ndarray, class_names: List[str] = None):
        if class_names is None:
            class_names = [f'Class {i}' for i in range(confusion_matrix.shape[0])]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(confusion_matrix, cmap='Blues')
        
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticklabels(class_names)
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                text = ax.text(j, i, confusion_matrix[i, j],
                             ha="center", va="center", color="black")
        
        ax.set_title("Confusion Matrix")
        fig.tight_layout()
        plt.colorbar(im, ax=ax)
        plt.savefig(self.save_dir / 'confusion_matrix.png', dpi=150)
        plt.close()
        
        print(f"Confusion matrix saved to {self.save_dir / 'confusion_matrix.png'}")
    
    def _save_history(self):
        history_file = self.save_dir / 'training_history.json'
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)


def save_model_architecture(model: nn.Module, save_path: str):
    architecture = {}
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            architecture[name] = {
                'type': type(module).__name__,
                'parameters': sum(p.numel() for p in module.parameters())
            }
    
    with open(save_path, 'w') as f:
        json.dump(architecture, f, indent=2)
    
    print(f"Model architecture saved to {save_path}")


def compute_segmentation_metrics(pred: torch.Tensor, target: torch.Tensor,
                                 num_classes: int = 13) -> Dict[str, float]:
    pred_flat = pred.view(-1).cpu().numpy()
    target_flat = target.view(-1).cpu().numpy()
    
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(target_flat, pred_flat):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            confusion[t, p] += 1
    
    tp = np.diag(confusion)
    fp = np.sum(confusion, axis=0) - tp
    fn = np.sum(confusion, axis=1) - tp
    
    iou = tp / (tp + fp + fn + 1e-8)
    mean_iou = np.mean(iou)
    
    accuracy = 100.0 * np.sum(tp) / (np.sum(confusion) + 1e-8)
    
    return {
        'accuracy': accuracy,
        'mean_iou': mean_iou,
        'class_iou': iou.tolist()
    }


class CheckpointManager:
    def __init__(self, save_dir: str = 'checkpoints', max_keep: int = 5):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_keep = max_keep
        self.checkpoints = []
    
    def save_checkpoint(self, epoch: int, model: nn.Module, 
                       optimizer, metrics: Dict, filename: str = None):
        if filename is None:
            filename = f'checkpoint_epoch_{epoch}.pth'
        
        save_path = self.save_dir / filename
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }
        
        torch.save(checkpoint, save_path)
        
        self.checkpoints.append(save_path)
        
        if len(self.checkpoints) > self.max_keep:
            old_checkpoint = self.checkpoints.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()
        
        print(f"Checkpoint saved: {save_path}")
        
        return save_path
    
    def load_checkpoint(self, path: str, model: nn.Module, 
                      optimizer = None) -> Dict:
        checkpoint = torch.load(path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Checkpoint loaded: {path}")
        
        return checkpoint
    
    def load_latest_checkpoint(self, model: nn.Module, 
                              optimizer = None) -> Optional[Dict]:
        if not self.checkpoints:
            checkpoint_files = sorted(self.save_dir.glob('checkpoint_*.pth'))
            if not checkpoint_files:
                print("No checkpoints found")
                return None
            self.checkpoints = checkpoint_files
        
        latest = self.checkpoints[-1]
        return self.load_checkpoint(str(latest), model, optimizer)
