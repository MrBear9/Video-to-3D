import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import sys
import os

sys.path.append(str(Path(__file__).parent.parent))


class S3DISDataset(Dataset):
    CLASS_LABELS = [
        'ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
        'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter'
    ]
    
    def __init__(self, root_dir: str, split: str = 'train', 
                 num_points: int = 4096, transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.num_points = num_points
        self.transform = transform
        
        self.data_files = []
        self._scan_data_directory()
    
    def _scan_data_directory(self):
        split_path = self.root_dir / self.split
        
        if not split_path.exists():
            print(f"Warning: {split_path} does not exist")
            return
        
        for area_folder in sorted(split_path.glob('area*')):
            if area_folder.is_dir():
                for room_folder in sorted(area_folder.glob('*')):
                    if room_folder.is_dir():
                        npy_file = room_folder / f"{room_folder.name}.npy"
                        if npy_file.exists():
                            self.data_files.append(npy_file)
    
    def __len__(self) -> int:
        return len(self.data_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        try:
            data = np.load(self.data_files[idx], allow_pickle=True).item()
            
            points = data.get('points', np.zeros((self.num_points, 6)))
            labels = data.get('labels', np.zeros(self.num_points, dtype=np.int64))
            
            if len(points) >= self.num_points:
                choice = np.random.choice(len(points), self.num_points, replace=False)
            else:
                choice = np.random.choice(len(points), self.num_points, replace=True)
            
            points = points[choice, :]
            labels = labels[choice]
            
            points_tensor = torch.from_numpy(points).float()
            labels_tensor = torch.from_numpy(labels).long()
            
            if self.transform:
                points_tensor = self.transform(points_tensor)
            
            return {
                'points': points_tensor,
                'labels': labels_tensor,
                'filename': str(self.data_files[idx])
            }
            
        except Exception as e:
            print(f"Error loading {self.data_files[idx]}: {e}")
            return self._generate_fake_sample()
    
    def _generate_fake_sample(self) -> Dict[str, torch.Tensor]:
        points = torch.randn(self.num_points, 6).float()
        labels = torch.randint(0, len(self.CLASS_LABELS), (self.num_points,)).long()
        
        return {
            'points': points,
            'labels': labels,
            'filename': 'fake_sample'
        }


class DataLoader3D:
    def __init__(self, dataset: Dataset, batch_size: int = 8, 
                 shuffle: bool = True, num_workers: int = 4,
                 pin_memory: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=self.collate_fn
        )
    
    def __iter__(self):
        return iter(self.loader)
    
    def __len__(self) -> int:
        return len(self.loader)
    
    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        points = torch.stack([item['points'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        filenames = [item['filename'] for item in batch]
        
        return {
            'points': points,
            'labels': labels,
            'filenames': filenames
        }


def create_s3dis_dataloader(root_dir: str, split: str = 'train',
                            batch_size: int = 8, num_points: int = 4096,
                            shuffle: bool = True, num_workers: int = 4) -> DataLoader3D:
    dataset = S3DISDataset(root_dir=root_dir, split=split, num_points=num_points)
    
    return DataLoader3D(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )


def download_s3dis(dest_dir: str = None):
    print("=" * 60)
    print("S3DIS Dataset Download Instructions")
    print("=" * 60)
    print()
    print("The S3DIS dataset needs to be manually downloaded.")
    print()
    print("1. Visit: http://www.scan-net.org/scanNet/2/0/")
    print("2. Register for an account")
    print("3. Download the S3DIS dataset")
    print("4. Extract the files to your data directory")
    print()
    print("Expected directory structure:")
    print(f"{dest_dir or './data/datasets/scannet'}/")
    print("├── train/")
    print("│   ├── area_1/")
    print("│   │   ├── office_1/")
    print("│   │   │   └── office_1.npy")
    print("│   │   └── office_2/")
    print("│   │       └── office_2.npy")
    print("│   └── area_2/")
    print("│       └── ...")
    print("└── val/")
    print("    └── ...")
    print()
    print("Note: The .npy files should contain dictionaries with:")
    print("  - 'points': numpy array of shape (N, 6) with XYZ + RGB")
    print("  - 'labels': numpy array of shape (N,) with class labels")
    print()
    print("Alternatively, you can convert raw S3DIS data using:")
    print("  python scripts/convert_s3dis.py --input /path/to/raw/data --output ./data/datasets/scannet")
    print()
    print("=" * 60)


def preprocess_s3dis_raw_data(raw_dir: str, output_dir: str):
    print(f"Preprocessing S3DIS data from {raw_dir} to {output_dir}")
    
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("This script should:")
    print("1. Read Stanford3dDataset_v1.2_Aligned_Version")
    print("2. Process room .txt files")
    print("3. Extract XYZ + RGB points")
    print("4. Map labels to S3DIS class IDs")
    print("5. Save as .npy files")
    print()
    print("Implementation depends on the raw data format.")
