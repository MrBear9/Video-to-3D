'''
@File    :   dataset_loader.py
@Time    :   2026/05/30 17:13:07
@Author  :   Mr.Bear9 
@Github  :   https://github.com/MrBear9
'''
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class S3DISDataset(Dataset):
    CLASS_LABELS = [
        "ceiling", "floor", "wall", "beam", "column", "window", "door",
        "table", "chair", "sofa", "bookcase", "board", "clutter",
    ]

    def __init__(self, root_dir: str, split: str = "train",
                 num_points: int = 4096, transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.num_points = num_points
        self.transform = transform
        self.data_files: List[Path] = []
        self._scan_data_directory()

    def _scan_data_directory(self):
        split_path = self.root_dir / self.split
        if not split_path.exists():
            print(f"Warning: {split_path} does not exist")
            return

        flat_files = sorted(split_path.glob("*.npy"))
        if flat_files:
            self.data_files.extend(flat_files)
            return

        for area_folder in sorted(split_path.glob("area*")):
            if area_folder.is_dir():
                for room_folder in sorted(area_folder.glob("*")):
                    if room_folder.is_dir():
                        npy_file = room_folder / f"{room_folder.name}.npy"
                        if npy_file.exists():
                            self.data_files.append(npy_file)

    def __len__(self) -> int:
        return len(self.data_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        try:
            data = np.load(self.data_files[idx], allow_pickle=True).item()
            points = data.get("points", np.zeros((self.num_points, 6), dtype=np.float32))
            labels = data.get("labels", np.zeros(self.num_points, dtype=np.int64))

            if len(points) >= self.num_points:
                choice = np.random.choice(len(points), self.num_points, replace=False)
            else:
                choice = np.random.choice(len(points), self.num_points, replace=True)

            points_tensor = torch.from_numpy(points[choice]).float()
            labels_tensor = torch.from_numpy(labels[choice]).long()
            if self.transform:
                points_tensor = self.transform(points_tensor)

            return {
                "points": points_tensor,
                "labels": labels_tensor,
                "filename": str(self.data_files[idx]),
            }
        except Exception as e:
            print(f"Error loading {self.data_files[idx]}: {e}")
            return self._generate_fake_sample()

    def _generate_fake_sample(self) -> Dict[str, torch.Tensor]:
        return {
            "points": torch.randn(self.num_points, 6).float(),
            "labels": torch.randint(0, len(self.CLASS_LABELS), (self.num_points,)).long(),
            "filename": "fake_sample",
        }


class DataLoader3D:
    def __init__(self, dataset: Dataset, batch_size: int = 8,
                 shuffle: bool = True, num_workers: int = 4,
                 pin_memory: bool = True):
        self.dataset = dataset
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=self.collate_fn,
        )

    def __iter__(self):
        return iter(self.loader)

    def __len__(self) -> int:
        return len(self.loader)

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        return {
            "points": torch.stack([item["points"] for item in batch]),
            "labels": torch.stack([item["labels"] for item in batch]),
            "filenames": [item["filename"] for item in batch],
        }


def create_s3dis_dataloader(root_dir: str, split: str = "train",
                            batch_size: int = 8, num_points: int = 4096,
                            shuffle: bool = True, num_workers: int = 4) -> DataLoader3D:
    dataset = S3DISDataset(root_dir=root_dir, split=split, num_points=num_points)
    return DataLoader3D(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def download_s3dis(dest_dir: str = None):
    print("=" * 60)
    print("S3DIS Dataset Instructions")
    print("=" * 60)
    print("Download S3DIS from http://buildingparser.stanford.edu/dataset.html")
    print("Extract Stanford3dDataset_v1.2_Aligned_Version, then run:")
    print("  python data/preprocess_scannet.py --raw data/datasets/S3DIS/Stanford3dDataset_v1.2_Aligned_Version --output data/datasets/S3DIS/processed")
    print()
    print("Expected processed structure:")
    print(f"{dest_dir or './data/datasets/S3DIS'}/")
    print("  Stanford3dDataset_v1.2_Aligned_Version/")
    print("    Area_1/office_1/Annotations/*.txt")
    print("  processed/")
    print("    train/Area_1_office_1.npy")
    print("    val/Area_5_office_1.npy")
    print("    test/Area_6_office_1.npy")
    print()
    print("Each .npy file contains:")
    print("  - 'points': numpy array of shape (N, 6), XYZ + RGB")
    print("  - 'labels': numpy array of shape (N,), S3DIS 13-class labels")
    print("=" * 60)


def preprocess_s3dis_raw_data(raw_dir: str, output_dir: str):
    from data.preprocess_scannet import S3DISPreprocessor

    preprocessor = S3DISPreprocessor(raw_path=raw_dir, output_path=output_dir)
    preprocessor.process_all_areas()
    preprocessor.generate_statistics()
