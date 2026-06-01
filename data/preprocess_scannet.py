'''
@File    :   preprocess_scannet.py
@Time    :   2026/05/30 17:14:26
@Author  :   Mr.Bear9 
@Github  :   https://github.com/MrBear9
'''


import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


class S3DISPreprocessor:
    """Convert raw S3DIS aligned rooms to the project's .npy format.

    Raw S3DIS rooms are stored as:
        Area_*/room_name/Annotations/object_instance.txt

    Each annotation file contains XYZRGB rows. This preprocessor merges all
    annotation files in one room into:
        {"points": float32[N, 6], "labels": int64[N], "area": ..., "room": ...}

    The output split is flat and compatible with training.train_segmentation:
        data/datasets/S3DIS/processed/train/*.npy
        data/datasets/S3DIS/processed/val/*.npy
        data/datasets/S3DIS/processed/test/*.npy
    """

    CLASS_LABELS = [
        "ceiling", "floor", "wall", "beam", "column", "window", "door",
        "table", "chair", "sofa", "bookcase", "board", "clutter",
    ]

    LABEL_MAP: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_LABELS)}
    LABEL_ALIASES = {
        "stairs": "clutter",
        "foor": "floor",
        "floor": "floor",
        "ceiiling": "ceiling",
        "ceiling": "ceiling",
    }

    def __init__(
        self,
        raw_path: str = "data/datasets/S3DIS/Stanford3dDataset_v1.2_Aligned_Version",
        output_path: str = "data/datasets/S3DIS/processed",
        train_areas: Iterable[str] = ("Area_1", "Area_2", "Area_3", "Area_4"),
        val_areas: Iterable[str] = ("Area_5",),
        test_areas: Iterable[str] = ("Area_6",),
        min_points: int = 256,
    ):
        self.raw_path = Path(raw_path)
        self.output_path = Path(output_path)
        self.train_areas = set(train_areas)
        self.val_areas = set(val_areas)
        self.test_areas = set(test_areas)
        self.min_points = min_points

    def create_output_dirs(self):
        for split in ("train", "val", "test"):
            (self.output_path / split).mkdir(parents=True, exist_ok=True)

    def get_area_split(self, area_name: str) -> str:
        if area_name in self.val_areas:
            return "val"
        if area_name in self.test_areas:
            return "test"
        return "train"

    def extract_label_from_filename(self, stem: str) -> int:
        label_name = stem.split("_")[0].lower()
        label_name = self.LABEL_ALIASES.get(label_name, label_name)
        return self.LABEL_MAP.get(label_name, self.LABEL_MAP["clutter"])

    def iter_rooms(self) -> Iterable[Tuple[Path, Path]]:
        if not self.raw_path.exists():
            raise FileNotFoundError(f"S3DIS raw directory not found: {self.raw_path}")
        for area_dir in sorted(self.raw_path.glob("Area_*")):
            if not area_dir.is_dir():
                continue
            for room_dir in sorted(area_dir.iterdir()):
                if room_dir.is_dir() and not room_dir.name.startswith("."):
                    yield area_dir, room_dir

    def process_room(self, area_dir: Path, room_dir: Path) -> int:
        annotations_dir = room_dir / "Annotations"
        if not annotations_dir.exists():
            return 0

        room_points: List[np.ndarray] = []
        room_labels: List[np.ndarray] = []
        object_counts: Dict[str, int] = {}

        for annotation_file in sorted(annotations_dir.glob("*.txt")):
            try:
                data = np.loadtxt(annotation_file)
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                if data.shape[1] < 6:
                    continue
                points = data[:, :6].astype(np.float32)
                points[:, 3:6] = np.clip(points[:, 3:6], 0, 255)
                label = self.extract_label_from_filename(annotation_file.stem)
                labels = np.full(len(points), label, dtype=np.int64)

                label_name = self.CLASS_LABELS[label]
                object_counts[label_name] = object_counts.get(label_name, 0) + int(len(points))
                room_points.append(points)
                room_labels.append(labels)
            except Exception as e:
                print(f"Warning: failed to load {annotation_file}: {e}")

        if not room_points:
            return 0

        all_points = np.vstack(room_points).astype(np.float32)
        all_labels = np.concatenate(room_labels).astype(np.int64)
        if len(all_points) < self.min_points:
            return 0

        split = self.get_area_split(area_dir.name)
        output_file = self.output_path / split / f"{area_dir.name}_{room_dir.name}.npy"
        np.save(output_file, {
            "points": all_points,
            "labels": all_labels,
            "area": area_dir.name,
            "room": room_dir.name,
            "source": str(room_dir),
            "class_labels": self.CLASS_LABELS,
            "label_counts": object_counts,
        })

        print(f"Saved {output_file} with {len(all_points)} points")
        return len(all_points)

    def process_all_areas(self):
        self.create_output_dirs()
        totals = {"train": 0, "val": 0, "test": 0}
        files = {"train": 0, "val": 0, "test": 0}

        for area_dir, room_dir in self.iter_rooms():
            split = self.get_area_split(area_dir.name)
            points = self.process_room(area_dir, room_dir)
            if points > 0:
                totals[split] += points
                files[split] += 1

        summary = {
            "raw_path": str(self.raw_path),
            "output_path": str(self.output_path),
            "class_labels": self.CLASS_LABELS,
            "splits": {
                split: {"files": files[split], "points": totals[split]}
                for split in ("train", "val", "test")
            },
        }
        summary_path = self.output_path / "summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        print("\nPreprocessing complete!")
        print(f"Output directory: {self.output_path}")
        for split in ("train", "val", "test"):
            print(f"{split}: {files[split]} files, {totals[split]} points")

    def generate_statistics(self):
        print("\nGenerating statistics...")
        for split in ("train", "val", "test"):
            split_dir = self.output_path / split
            if not split_dir.exists():
                continue
            total_points = 0
            label_counts = {idx: 0 for idx in range(len(self.CLASS_LABELS))}
            for npy_file in split_dir.glob("*.npy"):
                try:
                    data = np.load(npy_file, allow_pickle=True).item()
                    labels = data["labels"]
                    total_points += len(labels)
                    unique, counts = np.unique(labels, return_counts=True)
                    for label, count in zip(unique, counts):
                        label_counts[int(label)] += int(count)
                except Exception as e:
                    print(f"Warning: failed to inspect {npy_file}: {e}")

            print(f"\n{split} statistics:")
            print(f"  Total points: {total_points}")
            for label_id, label_name in enumerate(self.CLASS_LABELS):
                count = label_counts.get(label_id, 0)
                percentage = (count / total_points * 100.0) if total_points else 0.0
                print(f"    {label_name:12s}: {count:10d} ({percentage:5.2f}%)")


ScanNetPreprocessor = S3DISPreprocessor


def main():
    parser = argparse.ArgumentParser(description="Preprocess raw S3DIS aligned dataset")
    parser.add_argument(
        "--raw",
        default="data/datasets/S3DIS/Stanford3dDataset_v1.2_Aligned_Version",
        help="Raw Stanford3dDataset_v1.2_Aligned_Version directory",
    )
    parser.add_argument(
        "--output",
        default="data/datasets/S3DIS/processed",
        help="Output directory containing train/val/test .npy files",
    )
    parser.add_argument("--min-points", type=int, default=256)
    args = parser.parse_args()

    preprocessor = S3DISPreprocessor(
        raw_path=args.raw,
        output_path=args.output,
        min_points=args.min_points,
    )
    print("Starting S3DIS data preprocessing...")
    preprocessor.process_all_areas()
    preprocessor.generate_statistics()


if __name__ == "__main__":
    main()
