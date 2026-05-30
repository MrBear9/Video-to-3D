import argparse
import json
from pathlib import Path
import sys
import shutil

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.model_io import ModelIO
from core.rgbd_tsdf_reconstructor import RGBDTSDFReconstructor, write_depth_png


S3DIS_LABELS = {
    0: "ceiling",
    1: "floor",
    2: "wall",
    3: "beam",
    4: "column",
    5: "window",
    6: "door",
    7: "table",
    8: "chair",
    9: "sofa",
    10: "bookcase",
    11: "board",
    12: "clutter",
}

LABEL_COLORS = np.array([
    [0.62, 0.62, 0.62],
    [0.55, 0.55, 0.50],
    [0.70, 0.80, 0.95],
    [0.95, 0.65, 0.30],
    [0.70, 0.45, 0.85],
    [0.25, 0.62, 0.95],
    [0.95, 0.72, 0.28],
    [0.45, 0.78, 0.42],
    [0.95, 0.32, 0.28],
    [0.72, 0.45, 0.32],
    [0.48, 0.36, 0.25],
    [0.92, 0.82, 0.24],
    [0.55, 0.55, 0.60],
], dtype=np.float32)


def look_at(eye, target, up=np.array([0.0, 0.0, 1.0], dtype=np.float32)):
    forward = target - eye
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    if abs(float(np.dot(forward, up))) > 0.96:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-8)
    true_up = np.cross(right, forward)
    down = -true_up
    return np.stack([right, down, forward], axis=0)


def find_default_scene(processed_root=ROOT / "data" / "datasets" / "S3DIS" / "processed"):
    candidates = []
    for split in ("test", "val", "train"):
        split_dir = Path(processed_root) / split
        for npy_file in sorted(split_dir.glob("*.npy")):
            try:
                data = np.load(npy_file, allow_pickle=True).item()
                labels = data["labels"]
                points = data["points"]
                unique_labels = np.unique(labels)
                object_labels = [label for label in unique_labels if int(label) not in (0, 1, 2)]
                score = len(object_labels) * 1_000_000 + min(len(points), 1_000_000)
                candidates.append((score, npy_file))
            except Exception:
                continue
        if candidates:
            break
    if not candidates:
        raise FileNotFoundError(
            "No processed S3DIS .npy scene found. Run data/preprocess_scannet.py first."
        )
    return max(candidates, key=lambda item: item[0])[1]


def load_scene(path, max_points=180000):
    data = np.load(path, allow_pickle=True).item()
    points6 = data["points"].astype(np.float32)
    labels = data["labels"].astype(np.int64)
    xyz = points6[:, :3]
    rgb = np.clip(points6[:, 3:6] / 255.0, 0, 1)

    rng = np.random.default_rng(52252313128)
    if len(xyz) > max_points:
        idx = rng.choice(len(xyz), max_points, replace=False)
        xyz, rgb, labels = xyz[idx], rgb[idx], labels[idx]

    center = np.mean(xyz, axis=0)
    xyz = xyz - center
    scale = np.percentile(np.linalg.norm(xyz[:, :2], axis=1), 92)
    if scale > 1e-8:
        xyz = xyz / scale * 3.0
    return xyz, rgb, labels


def render_frame(xyz, rgb, eye, target, size=720, focal=650.0, return_depth=False):
    R = look_at(eye.astype(np.float32), target.astype(np.float32))
    cam = (xyz - eye) @ R.T
    valid = cam[:, 2] > 0.08
    cam = cam[valid]
    colors = rgb[valid]
    if len(cam) == 0:
        image = np.full((size, size, 3), 245, dtype=np.uint8)
        if return_depth:
            return image, np.zeros((size, size), dtype=np.float32), R
        return image

    u = (cam[:, 0] / cam[:, 2] * focal + size / 2).astype(np.int32)
    v = (cam[:, 1] / cam[:, 2] * focal + size / 2).astype(np.int32)
    valid2 = (u >= 0) & (u < size) & (v >= 0) & (v < size)
    u, v, z, colors = u[valid2], v[valid2], cam[:, 2][valid2], colors[valid2]

    order = np.argsort(z)[::-1]
    image = np.full((size, size, 3), 238, dtype=np.uint8)
    depth = np.full((size, size), np.inf, dtype=np.float32)
    for px, py, pz, col in zip(u[order], v[order], z[order], colors[order]):
        if pz < depth[py, px]:
            x0, x1 = max(0, px - 1), min(size, px + 2)
            y0, y1 = max(0, py - 1), min(size, py + 2)
            depth[y0:y1, x0:x1] = pz
            image[y0:y1, x0:x1] = (col * 255).astype(np.uint8)
    image = cv2.medianBlur(image, 3)
    if return_depth:
        depth_out = depth.copy()
        depth_out[~np.isfinite(depth_out)] = 0.0
        return image, depth_out, R
    return image


def make_camera_path(xyz, frames=144):
    xy = xyz[:, :2]
    lo = np.percentile(xy, 10, axis=0)
    hi = np.percentile(xy, 90, axis=0)
    z_mid = np.percentile(xyz[:, 2], 55)
    z_eye = z_mid + 0.6
    direction = hi - lo
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    side_dir = np.array([-direction[1], direction[0]], dtype=np.float32)
    path = []
    for i in range(frames):
        t = i / max(frames - 1, 1)
        smooth = t * t * (3.0 - 2.0 * t)
        xy_pos = lo * (1 - smooth) + hi * smooth
        lateral = np.sin(t * np.pi) * 0.18
        eye_xy = xy_pos + side_dir * lateral
        look_ahead = min(1.0, smooth + 0.12)
        target_xy = lo * (1 - look_ahead) + hi * look_ahead
        eye = np.array([eye_xy[0], eye_xy[1], z_eye], dtype=np.float32)
        target = np.array([target_xy[0], target_xy[1], z_mid - 0.15], dtype=np.float32)
        path.append((eye, target))
    return path


def write_point_cloud_ply(path, xyz, rgb):
    with Path(path).open("w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        colors = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        for p, c in zip(xyz, colors):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]}\n")


def camera_to_world(eye, R_world_to_cam):
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = R_world_to_cam.T
    pose[:3, 3] = eye
    return pose


def export_label_instances(out_dir, name, xyz, rgb, labels):
    instance_dir = out_dir / f"{name}_instances"
    instance_dir.mkdir(parents=True, exist_ok=True)
    instances = []
    for label in sorted(np.unique(labels)):
        mask = labels == label
        if mask.sum() < 500:
            continue
        instance_name = S3DIS_LABELS.get(int(label), f"label_{int(label)}")
        ply_path = instance_dir / f"{instance_name}_{int(label)}.ply"
        write_point_cloud_ply(ply_path, xyz[mask], rgb[mask])
        instances.append({
            "label_id": int(label),
            "name": instance_name,
            "points": int(mask.sum()),
            "path": str(ply_path.relative_to(ROOT)),
        })
    return instances


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene",
        default="auto",
        help="Processed S3DIS .npy file, or 'auto' to choose a rich test/val/train room",
    )
    parser.add_argument("--name", default=None)
    parser.add_argument("--frames", type=int, default=144)
    parser.add_argument("--size", type=int, default=720)
    parser.add_argument("--reconstruct", action="store_true")
    args = parser.parse_args()

    if args.scene == "auto":
        scene_path = find_default_scene()
    else:
        scene_path = Path(args.scene)
        if not scene_path.is_absolute():
            scene_path = ROOT / scene_path
    scene_name = args.name or f"s3dis_{scene_path.stem.lower()}"

    video_path = ROOT / "data" / "input_videos" / f"{scene_name}.mp4"
    rgbd_dir = ROOT / "data" / "input_videos" / f"{scene_name}_rgbd"
    out_dir = ROOT / "data" / "output_models"
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    for sub in ("color", "depth", "pose"):
        sub_dir = rgbd_dir / sub
        if sub_dir.exists():
            shutil.rmtree(sub_dir)
        sub_dir.mkdir(parents=True, exist_ok=True)

    xyz, rgb, labels = load_scene(scene_path)
    label_rgb = LABEL_COLORS[np.clip(labels, 0, len(LABEL_COLORS) - 1)]

    focal = 650.0 * (args.size / 720.0)
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 24, (args.size, args.size))
    for frame_idx, (eye, target) in enumerate(make_camera_path(xyz, args.frames)):
        frame, depth, R_wc = render_frame(xyz, rgb, eye, target, size=args.size, focal=focal, return_depth=True)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(rgbd_dir / "color" / f"{frame_idx:05d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        write_depth_png(rgbd_dir / "depth" / f"{frame_idx:05d}.png", depth)
        np.savetxt(rgbd_dir / "pose" / f"{frame_idx:05d}.txt", camera_to_world(eye, R_wc), fmt="%.8f")
    writer.release()

    gt_ply = out_dir / f"{scene_name}_source_scene.ply"
    seg_ply = out_dir / f"{scene_name}_segmented_labels.ply"
    write_point_cloud_ply(gt_ply, xyz, rgb)
    write_point_cloud_ply(seg_ply, xyz, label_rgb)
    instances = export_label_instances(out_dir, scene_name, xyz, rgb, labels)

    meta = {
        "width": args.size,
        "height": args.size,
        "fx": focal,
        "fy": focal,
        "cx": args.size / 2.0,
        "cy": args.size / 2.0,
        "depth_scale": 1000.0,
        "depth_trunc": 8.0,
        "frames": args.frames,
        "source_scene": str(scene_path.relative_to(ROOT)),
    }
    (rgbd_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "scene": str(scene_path.relative_to(ROOT)),
        "video": str(video_path.relative_to(ROOT)),
        "source_ply": str(gt_ply.relative_to(ROOT)),
        "segmented_ply": str(seg_ply.relative_to(ROOT)),
        "rgbd_project": str(rgbd_dir.relative_to(ROOT)),
        "instances": instances,
        "points": int(len(xyz)),
        "frames": args.frames,
        "labels": {S3DIS_LABELS.get(int(k), str(int(k))): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
    }

    if args.reconstruct:
        result = RGBDTSDFReconstructor().reconstruct_from_project(str(rgbd_dir))
        recon_ply = out_dir / f"{scene_name}_tsdf_mesh.ply"
        ModelIO.export_model(result.mesh, str(recon_ply))
        summary.update({
            "reconstructed_ply": str(recon_ply.relative_to(ROOT)),
            "reconstruction_method": result.volume_type,
            "reconstructed_vertices": int(len(result.mesh.vertices)),
            "reconstructed_faces": int(len(result.mesh.triangles)),
            "integrated_frames": int(result.frames),
        })

    summary_path = out_dir / f"{scene_name}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
