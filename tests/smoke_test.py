import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from models.segmentation_model import (SegmentationModel, PointNetSetAbstraction,
                                        farthest_point_sample, index_points,
                                        query_ball_point, square_distance)
from training.train_segmentation import SegmentationTrainer, SegmentationDataset


def test_fps():
    print("[1] Testing farthest_point_sample...")
    xyz = torch.randn(2, 1024, 3)
    idx = farthest_point_sample(xyz, 512)
    assert idx.shape == (2, 512), f"Expected (2, 512), got {idx.shape}"
    print("    PASSED")


def test_index_points():
    print("[2] Testing index_points...")
    pts = torch.randn(2, 1024, 3)
    idx = torch.tensor([[0, 1, 2], [3, 4, 5]])
    res = index_points(pts, idx)
    assert res.shape == (2, 2, 3), f"Expected (2, 2, 3), got {res.shape}"
    print("    PASSED")


def test_query_ball():
    print("[3] Testing query_ball_point...")
    xyz = torch.randn(2, 100, 3)
    new_xyz = torch.randn(2, 10, 3)
    idx = query_ball_point(0.5, 32, xyz, new_xyz)
    assert idx.shape == (2, 10, 32), f"Expected (2, 10, 32), got {idx.shape}"
    print("    PASSED")


def test_square_distance():
    print("[4] Testing square_distance...")
    src = torch.randn(2, 10, 3)
    dst = torch.randn(2, 20, 3)
    dist = square_distance(src, dst)
    assert dist.shape == (2, 10, 20), f"Expected (2, 10, 20), got {dist.shape}"
    print("    PASSED")


def test_sa1():
    print("[5] Testing SA1 layer...")
    sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32,
                                  in_channel=9, mlp=[64, 64, 128])
    coords = torch.randn(2, 3, 2048)
    features = torch.randn(2, 6, 2048)
    new_xyz, new_pts = sa1(coords, features)
    assert new_xyz.shape == (2, 3, 512), f"SA1 coords: expected (2,3,512), got {new_xyz.shape}"
    assert new_pts.shape == (2, 128, 512), f"SA1 points: expected (2,128,512), got {new_pts.shape}"
    print("    PASSED")


def test_sa2():
    print("[6] Testing SA2 layer...")
    sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64,
                                  in_channel=131, mlp=[128, 128, 256])
    l1_xyz = torch.randn(2, 3, 512)
    l1_pts = torch.randn(2, 128, 512)
    new_xyz, new_pts = sa2(l1_xyz, l1_pts)
    assert new_xyz.shape == (2, 3, 128), f"SA2 coords: expected (2,3,128), got {new_xyz.shape}"
    assert new_pts.shape == (2, 256, 128), f"SA2 points: expected (2,256,128), got {new_pts.shape}"
    print("    PASSED")


def test_sa3():
    print("[7] Testing SA3 layer (group_all)...")
    sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None,
                                  in_channel=259, mlp=[256, 512, 1024],
                                  group_all=True)
    l2_xyz = torch.randn(2, 3, 128)
    l2_pts = torch.randn(2, 256, 128)
    new_xyz, new_pts = sa3(l2_xyz, l2_pts)
    assert new_xyz.shape == (2, 3, 1), f"SA3 coords: expected (2,3,1), got {new_xyz.shape}"
    assert new_pts.shape == (2, 1024, 1), f"SA3 points: expected (2,1024,1), got {new_pts.shape}"
    print("    PASSED")


def test_model_forward():
    print("[8] Testing full SegmentationModel forward...")
    model = SegmentationModel(num_classes=13, in_channels=6)

    x = torch.randn(4, 6, 2048)
    out, aux = model(x)

    assert out.shape == (4, 13, 2048), f"Model output: expected (4,13,2048), got {out.shape}"
    print("    PASSED")


def test_model_backward():
    print("[9] Testing backward pass...")
    model = SegmentationModel(num_classes=13, in_channels=6)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    x = torch.randn(4, 6, 2048)
    labels = torch.randint(0, 13, (4, 2048))

    out, _ = model(x)

    out_flat = out.permute(0, 2, 1).contiguous().view(-1, 13)
    labels_flat = labels.view(-1)

    loss = criterion(out_flat, labels_flat)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    grads_ok = True
    for name, param in model.named_parameters():
        if param.grad is None:
            grads_ok = False
            print(f"    WARNING: No grad for {name}")

    assert grads_ok, "Some parameters have no gradients!"
    print(f"    Loss: {loss.item():.4f}")
    print("    PASSED")


def test_multiple_input_shapes():
    print("[10] Testing multiple input sizes...")
    model = SegmentationModel(num_classes=13, in_channels=6)

    for N in [1024, 2048, 4096, 8192]:
        x = torch.randn(2, 6, N)
        out, _ = model(x)
        out_N = out.shape[-1]
        assert out.shape[1] == 13, f"N={N}: wrong output channels"
        assert out_N > 0, f"N={N}: zero output points"
        print(f"    N={N:5d} -> out_N={out_N:4d}")
    print("    PASSED")


def test_training_loop():
    print("[11] Testing mini training loop...")
    from training.train_segmentation import SegmentationTrainer, SegmentationDataset
    import tempfile
    import os
    import numpy as np

    tmpdir = tempfile.mkdtemp()
    train_dir = Path(tmpdir) / 'train'
    train_dir.mkdir(parents=True)

    for i in range(10):
        data = {
            'points': np.random.randn(2048, 6).astype(np.float32),
            'labels': np.random.randint(0, 13, 2048).astype(np.int64)
        }
        np.save(train_dir / f'room_{i}.npy', data)

    dataset = SegmentationDataset(data_path=tmpdir, split='train', num_points=2048)
    trainer = SegmentationTrainer({
        'device': 'cpu', 'num_classes': 13, 'in_channels': 6,
        'batch_size': 2, 'num_epochs': 1, 'learning_rate': 0.001
    })
    trainer.run(dataset, epochs=2)

    assert len(trainer.train_losses) == 2, f"Expected 2 losses, got {len(trainer.train_losses)}"
    print(f"    Epoch losses: {trainer.train_losses}")
    print("    PASSED")

    import shutil
    shutil.rmtree(tmpdir)


if __name__ == '__main__':
    print("=" * 60)
    print("  SEGMENTATION MODEL SMOKE TEST")
    print("=" * 60)

    try:
        test_fps()
        test_index_points()
        test_query_ball()
        test_square_distance()
        test_sa1()
        test_sa2()
        test_sa3()
        test_model_forward()
        test_model_backward()
        test_multiple_input_shapes()
        test_training_loop()

        print()
        print("=" * 60)
        print("  ALL TESTS PASSED!")
        print("=" * 60)
    except Exception as e:
        print()
        print("=" * 60)
        print(f"  TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)