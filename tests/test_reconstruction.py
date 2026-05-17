import unittest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from core.nerf_reconstructor import NeRFReconstructor, NeRFNetwork, PositionalEncoding
from core.gaussian_splatting import GaussianSplattingReconstructor, Gaussian3D


class TestNeRFNetwork(unittest.TestCase):
    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.network = NeRFNetwork(input_dim=3, hidden_dim=128, num_layers=4, num_frequencies=4)
        self.network.to(self.device)
    
    def test_positional_encoding(self):
        pos_enc = PositionalEncoding(num_frequencies=4)
        x = torch.randn(10, 3)
        encoded = pos_enc(x)
        
        self.assertEqual(encoded.shape[0], 10)
        self.assertGreater(encoded.shape[1], 3)
    
    def test_network_forward(self):
        x = torch.randn(16, 3).to(self.device)
        d = torch.randn(16, 3).to(self.device)
        
        rgb, sigma = self.network(x, d)
        
        self.assertEqual(rgb.shape, (16, 3))
        self.assertEqual(sigma.shape, (16, 1))
        self.assertTrue(torch.all(rgb >= 0) and torch.all(rgb <= 1))
        self.assertTrue(torch.all(sigma >= 0))
    
    def test_network_architecture(self):
        num_params = sum(p.numel() for p in self.network.parameters())
        self.assertGreater(num_params, 0)


class TestNeRFReconstructor(unittest.TestCase):
    def setUp(self):
        self.config = {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'num_frequencies': 4,
            'hidden_dim': 64,
            'num_layers': 4
        }
        self.reconstructor = NeRFReconstructor(self.config)
    
    def test_initialization(self):
        self.assertIsNotNone(self.reconstructor.network)
        self.assertEqual(self.reconstructor.device, self.config['device'])
    
    def test_pose_estimation(self):
        frames = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(5)]
        poses = self.reconstructor.estimate_poses(frames)
        
        self.assertEqual(poses.shape, (5, 4, 4))
    
    def test_save_load_checkpoint(self):
        checkpoint_path = 'test_nerf_checkpoint.pth'
        self.reconstructor.save_checkpoint(checkpoint_path)
        
        new_reconstructor = NeRFReconstructor(self.config)
        new_reconstructor.load_checkpoint(checkpoint_path)
        
        self.assertIsNotNone(new_reconstructor.network)


class TestGaussian3D(unittest.TestCase):
    def test_gaussian_creation(self):
        position = np.array([0.0, 0.0, 0.0])
        color = np.array([1.0, 0.0, 0.0])
        scale = np.array([1.0, 1.0, 1.0])
        rotation = np.array([1.0, 0.0, 0.0, 0.0])
        
        gaussian = Gaussian3D(position, color, scale, rotation)
        
        self.assertEqual(gaussian.position.shape, (3,))
        self.assertEqual(gaussian.color.shape, (3,))
        self.assertEqual(gaussian.scale.shape, (3,))
        self.assertEqual(gaussian.rotation.shape, (4,))
        self.assertEqual(gaussian.opacity, 1.0)


class TestGaussianSplattingReconstructor(unittest.TestCase):
    def setUp(self):
        self.config = {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'max_iterations': 10
        }
        self.reconstructor = GaussianSplattingReconstructor(self.config)
    
    def test_initialization(self):
        self.assertEqual(self.reconstructor.device, self.config['device'])
        self.assertEqual(self.reconstructor.max_iterations, 10)
        self.assertEqual(len(self.reconstructor.gaussians), 0)
    
    def test_gaussian_initialization(self):
        point_cloud = np.random.randn(1000, 3).astype(np.float32)
        colors = np.random.rand(1000, 3).astype(np.float32)
        
        self.reconstructor.initialize_gaussians(point_cloud, colors)
        
        self.assertIsNotNone(self.reconstructor.positions)
        self.assertIsNotNone(self.reconstructor.colors)
        self.assertIsNotNone(self.reconstructor.scales)
        self.assertIsNotNone(self.reconstructor.rotations)
        self.assertIsNotNone(self.reconstructor.opacities)
    
    def test_checkpoint_save_load(self):
        point_cloud = np.random.randn(100, 3).astype(np.float32)
        self.reconstructor.initialize_gaussians(point_cloud)
        
        checkpoint_path = 'test_gs_checkpoint.pth'
        self.reconstructor.save_checkpoint(checkpoint_path)
        
        new_reconstructor = GaussianSplattingReconstructor(self.config)
        new_reconstructor.load_checkpoint(checkpoint_path)
        
        self.assertIsNotNone(new_reconstructor.positions)


if __name__ == '__main__':
    unittest.main()
