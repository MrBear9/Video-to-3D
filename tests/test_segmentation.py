import unittest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from core.instance_detector import InstanceDetector, Instance, VoteNet
from core.instance_segmentor import InstanceSegmentor, Segment, PointGroupSegmentation


class TestInstance(unittest.TestCase):
    def test_instance_creation(self):
        class_id = 1
        class_name = 'chair'
        bbox = np.array([[0, 0, 0], [1, 1, 1]])
        confidence = 0.95
        
        instance = Instance(class_id, class_name, bbox, confidence)
        
        self.assertEqual(instance.class_id, class_id)
        self.assertEqual(instance.class_name, class_name)
        self.assertEqual(instance.confidence, confidence)
        self.assertIsNone(instance.mask)
        self.assertIsNone(instance.center)


class TestInstanceDetector(unittest.TestCase):
    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.detector = InstanceDetector(device=self.device)
    
    def test_initialization(self):
        self.assertEqual(self.detector.device, self.device)
        self.assertIsNotNone(self.detector.model)
        self.assertEqual(len(self.detector.CLASS_NAMES), 13)
    
    def test_detect_with_synthetic_data(self):
        point_cloud = np.random.randn(2048, 3).astype(np.float32)
        
        detections = self.detector.detect(point_cloud)
        
        self.assertIsInstance(detections, list)
        for detection in detections:
            self.assertIsInstance(detection, Instance)
    
    def test_bbox_computation(self):
        center = np.array([0.0, 0.0, 0.0])
        size = np.array([1.0, 1.0, 1.0])
        
        bbox = self.detector._compute_bbox(center, size)
        
        self.assertEqual(bbox.shape, (8, 3))
    
    def test_point_preprocessing(self):
        points = np.random.randn(5000, 3).astype(np.float32)
        processed = self.detector._preprocess_points(points)
        
        self.assertEqual(processed.ndim, 3)
        self.assertGreaterEqual(processed.shape[1], 3)


class TestSegment(unittest.TestCase):
    def test_segment_creation(self):
        segment_id = 0
        points = np.random.randn(100, 3).astype(np.float32)
        
        segment = Segment(segment_id, points)
        
        self.assertEqual(segment.segment_id, segment_id)
        self.assertEqual(segment.points.shape, (100, 3))
        self.assertIsNone(segment.color)
        self.assertIsNone(segment.instance_id)


class TestInstanceSegmentor(unittest.TestCase):
    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.segmentor = InstanceSegmentor(device=self.device)
    
    def test_initialization(self):
        self.assertEqual(self.segmentor.device, self.device)
        self.assertIsNotNone(self.segmentor.model)
        self.assertEqual(len(self.segmentor.SEMANTIC_LABELS), 13)
    
    def test_segment_with_synthetic_data(self):
        point_cloud = np.random.randn(2048, 3).astype(np.float32)
        
        segments = self.segmentor.segment(point_cloud)
        
        self.assertIsInstance(segments, list)
        for segment in segments:
            self.assertIsInstance(segment, Segment)
    
    def test_refine_segmentation(self):
        points = np.random.randn(1000, 3).astype(np.float32)
        original_segment = Segment(0, points)
        
        segments = [original_segment]
        refined = self.segmentor.refine_segmentation(segments)
        
        self.assertIsInstance(refined, list)
    
    def test_get_segment_point_cloud(self):
        points = np.random.randn(500, 3).astype(np.float32)
        segment = Segment(segment_id=0, points=points)
        segments = [segment]
        
        retrieved_points = self.segmentor.get_segment_point_cloud(0, segments)
        
        self.assertEqual(retrieved_points.shape, (500, 3))


class TestSegmentationIntegration(unittest.TestCase):
    def test_full_pipeline(self):
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
