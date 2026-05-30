'''
@File    :   video_processor.py
@Time    :   2026/05/30 17:14:03
@Author  :   Mr.Bear9 
@Github  :   https://github.com/MrBear9
'''


import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict
from config import Config


class VideoProcessor:
    def __init__(self, video_path: str):
        self.video_path = Path(video_path)
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.current_frame_idx = 0
    
    def get_video_info(self) -> Dict:
        return {
            'path': str(self.video_path),
            'resolution': (self.width, self.height),
            'fps': self.fps,
            'frame_count': self.frame_count,
            'duration': self.frame_count / self.fps if self.fps > 0 else 0
        }
    
    def extract_frames(self, output_dir: str, sample_rate: int = 1) -> List[Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        frames = []
        frame_idx = 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            if frame_idx % sample_rate == 0:
                frame_path = output_dir / f'frame_{frame_idx:06d}.jpg'
                cv2.imwrite(str(frame_path), frame)
                frames.append(frame_path)
            
            frame_idx += 1
        
        return frames
    
    def get_frame(self, frame_idx: int) -> np.ndarray:
        if frame_idx < 0 or frame_idx >= self.frame_count:
            raise ValueError(f"Frame index out of range: {frame_idx}")
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if ret:
            self.current_frame_idx = frame_idx
            return frame
        return None
    
    def preprocess_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        processed_frames = []
        for frame in frames:
            denoised = cv2.GaussianBlur(frame, (5, 5), 0)
            processed_frames.append(denoised)
        return processed_frames
    
    def play_video(self, window_name: str = 'Video Player'):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            cv2.imshow(window_name, frame)
            if cv2.waitKey(int(1000 / self.fps)) & 0xFF == ord('q'):
                break
        
        cv2.destroyWindow(window_name)
    
    def release(self):
        if self.cap.isOpened():
            self.cap.release()
    
    def __del__(self):
        self.release()