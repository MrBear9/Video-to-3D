
import os
from pathlib import Path

class Config:
    BASE_DIR = Path(__file__).parent
    
    DATA_DIR = BASE_DIR / 'data'
    INPUT_VIDEOS_DIR = DATA_DIR / 'input_videos'
    OUTPUT_MODELS_DIR = DATA_DIR / 'output_models'
    CHECKPOINTS_DIR = DATA_DIR / 'checkpoints'
    DATASETS_DIR = DATA_DIR / 'datasets'
    
    SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv']
    SUPPORTED_MODEL_IMPORT_FORMATS = ['.obj', '.ply', '.glb', '.gltf', '.stl', '.fbx']
    SUPPORTED_MODEL_EXPORT_FORMATS = ['.obj', '.ply', '.glb', '.stl']
    
    DEVICE = 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'
    
    VIDEO_DEFAULT_SAMPLE_RATE = 1
    VIDEO_DEFAULT_OUTPUT_FORMAT = 'jpg'
    
    NERF_DEFAULT_EPOCHS = 1000
    NERF_DEFAULT_BATCH_SIZE = 1024
    
    GAUSSIAN_SPLATTING_DEFAULT_ITERATIONS = 30000
    
    @classmethod
    def ensure_dirs(cls):
        cls.INPUT_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
        cls.OUTPUT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        cls.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.DATASETS_DIR.mkdir(parents=True, exist_ok=True)

Config.ensure_dirs()

