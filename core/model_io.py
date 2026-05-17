
from pathlib import Path
from typing import Union, Dict
import numpy as np
import trimesh
import open3d as o3d
from config import Config


class ModelIO:
    SUPPORTED_IMPORT_FORMATS = Config.SUPPORTED_MODEL_IMPORT_FORMATS
    SUPPORTED_EXPORT_FORMATS = Config.SUPPORTED_MODEL_EXPORT_FORMATS
    
    @staticmethod
    def import_model(path: str) -> Union[trimesh.Trimesh, o3d.geometry.PointCloud]:
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        suffix = path_obj.suffix.lower()
        
        if suffix in ['.ply']:
            try:
                pcd = o3d.io.read_point_cloud(str(path_obj))
                if len(pcd.points) > 0:
                    return pcd
            except Exception:
                pass
        
        try:
            mesh = trimesh.load(path)
            return mesh
        except Exception as e:
            raise ValueError(f"Failed to import model: {e}")
    
    @staticmethod
    def export_model(model, path: str, format: str = None) -> bool:
        try:
            path_obj = Path(path)
            if format is None:
                format = path_obj.suffix.lower()
            else:
                if not format.startswith('.'):
                    format = '.' + format
            
            if isinstance(model, trimesh.Trimesh):
                model.export(str(path_obj))
                return True
            elif isinstance(model, o3d.geometry.PointCloud):
                o3d.io.write_point_cloud(str(path_obj), model)
                return True
            else:
                raise TypeError(f"Unsupported model type: {type(model)}")
        except Exception as e:
            print(f"Failed to export model: {e}")
            return False
    
    @staticmethod
    def convert_format(input_path: str, output_path: str) -> bool:
        try:
            model = ModelIO.import_model(input_path)
            return ModelIO.export_model(model, output_path)
        except Exception as e:
            print(f"Failed to convert format: {e}")
            return False
    
    @staticmethod
    def get_model_info(model) -> Dict:
        info = {}
        
        if isinstance(model, trimesh.Trimesh):
            info['type'] = 'mesh'
            info['vertices'] = len(model.vertices)
            info['faces'] = len(model.faces)
            info['bounds'] = model.bounds
            info['volume'] = model.volume if model.is_watertight else None
        elif isinstance(model, o3d.geometry.PointCloud):
            points = np.asarray(model.points)
            info['type'] = 'point_cloud'
            info['points'] = len(points)
            if len(points) > 0:
                info['bounds'] = np.array([points.min(axis=0), points.max(axis=0)])
            info['has_colors'] = model.has_colors()
            info['has_normals'] = model.has_normals()
        else:
            info['type'] = 'unknown'
        
        return info

