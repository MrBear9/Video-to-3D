
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
            elif isinstance(model, o3d.geometry.TriangleMesh):
                return bool(o3d.io.write_triangle_mesh(str(path_obj), model))
            elif isinstance(model, o3d.geometry.PointCloud):
                o3d.io.write_point_cloud(str(path_obj), model)
                return True
            elif hasattr(model, 'points') and hasattr(model, 'colors'):
                points = np.asarray(model.points)
                colors = np.asarray(model.colors)
                if path_obj.suffix.lower() == '.ply':
                    ModelIO._write_simple_point_cloud_ply(path_obj, points, colors)
                    return True
                raise TypeError('Simple point clouds can currently be exported as PLY')
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
        elif isinstance(model, o3d.geometry.TriangleMesh):
            vertices = np.asarray(model.vertices)
            info['type'] = 'mesh'
            info['vertices'] = len(vertices)
            info['faces'] = len(np.asarray(model.triangles))
            if len(vertices) > 0:
                info['bounds'] = np.array([vertices.min(axis=0), vertices.max(axis=0)])
            info['has_colors'] = model.has_vertex_colors()
            info['has_normals'] = model.has_vertex_normals()
        elif hasattr(model, 'points'):
            points = np.asarray(model.points)
            info['type'] = 'point_cloud'
            info['points'] = len(points)
            if len(points) > 0:
                info['bounds'] = np.array([points.min(axis=0), points.max(axis=0)])
            info['has_colors'] = hasattr(model, 'colors')
            info['has_normals'] = False
        else:
            info['type'] = 'unknown'
        
        return info

    @staticmethod
    def _write_simple_point_cloud_ply(path_obj: Path, points: np.ndarray, colors: np.ndarray):
        colors = np.clip(colors, 0, 1)
        rgb = (colors * 255).astype(np.uint8)
        with path_obj.open('w', encoding='ascii') as f:
            f.write('ply\nformat ascii 1.0\n')
            f.write(f'element vertex {len(points)}\n')
            f.write('property float x\nproperty float y\nproperty float z\n')
            f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
            f.write('end_header\n')
            for p, c in zip(points, rgb):
                f.write(f'{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]}\n')
