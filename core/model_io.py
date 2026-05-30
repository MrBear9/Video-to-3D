'''
@File    :   model_io.py
@Time    :   2026/05/30 17:13:51
@Author  :   Mr.Bear9 
@Github  :   https://github.com/MrBear9
'''


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
                if format in ['.obj', '.stl', '.glb', '.gltf']:
                    model.compute_vertex_normals()
                return bool(o3d.io.write_triangle_mesh(str(path_obj), model))
            elif isinstance(model, o3d.geometry.PointCloud):
                if format == '.ply':
                    return bool(o3d.io.write_point_cloud(str(path_obj), model))
                mesh = ModelIO._point_cloud_to_mesh(model)
                if mesh is None or len(mesh.triangles) == 0:
                    raise TypeError('Point cloud export to mesh failed; please export point clouds as PLY')
                return bool(o3d.io.write_triangle_mesh(str(path_obj), mesh))
            elif hasattr(model, 'points') and hasattr(model, 'colors'):
                points = np.asarray(model.points)
                colors = np.asarray(model.colors)
                if path_obj.suffix.lower() == '.ply':
                    ModelIO._write_simple_point_cloud_ply(path_obj, points, colors)
                    return True
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
                mesh = ModelIO._point_cloud_to_mesh(pcd)
                if mesh is None or len(mesh.triangles) == 0:
                    raise TypeError('Simple point clouds can currently be exported as PLY')
                return bool(o3d.io.write_triangle_mesh(str(path_obj), mesh))
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

    @staticmethod
    def _point_cloud_to_mesh(pcd: o3d.geometry.PointCloud):
        if len(pcd.points) < 4:
            return None
        working = o3d.geometry.PointCloud(pcd)
        if not working.has_normals():
            working.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=Config.EXPORT_NORMAL_RADIUS,
                    max_nn=Config.EXPORT_NORMAL_MAX_NN
                )
            )
            try:
                working.orient_normals_consistent_tangent_plane(30)
            except Exception:
                pass

        distances = working.compute_nearest_neighbor_distance()
        avg_dist = float(np.mean(distances)) if distances else 0.03
        radii = [avg_dist * 1.5, avg_dist * 3.0, avg_dist * 5.0]
        try:
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                working,
                o3d.utility.DoubleVector(radii)
            )
            if len(mesh.triangles) > 0:
                mesh.compute_vertex_normals()
                return mesh
        except Exception:
            pass

        try:
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                working,
                depth=Config.EXPORT_POISSON_DEPTH
            )
            if len(mesh.triangles) > 0:
                density = np.asarray(densities)
                keep = density > np.quantile(density, 0.05)
                mesh.remove_vertices_by_mask(~keep)
                mesh.compute_vertex_normals()
                return mesh
        except Exception:
            pass
        return None
