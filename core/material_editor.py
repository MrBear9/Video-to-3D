
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import trimesh


@dataclass
class MaterialProperties:
    name: str
    albedo: tuple
    roughness: float
    metallic: float
    transparency: float = 0.0
    ior: float = 1.5
    texture: Optional[np.ndarray] = None


class MaterialEditor:
    MATERIALS: Dict[str, MaterialProperties] = {
        'metal': MaterialProperties(
            name='metal',
            albedo=(0.8, 0.8, 0.8),
            roughness=0.1,
            metallic=1.0
        ),
        'wood': MaterialProperties(
            name='wood',
            albedo=(0.6, 0.4, 0.2),
            roughness=0.8,
            metallic=0.0
        ),
        'plastic': MaterialProperties(
            name='plastic',
            albedo=(0.9, 0.2, 0.2),
            roughness=0.3,
            metallic=0.0
        ),
        'glass': MaterialProperties(
            name='glass',
            albedo=(0.9, 0.95, 1.0),
            roughness=0.05,
            metallic=0.0,
            transparency=0.9,
            ior=1.5
        ),
        'stone': MaterialProperties(
            name='stone',
            albedo=(0.5, 0.5, 0.5),
            roughness=0.9,
            metallic=0.0
        ),
        'ceramic': MaterialProperties(
            name='ceramic',
            albedo=(0.95, 0.95, 0.95),
            roughness=0.2,
            metallic=0.0
        ),
        'fabric': MaterialProperties(
            name='fabric',
            albedo=(0.3, 0.4, 0.6),
            roughness=0.95,
            metallic=0.0
        )
    }
    
    def __init__(self):
        pass
    
    def apply_material(self, mesh: trimesh.Trimesh, material_type: str) -> trimesh.Trimesh:
        if material_type not in self.MATERIALS:
            raise ValueError(f"Unknown material type: {material_type}")
        
        mat = self.MATERIALS[material_type]
        new_mesh = mesh.copy()
        
        if hasattr(new_mesh.visual, 'vertex_colors'):
            colors = np.ones((len(new_mesh.vertices), 4), dtype=np.float32)
            colors[:, 0:3] = mat.albedo
            colors[:, 3] = 1.0 - mat.transparency
            new_mesh.visual.vertex_colors = (colors * 255).astype(np.uint8)
        elif hasattr(new_mesh.visual, 'material'):
            new_mesh.visual.material = trimesh.visual.material.SimpleMaterial(
                diffuse=mat.albedo,
                metallic=mat.metallic,
                roughness=mat.roughness
            )
        
        return new_mesh
    
    def create_custom_material(self, properties: Dict) -> MaterialProperties:
        return MaterialProperties(
            name=properties.get('name', 'custom'),
            albedo=properties.get('albedo', (0.5, 0.5, 0.5)),
            roughness=properties.get('roughness', 0.5),
            metallic=properties.get('metallic', 0.0),
            transparency=properties.get('transparency', 0.0),
            ior=properties.get('ior', 1.5),
            texture=properties.get('texture')
        )
    
    def preview_material(self, mesh: trimesh.Trimesh, material: MaterialProperties) -> np.ndarray:
        temp_mesh = mesh.copy()
        if hasattr(temp_mesh.visual, 'vertex_colors'):
            colors = np.ones((len(temp_mesh.vertices), 4), dtype=np.float32)
            colors[:, 0:3] = material.albedo
            colors[:, 3] = 1.0 - material.transparency
            temp_mesh.visual.vertex_colors = (colors * 255).astype(np.uint8)
        
        return temp_mesh
    
    def export_material(self, material: MaterialProperties, path: str):
        pass

