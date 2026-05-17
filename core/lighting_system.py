
from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class Light:
    light_id: int
    light_type: str
    position: Optional[np.ndarray] = None
    direction: Optional[np.ndarray] = None
    color: tuple = (1.0, 1.0, 1.0)
    intensity: float = 1.0


@dataclass
class AmbientLight(Light):
    def __init__(self, light_id: int, color: tuple = (0.2, 0.2, 0.2), intensity: float = 0.3):
        super().__init__(light_id=light_id, light_type='ambient', color=color, intensity=intensity)


@dataclass
class DirectionalLight(Light):
    def __init__(self, light_id: int, direction: np.ndarray, color: tuple = (1.0, 1.0, 1.0), intensity: float = 1.0):
        super().__init__(light_id=light_id, light_type='directional', direction=direction, color=color, intensity=intensity)


@dataclass
class PointLight(Light):
    def __init__(self, light_id: int, position: np.ndarray, color: tuple = (1.0, 1.0, 1.0), intensity: float = 1.0):
        super().__init__(light_id=light_id, light_type='point', position=position, color=color, intensity=intensity)


@dataclass
class SpotLight(Light):
    def __init__(self, light_id: int, position: np.ndarray, direction: np.ndarray, 
                 color: tuple = (1.0, 1.0, 1.0), intensity: float = 1.0, cutoff: float = 0.5):
        super().__init__(light_id=light_id, light_type='spot', position=position, 
                        direction=direction, color=color, intensity=intensity)
        self.cutoff = cutoff


class LightingSystem:
    def __init__(self):
        self.lights: List[Light] = []
        self.next_id = 0
        self.ambient_light = AmbientLight(self._get_next_id(), color=(0.2, 0.2, 0.2), intensity=0.3)
        self.lights.append(self.ambient_light)
    
    def _get_next_id(self) -> int:
        current_id = self.next_id
        self.next_id += 1
        return current_id
    
    def add_light(self, light_type: str, **params) -> int:
        light_id = self._get_next_id()
        
        if light_type == 'ambient':
            light = AmbientLight(light_id, **params)
        elif light_type == 'directional':
            direction = params.get('direction', np.array([0, -1, 0]))
            light = DirectionalLight(light_id, direction=direction, **params)
        elif light_type == 'point':
            position = params.get('position', np.array([0, 0, 5]))
            light = PointLight(light_id, position=position, **params)
        elif light_type == 'spot':
            position = params.get('position', np.array([0, 0, 5]))
            direction = params.get('direction', np.array([0, 0, -1]))
            light = SpotLight(light_id, position=position, direction=direction, **params)
        else:
            raise ValueError(f"Unknown light type: {light_type}")
        
        self.lights.append(light)
        return light_id
    
    def remove_light(self, light_id: int) -> bool:
        for i, light in enumerate(self.lights):
            if light.light_id == light_id:
                if isinstance(light, AmbientLight):
                    return False
                self.lights.pop(i)
                return True
        return False
    
    def update_light(self, light_id: int, **params) -> bool:
        for light in self.lights:
            if light.light_id == light_id:
                for key, value in params.items():
                    if hasattr(light, key):
                        setattr(light, key, value)
                return True
        return False
    
    def set_ambient_light(self, color: tuple, intensity: float):
        self.ambient_light.color = color
        self.ambient_light.intensity = intensity
    
    def get_all_lights(self) -> List[Light]:
        return self.lights
    
    def render_with_lighting(self, scene, camera) -> np.ndarray:
        pass
    
    def reset(self):
        self.lights = []
        self.next_id = 0
        self.ambient_light = AmbientLight(self._get_next_id(), color=(0.2, 0.2, 0.2), intensity=0.3)
        self.lights.append(self.ambient_light)

