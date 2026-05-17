import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class NeRFModel(nn.Module):
    def __init__(self, input_dim: int = 3, hidden_dim: int = 256, 
                 num_layers: int = 8, num_frequencies: int = 10,
                 use_view_direction: bool = True):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_frequencies = num_frequencies
        self.use_view_direction = use_view_direction
        
        self.pos_encoder = PositionalEncoding(num_frequencies=num_frequencies)
        dir_frequencies = 4 if use_view_direction else 0
        self.dir_encoder = PositionalEncoding(num_frequencies=dir_frequencies) if use_view_direction else None
        
        self.input_dim_encoded = self.pos_encoder(
            torch.zeros(1, input_dim)
        ).shape[-1]
        
        self.layers = nn.ModuleDict()
        self.layers['fc0'] = nn.Linear(self.input_dim_encoded, hidden_dim)
        
        for i in range(1, num_layers):
            if i == 4:
                self.layers[f'fc{i}'] = nn.Linear(hidden_dim + self.input_dim_encoded, hidden_dim)
            else:
                self.layers[f'fc{i}'] = nn.Linear(hidden_dim, hidden_dim)
        
        self.layers['sigma'] = nn.Linear(hidden_dim, 1)
        self.layers['feature'] = nn.Linear(hidden_dim, hidden_dim)
        
        if use_view_direction:
            dir_dim_encoded = self.dir_encoder(torch.zeros(1, 3)).shape[-1]
            self.layers['rgb0'] = nn.Linear(hidden_dim + dir_dim_encoded, hidden_dim // 2)
            self.layers['rgb1'] = nn.Linear(hidden_dim // 2, 3)
        else:
            self.layers['rgb0'] = nn.Linear(hidden_dim, hidden_dim // 2)
            self.layers['rgb1'] = nn.Linear(hidden_dim // 2, 3)
        
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor, d: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x_encoded = self.pos_encoder(x)
        
        h = self.activation(self.layers['fc0'](x_encoded))
        
        for i in range(1, 4):
            h = self.activation(self.layers[f'fc{i}'](h))
        
        if 4 < self.num_layers:
            h = self.activation(self.layers['fc4'](torch.cat([h, x_encoded], dim=-1)))
            for i in range(5, self.num_layers):
                h = self.activation(self.layers[f'fc{i}'](h))
        
        sigma = self.activation(self.layers['sigma'](h))
        
        features = self.layers['feature'](h)
        
        if self.use_view_direction and d is not None:
            d_encoded = self.dir_encoder(d)
            rgb_input = torch.cat([features, d_encoded], dim=-1)
        else:
            rgb_input = features
        
        rgb = self.activation(self.layers['rgb0'](rgb_input))
        rgb = torch.sigmoid(self.layers['rgb1'](rgb))
        
        return rgb, sigma
    
    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PositionalEncoding(nn.Module):
    def __init__(self, num_frequencies: int = 10, include_input: bool = True):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        self.freq_bands = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = [x] if self.include_input else []
        
        for freq in self.freq_bands:
            encoded.append(torch.sin(freq * 3.14159265 * x))
            encoded.append(torch.cos(freq * 3.14159265 * x))
        
        return torch.cat(encoded, dim=-1)
    
    @property
    def output_dim(self) -> int:
        return self.input_dim * (2 * self.num_frequencies + 1) if self.include_input else \
               self.input_dim * 2 * self.num_frequencies
