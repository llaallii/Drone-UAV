# train/networks.py
"""
Neural network architectures for drone training.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any

try:
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    import gymnasium as gym
except ImportError:
    print("❌ Required packages not found")
    raise


class CustomCNN(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for visual inputs.
    
    Processes multi-modal observations (state + image) and fuses them
    for downstream policy and value networks.
    """
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # Extract observation space info
        if hasattr(observation_space, 'spaces'):
            # Dict observation space
            img_shape = observation_space.spaces['image'].shape
            state_dim = observation_space.spaces['state'].shape[0]
        else:
            raise ValueError("Expected Dict observation space with 'image' and 'state' keys")
        
        print(f"🧠 CNN Input - Image shape: {img_shape}, State dim: {state_dim}")
        
        # CNN for image processing (based on RAPID paper architecture)
        self.cnn = nn.Sequential(
            # First conv block
            nn.Conv2d(img_shape[2], 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            
            # Fourth conv block
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Calculate CNN output size
        with torch.no_grad():
            sample_input = torch.zeros(1, img_shape[2], img_shape[0], img_shape[1])
            cnn_output = self.cnn(sample_input)
            cnn_output_size = cnn_output.view(1, -1).shape[1]
            print(f"🧠 CNN output size: {cnn_output_size}")
        
        # State processing network
        self.state_processor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU()
        )
        
        # Feature fusion network
        self.fusion = nn.Sequential(
            nn.Linear(cnn_output_size + 64, features_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(features_dim, features_dim),
            nn.LeakyReLU()
        )
        
        print(f"🧠 Network created with {features_dim} output features")
        
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through the network."""
        try:
            # Process image
            image = observations['image'].float() / 255.0
            
            # Handle batch dimensions
            if len(image.shape) == 4:  # (B, H, W, C)
                image = image.permute(0, 3, 1, 2)  # Convert to (B, C, H, W)
            elif len(image.shape) == 3:  # (H, W, C)
                image = image.permute(2, 0, 1).unsqueeze(0)  # Convert to (1, C, H, W)
            else:
                raise ValueError(f"Unexpected image shape: {image.shape}")
            
            # CNN processing
            cnn_features = self.cnn(image)
            cnn_features = cnn_features.reshape(cnn_features.shape[0], -1)
            
            # State processing
            state_features = self.state_processor(observations['state'])
            
            # Feature fusion
            combined = torch.cat([cnn_features, state_features], dim=1)
            output = self.fusion(combined)
            
            return output
            
        except Exception as e:
            print(f"❌ Error in CustomCNN forward pass: {e}")
            print(f"Image shape: {observations['image'].shape}")
            print(f"State shape: {observations['state'].shape}")
            raise


class SimpleVisualizationCNN(BaseFeaturesExtractor):
    """
    Simplified CNN for faster training during live visualization.
    
    Uses fewer parameters for real-time performance during live training.
    """
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        
        if hasattr(observation_space, 'spaces'):
            img_shape = observation_space.spaces['image'].shape
            state_dim = observation_space.spaces['state'].shape[0]
        else:
            raise ValueError("Expected Dict observation space")
        
        # Simplified CNN for speed
        self.cnn = nn.Sequential(
            nn.Conv2d(img_shape[2], 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2))
        )
        
        # Calculate output size
        with torch.no_grad():
            sample_input = torch.zeros(1, img_shape[2], img_shape[0], img_shape[1])
            cnn_output = self.cnn(sample_input)
            cnn_output_size = cnn_output.view(1, -1).shape[1]
        
        # Simple state processor
        self.state_processor = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU()
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(cnn_output_size + 32, features_dim),
            nn.ReLU()
        )
        
        print(f"🏃 Fast CNN created for live visualization ({features_dim} features)")
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through simplified network."""
        image = observations['image'].float() / 255.0
        
        if len(image.shape) == 4:
            image = image.permute(0, 3, 1, 2)
        elif len(image.shape) == 3:
            image = image.permute(2, 0, 1).unsqueeze(0)
        
        cnn_features = self.cnn(image).reshape(image.shape[0], -1)
        state_features = self.state_processor(observations['state'])
        
        combined = torch.cat([cnn_features, state_features], dim=1)
        return self.fusion(combined)


class AttentionCNN(BaseFeaturesExtractor):
    """
    CNN with attention mechanisms for better feature learning.
    
    Implements spatial attention to focus on important regions
    of the input images (obstacles, goals, etc.).
    """
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        if hasattr(observation_space, 'spaces'):
            img_shape = observation_space.spaces['image'].shape
            state_dim = observation_space.spaces['state'].shape[0]
        else:
            raise ValueError("Expected Dict observation space")
        
        # Feature extraction CNN
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(img_shape[2], 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
        )
        
        # Spatial attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Calculate feature map size
        with torch.no_grad():
            sample_input = torch.zeros(1, img_shape[2], img_shape[0], img_shape[1])
            features = self.feature_extractor(sample_input)
            self.feature_map_size = features.shape[2] * features.shape[3]
            feature_channels = features.shape[1]
        
        # State processing
        self.state_processor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU()
        )
        
        # Final fusion
        self.fusion = nn.Sequential(
            nn.Linear(feature_channels + 64, features_dim),
            nn.LeakyReLU(),
            nn.Linear(features_dim, features_dim)
        )
        
        print(f"🎯 Attention CNN created with {features_dim} features")
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass with attention mechanism."""
        image = observations['image'].float() / 255.0
        
        if len(image.shape) == 4:
            image = image.permute(0, 3, 1, 2)
        elif len(image.shape) == 3:
            image = image.permute(2, 0, 1).unsqueeze(0)
        
        # Extract features
        features = self.feature_extractor(image)
        
        # Apply spatial attention
        attention_weights = self.attention(features)
        attended_features = features * attention_weights
        
        # Global average pooling
        pooled_features = torch.mean(attended_features, dim=(2, 3))
        
        # Process state
        state_features = self.state_processor(observations['state'])
        
        # Fusion
        combined = torch.cat([pooled_features, state_features], dim=1)
        return self.fusion(combined)


def get_network_by_name(name: str, observation_space: gym.Space, features_dim: int = 256) -> BaseFeaturesExtractor:
    """Factory function to get network by name."""
    
    networks = {
        'custom': CustomCNN,
        'simple': SimpleVisualizationCNN,
        'attention': AttentionCNN
    }
    
    if name.lower() not in networks:
        raise ValueError(f"Unknown network: {name}. Available: {list(networks.keys())}")
    
    network_class = networks[name.lower()]
    return network_class(observation_space, features_dim)


def count_parameters(model: nn.Module) -> int:
    """Count total parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_network_summary(model: nn.Module, model_name: str = "Model"):
    """Print a summary of the network architecture."""
    total_params = count_parameters(model)
    print(f"\n🧠 {model_name} Summary:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Memory estimate: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    
    # Print layer info
    for name, module in model.named_children():
        if hasattr(module, '__len__'):
            print(f"   {name}: {len(module)} layers")
        else:
            print(f"   {name}: {module.__class__.__name__}")