"""
Neural network architecture for drone vision-based control.
"""

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
from crazy_flie_env.utils.logging_utils import get_logger

logger = get_logger(__name__)


class CustomCNN(BaseFeaturesExtractor):
    """
    Custom CNN for processing multi-modal observations (state + image).

    Processes camera images through CNN and fuses with state vector
    for policy and value networks.
    """

    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # Extract dimensions from observation space
        if hasattr(observation_space, 'spaces'):
            img_shape = observation_space.spaces['image'].shape  # (H, W, C)
            state_dim = observation_space.spaces['state'].shape[0]
        else:
            raise ValueError("Expected Dict observation space with 'image' and 'state'")

        logger.debug(f"CNN Input - Image: {img_shape}, State: {state_dim}D")

        # CNN for image processing
        self.cnn = nn.Sequential(
            # Conv block 1
            nn.Conv2d(img_shape[2], 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),

            # Conv block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),

            # Conv block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),

            # Conv block 4
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),

            # Global pooling
            nn.AdaptiveAvgPool2d((4, 4))
        )

        # Calculate CNN output size
        with torch.no_grad():
            sample = torch.zeros(1, img_shape[2], img_shape[0], img_shape[1])
            cnn_out = self.cnn(sample)
            cnn_output_size = cnn_out.view(1, -1).shape[1]

        logger.debug(f"CNN output: {cnn_output_size}D")

        # State processing
        self.state_processor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU()
        )

        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(cnn_output_size + 64, features_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(features_dim, features_dim),
            nn.LeakyReLU()
        )

        logger.debug(f"Output features: {features_dim}D")

    def forward(self, observations: dict) -> torch.Tensor:
        """Process observations through network."""

        # Process image: normalize and convert to (B, C, H, W)
        image = observations['image'].float() / 255.0

        if len(image.shape) == 4:  # (B, H, W, C)
            image = image.permute(0, 3, 1, 2)
        elif len(image.shape) == 3:  # (H, W, C)
            image = image.permute(2, 0, 1).unsqueeze(0)

        # Extract features
        cnn_features = self.cnn(image).reshape(image.shape[0], -1)
        state_features = self.state_processor(observations['state'])

        # Fuse and return
        combined = torch.cat([cnn_features, state_features], dim=1)
        return self.fusion(combined)
