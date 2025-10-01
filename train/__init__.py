from .config import TrainingConfig
from .networks import CustomCNN
from .callbacks import RenderCallback, LiveMetricsCallback, DroneStatsCallback
__version__ = "1.0.0"
__author__ = "Drone Navigation Team"
__description__ = "Modular RL training system for autonomous drone navigation"

# Main exports - what users typically need
__all__ = [
    # Core components
    "TrainingConfig",
    "CustomCNN",
    # Callbacks
    "RenderCallback",
    "LiveMetricsCallback",
    "DroneStatsCallback",
]


