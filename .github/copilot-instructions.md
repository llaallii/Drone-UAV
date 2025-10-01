# CrazyFlie RL Environment - AI Coding Guide

## Project Architecture

This is a **modular reinforcement learning environment** for training autonomous drone navigation using MuJoCo physics simulation and Stable-Baselines3. The codebase separates physics, vision, rewards, and training into distinct modules.

### Core Components & Data Flow

- **`crazy_flie_env/`**: Main environment package with modular design
  - `core/environment.py`: Main `CrazyFlieEnv(gym.Env)` - coordinates all subsystems
  - `physics/`: MuJoCo simulation + PID controllers (`dynamics.py`, `controller.py`) 
  - `vision/`: Multi-camera system with FPV + chase views (`cameras.py`, `rendering.py`)
  - `rewards/`: Configurable reward components for different flight tasks
  - `utils/config.py`: Central configuration with `EnvConfig` class

- **`train/`**: Notebook-friendly training utilities
  - `simple_train.py`: Main entry points (`train_ppo()`, `train_sac()`)
  - `config.py`: `TrainingConfig` dataclass with sensible defaults
  - `networks.py`: Custom CNN for vision + state fusion (`CustomCNN`)
  - Automatically handles model saving to `models/ppo_drone_YYYYMMDD_HHMMSS/`

- **`bitcraze_crazyflie_2/`**: MuJoCo XML model files (`scene.xml`, `cf2.xml`)

## Key Patterns & Conventions

### Environment Configuration
```python
# Always use EnvConfig for environment setup
from crazy_flie_env.utils.config import EnvConfig
config = EnvConfig(
    dt=0.02,  # 50Hz control loop
    max_episode_steps=100000,
    initial_height=0.3  # 30cm spawn height
)
```

### Multi-Modal Observations
Environment returns `Dict[str, np.ndarray]` observations:
- `'state'`: 12D vector (position, velocity, orientation, angular velocity)
- `'image'`: Camera feed (configurable resolution, multiple camera views available)

**Critical**: Use `ObservationFixWrapper` in training to fix SB3 dtype issues:
```python
env = ObservationFixWrapper(Monitor(CrazyFlieEnv(config)))
```

### Training Workflow (Notebook-Optimized)
```python
from train.simple_train import train_ppo
from train.config import TrainingConfig

config = TrainingConfig(total_timesteps=100_000, num_envs=4)
model, results = train_ppo(config)  # Returns both model and paths

# Test immediately after training
from train.test_utils import quick_test
quick_test(results['final_model_path'])
```

### Physics & Control Architecture
- **Two-layer control**: High-level RL actions → PID controllers → MuJoCo actuators
- Actions: `[roll, pitch, yaw_rate, thrust]` normalized to `[-1,1]` or `[0,1]`
- PID controllers in `physics/controller.py` handle low-level stabilization
- Physics timestep: `dt=0.02` with `physics_steps=10` sub-steps per control

## Developer Workflows

### Training Models
- **From notebooks**: Use `train/simple_train.py` functions directly
- **From scripts**: Import and configure via `TrainingConfig`
- Models auto-save to `models/` with timestamp directories
- Logs auto-save to `logs/` with TensorBoard integration

### Testing & Debugging
- `train/test_utils.py`: `test_model()`, `quick_test()`, `visualize_episode()`
- Use `quick_test()` for immediate visual verification after training
- Camera angles configurable in `test_camera_angles.py` for debugging views

### Adding New Reward Functions
Extend `rewards/reward_functions.py` `RewardCalculator` class:
```python
def _setup_reward_components(self):
    return {
        'height': self._height_reward,
        'stability': self._stability_reward,
        'custom_task': self._your_custom_reward  # Add here
    }
```

## Integration Points & Dependencies

### External Dependencies
- **MuJoCo**: Physics simulation (requires `scene.xml` in `bitcraze_crazyflie_2/`)
- **Stable-Baselines3**: RL algorithms (PPO, SAC)
- **Gymnasium**: Environment interface
- Models expect specific XML structure with camera names: `"drone_fpv"`, `"chase_cam"`

### Configuration Coupling
- `EnvConfig.model_path` must point to valid MuJoCo XML files
- Camera system expects specific camera names in XML model
- Action/observation bounds configured in `EnvConfig` - modify here for different drones

### Critical File Dependencies
- `bitcraze_crazyflie_2/scene.xml`: Main MuJoCo scene (includes `cf2.xml`)
- `crazy_flie_env/utils/config.py`: Central configuration - most behavior tuned here
- `train/simple_train.py`: Entry point for all training workflows

## Logging System

The project uses a centralized logging system (no print statements):
- **Automatic initialization**: Logging starts when calling `train_ppo()` or `train_sac()`
- **Timestamped log files**: Created in `logs/drone_training_YYYYMMDD_HHMMSS.log`
- **Console level**: Only warnings/errors by default (keeps output clean)
- **File level**: Everything (DEBUG, INFO, WARNING, ERROR) with emoji formatting
- **Usage**: `from crazy_flie_env.utils.logging_utils import get_logger; logger = get_logger(__name__)`

## Common Gotchas

1. **Observation dtype issues**: Always use `ObservationFixWrapper` with SB3
2. **Model paths**: Use absolute paths, auto-generated timestamped directories
3. **Camera setup**: Ensure XML model has required camera names before environment init
4. **Multi-env training**: `num_envs>1` requires careful config to avoid XML conflicts
5. **Logging**: Never use `print()` - use `logger.info()`, `logger.warning()`, `logger.error()` instead