# 🚁 Modular Drone Training System

A flexible, algorithm-agnostic reinforcement learning system for training autonomous drone navigation policies with real-time visualization capabilities.

## 📁 Project Structure

```
train/
├── __init__.py           # Package initialization
├── main.py              # Main entry point with CLI
├── config.py            # Training configuration
├── manager.py           # Training orchestration
├── algorithms.py        # Algorithm implementations
├── networks.py          # Neural network architectures
├── environments.py      # Environment factory and wrappers
├── callbacks.py         # Training callbacks and monitoring
├── utils.py             # Utility functions
├── examples/            # Example configurations and scripts
└── README.md           # This file
```

## 🚀 Quick Start

### Basic Usage

```bash
# Interactive mode - guided setup
python -m train --mode interactive

# Quick PPO training with live visualization
python -m train --algorithm PPO --live_training --timesteps 1000000

# Fast SAC training without visualization
python -m train --algorithm SAC --timesteps 500000 --num_envs 8
```

### Live Training Experience

When you enable live training, you'll see:
- 🎥 Real-time 3D environment rendering
- 📊 Live reward and performance plots
- 📈 Training progress monitoring
- 🎯 Performance statistics

```bash
# Enable live training with custom settings
python -m train --live_training --render_freq 500 --algorithm PPO
```

## 🧠 Supported Algorithms

| Algorithm | Description | Best For |
|-----------|-------------|----------|
| **PPO** | Proximal Policy Optimization | General purpose, stable |
| **SAC** | Soft Actor-Critic | Continuous control, sample efficient |
| **RAPID** | Inverse Reinforcement Learning | Learning from demonstrations |
| **A2C** | Advantage Actor-Critic | Fast training, simple |
| **Custom** | Your own algorithm | Research and experimentation |

### Algorithm Selection Guide

```bash
# Stable and reliable (recommended for beginners)
python -m train --algorithm PPO

# Sample efficient for continuous control
python -m train --algorithm SAC

# Learning from expert demonstrations
python -m train --algorithm RAPID

# Quick prototyping
python -m train --algorithm A2C
```

## 🛠️ Configuration

### Interactive Configuration

The interactive mode walks you through all options:

```bash
python -m train --mode interactive
```

### Direct Configuration

```bash
python -m train \
  --algorithm PPO \
  --timesteps 1000000 \
  --num_envs 8 \
  --learning_rate 3e-4 \
  --batch_size 64 \
  --live_training \
  --render_freq 1000
```

### Configuration Files

Create reusable configurations:

```python
from train.config import TrainingConfig

config = TrainingConfig(
    algorithm="PPO",
    total_timesteps=1_000_000,
    num_envs=8,
    learning_rate=3e-4,
    enable_live_training=True,
    render_freq=500
)
```

## 🎯 Training Modes

### 1. Interactive Mode (Recommended)
```bash
python -m train --mode interactive
```
- Guided configuration setup
- Perfect for beginners
- Asks about live visualization
- Validates all settings

### 2. Configuration Mode
```bash
python -m train --mode config --algorithm PPO --timesteps 1000000
```
- Direct command-line configuration
- Great for scripts and automation
- All parameters configurable

### 3. Benchmark Mode
```bash
python -m train --mode benchmark
```
- Compares multiple algorithms
- Automatic performance evaluation
- Generates comparison reports

### 4. Test Mode
```bash
python -m train --mode test --test_model models/ppo_drone_20241201/final_model
```
- Evaluate trained models
- Performance analysis
- Video generation

### 5. Quick Test Mode
```bash
python -m train --mode quick_test --algorithm PPO
```
- Verify system setup
- Fast development testing
- 10,000 timesteps only

## 🎥 Live Training Visualization

### Enabling Live Training

```bash
# Enable during training
python -m train --live_training

# Customize render frequency
python -m train --live_training --render_freq 500
```

### What You'll See

1. **Real-time Environment**: Watch the drone learn to fly in 3D
2. **Live Reward Plots**: Episode rewards updating in real-time
3. **Training Progress**: Learning curves and statistics
4. **Performance Metrics**: Success rates, crash rates, etc.
5. **System Stats**: FPS, memory usage, training speed

### Performance Impact

Live training adds ~30% overhead but provides valuable insights:
- Monitor training stability
- Detect issues early
- Understand drone behavior
- Adjust hyperparameters on-the-fly

## 🔧 Advanced Usage

### Custom Algorithms

Implement your own algorithms:

```python
from train.algorithms import BaseTrainingAlgorithm

class MyCustomAlgorithm(BaseTrainingAlgorithm):
    def create_model(self, env):
        # Your model creation logic
        pass
    
    def train(self, env, callbacks=None):
        # Your training logic
        pass

# Register your algorithm
from train.algorithms import AlgorithmFactory
AlgorithmFactory.register_algorithm('MYCUSTOM', MyCustomAlgorithm)
```

### Custom Networks

Create specialized network architectures:

```python
from train.networks import CustomCNN

class MyCustomCNN(CustomCNN):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        # Your custom architecture
```

### Environment Wrappers

Add custom environment modifications:

```python
from train.environments import EnvironmentFactory

# Custom reward wrapper
class MyRewardWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # Modify reward logic
        return obs, modified_reward, done, info
```

### Custom Callbacks

Monitor specific metrics:

```python
from train.callbacks import BaseCallback

class MyCustomCallback(BaseCallback):
    def _on_step(self):
        # Your monitoring logic
        return True
```

## 📊 Model Evaluation

### Testing Trained Models

```bash
# Test with visualization
python -m train --mode test --test_model models/ppo_drone_20241201/final_model --test_episodes 20

# Quick evaluation
python -m train --mode test --test_model models/ppo_drone_20241201/final_model --test_episodes 5
```

### Benchmark Multiple Models

```python
from train.utils import benchmark_multiple_models

models = {
    'PPO_v1': 'models/ppo_drone_20241201/final_model.zip',
    'SAC_v1': 'models/sac_drone_20241201/final_model.zip'
}

results = benchmark_multiple_models(models, num_episodes=10)
```

### Generate Training Reports

```bash
# Comprehensive report
python -m train --mode report --test_model models/ppo_drone_20241201/

# Quick report
python -m train --mode report --test_model models/ppo_drone_20241201/ --test_episodes 5
```

## 🔍 Debugging and Monitoring

### System Information

```bash
# Check system capabilities
python -m train --system_info

# List available algorithms
python -m train --list_algorithms
```

### Training Diagnostics

```python
from train.utils import analyze_training_logs, plot_training_curves

# Analyze logs
analysis = analyze_training_logs('logs/ppo_drone_20241201/')

# Plot training curves
plot_training_curves('logs/ppo_drone_20241201/', 'training_curves.png')
```

### Performance Monitoring

The system automatically tracks:
- Training FPS
- Memory usage
- Episode statistics
- Crash rates
- Success rates

## 🎛️ Hyperparameter Tuning

### Manual Tuning

```bash
# Adjust learning rate
python -m train --algorithm PPO --learning_rate 1e-4

# Modify batch size
python -m train --algorithm PPO --batch_size 128

# Change environment count
python -m train --algorithm PPO --num_envs 16
```

### Systematic Search

```python
from train.config import TrainingConfig
from train.manager import TrainingManager

# Test different learning rates
learning_rates = [1e-4, 3e-4, 1e-3]
results = {}

for lr in learning_rates:
    config = TrainingConfig(
        algorithm="PPO",
        learning_rate=lr,
        total_timesteps=100_000  # Shorter for tuning
    )
    
    with TrainingManager(config) as manager:
        model, model_dir = manager.train()
        # Evaluate and store results
```

## 🚀 Production Deployment

### Model Export

```python
from train.utils import export_model_for_deployment

# Export for deployment
export_model_for_deployment(
    'models/ppo_drone_20241201/final_model.zip',
    'deployed_model.zip',
    format='onnx'
)
```

### Continuous Training

```python
# Resume from checkpoint
config = TrainingConfig(algorithm="PPO")

with TrainingManager(config) as manager:
    # Resume training
    model = manager.resume_training(
        'models/ppo_drone_20241201/checkpoint_1000000.zip',
        additional_timesteps=500_000
    )
```

## 🛡️ Safety and Validation

### Automatic Safety Monitoring

The system includes:
- Crash detection and logging
- Performance degradation alerts
- Training stability monitoring
- Early stopping on failure

### Model Validation

```python
from train.utils import validate_model_file

# Validate model before deployment
is_valid = validate_model_file('models/ppo_drone_20241201/final_model.zip')
```

## 📚 Examples

### Basic Training Example

```python
from train import TrainingConfig, TrainingManager

# Create configuration
config = TrainingConfig(
    algorithm="PPO",
    total_timesteps=1_000_000,
    enable_live_training=True
)

# Train model
with TrainingManager(config) as manager:
    model, model_dir = manager.train()
    print(f"Training completed: {model_dir}")
```

### Custom Training Pipeline

```python
from train import TrainingConfig, TrainingManager
from train.callbacks import LiveVisualizationCallback

# Advanced configuration
config = TrainingConfig(
    algorithm="SAC",
    total_timesteps=2_000_000,
    num_envs=8,
    learning_rate=1e-4,
    batch_size=256,
    enable_live_training=True
)

# Custom training
with TrainingManager(config) as manager:
    # Add custom callbacks
    callbacks = manager.create_callbacks()
    
    # Train with monitoring
    model, model_dir = manager.train()
```

### Evaluation Pipeline

```python
from train.utils import test_trained_model, create_training_report

# Test model
avg_reward, success_rate = test_trained_model(
    model_path='models/ppo_drone_20241201/final_model.zip',
    algorithm='PPO',
    num_episodes=20,
    render=True
)

# Generate report
report = create_training_report('models/ppo_drone_20241201/')
print(report)
```

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size or number of environments
   python -m train --algorithm PPO --batch_size 32 --num_envs 4
   ```

2. **Slow Training**
   ```bash
   # Disable live training for faster training
   python -m train --algorithm PPO --timesteps 1000000
   ```

3. **Environment Crashes**
   ```bash
   # Run quick test to verify setup
   python -m train --mode quick_test
   ```

### Getting Help

- Check system info: `python -m train --system_info`
- Run quick test: `python -m train --mode quick_test`
- Use interactive mode: `python -m train --mode interactive`

## 📈 Performance Tips

### Optimization Strategies

1. **Use GPU**: Ensure CUDA is available
2. **Parallel Environments**: Use 4-8 environments
3. **Batch Size**: Start with 64, increase if GPU allows
4. **Live Training**: Disable for maximum speed
5. **Algorithm Choice**: SAC for sample efficiency, PPO for stability

### Recommended Settings

```bash
# Fast training (no visualization)
python -m train --algorithm SAC --num_envs 8 --batch_size 256

# Stable training with monitoring
python -m train --algorithm PPO --num_envs 4 --live_training --render_freq 2000

# Development/debugging
python -m train --algorithm PPO --timesteps 50000 --live_training --render_freq 500
```

## 📝 License and Contributing

This modular training system is designed to be:
- **Extensible**: Easy to add new algorithms
- **Configurable**: Flexible parameter tuning
- **Observable**: Rich monitoring and visualization
- **Reliable**: Robust error handling and validation

Contribute by:
- Adding new algorithms
- Improving visualizations
- Enhancing environment wrappers
- Optimizing performance