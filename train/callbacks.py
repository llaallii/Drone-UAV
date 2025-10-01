"""
Custom callbacks for training visualization and monitoring.
"""

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from crazy_flie_env.utils.logging_utils import get_logger

logger = get_logger(__name__)


class RenderCallback(BaseCallback):
    """
    Callback for rendering the environment during training.

    This allows live visualization of the drone training process
    without significantly impacting training performance.

    Args:
        render_freq: Render every N steps (default: 1000)
        verbose: Verbosity level (0: no output, 1: info)
    """

    def __init__(self, render_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.render_freq = render_freq
        self.render_count = 0

    def _on_step(self) -> bool:
        """
        Called at every step of the environment.

        Returns:
            True to continue training, False to stop
        """
        # Only render at specified frequency
        if self.n_calls % self.render_freq == 0:
            try:
                # Get the base environment from the vectorized wrapper
                if isinstance(self.training_env, VecEnv):
                    # For vectorized environments, render the first environment
                    env = self.training_env.envs[0]

                    # Unwrap to get the actual CrazyFlieEnv
                    while hasattr(env, 'env'):
                        env = env.env

                    # Call render on the base environment
                    if hasattr(env, 'render'):
                        env.render(mode='human')
                        self.render_count += 1

                        # Only print occasional status updates
                        if self.verbose > 0 and self.render_count == 1:
                            logger.info(f"Rendering enabled - viewer should be visible")
                        elif self.verbose > 0 and self.render_count % 50 == 0:
                            logger.info(f"Rendered {self.render_count} frames "
                                  f"(step {self.n_calls}/{self.num_timesteps})")
                else:
                    # For non-vectorized environments
                    self.training_env.render(mode='human')
                    self.render_count += 1

            except Exception as e:
                # More informative error handling
                if self.verbose > 0:
                    if self.render_count == 0:
                        # First render attempt failed
                        logger.warning(f"Rendering failed to start: {e}")
                        logger.info("   This is common in Jupyter notebooks on Windows.")
                        logger.info("   Training will continue without visualization.")
                        logger.info("   To fix: Run training script directly with Python (not in Jupyter)")
                    elif self.render_count % 100 == 0:
                        # Occasional reminder if rendering keeps failing
                        logger.warning(f"Rendering error at step {self.n_calls}: {e}")

        return True

    def _on_training_end(self) -> None:
        """Called at the end of training."""
        if self.verbose > 0:
            logger.info(f"\n✅ Training complete. Total frames rendered: {self.render_count}")


class LiveMetricsCallback(BaseCallback):
    """
    Callback for displaying live training metrics.

    Prints useful metrics during training like episode rewards,
    success rate, and drone statistics.

    Args:
        print_freq: Print metrics every N steps (default: 5000)
        verbose: Verbosity level
    """

    def __init__(self, print_freq: int = 5000, verbose: int = 1):
        super().__init__(verbose)
        self.print_freq = print_freq
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        """Called at every step."""
        # Collect episode statistics if available
        if len(self.model.ep_info_buffer) > 0:
            for info in self.model.ep_info_buffer:
                if 'r' in info:
                    self.episode_rewards.append(info['r'])
                if 'l' in info:
                    self.episode_lengths.append(info['l'])

        # Print metrics at specified frequency
        if self.n_calls % self.print_freq == 0 and len(self.episode_rewards) > 0:
            mean_reward = np.mean(self.episode_rewards[-100:])
            mean_length = np.mean(self.episode_lengths[-100:])

            if self.verbose > 0:
                logger.info(f"\n📊 Training Metrics (Step {self.n_calls}):")
                logger.info(f"   Mean Episode Reward: {mean_reward:.2f}")
                logger.info(f"   Mean Episode Length: {mean_length:.1f}")
                logger.info(f"   Episodes Completed: {len(self.episode_rewards)}")

        return True


class DroneStatsCallback(BaseCallback):
    """
    Callback for tracking drone-specific statistics during training.

    Monitors crash rate, flight stability, altitude maintenance, etc.

    Args:
        stats_freq: Compute stats every N steps (default: 10000)
        verbose: Verbosity level
    """

    def __init__(self, stats_freq: int = 10000, verbose: int = 1):
        super().__init__(verbose)
        self.stats_freq = stats_freq
        self.crash_count = 0
        self.total_episodes = 0
        self.successful_flights = 0

    def _on_step(self) -> bool:
        """Called at every step."""
        # Check if episode ended
        if self.locals.get('dones') is not None:
            dones = self.locals['dones']
            infos = self.locals.get('infos', [])

            for i, done in enumerate(dones):
                if done:
                    self.total_episodes += 1

                    # Check if crashed
                    if i < len(infos) and 'is_crashed' in infos[i]:
                        if infos[i]['is_crashed']:
                            self.crash_count += 1
                        else:
                            self.successful_flights += 1

        # Print stats at specified frequency
        if self.n_calls % self.stats_freq == 0 and self.total_episodes > 0:
            crash_rate = (self.crash_count / self.total_episodes) * 100
            success_rate = (self.successful_flights / self.total_episodes) * 100

            if self.verbose > 0:
                logger.info(f"\n🚁 Drone Statistics (Step {self.n_calls}):")
                logger.info(f"   Total Episodes: {self.total_episodes}")
                logger.info(f"   Crash Rate: {crash_rate:.1f}%")
                logger.info(f"   Success Rate: {success_rate:.1f}%")

        return True
