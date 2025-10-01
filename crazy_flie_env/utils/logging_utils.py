"""
Central logging configuration for the CrazyFlie RL Environment.

This module provides a centralized logging system that:
- Creates timestamped log files in the logs/ directory
- Shows only warnings and errors on console
- Provides different log levels for different types of messages
- Maintains the existing emoji-based messaging style in log files
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class EmojiFormatter(logging.Formatter):
    """Custom formatter that maintains emoji-based message styling."""
    
    def format(self, record):
        # Add emoji prefix based on log level
        emoji_map = {
            'DEBUG': '🔍',
            'INFO': '✅',
            'WARNING': '⚠️',
            'ERROR': '❌',
            'CRITICAL': '🚨'
        }
        
        emoji = emoji_map.get(record.levelname, '📝')
        
        # Format the message with emoji and timestamp
        original_format = self._style._fmt
        self._style._fmt = f'%(asctime)s - {emoji} %(name)s - %(levelname)s - %(message)s'
        
        formatted = super().format(record)
        
        # Restore original format
        self._style._fmt = original_format
        
        return formatted


def setup_logging(log_dir: Optional[str] = None, 
                 console_level: int = logging.WARNING,
                 file_level: int = logging.DEBUG) -> str:
    """
    Set up centralized logging for the drone RL environment.
    
    Args:
        log_dir: Directory to store log files. If None, uses 'logs' in current directory
        console_level: Minimum level for console output (default: WARNING)
        file_level: Minimum level for file output (default: DEBUG)
        
    Returns:
        Path to the created log file
    """
    # Create log directory if it doesn't exist
    if log_dir is None:
        log_dir = "logs"
    
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"drone_training_{timestamp}.log"
    log_filepath = os.path.join(log_dir, log_filename)
    
    # Remove any existing handlers to avoid duplicates
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set root logger level to DEBUG to capture everything
    root_logger.setLevel(logging.DEBUG)
    
    # Create file handler with emoji formatter
    file_handler = logging.FileHandler(log_filepath, mode='w', encoding='utf-8')
    file_handler.setLevel(file_level)
    file_formatter = EmojiFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Create console handler (only warnings and errors)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_formatter = EmojiFormatter(
        '%(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Log the initial setup
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized - Log file: {log_filepath}")
    logger.info(f"Console level: {logging.getLevelName(console_level)}, File level: {logging.getLevelName(file_level)}")
    
    return log_filepath


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Usually __name__ from the calling module
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def log_system_info():
    """Log system information at startup."""
    logger = get_logger(__name__)
    
    logger.info("=== CrazyFlie RL Environment - System Information ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    
    # Log key environment variables if they exist
    env_vars = ['CUDA_VISIBLE_DEVICES', 'MUJOCO_GL', 'PYTHONPATH']
    for var in env_vars:
        value = os.environ.get(var)
        if value:
            logger.info(f"{var}: {value}")
    
    logger.info("=" * 60)


def log_training_start(config_info: dict):
    """Log training configuration at start."""
    logger = get_logger(__name__)
    
    logger.info("=== Training Configuration ===")
    for key, value in config_info.items():
        logger.info(f"{key}: {value}")
    logger.info("=" * 40)


def log_training_complete(results: dict):
    """Log training completion summary."""
    logger = get_logger(__name__)
    
    logger.info("=== Training Complete ===")
    for key, value in results.items():
        logger.info(f"{key}: {value}")
    logger.info("=" * 30)