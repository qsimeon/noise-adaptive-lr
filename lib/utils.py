"""Utility functions for noise-aware learning rate optimization.

This module provides helper functions for visualization, metrics computation,
and analysis of the optimization process.
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from collections import defaultdict
import warnings


def compute_signal_to_noise_ratio(
    smooth_losses: List[float],
    noise_estimates: List[float]
) -> float:
    """Compute the signal-to-noise ratio of the loss landscape.
    
    Args:
        smooth_losses: List of smoothed loss values.
        noise_estimates: List of noise estimates.
        
    Returns:
        Signal-to-noise ratio (higher is better).
    """
    if not smooth_losses or not noise_estimates:
        return 0.0
    
    signal_power = np.var(smooth_losses) if len(smooth_losses) > 1 else 0.0
    noise_power = np.mean(np.array(noise_estimates) ** 2)
    
    if noise_power < 1e-10:
        return float('inf')
    
    return float(signal_power / noise_power)


def detect_local_minima(
    losses: List[float],
    window_size: int = 5,
    threshold: float = 0.01
) -> List[int]:
    """Detect potential local minima in the loss trajectory.
    
    Args:
        losses: List of loss values.
        window_size: Size of window for local comparison.
        threshold: Minimum depth threshold for considering a minimum.
        
    Returns:
        List of indices where local minima occur.
    """
    if len(losses) < window_size * 2 + 1:
        return []
    
    minima_indices = []
    losses_array = np.array(losses)
    
    for i in range(window_size, len(losses) - window_size):
        left_window = losses_array[i - window_size:i]
        right_window = losses_array[i + 1:i + window_size + 1]
        current = losses_array[i]
        
        # Check if current point is lower than neighbors
        if np.all(current <= left_window) and np.all(current <= right_window):
            # Check if the depth is significant
            depth = min(np.min(left_window) - current, np.min(right_window) - current)
            if depth >= threshold:
                minima_indices.append(i)
    
    return minima_indices


def estimate_gradient_noise(
    gradients: List[np.ndarray],
    window_size: int = 10
) -> float:
    """Estimate the noise level in gradient estimates.
    
    Args:
        gradients: List of gradient arrays.
        window_size: Window size for noise estimation.
        
    Returns:
        Estimated gradient noise magnitude.
    """
    if len(gradients) < window_size:
        return 0.0
    
    recent_grads = gradients[-window_size:]
    
    # Compute mean gradient direction
    mean_grad = np.mean(recent_grads, axis=0)
    
    # Compute variance around mean
    deviations = [np.linalg.norm(g - mean_grad) for g in recent_grads]
    
    return float(np.mean(deviations))


def smooth_trajectory(
    values: List[float],
    method: str = 'exponential',
    alpha: float = 0.3,
    window_size: int = 5
) -> List[float]:
    """Smooth a trajectory of values using various methods.
    
    Args:
        values: List of values to smooth.
        method: Smoothing method ('exponential', 'moving_average', 'gaussian').
        alpha: Smoothing parameter for exponential smoothing.
        window_size: Window size for moving average.
        
    Returns:
        Smoothed trajectory.
    """
    if not values:
        return []
    
    if method == 'exponential':
        smoothed = [values[0]]
        for v in values[1:]:
            smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])
        return smoothed
    
    elif method == 'moving_average':
        smoothed = []
        for i in range(len(values)):
            start = max(0, i - window_size // 2)
            end = min(len(values), i + window_size // 2 + 1)
            smoothed.append(float(np.mean(values[start:end])))
        return smoothed
    
    elif method == 'gaussian':
        # Simple Gaussian smoothing
        smoothed = []
        sigma = window_size / 3.0
        for i in range(len(values)):
            weights = []
            weighted_sum = 0.0
            weight_total = 0.0
            
            for j in range(max(0, i - window_size), min(len(values), i + window_size + 1)):
                weight = np.exp(-((i - j) ** 2) / (2 * sigma ** 2))
                weighted_sum += values[j] * weight
                weight_total += weight
            
            smoothed.append(weighted_sum / weight_total if weight_total > 0 else values[i])
        return smoothed
    
    else:
        return values


def compute_optimization_metrics(
    losses: List[float],
    learning_rates: List[float],
    smooth_losses: Optional[List[float]] = None
) -> Dict[str, float]:
    """Compute various metrics about the optimization process.
    
    Args:
        losses: List of raw loss values.
        learning_rates: List of learning rates used.
        smooth_losses: Optional list of smoothed losses.
        
    Returns:
        Dictionary of computed metrics.
    """
    if not losses:
        return {}
    
    losses_array = np.array(losses)
    lr_array = np.array(learning_rates)
    
    metrics = {
        'final_loss': float(losses[-1]),
        'min_loss': float(np.min(losses_array)),
        'max_loss': float(np.max(losses_array)),
        'mean_loss': float(np.mean(losses_array)),
        'loss_std': float(np.std(losses_array)),
        'total_improvement': float(losses[0] - losses[-1]) if len(losses) > 1 else 0.0,
        'mean_lr': float(np.mean(lr_array)),
        'lr_std': float(np.std(lr_array)),
        'lr_adaptations': int(np.sum(np.abs(np.diff(lr_array)) > 1e-8))
    }
    
    if smooth_losses and len(smooth_losses) == len(losses):
        noise = np.abs(losses_array - np.array(smooth_losses))
        metrics['mean_noise'] = float(np.mean(noise))
        metrics['max_noise'] = float(np.max(noise))
        metrics['snr'] = compute_signal_to_noise_ratio(smooth_losses, noise.tolist())
    
    # Compute convergence rate
    if len(losses) > 10:
        recent_losses = losses_array[-10:]
        convergence_rate = float(np.mean(np.abs(np.diff(recent_losses))))
        metrics['convergence_rate'] = convergence_rate
    
    return metrics


def analyze_stuck_periods(
    smooth_losses: List[float],
    threshold: float = 0.001,
    min_duration: int = 5
) -> List[Tuple[int, int]]:
    """Analyze periods where optimization appears stuck.
    
    Args:
        smooth_losses: List of smoothed loss values.
        threshold: Threshold for considering progress as stuck.
        min_duration: Minimum duration to consider a stuck period.
        
    Returns:
        List of (start_idx, end_idx) tuples for stuck periods.
    """
    if len(smooth_losses) < min_duration:
        return []
    
    stuck_periods = []
    current_start = None
    
    for i in range(1, len(smooth_losses)):
        improvement = smooth_losses[i - 1] - smooth_losses[i]
        relative_improvement = improvement / (abs(smooth_losses[i - 1]) + 1e-8)
        
        if relative_improvement < threshold:
            if current_start is None:
                current_start = i - 1
        else:
            if current_start is not None:
                duration = i - current_start
                if duration >= min_duration:
                    stuck_periods.append((current_start, i - 1))
                current_start = None
    
    # Handle case where stuck period extends to end
    if current_start is not None:
        duration = len(smooth_losses) - current_start
        if duration >= min_duration:
            stuck_periods.append((current_start, len(smooth_losses) - 1))
    
    return stuck_periods


def create_lr_schedule_summary(
    learning_rates: List[float],
    losses: List[float]
) -> Dict[str, Any]:
    """Create a summary of learning rate schedule behavior.
    
    Args:
        learning_rates: List of learning rates.
        losses: List of corresponding losses.
        
    Returns:
        Dictionary with schedule summary statistics.
    """
    if not learning_rates or not losses:
        return {}
    
    lr_array = np.array(learning_rates)
    loss_array = np.array(losses)
    
    # Find increases and decreases
    lr_changes = np.diff(lr_array)
    increases = np.sum(lr_changes > 1e-8)
    decreases = np.sum(lr_changes < -1e-8)
    
    # Compute correlation between LR and loss change
    if len(losses) > 1:
        loss_changes = np.diff(loss_array)
        if len(loss_changes) > 0 and np.std(lr_changes) > 1e-8:
            correlation = float(np.corrcoef(lr_changes, loss_changes)[0, 1])
        else:
            correlation = 0.0
    else:
        correlation = 0.0
    
    return {
        'num_increases': int(increases),
        'num_decreases': int(decreases),
        'total_adaptations': int(increases + decreases),
        'lr_range': (float(np.min(lr_array)), float(np.max(lr_array))),
        'lr_loss_correlation': correlation,
        'adaptation_frequency': float((increases + decreases) / len(learning_rates))
    }


def validate_optimizer_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate optimizer configuration parameters.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        Tuple of (is_valid, list_of_errors).
    """
    errors = []
    
    # Check learning rate bounds
    if 'base_lr' in config and config['base_lr'] <= 0:
        errors.append("base_lr must be positive")
    
    if 'min_lr' in config and config['min_lr'] < 0:
        errors.append("min_lr must be non-negative")
    
    if 'max_lr' in config and config['max_lr'] <= 0:
        errors.append("max_lr must be positive")
    
    if 'min_lr' in config and 'max_lr' in config:
        if config['min_lr'] >= config['max_lr']:
            errors.append("min_lr must be less than max_lr")
    
    # Check window size
    if 'window_size' in config and config['window_size'] < 2:
        errors.append("window_size must be at least 2")
    
    # Check alpha
    if 'alpha' in config:
        if config['alpha'] <= 0 or config['alpha'] > 1:
            errors.append("alpha must be in range (0, 1]")
    
    # Check adaptation factor
    if 'adaptation_factor' in config and config['adaptation_factor'] <= 1:
        errors.append("adaptation_factor must be greater than 1")
    
    # Check patience
    if 'patience' in config and config['patience'] < 1:
        errors.append("patience must be at least 1")
    
    # Check smoothing method
    valid_methods = ['moving_average', 'exponential', 'savitzky_golay']
    if 'smoothing_method' in config:
        if config['smoothing_method'] not in valid_methods:
            errors.append(f"smoothing_method must be one of {valid_methods}")
    
    return len(errors) == 0, errors


def _compute_running_statistics(
    values: List[float],
    window_size: int = 10
) -> Dict[str, List[float]]:
    """Compute running statistics over a sliding window.
    
    Args:
        values: List of values.
        window_size: Size of sliding window.
        
    Returns:
        Dictionary with running mean and std.
    """
    running_mean = []
    running_std = []
    
    for i in range(len(values)):
        start = max(0, i - window_size + 1)
        window = values[start:i + 1]
        running_mean.append(float(np.mean(window)))
        running_std.append(float(np.std(window)))
    
    return {
        'mean': running_mean,
        'std': running_std
    }
