"""Core module for adaptive learning rate optimization with noise estimation.

This module provides classes and functions for estimating the smooth loss landscape
and adapting learning rates to avoid getting stuck in noisy local minima.
"""

from typing import Dict, List, Optional, Tuple, Callable, Any
import numpy as np
from collections import deque
import warnings


class LossLandscapeEstimator:
    """Estimates the smooth loss landscape by filtering out noise.
    
    This class maintains a history of loss values and gradients to estimate
    the underlying smooth loss function, separating signal from noise.
    
    Attributes:
        window_size: Number of recent samples to consider for smoothing.
        smoothing_method: Method to use for smoothing ('moving_average', 'exponential', 'savitzky_golay').
        alpha: Smoothing factor for exponential smoothing (0 < alpha <= 1).
    """
    
    def __init__(
        self,
        window_size: int = 20,
        smoothing_method: str = 'exponential',
        alpha: float = 0.3
    ):
        """Initialize the loss landscape estimator.
        
        Args:
            window_size: Number of samples for smoothing window.
            smoothing_method: Smoothing technique to use.
            alpha: Exponential smoothing factor.
        """
        self.window_size = window_size
        self.smoothing_method = smoothing_method
        self.alpha = alpha
        
        self._loss_history: deque = deque(maxlen=window_size)
        self._smooth_loss_history: List[float] = []
        self._noise_estimates: List[float] = []
        self._current_smooth_loss: Optional[float] = None
    
    def update(self, loss: float) -> Tuple[float, float]:
        """Update the estimator with a new loss value.
        
        Args:
            loss: Current loss value.
            
        Returns:
            Tuple of (smooth_loss, noise_estimate).
        """
        self._loss_history.append(loss)
        
        if len(self._loss_history) < 2:
            self._current_smooth_loss = loss
            noise_estimate = 0.0
        else:
            smooth_loss = self._compute_smooth_loss()
            self._current_smooth_loss = smooth_loss
            noise_estimate = abs(loss - smooth_loss)
            
            self._smooth_loss_history.append(smooth_loss)
            self._noise_estimates.append(noise_estimate)
        
        return self._current_smooth_loss, noise_estimate
    
    def _compute_smooth_loss(self) -> float:
        """Compute the smoothed loss based on the selected method.
        
        Returns:
            Smoothed loss value.
        """
        if self.smoothing_method == 'moving_average':
            return np.mean(self._loss_history)
        
        elif self.smoothing_method == 'exponential':
            if self._current_smooth_loss is None:
                return self._loss_history[0]
            return self.alpha * self._loss_history[-1] + (1 - self.alpha) * self._current_smooth_loss
        
        elif self.smoothing_method == 'savitzky_golay':
            if len(self._loss_history) < 5:
                return np.mean(self._loss_history)
            # Simple polynomial smoothing
            losses = np.array(self._loss_history)
            return np.median(losses[-5:])
        
        else:
            return np.mean(self._loss_history)
    
    def get_noise_statistics(self) -> Dict[str, float]:
        """Get statistics about the estimated noise.
        
        Returns:
            Dictionary containing noise statistics (mean, std, max).
        """
        if not self._noise_estimates:
            return {'mean': 0.0, 'std': 0.0, 'max': 0.0, 'recent': 0.0}
        
        recent_window = min(10, len(self._noise_estimates))
        recent_noise = self._noise_estimates[-recent_window:]
        
        return {
            'mean': float(np.mean(self._noise_estimates)),
            'std': float(np.std(self._noise_estimates)),
            'max': float(np.max(self._noise_estimates)),
            'recent': float(np.mean(recent_noise))
        }
    
    def get_smooth_loss(self) -> Optional[float]:
        """Get the current smooth loss estimate.
        
        Returns:
            Current smooth loss value or None if not yet computed.
        """
        return self._current_smooth_loss


class AdaptiveLearningRateScheduler:
    """Adapts learning rate based on noise estimates to avoid local minima.
    
    This scheduler adjusts the learning rate dynamically based on the estimated
    noise in the loss landscape, increasing it when stuck in noisy regions.
    
    Attributes:
        base_lr: Base learning rate.
        min_lr: Minimum allowed learning rate.
        max_lr: Maximum allowed learning rate.
        noise_threshold: Noise level threshold for adaptation.
        adaptation_factor: Factor by which to adjust learning rate.
    """
    
    def __init__(
        self,
        base_lr: float = 0.001,
        min_lr: float = 1e-6,
        max_lr: float = 0.1,
        noise_threshold: float = 0.1,
        adaptation_factor: float = 1.5,
        patience: int = 5
    ):
        """Initialize the adaptive learning rate scheduler.
        
        Args:
            base_lr: Initial learning rate.
            min_lr: Minimum learning rate bound.
            max_lr: Maximum learning rate bound.
            noise_threshold: Threshold for noise-based adaptation.
            adaptation_factor: Multiplicative factor for LR adjustment.
            patience: Number of steps before adapting.
        """
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.noise_threshold = noise_threshold
        self.adaptation_factor = adaptation_factor
        self.patience = patience
        
        self._current_lr = base_lr
        self._stuck_counter = 0
        self._improvement_history: deque = deque(maxlen=patience)
    
    def step(
        self,
        smooth_loss: float,
        noise_estimate: float,
        current_loss: float
    ) -> float:
        """Compute the adapted learning rate for the current step.
        
        Args:
            smooth_loss: Smoothed loss estimate.
            noise_estimate: Estimated noise magnitude.
            current_loss: Current raw loss value.
            
        Returns:
            Adapted learning rate.
        """
        self._improvement_history.append(smooth_loss)
        
        # Check if we're stuck (no improvement in smooth loss)
        is_stuck = self._detect_stuck_state()
        
        # Check if noise is high relative to loss magnitude
        noise_ratio = noise_estimate / (abs(smooth_loss) + 1e-8)
        high_noise = noise_ratio > self.noise_threshold
        
        if is_stuck and high_noise:
            # Increase learning rate to escape noisy local minimum
            self._current_lr = min(self._current_lr * self.adaptation_factor, self.max_lr)
            self._stuck_counter = 0
        elif not is_stuck and not high_noise:
            # Decrease learning rate for fine-tuning in smooth regions
            self._current_lr = max(self._current_lr / (self.adaptation_factor ** 0.5), self.min_lr)
        
        return self._current_lr
    
    def _detect_stuck_state(self) -> bool:
        """Detect if optimization is stuck based on loss history.
        
        Returns:
            True if stuck, False otherwise.
        """
        if len(self._improvement_history) < self.patience:
            return False
        
        losses = list(self._improvement_history)
        # Check if loss hasn't improved significantly
        recent_improvement = losses[0] - losses[-1]
        relative_improvement = recent_improvement / (abs(losses[0]) + 1e-8)
        
        if relative_improvement < 0.001:  # Less than 0.1% improvement
            self._stuck_counter += 1
            return self._stuck_counter >= self.patience
        else:
            self._stuck_counter = 0
            return False
    
    def get_current_lr(self) -> float:
        """Get the current learning rate.
        
        Returns:
            Current learning rate value.
        """
        return self._current_lr
    
    def reset(self) -> None:
        """Reset the scheduler to initial state."""
        self._current_lr = self.base_lr
        self._stuck_counter = 0
        self._improvement_history.clear()


class NoiseAwareLearningRateOptimizer:
    """Combined optimizer that estimates noise and adapts learning rate.
    
    This class integrates loss landscape estimation and adaptive learning rate
    scheduling to provide a complete solution for noise-aware optimization.
    """
    
    def __init__(
        self,
        base_lr: float = 0.001,
        window_size: int = 20,
        smoothing_method: str = 'exponential',
        alpha: float = 0.3,
        min_lr: float = 1e-6,
        max_lr: float = 0.1,
        noise_threshold: float = 0.1,
        adaptation_factor: float = 1.5,
        patience: int = 5
    ):
        """Initialize the noise-aware optimizer.
        
        Args:
            base_lr: Base learning rate.
            window_size: Window size for loss smoothing.
            smoothing_method: Method for smoothing.
            alpha: Exponential smoothing factor.
            min_lr: Minimum learning rate.
            max_lr: Maximum learning rate.
            noise_threshold: Noise threshold for adaptation.
            adaptation_factor: LR adaptation factor.
            patience: Patience for stuck detection.
        """
        self.estimator = LossLandscapeEstimator(
            window_size=window_size,
            smoothing_method=smoothing_method,
            alpha=alpha
        )
        
        self.scheduler = AdaptiveLearningRateScheduler(
            base_lr=base_lr,
            min_lr=min_lr,
            max_lr=max_lr,
            noise_threshold=noise_threshold,
            adaptation_factor=adaptation_factor,
            patience=patience
        )
        
        self._step_count = 0
    
    def step(self, loss: float) -> Dict[str, Any]:
        """Process a training step and compute adapted learning rate.
        
        Args:
            loss: Current loss value.
            
        Returns:
            Dictionary containing learning rate and diagnostic information.
        """
        # Update loss landscape estimate
        smooth_loss, noise_estimate = self.estimator.update(loss)
        
        # Adapt learning rate
        new_lr = self.scheduler.step(smooth_loss, noise_estimate, loss)
        
        self._step_count += 1
        
        # Get noise statistics
        noise_stats = self.estimator.get_noise_statistics()
        
        return {
            'learning_rate': new_lr,
            'smooth_loss': smooth_loss,
            'noise_estimate': noise_estimate,
            'noise_stats': noise_stats,
            'step': self._step_count,
            'raw_loss': loss
        }
    
    def get_learning_rate(self) -> float:
        """Get the current learning rate.
        
        Returns:
            Current learning rate.
        """
        return self.scheduler.get_current_lr()
    
    def reset(self) -> None:
        """Reset the optimizer to initial state."""
        self.scheduler.reset()
        self._step_count = 0


def create_optimizer(
    base_lr: float = 0.001,
    config: Optional[Dict[str, Any]] = None
) -> NoiseAwareLearningRateOptimizer:
    """Factory function to create a noise-aware optimizer with configuration.
    
    Args:
        base_lr: Base learning rate.
        config: Optional configuration dictionary.
        
    Returns:
        Configured NoiseAwareLearningRateOptimizer instance.
    """
    if config is None:
        config = {}
    
    return NoiseAwareLearningRateOptimizer(
        base_lr=base_lr,
        window_size=config.get('window_size', 20),
        smoothing_method=config.get('smoothing_method', 'exponential'),
        alpha=config.get('alpha', 0.3),
        min_lr=config.get('min_lr', 1e-6),
        max_lr=config.get('max_lr', 0.1),
        noise_threshold=config.get('noise_threshold', 0.1),
        adaptation_factor=config.get('adaptation_factor', 1.5),
        patience=config.get('patience', 5)
    )
