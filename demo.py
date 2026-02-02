"""
Noise-Aware Learning Rate Adaptation Demo

This demo demonstrates how to use noise-aware optimization to avoid getting stuck
in local minima caused by noisy gradients. It shows:
1. How small learning rates can get stuck in noisy local minima
2. How to estimate the smooth loss landscape
3. How to adapt learning rates based on noise estimates
4. Comparison between standard and noise-aware optimization
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import sys
import os

# Add lib directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

# Import from available modules
from core import (
    LossLandscapeEstimator,
    AdaptiveLearningRateScheduler,
    NoiseAwareLearningRateOptimizer,
    create_optimizer
)
from utils import (
    compute_signal_to_noise_ratio,
    detect_local_minima,
    estimate_gradient_noise,
    smooth_trajectory,
    compute_optimization_metrics,
    analyze_stuck_periods,
    create_lr_schedule_summary,
    validate_optimizer_config
)


class NoisyLossFunction:
    """
    A synthetic loss function with noise to simulate real-world optimization.
    Contains multiple local minima and a global minimum.
    """
    
    def __init__(self, noise_level: float = 0.1, seed: int = 42):
        """
        Initialize the noisy loss function.
        
        Args:
            noise_level: Standard deviation of Gaussian noise
            seed: Random seed for reproducibility
        """
        self.noise_level = noise_level
        self.rng = np.random.RandomState(seed)
        
    def compute_loss(self, params: np.ndarray, add_noise: bool = True) -> float:
        """
        Compute loss with optional noise.
        Loss landscape: f(x) = x^2 + 0.5*sin(10*x) + noise
        This creates a quadratic bowl with many small local minima.
        
        Args:
            params: Current parameters
            add_noise: Whether to add noise to the loss
            
        Returns:
            Loss value
        """
        x = params[0]
        # Smooth loss: quadratic with sinusoidal perturbations
        smooth_loss = x**2 + 0.5 * np.sin(10 * x)
        
        if add_noise:
            noise = self.rng.normal(0, self.noise_level)
            return smooth_loss + noise
        return smooth_loss
    
    def compute_gradient(self, params: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        Compute gradient with optional noise.
        
        Args:
            params: Current parameters
            add_noise: Whether to add noise to the gradient
            
        Returns:
            Gradient vector
        """
        x = params[0]
        # Smooth gradient
        smooth_grad = 2 * x + 5 * np.cos(10 * x)
        
        if add_noise:
            noise = self.rng.normal(0, self.noise_level * 2)
            return np.array([smooth_grad + noise])
        return np.array([smooth_grad])


def run_standard_sgd(
    loss_fn: NoisyLossFunction,
    initial_params: np.ndarray,
    learning_rate: float,
    num_steps: int
) -> Tuple[List[float], List[float], List[np.ndarray]]:
    """
    Run standard SGD with fixed learning rate.
    
    Args:
        loss_fn: Loss function to optimize
        initial_params: Starting parameters
        learning_rate: Fixed learning rate
        num_steps: Number of optimization steps
        
    Returns:
        Tuple of (losses, smooth_losses, parameter_history)
    """
    params = initial_params.copy()
    losses = []
    smooth_losses = []
    param_history = [params.copy()]
    
    for step in range(num_steps):
        # Compute noisy loss and gradient
        loss = loss_fn.compute_loss(params, add_noise=True)
        smooth_loss = loss_fn.compute_loss(params, add_noise=False)
        grad = loss_fn.compute_gradient(params, add_noise=True)
        
        losses.append(loss)
        smooth_losses.append(smooth_loss)
        
        # Standard SGD update
        params = params - learning_rate * grad
        param_history.append(params.copy())
    
    return losses, smooth_losses, param_history


def run_noise_aware_optimization(
    loss_fn: NoisyLossFunction,
    initial_params: np.ndarray,
    base_learning_rate: float,
    num_steps: int,
    config: Dict = None
) -> Tuple[List[float], List[float], List[float], List[np.ndarray]]:
    """
    Run noise-aware optimization with adaptive learning rate.
    
    Args:
        loss_fn: Loss function to optimize
        initial_params: Starting parameters
        base_learning_rate: Base learning rate
        num_steps: Number of optimization steps
        config: Configuration for noise-aware optimizer
        
    Returns:
        Tuple of (losses, smooth_losses, learning_rates, parameter_history)
    """
    params = initial_params.copy()
    
    # Create noise-aware optimizer
    optimizer = create_optimizer(base_lr=base_learning_rate, config=config)
    
    # Create loss landscape estimator
    landscape_estimator = LossLandscapeEstimator(
        smoothing_factor=0.1,
        noise_window=20
    )
    
    losses = []
    smooth_losses = []
    learning_rates = []
    param_history = [params.copy()]
    gradients = []
    
    for step in range(num_steps):
        # Compute noisy loss and gradient
        loss = loss_fn.compute_loss(params, add_noise=True)
        smooth_loss = loss_fn.compute_loss(params, add_noise=False)
        grad = loss_fn.compute_gradient(params, add_noise=True)
        
        losses.append(loss)
        smooth_losses.append(smooth_loss)
        gradients.append(grad)
        
        # Update landscape estimator
        landscape_estimator.update(loss)
        
        # Get noise statistics
        noise_stats = landscape_estimator.get_noise_statistics()
        
        # Estimate gradient noise if we have enough history
        if len(gradients) >= 10:
            grad_noise = estimate_gradient_noise(gradients[-10:], window_size=10)
        else:
            grad_noise = 0.1
        
        # Perform noise-aware optimization step
        params, lr = optimizer.step(
            params=params,
            gradient=grad,
            loss=loss,
            noise_estimate=noise_stats.get('noise_std', 0.1)
        )
        
        learning_rates.append(lr)
        param_history.append(params.copy())
    
    return losses, smooth_losses, learning_rates, param_history


def visualize_comparison(
    standard_results: Tuple,
    noise_aware_results: Tuple,
    loss_fn: NoisyLossFunction,
    save_path: str = 'optimization_comparison.png'
):
    """
    Visualize comparison between standard and noise-aware optimization.
    
    Args:
        standard_results: Results from standard SGD
        noise_aware_results: Results from noise-aware optimization
        loss_fn: Loss function for landscape visualization
        save_path: Path to save the figure
    """
    std_losses, std_smooth, std_params = standard_results
    na_losses, na_smooth, na_lrs, na_params = noise_aware_results
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Loss trajectories
    ax1 = axes[0, 0]
    ax1.plot(std_losses, alpha=0.3, label='Standard SGD (noisy)', color='blue')
    ax1.plot(std_smooth, linewidth=2, label='Standard SGD (smooth)', color='blue')
    ax1.plot(na_losses, alpha=0.3, label='Noise-Aware (noisy)', color='red')
    ax1.plot(na_smooth, linewidth=2, label='Noise-Aware (smooth)', color='red')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Trajectories Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Smoothed losses with local minima detection
    ax2 = axes[0, 1]
    smoothed_std = smooth_trajectory(std_smooth, method='exponential', alpha=0.2)
    smoothed_na = smooth_trajectory(na_smooth, method='exponential', alpha=0.2)
    
    ax2.plot(smoothed_std, linewidth=2, label='Standard SGD', color='blue')
    ax2.plot(smoothed_na, linewidth=2, label='Noise-Aware', color='red')
    
    # Detect local minima
    std_minima = detect_local_minima(smoothed_std, window_size=10, threshold=0.01)
    na_minima = detect_local_minima(smoothed_na, window_size=10, threshold=0.01)
    
    ax2.scatter(std_minima, [smoothed_std[i] for i in std_minima],
                color='blue', marker='x', s=100, label='Standard local minima')
    ax2.scatter(na_minima, [smoothed_na[i] for i in na_minima],
                color='red', marker='x', s=100, label='Noise-Aware local minima')
    
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Smoothed Loss')
    ax2.set_title('Smoothed Loss with Local Minima Detection')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Parameter trajectories on loss landscape
    ax3 = axes[1, 0]
    x_range = np.linspace(-2, 2, 200)
    landscape = [loss_fn.compute_loss(np.array([x]), add_noise=False) for x in x_range]
    
    ax3.plot(x_range, landscape, 'k-', linewidth=2, label='True Loss Landscape')
    
    std_x = [p[0] for p in std_params]
    na_x = [p[0] for p in na_params]
    
    ax3.plot(std_x, std_smooth, 'b.-', alpha=0.6, label='Standard SGD', markersize=3)
    ax3.plot(na_x, na_smooth, 'r.-', alpha=0.6, label='Noise-Aware', markersize=3)
    
    ax3.scatter([std_x[0]], [std_smooth[0]], color='green', s=100, 
                marker='o', zorder=5, label='Start')
    ax3.scatter([std_x[-1]], [std_smooth[-1]], color='blue', s=100, 
                marker='*', zorder=5, label='Standard End')
    ax3.scatter([na_x[-1]], [na_smooth[-1]], color='red', s=100, 
                marker='*', zorder=5, label='Noise-Aware End')
    
    ax3.set_xlabel('Parameter Value')
    ax3.set_ylabel('Loss')
    ax3.set_title('Optimization Trajectory on Loss Landscape')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Adaptive learning rate
    ax4 = axes[1, 1]
    ax4.plot(na_lrs, linewidth=2, color='purple')
    ax4.axhline(y=0.001, color='blue', linestyle='--', label='Standard LR (fixed)')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Learning Rate')
    ax4.set_title('Adaptive Learning Rate Schedule')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")
    plt.close()


def print_optimization_analysis(
    standard_results: Tuple,
    noise_aware_results: Tuple,
    title: str = "Optimization Analysis"
):
    """
    Print detailed analysis of optimization results.
    
    Args:
        standard_results: Results from standard SGD
        noise_aware_results: Results from noise-aware optimization
        title: Title for the analysis
    """
    std_losses, std_smooth, std_params = standard_results
    na_losses, na_smooth, na_lrs, na_params = noise_aware_results
    
    print("\n" + "="*70)
    print(f"{title:^70}")
    print("="*70)
    
    # Standard SGD metrics
    print("\n📊 STANDARD SGD (Fixed Learning Rate)")
    print("-" * 70)
    std_metrics = compute_optimization_metrics(std_losses, [0.001] * len(std_losses), std_smooth)
    print(f"  Final Loss (noisy):        {std_losses[-1]:.6f}")
    print(f"  Final Loss (smooth):       {std_smooth[-1]:.6f}")
    print(f"  Loss Reduction:            {std_metrics['loss_reduction']:.6f}")
    print(f"  Convergence Rate:          {std_metrics['convergence_rate']:.6f}")
    print(f"  Average Loss Variance:     {std_metrics['avg_loss_variance']:.6f}")
    
    # Detect stuck periods
    std_stuck = analyze_stuck_periods(std_smooth, threshold=0.001, min_duration=5)
    print(f"  Stuck Periods Detected:    {len(std_stuck)}")
    if std_stuck:
        total_stuck = sum(end - start for start, end in std_stuck)
        print(f"  Total Stuck Iterations:    {total_stuck}")
    
    # Detect local minima
    std_minima = detect_local_minima(std_smooth, window_size=10, threshold=0.01)
    print(f"  Local Minima Encountered:  {len(std_minima)}")
    
    # Noise-aware optimization metrics
    print("\n🎯 NOISE-AWARE OPTIMIZATION (Adaptive Learning Rate)")
    print("-" * 70)
    na_metrics = compute_optimization_metrics(na_losses, na_lrs, na_smooth)
    print(f"  Final Loss (noisy):        {na_losses[-1]:.6f}")
    print(f"  Final Loss (smooth):       {na_smooth[-1]:.6f}")
    print(f"  Loss Reduction:            {na_metrics['loss_reduction']:.6f}")
    print(f"  Convergence Rate:          {na_metrics['convergence_rate']:.6f}")
    print(f"  Average Loss Variance:     {na_metrics['avg_loss_variance']:.6f}")
    
    # Detect stuck periods
    na_stuck = analyze_stuck_periods(na_smooth, threshold=0.001, min_duration=5)
    print(f"  Stuck Periods Detected:    {len(na_stuck)}")
    if na_stuck:
        total_stuck = sum(end - start for start, end in na_stuck)
        print(f"  Total Stuck Iterations:    {total_stuck}")
    
    # Detect local minima
    na_minima = detect_local_minima(na_smooth, window_size=10, threshold=0.01)
    print(f"  Local Minima Encountered:  {len(na_minima)}")
    
    # Learning rate schedule summary
    lr_summary = create_lr_schedule_summary(na_lrs, na_losses)
    print(f"\n  Learning Rate Statistics:")
    print(f"    Min LR:                  {lr_summary['min_lr']:.6e}")
    print(f"    Max LR:                  {lr_summary['max_lr']:.6e}")
    print(f"    Mean LR:                 {lr_summary['mean_lr']:.6e}")
    print(f"    LR Adaptations:          {lr_summary['num_changes']}")
    
    # Signal-to-noise ratio
    if len(std_smooth) > 0 and len(std_losses) > 0:
        std_noise = [abs(std_losses[i] - std_smooth[i]) for i in range(len(std_losses))]
        std_snr = compute_signal_to_noise_ratio(std_smooth, std_noise)
        print(f"\n  Standard SGD SNR:          {std_snr:.4f}")
    
    if len(na_smooth) > 0 and len(na_losses) > 0:
        na_noise = [abs(na_losses[i] - na_smooth[i]) for i in range(len(na_losses))]
        na_snr = compute_signal_to_noise_ratio(na_smooth, na_noise)
        print(f"  Noise-Aware SNR:           {na_snr:.4f}")
    
    # Comparison
    print("\n📈 IMPROVEMENT SUMMARY")
    print("-" * 70)
    improvement = ((std_smooth[-1] - na_smooth[-1]) / std_smooth[-1]) * 100
    print(f"  Final Loss Improvement:    {improvement:.2f}%")
    
    stuck_reduction = len(std_stuck) - len(na_stuck)
    print(f"  Stuck Period Reduction:    {stuck_reduction} periods")
    
    minima_reduction = len(std_minima) - len(na_minima)
    print(f"  Local Minima Reduction:    {minima_reduction} minima")
    
    print("="*70 + "\n")


def demonstrate_config_validation():
    """Demonstrate optimizer configuration validation."""
    print("\n" + "="*70)
    print("OPTIMIZER CONFIGURATION VALIDATION".center(70))
    print("="*70 + "\n")
    
    # Valid configuration
    valid_config = {
        'noise_threshold': 0.1,
        'lr_increase_factor': 1.5,
        'lr_decrease_factor': 0.8,
        'min_lr': 1e-6,
        'max_lr': 0.1
    }
    
    is_valid, errors = validate_optimizer_config(valid_config)
    print("✅ Valid Configuration:")
    print(f"   {valid_config}")
    print(f"   Valid: {is_valid}, Errors: {errors}\n")
    
    # Invalid configuration
    invalid_config = {
        'noise_threshold': -0.1,  # Negative threshold
        'lr_increase_factor': 0.5,  # Should be > 1
        'min_lr': 0.1,
        'max_lr': 0.01  # min_lr > max_lr
    }
    
    is_valid, errors = validate_optimizer_config(invalid_config)
    print("❌ Invalid Configuration:")
    print(f"   {invalid_config}")
    print(f"   Valid: {is_valid}")
    print(f"   Errors:")
    for error in errors:
        print(f"     - {error}")
    
    print("="*70 + "\n")


def main():
    """Main demonstration function."""
    print("\n" + "="*70)
    print("NOISE-AWARE LEARNING RATE ADAPTATION DEMO".center(70))
    print("="*70)
    print("\nThis demo shows how noise-aware optimization can help avoid")
    print("getting stuck in local minima caused by noisy gradients.\n")
    
    # Demonstrate configuration validation
    demonstrate_config_validation()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create noisy loss function
    print("🔧 Setting up noisy loss function...")
    loss_fn = NoisyLossFunction(noise_level=0.15, seed=42)
    
    # Initial parameters (start far from optimum)
    initial_params = np.array([1.5])
    
    # Optimization parameters
    base_lr = 0.001  # Very small learning rate
    num_steps = 300
    
    print(f"   Initial parameter: {initial_params[0]:.4f}")
    print(f"   Base learning rate: {base_lr}")
    print(f"   Number of steps: {num_steps}")
    print(f"   Noise level: 0.15\n")
    
    # Run standard SGD
    print("🏃 Running standard SGD with fixed learning rate...")
    std_results = run_standard_sgd(
        loss_fn=loss_fn,
        initial_params=initial_params,
        learning_rate=base_lr,
        num_steps=num_steps
    )
    print("   ✓ Standard SGD completed\n")
    
    # Configure noise-aware optimizer
    optimizer_config = {
        'noise_threshold': 0.1,
        'lr_increase_factor': 2.0,
        'lr_decrease_factor': 0.7,
        'min_lr': 1e-5,
        'max_lr': 0.05,
        'adaptation_rate': 0.1
    }
    
    # Run noise-aware optimization
    print("🎯 Running noise-aware optimization with adaptive learning rate...")
    na_results = run_noise_aware_optimization(
        loss_fn=loss_fn,
        initial_params=initial_params,
        base_learning_rate=base_lr,
        num_steps=num_steps,
        config=optimizer_config
    )
    print("   ✓ Noise-aware optimization completed\n")
    
    # Print detailed analysis
    print_optimization_analysis(std_results, na_results)
    
    # Visualize results
    print("📊 Generating visualization...")
    visualize_comparison(std_results, na_results, loss_fn)
    print("   ✓ Visualization complete\n")
    
    # Additional demonstration: Show landscape estimation
    print("🔍 LOSS LANDSCAPE ESTIMATION DEMO")
    print("-" * 70)
    
    estimator = LossLandscapeEstimator(smoothing_factor=0.1, noise_window=20)
    
    # Simulate some noisy losses
    test_losses = na_results[0][:50]  # First 50 losses
    
    for i, loss in enumerate(test_losses):
        estimator.update(loss)
        if i % 10 == 0 and i > 0:
            smooth_loss = estimator.get_smooth_loss()
            noise_stats = estimator.get_noise_statistics()
            print(f"  Step {i:3d}: Noisy={loss:.4f}, Smooth={smooth_loss:.4f}, "
                  f"Noise STD={noise_stats.get('noise_std', 0):.4f}")
    
    print("-" * 70 + "\n")
    
    print("✅ Demo completed successfully!")
    print("\nKey Takeaways:")
    print("  1. Small fixed learning rates can get stuck in noisy local minima")
    print("  2. Estimating the smooth loss landscape helps identify true progress")
    print("  3. Adapting learning rate based on noise helps escape local minima")
    print("  4. Noise-aware optimization typically achieves better final loss")
    print("  5. Adaptive learning rates reduce time stuck in local minima\n")


if __name__ == "__main__":
    main()
