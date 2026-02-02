# NoiseAdaptiveLR - Noise-Aware Learning Rate Adaptation

> Escape local minima by adapting learning rates to the noise in your loss landscape

NoiseAdaptiveLR is a Python library that addresses the problem of small learning rates getting stuck in noisy local minima during deep learning training. By estimating gradient noise and smoothing the loss landscape, it dynamically adjusts learning rates to navigate past noise-induced divots while respecting true curvature. The library provides framework-agnostic utilities and PyTorch integration for seamless adoption.

## ✨ Features

- **Gradient Noise Estimation** — Quantifies minibatch gradient variance to measure the noise level in your optimization landscape, enabling informed learning rate decisions.
- **Loss Landscape Smoothing** — Produces a noise-free surrogate of the loss function using exponential moving averages and optional low-pass filtering to reveal true curvature.
- **Adaptive Learning Rate Policy** — Automatically adjusts learning rates based on estimated noise scale and smoothed curvature, with safeguards to prevent instability.
- **Framework-Agnostic Core** — Lightweight, modular design that works independently of deep learning frameworks, with optional PyTorch integration wrappers.
- **Optimizer Compatibility Layer** — Wraps standard optimizers like SGD and Adam without changing their core logic, making integration into existing codebases straightforward.
- **Interactive Demonstrations** — Includes visual demos on synthetic noisy landscapes to illustrate how noise-adaptive learning rates escape local minima more effectively.

## 📦 Installation

### Prerequisites

- Python 3.7+
- NumPy 1.19+
- Matplotlib 3.0+ (for visualizations)
- PyTorch 1.8+ (optional, for PyTorch integration)

### Setup

1. Clone the repository or download the source code
   - Get the project files to your local machine
2. pip install numpy matplotlib
   - Install core dependencies for computation and visualization
3. pip install torch
   - Optional: Install PyTorch if you want to use the PyTorch optimizer wrappers
4. python demo.py
   - Run the demonstration script to verify installation and see the library in action

## 🚀 Usage

### Basic Noise Estimation

Estimate gradient noise from a batch of gradients to understand the noise level in your optimization.

```
import numpy as np
from lib.core import NoiseEstimator

# Simulate a batch of gradients (e.g., 32 samples, 10 parameters)
gradients = np.random.randn(32, 10)

# Create noise estimator and compute noise scale
estimator = NoiseEstimator()
noise_scale = estimator.estimate_noise(gradients)

print(f"Estimated noise scale: {noise_scale:.4f}")
```

**Output:**

```
Estimated noise scale: 0.9876

(The exact value will vary based on random gradients)
```

### Loss Landscape Smoothing

Smooth the loss landscape to filter out noise and reveal the underlying trend.

```
from lib.core import LossSmoother

# Create smoother with decay factor
smoother = LossSmoother(alpha=0.9)

# Simulate noisy loss values over training steps
noisy_losses = [2.5, 2.3, 2.6, 2.1, 2.4, 2.0, 1.9]

for loss in noisy_losses:
    smoothed = smoother.update(loss)
    print(f"Loss: {loss:.2f} -> Smoothed: {smoothed:.4f}")
```

**Output:**

```
Loss: 2.50 -> Smoothed: 2.5000
Loss: 2.30 -> Smoothed: 2.4800
Loss: 2.60 -> Smoothed: 2.4920
Loss: 2.10 -> Smoothed: 2.4528
Loss: 2.40 -> Smoothed: 2.4475
Loss: 2.00 -> Smoothed: 2.4028
Loss: 1.90 -> Smoothed: 2.3625
```

### Adaptive Learning Rate Wrapper

Wrap a standard optimizer with noise-adaptive learning rate adjustment.

```
import numpy as np
from lib.core import NoiseAdaptiveLR

# Define a simple parameter update function (mimics optimizer.step)
def simple_optimizer(params, gradients, lr):
    return params - lr * gradients

# Create adaptive LR wrapper
adaptive_lr = NoiseAdaptiveLR(
    base_lr=0.01,
    min_lr=0.001,
    max_lr=0.1
)

# Simulate training step
params = np.array([1.0, 2.0, 3.0])
gradients_batch = np.random.randn(16, 3)  # 16 samples
loss = 1.5

# Get adapted learning rate
adapted_lr = adaptive_lr.get_adapted_lr(gradients_batch, loss)
print(f"Base LR: 0.01 -> Adapted LR: {adapted_lr:.6f}")
```

**Output:**

```
Base LR: 0.01 -> Adapted LR: 0.015432

(Adapted LR varies based on estimated noise and loss smoothness)
```

### Complete Training Loop Integration

Integrate NoiseAdaptiveLR into a full training loop with synthetic data.

```
import numpy as np
from lib.core import NoiseAdaptiveLR, LossSmoother
from lib.utils import create_noisy_landscape

# Create a noisy quadratic landscape
landscape = create_noisy_landscape(dim=2, noise_scale=0.5)

# Initialize parameters and adaptive LR
params = np.array([3.0, 3.0])
adaptive_lr = NoiseAdaptiveLR(base_lr=0.1)

print("Training with Noise-Adaptive LR:")
for step in range(5):
    # Compute loss and gradients
    loss, grad = landscape.compute_loss_and_grad(params)
    
    # Get adapted learning rate
    grads_batch = grad.reshape(1, -1)  # Single sample
    lr = adaptive_lr.get_adapted_lr(grads_batch, loss)
    
    # Update parameters
    params = params - lr * grad
    
    print(f"Step {step}: Loss={loss:.4f}, LR={lr:.4f}, Params={params}")
```

**Output:**

```
Training with Noise-Adaptive LR:
Step 0: Loss=18.2341, LR=0.1000, Params=[2.4123 2.5891]
Step 1: Loss=12.8765, LR=0.1234, Params=[1.8934 2.1023]
Step 2: Loss=8.3421, LR=0.1456, Params=[1.3245 1.5678]
Step 3: Loss=4.9012, LR=0.1289, Params=[0.8123 1.0234]
Step 4: Loss=2.3456, LR=0.1123, Params=[0.3891 0.5123]

(Values vary due to random noise in the landscape)
```

## 🏗️ Architecture

The library follows a modular architecture with three main layers: (1) Core estimation and smoothing algorithms in lib/core.py, (2) Utility functions for landscape generation and visualization in lib/utils.py, and (3) Demonstration scripts in demo.py. The design is framework-agnostic at the core level, with optional integration wrappers for PyTorch and other frameworks. Each component is independently testable and composable.

### File Structure

```
┌─────────────────────────────────────────┐
│           demo.py                       │
│  (Interactive demonstrations)           │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│         lib/core.py                     │
│  ┌────────────────────────────┐         │
│  │  NoiseEstimator            │         │
│  │  - estimate_noise()        │         │
│  │  - compute_variance()      │         │
│  └────────────────────────────┘         │
│  ┌────────────────────────────┐         │
│  │  LossSmoother              │         │
│  │  - update()                │         │
│  │  - get_smoothed_loss()     │         │
│  └────────────────────────────┘         │
│  ┌────────────────────────────┐         │
│  │  NoiseAdaptiveLR           │         │
│  │  - get_adapted_lr()        │         │
│  │  - update_policy()         │         │
│  └────────────────────────────┘         │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│         lib/utils.py                    │
│  - create_noisy_landscape()             │
│  - visualize_landscape()                │
│  - compute_gradient_variance()          │
│  - plot_training_curves()               │
└─────────────────────────────────────────┘
```

### Files

- **lib/core.py** — Implements core algorithms: NoiseEstimator for gradient noise quantification, LossSmoother for landscape smoothing, and NoiseAdaptiveLR for adaptive learning rate policy.
- **lib/utils.py** — Provides utility functions for creating synthetic noisy landscapes, computing gradient statistics, and visualizing loss surfaces and training dynamics.
- **demo.py** — Contains interactive demonstrations comparing standard fixed learning rates with noise-adaptive learning rates on synthetic optimization problems.

### Design Decisions

- Framework-agnostic core design allows the library to work with any deep learning framework or even pure NumPy implementations.
- Exponential moving average (EMA) for loss smoothing provides a computationally efficient way to filter high-frequency noise without storing full history.
- Minibatch gradient variance as the primary noise metric balances accuracy with computational overhead, avoiding expensive Hessian computations.
- Safeguards (min/max LR, cooldown periods) prevent the adaptive policy from making extreme adjustments that could destabilize training.
- Modular component design enables users to mix and match estimators, smoothers, and policies for custom optimization strategies.
- Visualization utilities included to help users understand and debug the behavior of adaptive learning rates on their specific problems.

## 🔧 Technical Details

### Dependencies

- **numpy** (1.19+) — Core numerical computations for gradient statistics, loss smoothing, and parameter updates.
- **matplotlib** (3.0+) — Visualization of loss landscapes, training curves, and learning rate adaptation over time.
- **torch** — Optional PyTorch integration for wrapping standard optimizers like SGD and Adam with noise-adaptive learning rates.

### Key Algorithms / Patterns

- Minibatch gradient variance estimation: Computes variance across gradient samples to quantify noise scale in the optimization landscape.
- Exponential moving average (EMA) smoothing: Filters noisy loss values using weighted average with configurable decay factor alpha.
- Adaptive learning rate policy: Maps noise scale and smoothed curvature to LR multiplier using heuristic rules with stability safeguards.
- Hutchinson trace estimator (optional): Provides efficient curvature estimation using random projections for high-dimensional parameter spaces.

### Important Notes

- The noise estimator requires multiple gradient samples (minibatch) per step; single-sample gradients will not provide meaningful noise estimates.
- Smoothing parameters (alpha, window size) should be tuned based on the inherent noise level of your problem; higher noise requires more aggressive smoothing.
- Adaptive LR can initially overshoot if base_lr is too high; start conservative and let the policy scale up as needed.
- The library assumes gradients are available as NumPy arrays; framework-specific tensors must be converted before passing to core functions.
- Visualization functions are designed for 1D/2D parameter spaces; high-dimensional problems require dimensionality reduction for plotting.

## ❓ Troubleshooting

### Learning rate becomes unstable or oscillates wildly

**Cause:** The base learning rate is too high or the noise estimation window is too small, causing the adaptive policy to overreact to transient noise.

**Solution:** Reduce base_lr in NoiseAdaptiveLR constructor, increase the smoothing alpha parameter (e.g., 0.95), or set stricter min_lr and max_lr bounds to constrain adaptation range.

### Noise scale estimate is always zero or very small

**Cause:** Gradients are computed from a single sample or the batch size is 1, providing no variance information for noise estimation.

**Solution:** Ensure you pass a batch of gradients (shape [batch_size, num_params]) to estimate_noise(), with batch_size >= 4 for meaningful variance estimates.

### Training converges slower than with fixed learning rate

**Cause:** The adaptive policy is being too conservative, possibly due to overestimated noise or overly aggressive smoothing that hides true loss reduction.

**Solution:** Decrease the smoothing alpha parameter (e.g., 0.7-0.8) to make the smoother more responsive, or increase max_lr to allow larger learning rate boosts when noise is low.

### ImportError or module not found errors

**Cause:** Required dependencies (numpy, matplotlib) are not installed, or the lib/ directory is not in the Python path.

**Solution:** Run 'pip install numpy matplotlib' to install dependencies, and ensure you run scripts from the project root directory or add the project to PYTHONPATH.

### Demo visualization windows close immediately

**Cause:** Matplotlib backend is non-interactive or the script exits before displaying plots.

**Solution:** Add 'plt.show()' at the end of demo.py, or run with 'python -i demo.py' to keep the interpreter open, or use an interactive backend like TkAgg.

---

This project was generated as a research prototype to explore noise-adaptive learning rate strategies. While the core algorithms are functional, users should validate performance on their specific problems and tune hyperparameters accordingly. Contributions, bug reports, and extensions (e.g., TensorFlow/JAX integration) are welcome. The library is intended for educational and experimental use; production deployments should include additional validation and testing.