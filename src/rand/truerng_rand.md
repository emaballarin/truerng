# TrueRNG Random Number Generator

Hardware random number generation using TrueRNG Pro devices, with NumPy, PyTorch, and JAX interfaces.

## Overview

`truerng_rand.py` provides a Python interface to the TrueRNG Pro hardware random number generator. It reads whitened random bytes from the device (MODE_NORMAL) and converts them to random floats, integers, and samples compatible with popular numerical computing frameworks.

The module offers three APIs:
- **NumPy API**: `TrueRNG` class and module-level functions
- **PyTorch API**: `TrueRNGTorch` class (optional, requires torch)
- **JAX API**: `TrueRNGJax` class (optional, requires jax)

## Requirements

### Hardware
- TrueRNG Pro v1 or compatible device (USB VID:PID `16D0:0AA0`)

### Python
- Python 3.10+

### Dependencies
```
pyserial
numpy
```

### Optional Dependencies
```
torch    # For PyTorch API
jax      # For JAX API
```

## Installation

```bash
pip install pyserial numpy

# Optional: for PyTorch support
pip install torch

# Optional: for JAX support
pip install jax jaxlib
```

## Quick Start

### NumPy
```python
from truerng_rand import TrueRNG, rand

# Using context manager (recommended)
with TrueRNG() as rng:
    x = rng.rand(10)        # Array of 10 random floats in [0, 1)
    y = rng.randint(1, 100, size=5)  # 5 random integers in [1, 100)

# Module-level (auto-manages connection)
x = rand(3, 4)              # 3x4 array of random floats
```

### PyTorch
```python
from truerng_rand import TrueRNGTorch

with TrueRNGTorch() as rng:
    x = rng.rand(3, 4)      # torch.Tensor, shape (3, 4)
    y = rng.randn(10)       # Normal distribution N(0, 1)
```

### JAX
```python
from truerng_rand import TrueRNGJax

with TrueRNGJax() as rng:
    x = rng.uniform(shape=(3, 4))   # jax.Array in [0, 1)
    y = rng.normal(shape=(10,))     # Standard normal
```

## API Reference

### NumPy API

#### Class: `TrueRNG`

```python
TrueRNG(port: str | None = None, buffer_size: int = 8192)
```

**Parameters:**
- `port`: Serial port path (e.g., `/dev/ttyACM0`). Auto-detected if `None`.
- `buffer_size`: Internal buffer size for efficient reads.

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `rand` | `(*shape: int) -> np.float64 \| np.ndarray` | Uniform floats in [0, 1) |
| `randint` | `(low, high=None, size=None) -> np.int64 \| np.ndarray` | Random integers in [low, high) |
| `choice` | `(a, size=None, replace=True) -> np.ndarray` | Random sample from array or range |
| `connect` | `() -> None` | Establish device connection |
| `disconnect` | `() -> None` | Close device connection |

**Context Manager:**
```python
with TrueRNG() as rng:
    # rng is connected
    x = rng.rand(10)
# rng is automatically disconnected
```

#### Module Functions

```python
rand(*shape: int) -> np.float64 | np.ndarray
randint(low: int, high: int | None = None, size = None) -> np.int64 | np.ndarray
choice(a, size=None, replace=True) -> np.ndarray
```

These functions use a global singleton that auto-connects on first use.

---

### PyTorch API

#### Class: `TrueRNGTorch`

```python
TrueRNGTorch(port: str | None = None, device: str | torch.device = "cpu")
```

**Parameters:**
- `port`: Serial port path. Auto-detected if `None`.
- `device`: PyTorch device for output tensors.

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `rand` | `(*size, dtype=torch.float64) -> Tensor` | Uniform in [0, 1) |
| `randn` | `(*size, dtype=torch.float64) -> Tensor` | Standard normal N(0, 1) |
| `randint` | `(low, high=None, size=(1,), dtype=torch.int64) -> Tensor` | Random integers |
| `uniform` | `(low=0.0, high=1.0, size=(1,), dtype=torch.float64) -> Tensor` | Uniform in [low, high) |
| `normal` | `(mean=0.0, std=1.0, size=(1,), dtype=torch.float64) -> Tensor` | Normal N(mean, std²) |

#### Module Functions

```python
torch_rand(*size, dtype=torch.float64, device="cpu") -> torch.Tensor
torch_randn(*size, dtype=torch.float64, device="cpu") -> torch.Tensor
```

---

### JAX API

#### Class: `TrueRNGJax`

```python
TrueRNGJax(port: str | None = None)
```

**Note:** Unlike JAX's functional random API, this class is stateful since it reads from hardware.

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `uniform` | `(shape=(), dtype=jnp.float64, minval=0.0, maxval=1.0) -> Array` | Uniform in [minval, maxval) |
| `normal` | `(shape=(), dtype=jnp.float64) -> Array` | Standard normal N(0, 1) |
| `randint` | `(minval, maxval, shape=(), dtype=jnp.int64) -> Array` | Random integers |
| `choice` | `(a, shape=(), replace=True) -> Array` | Random sample |

#### Module Functions

```python
jax_uniform(shape=(), dtype=jnp.float64, minval=0.0, maxval=1.0) -> jax.Array
jax_normal(shape=(), dtype=jnp.float64) -> jax.Array
```

---

## Examples

### NumPy: Monte Carlo Pi Estimation

```python
from truerng_rand import TrueRNG
import numpy as np

with TrueRNG() as rng:
    n = 1_000_000
    x = rng.rand(n)
    y = rng.rand(n)
    inside = np.sum(x**2 + y**2 < 1)
    pi_estimate = 4 * inside / n
    print(f"Pi ≈ {pi_estimate:.6f}")
```

### NumPy: Secure Password Generation

```python
from truerng_rand import TrueRNG
import string

with TrueRNG() as rng:
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
    password = "".join(rng.choice(list(alphabet), size=16))
    print(f"Password: {password}")
```

### PyTorch: Random Weight Initialization

```python
from truerng_rand import TrueRNGTorch
import torch.nn as nn

with TrueRNGTorch(device="cuda") as rng:
    model = nn.Linear(128, 64)
    with torch.no_grad():
        model.weight.copy_(rng.randn(64, 128) * 0.01)
        model.bias.copy_(rng.rand(64) * 0.01)
```

### JAX: Stochastic Gradient Descent with Hardware Noise

```python
from truerng_rand import TrueRNGJax
import jax.numpy as jnp

with TrueRNGJax() as rng:
    # Add hardware random noise to gradients
    gradients = jnp.ones((100,))
    noise = rng.normal(shape=(100,)) * 0.01
    noisy_gradients = gradients + noise
```

### Command Line

```bash
# Generate 10 random floats (default)
python truerng_rand.py

# Generate 100 random floats
python truerng_rand.py 100

# Generate 3x4 array
python truerng_rand.py 3 4
```

## Notes

### Device Modes

The module uses `MODE_NORMAL`, which provides whitened random output suitable for cryptographic and statistical applications. The raw ADC modes (`MODE_RAW_BIN`, `MODE_RAW_ASC`) are not used by this module.

### Normal Distribution

The `randn()` and `normal()` methods use the Box-Muller transform to convert uniform samples to normal distribution. Extreme tail values beyond ~37σ are clipped due to floating-point precision limits.

### Performance

- Hardware RNG throughput is limited by USB serial speed (~1.2 MB/s in NORMAL mode)
- The module uses internal buffering (default 8KB) to reduce serial read overhead
- For high-throughput applications, consider collecting a batch of random data upfront

### Thread Safety

The module-level singleton functions (`rand()`, `torch_rand()`, etc.) are **not** thread-safe. For multi-threaded use, create separate `TrueRNG` instances per thread.

### Differences from Standard APIs

| Feature | This Module | NumPy/PyTorch/JAX |
|---------|-------------|-------------------|
| Randomness source | Hardware (TrueRNG) | Software (PRNG) |
| Reproducibility | Not reproducible | Seedable |
| Speed | Limited by USB | Very fast |
| Cryptographic quality | Yes (whitened) | No (unless using secrets) |
