#!/usr/bin/python3
"""Generate random floats in [0, 1] from TrueRNG in NORMAL mode.

This module provides numpy, PyTorch, and JAX interfaces for generating hardware
random numbers using the TrueRNG device's whitened output (MODE_NORMAL).

Usage as script:
    python truerng_rand.py              # Generate 10 random floats
    python truerng_rand.py 100          # Generate 100 random floats
    python truerng_rand.py 5 3          # Generate 5x3 array

NumPy usage:
    from truerng_rand import TrueRNG, rand

    with TrueRNG() as rng:
        x = rng.rand()        # Single float
        y = rng.rand(10)      # Array of 10 floats
        z = rng.rand(3, 4)    # 3x4 array

    # Module-level (auto-manages connection)
    x = rand(10)

PyTorch usage:
    from truerng_rand import TrueRNGTorch

    with TrueRNGTorch() as rng:
        x = rng.rand(3, 4)              # torch.Tensor [0, 1)
        y = rng.randn(10)               # Normal distribution
        z = rng.randint(0, 100, (5,))   # Integer tensor

JAX usage:
    from truerng_rand import TrueRNGJax

    with TrueRNGJax() as rng:
        x = rng.uniform(shape=(3, 4))   # jax.Array [0, 1)
        y = rng.normal(shape=(10,))     # Normal distribution
"""

import atexit
import sys
from typing import Self

import numpy as np
import serial

from truerng_utils import get_first_truerng, mode_change, reset_serial_port

__all__ = [
    # NumPy API
    "TrueRNG",
    "rand",
    "randint",
    "choice",
    # PyTorch API (None if torch unavailable)
    "TrueRNGTorch",
    "torch_rand",
    "torch_randn",
    # JAX API (None if jax unavailable)
    "TrueRNGJax",
    "jax_uniform",
    "jax_normal",
]


class TrueRNG:
    """Hardware random number generator using TrueRNG device.

    Provides numpy-like interface for generating random floats in [0, 1].
    Uses NORMAL mode (whitened output) for cryptographic quality randomness.
    """

    # Bytes per float (8 bytes = 64 bits of precision)
    BYTES_PER_FLOAT = 8
    MAX_UINT64 = 2**64 - 1

    def __init__(self, port: str | None = None, buffer_size: int = 8192):
        """Initialize TrueRNG connection.

        Args:
            port: Serial port path. Auto-detected if None.
            buffer_size: Size of internal byte buffer for efficiency.
        """
        self._port = port
        self._buffer_size = buffer_size
        self._ser: serial.Serial | None = None
        self._buffer = b""
        self._connected = False

    def connect(self) -> None:
        """Establish connection to the TrueRNG device."""
        if self._connected:
            return

        # Find device if port not specified
        if self._port is None:
            device = get_first_truerng()
            if device is None:
                raise RuntimeError("No TrueRNG device found")
            self._port, _ = device

        # Switch to NORMAL mode
        if not mode_change("MODE_NORMAL", self._port):
            raise RuntimeError(f"Failed to switch to MODE_NORMAL on {self._port}")

        # Open serial connection
        try:
            self._ser = serial.Serial(port=self._port, timeout=10)
            if not self._ser.isOpen():
                self._ser.open()
            self._ser.setDTR(True)
            self._ser.flushInput()
            self._connected = True
        except serial.SerialException as e:
            raise RuntimeError(f"Failed to open {self._port}: {e}") from e

    def disconnect(self) -> None:
        """Close connection to the TrueRNG device."""
        if self._ser is not None:
            self._ser.close()
            self._ser = None
        if self._port is not None:
            reset_serial_port(self._port)
        self._connected = False
        self._buffer = b""

    def __enter__(self) -> Self:
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.disconnect()

    def _read_bytes(self, n: int) -> bytes:
        """Read n bytes from the device, using internal buffer.

        Args:
            n: Number of bytes to read.

        Returns:
            Exactly n random bytes.
        """
        if not self._connected:
            self.connect()

        # Refill buffer if needed
        while len(self._buffer) < n:
            try:
                chunk = self._ser.read(self._buffer_size)
                if not chunk:
                    raise RuntimeError("Timeout reading from TrueRNG")
                self._buffer += chunk
            except serial.SerialException as e:
                raise RuntimeError(f"Read error: {e}") from e

        # Extract requested bytes
        result = self._buffer[:n]
        self._buffer = self._buffer[n:]
        return result

    def rand(self, *shape: int) -> np.float64 | np.ndarray:
        """Generate random floats uniformly distributed in [0, 1).

        Args:
            *shape: Output shape. If empty, returns a single float.

        Returns:
            Random float or array of floats in [0, 1).

        Examples:
            rng.rand()       # Single float
            rng.rand(5)      # Array of 5 floats
            rng.rand(3, 4)   # 3x4 array
        """
        if not shape:
            # Single float
            raw = self._read_bytes(self.BYTES_PER_FLOAT)
            value = int.from_bytes(raw, byteorder="little")
            return np.float64(value / (self.MAX_UINT64 + 1))

        # Array of floats
        total = int(np.prod(shape))
        raw = self._read_bytes(total * self.BYTES_PER_FLOAT)

        # Convert bytes to uint64 array, then to float64
        values = np.frombuffer(raw, dtype=np.uint64)
        floats = values.astype(np.float64) / (self.MAX_UINT64 + 1)

        return floats.reshape(shape)

    def randint(
        self, low: int, high: int | None = None, size: int | tuple[int, ...] | None = None
    ) -> np.int64 | np.ndarray:
        """Generate random integers.

        Args:
            low: Lowest integer (inclusive), or upper bound if high is None.
            high: Upper bound (exclusive). If None, range is [0, low).
            size: Output shape.

        Returns:
            Random integer(s) in [low, high).
        """
        if high is None:
            low, high = 0, low

        if size is None:
            shape = ()
        elif isinstance(size, int):
            shape = (size,)
        else:
            shape = size

        # Generate floats and scale to integer range
        floats = self.rand(*shape) if shape else self.rand()
        return np.floor(floats * (high - low) + low).astype(np.int64)

    def choice(
        self, a: int | np.ndarray, size: int | tuple[int, ...] | None = None, replace: bool = True
    ) -> np.ndarray:
        """Random sample from a given array or range.

        Args:
            a: If int, sample from range(a). If array, sample from it.
            size: Output shape.
            replace: Whether to sample with replacement.

        Returns:
            Random sample(s).
        """
        if isinstance(a, int):
            arr = np.arange(a)
        else:
            arr = np.asarray(a)

        if size is None:
            n = 1
            squeeze = True
        elif isinstance(size, int):
            n = size
            squeeze = False
        else:
            n = int(np.prod(size))
            squeeze = False

        if replace:
            indices = self.randint(len(arr), size=n)
        else:
            if n > len(arr):
                raise ValueError("Cannot sample more elements than available without replacement")
            # Fisher-Yates shuffle using hardware randomness
            pool = list(range(len(arr)))
            indices = []
            for _ in range(n):
                idx = int(self.randint(len(pool)))
                indices.append(pool.pop(idx))
            indices = np.array(indices)

        result = arr[indices]

        if squeeze:
            return result[0]
        if size is not None and not isinstance(size, int):
            return result.reshape(size)
        return result


# Module-level singleton for convenience functions
_global_rng: TrueRNG | None = None


def _get_rng() -> TrueRNG:
    """Get or create the global TrueRNG instance."""
    global _global_rng
    if _global_rng is None:
        _global_rng = TrueRNG()
        _global_rng.connect()
        atexit.register(_global_rng.disconnect)
    return _global_rng


def rand(*shape: int) -> np.float64 | np.ndarray:
    """Generate random floats in [0, 1) using hardware RNG.

    Module-level convenience function. Automatically manages connection.

    Args:
        *shape: Output shape. If empty, returns a single float.

    Returns:
        Random float or array of floats in [0, 1).
    """
    return _get_rng().rand(*shape)


def randint(low: int, high: int | None = None, size: int | tuple[int, ...] | None = None) -> np.int64 | np.ndarray:
    """Generate random integers using hardware RNG.

    Args:
        low: Lowest integer (inclusive), or upper bound if high is None.
        high: Upper bound (exclusive).
        size: Output shape.

    Returns:
        Random integer(s).
    """
    return _get_rng().randint(low, high, size)


def choice(a: int | np.ndarray, size: int | tuple[int, ...] | None = None, replace: bool = True) -> np.ndarray:
    """Random sample using hardware RNG.

    Args:
        a: If int, sample from range(a). If array, sample from it.
        size: Output shape.
        replace: Whether to sample with replacement.

    Returns:
        Random sample(s).
    """
    return _get_rng().choice(a, size, replace)


# =============================================================================
# PyTorch API
# =============================================================================

try:
    import torch

    class TrueRNGTorch:
        """Hardware random number generator with PyTorch interface.

        Provides torch-like API for generating random tensors using TrueRNG.
        """

        def __init__(self, port: str | None = None, device: str | torch.device = "cpu"):
            """Initialize TrueRNG with PyTorch interface.

            Args:
                port: Serial port path. Auto-detected if None.
                device: PyTorch device for output tensors.
            """
            self._backend = TrueRNG(port=port)
            self._device = torch.device(device)

        def connect(self) -> None:
            """Establish connection to the TrueRNG device."""
            self._backend.connect()

        def disconnect(self) -> None:
            """Close connection to the TrueRNG device."""
            self._backend.disconnect()

        def __enter__(self) -> Self:
            """Context manager entry."""
            self.connect()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb) -> None:
            """Context manager exit."""
            self.disconnect()

        def rand(self, *size: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
            """Generate random floats uniformly distributed in [0, 1).

            Args:
                *size: Output shape.
                dtype: Output dtype (float32 or float64).

            Returns:
                Random tensor in [0, 1).
            """
            squeeze_output = not size
            if not size:
                size = (1,)
            arr = self._backend.rand(*size)
            tensor = torch.from_numpy(arr).to(dtype=dtype, device=self._device)
            return tensor.squeeze() if squeeze_output else tensor

        def randn(self, *size: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
            """Generate random floats from standard normal distribution.

            Uses Box-Muller transform on hardware uniform samples.

            Args:
                *size: Output shape.
                dtype: Output dtype (float32 or float64).

            Returns:
                Random tensor from N(0, 1).
            """
            squeeze_output = not size
            if not size:
                size = (1,)

            # Need pairs for Box-Muller
            total = int(np.prod(size))
            n_pairs = (total + 1) // 2

            u1 = self._backend.rand(n_pairs)
            u2 = self._backend.rand(n_pairs)

            # Box-Muller transform
            u1_safe = np.clip(u1, np.finfo(np.float64).tiny, 1.0)
            r = np.sqrt(-2.0 * np.log(u1_safe))
            theta = 2.0 * np.pi * u2
            z0 = r * np.cos(theta)
            z1 = r * np.sin(theta)

            # Interleave and trim
            normal = np.empty(2 * n_pairs)
            normal[0::2] = z0
            normal[1::2] = z1
            normal = normal[:total].reshape(size)

            tensor = torch.from_numpy(normal).to(dtype=dtype, device=self._device)
            return tensor.squeeze() if squeeze_output else tensor

        def randint(
            self,
            low: int,
            high: int | None = None,
            size: tuple[int, ...] = (1,),
            dtype: torch.dtype = torch.int64,
        ) -> torch.Tensor:
            """Generate random integers.

            Args:
                low: Lowest integer (inclusive), or upper bound if high is None.
                high: Upper bound (exclusive).
                size: Output shape.
                dtype: Output dtype.

            Returns:
                Random integer tensor.
            """
            if high is None:
                low, high = 0, low

            arr = self._backend.randint(low, high, size=size)
            return torch.from_numpy(arr).to(dtype=dtype, device=self._device)

        def uniform(
            self,
            low: float = 0.0,
            high: float = 1.0,
            size: tuple[int, ...] = (1,),
            dtype: torch.dtype = torch.float64,
        ) -> torch.Tensor:
            """Generate random floats uniformly distributed in [low, high).

            Args:
                low: Lower bound (inclusive).
                high: Upper bound (exclusive).
                size: Output shape.
                dtype: Output dtype.

            Returns:
                Random tensor in [low, high).
            """
            arr = self._backend.rand(*size)
            arr = arr * (high - low) + low
            return torch.from_numpy(arr).to(dtype=dtype, device=self._device)

        def normal(
            self,
            mean: float = 0.0,
            std: float = 1.0,
            size: tuple[int, ...] = (1,),
            dtype: torch.dtype = torch.float64,
        ) -> torch.Tensor:
            """Generate random floats from normal distribution.

            Args:
                mean: Mean of the distribution.
                std: Standard deviation.
                size: Output shape.
                dtype: Output dtype.

            Returns:
                Random tensor from N(mean, std^2).
            """
            tensor = self.randn(*size, dtype=dtype)
            return tensor * std + mean

    # Module-level PyTorch singleton
    _global_torch_rng: TrueRNGTorch | None = None

    def _get_torch_rng() -> TrueRNGTorch:
        """Get or create the global TrueRNGTorch instance."""
        global _global_torch_rng
        if _global_torch_rng is None:
            _global_torch_rng = TrueRNGTorch()
            _global_torch_rng.connect()
            atexit.register(_global_torch_rng.disconnect)
        return _global_torch_rng

    def torch_rand(*size: int, dtype: torch.dtype = torch.float64, device: str = "cpu") -> torch.Tensor:
        """Generate random tensor in [0, 1) using hardware RNG."""
        rng = _get_torch_rng()
        tensor = rng.rand(*size, dtype=dtype)
        return tensor.to(device) if device != "cpu" else tensor

    def torch_randn(*size: int, dtype: torch.dtype = torch.float64, device: str = "cpu") -> torch.Tensor:
        """Generate random tensor from N(0, 1) using hardware RNG."""
        rng = _get_torch_rng()
        tensor = rng.randn(*size, dtype=dtype)
        return tensor.to(device) if device != "cpu" else tensor

except ImportError:
    # PyTorch not available
    TrueRNGTorch = None  # type: ignore
    torch_rand = None  # type: ignore
    torch_randn = None  # type: ignore


# =============================================================================
# JAX API
# =============================================================================

try:
    import jax
    import jax.numpy as jnp

    class TrueRNGJax:
        """Hardware random number generator with JAX interface.

        Provides jax.random-like API for generating random arrays using TrueRNG.
        Note: This is stateful (unlike JAX's functional approach) since it
        reads from hardware.
        """

        def __init__(self, port: str | None = None):
            """Initialize TrueRNG with JAX interface.

            Args:
                port: Serial port path. Auto-detected if None.
            """
            self._backend = TrueRNG(port=port)

        def connect(self) -> None:
            """Establish connection to the TrueRNG device."""
            self._backend.connect()

        def disconnect(self) -> None:
            """Close connection to the TrueRNG device."""
            self._backend.disconnect()

        def __enter__(self) -> Self:
            """Context manager entry."""
            self.connect()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb) -> None:
            """Context manager exit."""
            self.disconnect()

        def uniform(
            self,
            shape: tuple[int, ...] = (),
            dtype: jnp.dtype = jnp.float64,
            minval: float = 0.0,
            maxval: float = 1.0,
        ) -> jax.Array:
            """Generate random floats uniformly distributed in [minval, maxval).

            Args:
                shape: Output shape.
                dtype: Output dtype.
                minval: Lower bound (inclusive).
                maxval: Upper bound (exclusive).

            Returns:
                Random array in [minval, maxval).
            """
            squeeze_output = not shape
            if not shape:
                shape = (1,)

            arr = self._backend.rand(*shape)
            arr = arr * (maxval - minval) + minval
            result = jnp.asarray(arr, dtype=dtype)
            return result.squeeze() if squeeze_output else result

        def normal(
            self,
            shape: tuple[int, ...] = (),
            dtype: jnp.dtype = jnp.float64,
        ) -> jax.Array:
            """Generate random floats from standard normal distribution.

            Uses Box-Muller transform on hardware uniform samples.

            Args:
                shape: Output shape.
                dtype: Output dtype.

            Returns:
                Random array from N(0, 1).
            """
            squeeze_output = not shape
            if not shape:
                shape = (1,)

            # Need pairs for Box-Muller
            total = int(np.prod(shape))
            n_pairs = (total + 1) // 2

            u1 = self._backend.rand(n_pairs)
            u2 = self._backend.rand(n_pairs)

            # Box-Muller transform
            u1_safe = np.clip(u1, np.finfo(np.float64).tiny, 1.0)
            r = np.sqrt(-2.0 * np.log(u1_safe))
            theta = 2.0 * np.pi * u2
            z0 = r * np.cos(theta)
            z1 = r * np.sin(theta)

            # Interleave and trim
            normal = np.empty(2 * n_pairs)
            normal[0::2] = z0
            normal[1::2] = z1
            normal = normal[:total].reshape(shape)

            result = jnp.asarray(normal, dtype=dtype)
            return result.squeeze() if squeeze_output else result

        def randint(
            self,
            minval: int,
            maxval: int,
            shape: tuple[int, ...] = (),
            dtype: jnp.dtype = jnp.int64,
        ) -> jax.Array:
            """Generate random integers in [minval, maxval).

            Args:
                minval: Lower bound (inclusive).
                maxval: Upper bound (exclusive).
                shape: Output shape.
                dtype: Output dtype.

            Returns:
                Random integer array.
            """
            if not shape:
                arr = self._backend.randint(minval, maxval)
                return jnp.asarray(arr, dtype=dtype)

            arr = self._backend.randint(minval, maxval, size=shape)
            return jnp.asarray(arr, dtype=dtype)

        def choice(
            self,
            a: int | jax.Array,
            shape: tuple[int, ...] = (),
            replace: bool = True,
        ) -> jax.Array:
            """Random sample from array or range.

            Args:
                a: If int, sample from range(a). If array, sample from it.
                shape: Output shape.
                replace: Whether to sample with replacement.

            Returns:
                Random sample(s).
            """
            if isinstance(a, int):
                arr_np = np.arange(a)
            else:
                arr_np = np.asarray(a)

            size = shape if shape else None
            result = self._backend.choice(arr_np, size=size, replace=replace)
            return jnp.asarray(result)

    # Module-level JAX singleton
    _global_jax_rng: TrueRNGJax | None = None

    def _get_jax_rng() -> TrueRNGJax:
        """Get or create the global TrueRNGJax instance."""
        global _global_jax_rng
        if _global_jax_rng is None:
            _global_jax_rng = TrueRNGJax()
            _global_jax_rng.connect()
            atexit.register(_global_jax_rng.disconnect)
        return _global_jax_rng

    def jax_uniform(
        shape: tuple[int, ...] = (),
        dtype: jnp.dtype = jnp.float64,
        minval: float = 0.0,
        maxval: float = 1.0,
    ) -> jax.Array:
        """Generate random array in [minval, maxval) using hardware RNG."""
        return _get_jax_rng().uniform(shape, dtype, minval, maxval)

    def jax_normal(shape: tuple[int, ...] = (), dtype: jnp.dtype = jnp.float64) -> jax.Array:
        """Generate random array from N(0, 1) using hardware RNG."""
        return _get_jax_rng().normal(shape, dtype)

except ImportError:
    # JAX not available
    TrueRNGJax = None  # type: ignore
    jax_uniform = None  # type: ignore
    jax_normal = None  # type: ignore


def main() -> int:
    """Command-line interface for generating random floats."""
    # Parse shape from command line
    if len(sys.argv) == 1:
        shape = (10,)
    else:
        try:
            shape = tuple(int(x) for x in sys.argv[1:])
        except ValueError:
            print("Usage: truerng_rand.py [dim1] [dim2] ...")
            return 1

    print(f"Generating random floats with shape {shape}...")

    try:
        with TrueRNG() as rng:
            result = rng.rand(*shape)
            print(result)
    except RuntimeError as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
