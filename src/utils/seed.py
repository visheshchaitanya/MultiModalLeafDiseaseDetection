"""
Reproducibility utilities for setting random seeds across different libraries.
Ensures deterministic behavior for experiments.
"""

import random
import numpy as np
import torch
import os
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed: Random seed value
        deterministic: If True, set PyTorch to use deterministic algorithms
                      (may impact performance)
    """
    # Python random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)

    # PyTorch CUDA (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Set deterministic behavior
    if deterministic:
        # Make CuDNN deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Set environment variable for hash randomization
        os.environ['PYTHONHASHSEED'] = str(seed)

        # Warn about potential performance impact
        logger.info(f"Seed set to {seed} with deterministic mode enabled.")
        logger.warning(
            "Deterministic mode may impact performance. "
            "Set deterministic=False for faster training if exact reproducibility is not required."
        )
    else:
        # Allow CuDNN to benchmark and choose fastest algorithm
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        logger.info(f"Seed set to {seed} with non-deterministic mode (faster training).")


def worker_init_fn(worker_id: int, seed: Optional[int] = None) -> None:
    """
    Initialize worker seeds for PyTorch DataLoader workers.
    Ensures each worker has a different but reproducible seed.

    Args:
        worker_id: Worker ID assigned by DataLoader
        seed: Base seed value (if None, uses current time)

    Usage:
        DataLoader(..., worker_init_fn=lambda worker_id: worker_init_fn(worker_id, seed=42))
    """
    if seed is None:
        seed = torch.initial_seed() % 2**32

    worker_seed = seed + worker_id

    # Set seeds for this worker
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def get_generator(seed: int) -> torch.Generator:
    """
    Get a PyTorch random number generator with a specific seed.

    Args:
        seed: Random seed value

    Returns:
        PyTorch Generator object

    Usage:
        generator = get_generator(42)
        DataLoader(..., generator=generator)
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


class SeedContext:
    """
    Context manager for temporary seed changes.
    Useful for making specific operations reproducible without affecting global state.
    """

    def __init__(self, seed: int):
        """
        Initialize seed context.

        Args:
            seed: Temporary seed value
        """
        self.seed = seed
        self.python_state = None
        self.numpy_state = None
        self.torch_state = None
        self.torch_cuda_state = None

    def __enter__(self):
        """Save current random states and set new seed."""
        # Save current states
        self.python_state = random.getstate()
        self.numpy_state = np.random.get_state()
        self.torch_state = torch.get_rng_state()

        if torch.cuda.is_available():
            self.torch_cuda_state = torch.cuda.get_rng_state_all()

        # Set new seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore previous random states."""
        random.setstate(self.python_state)
        np.random.set_state(self.numpy_state)
        torch.set_rng_state(self.torch_state)

        if torch.cuda.is_available() and self.torch_cuda_state is not None:
            torch.cuda.set_rng_state_all(self.torch_cuda_state)


def check_reproducibility(seed: int = 42, num_iterations: int = 5) -> bool:
    """
    Test if reproducibility is working correctly.

    Args:
        seed: Seed to test
        num_iterations: Number of test iterations

    Returns:
        True if reproducible, False otherwise
    """
    results = []

    for _ in range(num_iterations):
        set_seed(seed)

        # Generate some random numbers
        python_rand = random.random()
        numpy_rand = np.random.rand()
        torch_rand = torch.rand(1).item()

        results.append((python_rand, numpy_rand, torch_rand))

    # Check if all iterations produced the same results
    first_result = results[0]
    is_reproducible = all(r == first_result for r in results)

    if is_reproducible:
        logger.info(f"Reproducibility check PASSED with seed {seed}")
    else:
        logger.error(f"Reproducibility check FAILED with seed {seed}")
        logger.error(f"Results: {results}")

    return is_reproducible


if __name__ == "__main__":
    # Test seed functionality
    import logging
    logging.basicConfig(level=logging.INFO)

    print("Testing seed functionality...")
    print("=" * 80)

    # Test basic seed setting
    print("\nTest 1: Basic seed setting")
    set_seed(42)
    print(f"Python random: {random.random()}")
    print(f"NumPy random: {np.random.rand()}")
    print(f"PyTorch random: {torch.rand(1).item()}")

    # Reset and test reproducibility
    print("\nTest 2: Reproducibility check")
    set_seed(42)
    val1 = random.random()
    set_seed(42)
    val2 = random.random()
    print(f"Same values after reset? {val1 == val2}")

    # Test context manager
    print("\nTest 3: Seed context manager")
    set_seed(42)
    print(f"Before context: {random.random()}")

    with SeedContext(999):
        print(f"Inside context (seed=999): {random.random()}")

    print(f"After context (seed=42 restored): {random.random()}")

    # Full reproducibility check
    print("\nTest 4: Full reproducibility check")
    check_reproducibility(seed=42, num_iterations=5)
