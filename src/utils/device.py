"""
Device management utilities for PyTorch.
Handles GPU/CPU selection, device information, and memory management.
"""

import torch
import logging
from typing import Union, Optional

logger = logging.getLogger(__name__)


def get_device(
    use_cuda: bool = True,
    device_id: Optional[int] = None
) -> torch.device:
    """
    Get the appropriate device (GPU/CPU) for PyTorch operations.

    Args:
        use_cuda: Whether to use CUDA if available
        device_id: Specific GPU device ID to use (None for default)

    Returns:
        torch.device object
    """
    if use_cuda and torch.cuda.is_available():
        if device_id is not None:
            device = torch.device(f'cuda:{device_id}')
        else:
            device = torch.device('cuda')

        logger.info(f"Using device: {device} ({torch.cuda.get_device_name(device)})")
    else:
        device = torch.device('cpu')
        if use_cuda and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Using CPU instead.")
        logger.info("Using device: CPU")

    return device


def get_device_info(device: Optional[torch.device] = None) -> dict:
    """
    Get detailed information about the device.

    Args:
        device: PyTorch device (if None, uses default CUDA device or CPU)

    Returns:
        Dictionary containing device information
    """
    if device is None:
        device = get_device()

    info = {
        'device_type': device.type,
        'device_index': device.index,
    }

    if device.type == 'cuda':
        device_id = device.index if device.index is not None else 0

        info.update({
            'device_name': torch.cuda.get_device_name(device_id),
            'total_memory_gb': torch.cuda.get_device_properties(device_id).total_memory / (1024**3),
            'cuda_version': torch.version.cuda,
            'cudnn_version': torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            'num_gpus': torch.cuda.device_count(),
        })

    return info


def print_device_info(device: Optional[torch.device] = None) -> None:
    """
    Print detailed device information.

    Args:
        device: PyTorch device (if None, uses default)
    """
    info = get_device_info(device)

    logger.info("=" * 80)
    logger.info(f"{'Device Information':^80}")
    logger.info("=" * 80)

    for key, value in info.items():
        logger.info(f"{key}: {value}")

    logger.info("=" * 80)


def get_memory_info(device: Optional[torch.device] = None) -> dict:
    """
    Get GPU memory usage information.

    Args:
        device: PyTorch device (if None, uses default CUDA device)

    Returns:
        Dictionary with memory information in GB
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device.type != 'cuda':
        return {'error': 'Memory info only available for CUDA devices'}

    device_id = device.index if device.index is not None else 0

    allocated = torch.cuda.memory_allocated(device_id) / (1024**3)
    reserved = torch.cuda.memory_reserved(device_id) / (1024**3)
    total = torch.cuda.get_device_properties(device_id).total_memory / (1024**3)

    return {
        'allocated_gb': allocated,
        'reserved_gb': reserved,
        'total_gb': total,
        'free_gb': total - reserved,
        'utilization_%': (reserved / total) * 100
    }


def print_memory_info(device: Optional[torch.device] = None) -> None:
    """
    Print GPU memory usage information.

    Args:
        device: PyTorch device
    """
    info = get_memory_info(device)

    if 'error' in info:
        logger.info(info['error'])
        return

    logger.info("-" * 80)
    logger.info("GPU Memory Usage:")
    logger.info(f"  Allocated: {info['allocated_gb']:.2f} GB")
    logger.info(f"  Reserved:  {info['reserved_gb']:.2f} GB")
    logger.info(f"  Free:      {info['free_gb']:.2f} GB")
    logger.info(f"  Total:     {info['total_gb']:.2f} GB")
    logger.info(f"  Utilization: {info['utilization_%']:.1f}%")
    logger.info("-" * 80)


def clear_memory(device: Optional[torch.device] = None) -> None:
    """
    Clear GPU cache to free up memory.

    Args:
        device: PyTorch device
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info("GPU cache cleared")
    else:
        logger.info("Cache clearing only applicable to CUDA devices")


def move_to_device(
    data: Union[torch.Tensor, dict, list, tuple],
    device: torch.device
) -> Union[torch.Tensor, dict, list, tuple]:
    """
    Recursively move data to the specified device.
    Handles tensors, dictionaries, lists, and tuples.

    Args:
        data: Data to move (tensor, dict, list, or tuple)
        device: Target device

    Returns:
        Data moved to the device
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}
    elif isinstance(data, list):
        return [move_to_device(item, device) for item in data]
    elif isinstance(data, tuple):
        return tuple(move_to_device(item, device) for item in data)
    else:
        return data


def set_device_optimization(device: torch.device, mixed_precision: bool = True) -> None:
    """
    Set device-specific optimizations.

    Args:
        device: PyTorch device
        mixed_precision: Whether to enable automatic mixed precision
    """
    if device.type == 'cuda':
        # Enable TensorFloat-32 for matrix multiplications (Ampere+ GPUs)
        if hasattr(torch.backends.cuda, 'matmul'):
            torch.backends.cuda.matmul.allow_tf32 = True
            logger.info("TensorFloat-32 enabled for matrix multiplications")

        # Enable TensorFloat-32 for cuDNN (Ampere+ GPUs)
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = True
            logger.info("TensorFloat-32 enabled for cuDNN")

        # Enable cuDNN autotuner for optimal algorithms
        torch.backends.cudnn.benchmark = True
        logger.info("cuDNN autotuner enabled")

        if mixed_precision:
            logger.info("Mixed precision training enabled")


def get_num_gpus() -> int:
    """
    Get the number of available GPUs.

    Returns:
        Number of GPUs
    """
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def is_distributed() -> bool:
    """
    Check if distributed training is initialized.

    Returns:
        True if distributed, False otherwise
    """
    return torch.distributed.is_available() and torch.distributed.is_initialized()


class DeviceContext:
    """Context manager for temporary device changes."""

    def __init__(self, device: torch.device):
        """
        Initialize device context.

        Args:
            device: Target device
        """
        self.device = device
        self.previous_device = None

    def __enter__(self):
        """Set new device."""
        if self.device.type == 'cuda':
            self.previous_device = torch.cuda.current_device()
            torch.cuda.set_device(self.device)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore previous device."""
        if self.previous_device is not None:
            torch.cuda.set_device(self.previous_device)


if __name__ == "__main__":
    # Test device utilities
    import logging
    logging.basicConfig(level=logging.INFO)

    print("Testing device utilities...")
    print("=" * 80)

    # Get device
    device = get_device(use_cuda=True)
    print(f"\nDevice: {device}")

    # Print device info
    print_device_info(device)

    # Test number of GPUs
    num_gpus = get_num_gpus()
    print(f"\nNumber of GPUs: {num_gpus}")

    # Test memory info (if CUDA available)
    if device.type == 'cuda':
        print("\nMemory info before allocation:")
        print_memory_info(device)

        # Allocate some memory
        x = torch.randn(1000, 1000, device=device)

        print("\nMemory info after allocation:")
        print_memory_info(device)

        # Clear memory
        del x
        clear_memory(device)

        print("\nMemory info after clearing:")
        print_memory_info(device)

    # Test move_to_device
    print("\nTesting move_to_device...")
    test_data = {
        'tensor': torch.randn(10, 10),
        'list': [torch.randn(5), torch.randn(5)],
        'nested': {
            'a': torch.randn(3, 3),
            'b': 42
        }
    }
    moved_data = move_to_device(test_data, device)
    print(f"Data moved to: {moved_data['tensor'].device}")
