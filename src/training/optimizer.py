"""
Optimizer and learning rate scheduler setup.
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
    StepLR,
    ExponentialLR
)
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


def create_optimizer(
    model: torch.nn.Module,
    optimizer_config: Dict
) -> torch.optim.Optimizer:
    """
    Create optimizer from configuration.

    Args:
        model: PyTorch model
        optimizer_config: Optimizer configuration

    Returns:
        Configured optimizer
    """
    optimizer_type = optimizer_config.get('type', 'adamw').lower()
    lr = optimizer_config.get('learning_rate', 1e-4)
    weight_decay = optimizer_config.get('weight_decay', 1e-5)

    # Differential learning rates (if enabled)
    if optimizer_config.get('differential_lr', {}).get('enabled', False):
        param_groups = create_param_groups_with_differential_lr(
            model,
            base_lr=lr,
            image_encoder_lr=optimizer_config['differential_lr'].get('image_encoder_lr', 1e-5),
            other_lr=optimizer_config['differential_lr'].get('other_lr', 1e-4)
        )
    else:
        param_groups = model.parameters()

    # Create optimizer
    if optimizer_type == 'adam':
        optimizer = optim.Adam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=optimizer_config.get('betas', (0.9, 0.999)),
            eps=optimizer_config.get('eps', 1e-8),
            amsgrad=optimizer_config.get('amsgrad', False)
        )

    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=optimizer_config.get('betas', (0.9, 0.999)),
            eps=optimizer_config.get('eps', 1e-8),
            amsgrad=optimizer_config.get('amsgrad', False)
        )

    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(
            param_groups,
            lr=lr,
            momentum=optimizer_config.get('momentum', 0.9),
            weight_decay=weight_decay,
            nesterov=optimizer_config.get('nesterov', True)
        )

    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    logger.info(f"Optimizer created: {optimizer_type}, lr={lr}, weight_decay={weight_decay}")

    return optimizer


def create_param_groups_with_differential_lr(
    model: torch.nn.Module,
    base_lr: float,
    image_encoder_lr: float,
    other_lr: float
) -> List[Dict]:
    """
    Create parameter groups with differential learning rates.

    Args:
        model: PyTorch model
        base_lr: Base learning rate
        image_encoder_lr: Learning rate for image encoder
        other_lr: Learning rate for other components

    Returns:
        List of parameter groups
    """
    param_groups = []

    # Image encoder with lower learning rate (pretrained)
    if hasattr(model, 'image_encoder'):
        param_groups.append({
            'params': model.image_encoder.parameters(),
            'lr': image_encoder_lr
        })
        logger.info(f"Image encoder: lr={image_encoder_lr}")

    # Other components with higher learning rate
    other_params = []
    for name, module in model.named_children():
        if name != 'image_encoder':
            other_params.extend(module.parameters())

    if other_params:
        param_groups.append({
            'params': other_params,
            'lr': other_lr
        })
        logger.info(f"Other components: lr={other_lr}")

    return param_groups


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_config: Dict,
    num_training_steps: Optional[int] = None
):
    """
    Create learning rate scheduler from configuration.

    Args:
        optimizer: PyTorch optimizer
        scheduler_config: Scheduler configuration
        num_training_steps: Total number of training steps (for some schedulers)

    Returns:
        Configured scheduler
    """
    scheduler_type = scheduler_config.get('type', 'cosine').lower()

    if scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps or 100,
            eta_min=scheduler_config.get('min_lr', 1e-6)
        )
        logger.info(f"Scheduler: CosineAnnealingLR, min_lr={scheduler_config.get('min_lr', 1e-6)}")

    elif scheduler_type == 'cosine_warmup':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_config.get('T_0', 10),
            T_mult=scheduler_config.get('T_mult', 2),
            eta_min=scheduler_config.get('min_lr', 1e-6)
        )
        logger.info(f"Scheduler: CosineAnnealingWarmRestarts")

    elif scheduler_type == 'reduce_on_plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config.get('mode', 'min'),
            factor=scheduler_config.get('factor', 0.5),
            patience=scheduler_config.get('patience', 5),
            min_lr=scheduler_config.get('min_lr', 1e-6),
            verbose=True
        )
        logger.info(f"Scheduler: ReduceLROnPlateau, patience={scheduler_config.get('patience', 5)}")

    elif scheduler_type == 'step':
        scheduler = StepLR(
            optimizer,
            step_size=scheduler_config.get('step_size', 10),
            gamma=scheduler_config.get('gamma', 0.1)
        )
        logger.info(f"Scheduler: StepLR, step_size={scheduler_config.get('step_size', 10)}")

    elif scheduler_type == 'exponential':
        scheduler = ExponentialLR(
            optimizer,
            gamma=scheduler_config.get('gamma', 0.95)
        )
        logger.info(f"Scheduler: ExponentialLR, gamma={scheduler_config.get('gamma', 0.95)}")

    elif scheduler_type == 'none' or scheduler_type is None:
        scheduler = None
        logger.info("No scheduler used")

    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    return scheduler


class WarmupScheduler:
    """Learning rate scheduler with warmup."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        base_scheduler,
        warmup_start_lr: float = 1e-6
    ):
        """
        Initialize warmup scheduler.

        Args:
            optimizer: PyTorch optimizer
            warmup_epochs: Number of warmup epochs
            base_scheduler: Base scheduler to use after warmup
            warmup_start_lr: Starting learning rate for warmup
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_scheduler = base_scheduler
        self.warmup_start_lr = warmup_start_lr
        self.current_epoch = 0

        # Store initial learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self, epoch: Optional[int] = None, metric: Optional[float] = None):
        """
        Step the scheduler.

        Args:
            epoch: Current epoch
            metric: Metric value (for ReduceLROnPlateau)
        """
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1

        if self.current_epoch < self.warmup_epochs:
            # Warmup phase: linearly increase lr
            warmup_factor = self.current_epoch / self.warmup_epochs

            for i, param_group in enumerate(self.optimizer.param_groups):
                lr = self.warmup_start_lr + (self.base_lrs[i] - self.warmup_start_lr) * warmup_factor
                param_group['lr'] = lr

            logger.debug(f"Warmup epoch {self.current_epoch}/{self.warmup_epochs}, lr={lr:.6f}")
        else:
            # Regular scheduling
            if self.base_scheduler is not None:
                if isinstance(self.base_scheduler, ReduceLROnPlateau):
                    if metric is not None:
                        self.base_scheduler.step(metric)
                else:
                    self.base_scheduler.step()

    def get_last_lr(self):
        """Get last learning rate."""
        return [group['lr'] for group in self.optimizer.param_groups]


if __name__ == "__main__":
    # Test optimizer and scheduler creation
    import logging
    logging.basicConfig(level=logging.INFO)

    print("Testing Optimizer and Scheduler...")
    print("=" * 80)

    # Create a dummy model
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.image_encoder = torch.nn.Linear(10, 20)
            self.classifier = torch.nn.Linear(20, 4)

    model = DummyModel()

    # Test optimizer creation
    print("\nTest 1: Create AdamW optimizer")
    print("-" * 80)

    optimizer_config = {
        'type': 'adamw',
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'betas': (0.9, 0.999)
    }

    optimizer = create_optimizer(model, optimizer_config)
    print(f"Optimizer: {optimizer}")

    # Test differential learning rates
    print("\n\nTest 2: Differential learning rates")
    print("-" * 80)

    optimizer_config_diff = {
        'type': 'adamw',
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'differential_lr': {
            'enabled': True,
            'image_encoder_lr': 1e-5,
            'other_lr': 1e-4
        }
    }

    optimizer_diff = create_optimizer(model, optimizer_config_diff)
    for i, param_group in enumerate(optimizer_diff.param_groups):
        print(f"Parameter group {i}: lr={param_group['lr']}")

    # Test scheduler creation
    print("\n\nTest 3: Create schedulers")
    print("-" * 80)

    scheduler_configs = [
        {'type': 'cosine', 'min_lr': 1e-6},
        {'type': 'reduce_on_plateau', 'patience': 5, 'factor': 0.5},
        {'type': 'step', 'step_size': 10, 'gamma': 0.1}
    ]

    for config in scheduler_configs:
        scheduler = create_scheduler(optimizer, config, num_training_steps=100)
        print(f"  Scheduler type: {config['type']}")

    # Test warmup scheduler
    print("\n\nTest 4: Warmup scheduler")
    print("-" * 80)

    base_scheduler = CosineAnnealingLR(optimizer, T_max=100)
    warmup_scheduler = WarmupScheduler(
        optimizer,
        warmup_epochs=5,
        base_scheduler=base_scheduler
    )

    print("Learning rates during warmup:")
    for epoch in range(10):
        warmup_scheduler.step(epoch)
        lr = warmup_scheduler.get_last_lr()[0]
        print(f"  Epoch {epoch}: lr={lr:.6f}")

    print("\n" + "=" * 80)
    print("Optimizer and scheduler tests completed!")
