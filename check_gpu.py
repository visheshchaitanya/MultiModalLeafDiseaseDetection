import torch

print('=' * 60)
print('PyTorch CUDA Status:')
print('=' * 60)
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'CUDA Version: {torch.version.cuda}')
print(f'PyTorch Version: {torch.__version__}')
print(f'Number of GPUs: {torch.cuda.device_count()}')

if torch.cuda.is_available():
    print(f'GPU Name: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
    print(f'Current Device: {torch.cuda.current_device()}')
else:
    print('WARNING: No GPU detected!')

print('=' * 60)
