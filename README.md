# Multi-Modal Leaf Disease Detection

A deep learning system that combines leaf images with environmental sensor data to detect plant diseases and generate textual explanations.

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/license-MIT-green)

## Overview

This project implements a multi-modal deep learning model that:
- **Detects** plant diseases from leaf images (Healthy, Alternaria, Stemphylium, Marssonina)
- **Integrates** environmental sensor data (temperature, humidity, soil moisture)
- **Generates** natural language explanations for predictions
- **Provides** confidence scores and visual interpretations

### Key Features

- **Multi-Modal Fusion**: Combines visual and tabular data for improved accuracy
- **Transformer-Based Text Generation**: Produces human-readable disease explanations
- **Transfer Learning**: Uses pretrained ResNet-50 for efficient feature extraction
- **Comprehensive Evaluation**: Includes classification metrics (accuracy, F1) and NLG metrics (BLEU, ROUGE)
- **Explainability**: Supports Grad-CAM visualization and attention analysis
- **Production-Ready**: Includes training, evaluation, and inference scripts

## Architecture

```
Image (224×224×3) → ImageEncoder (ResNet-50) → [B, 512]
                                                    ↓
                                                 Concat → FusionModule → [B, 512]
                                                    ↑                        ↓
Sensors (3,) → TabularEncoder (MLP) → [B, 128]     ├→ ClassificationHead → [B, 4]
                                                    └→ TransformerDecoder → Text Output
```

**Components:**
- **Image Encoder**: ResNet-50 (pretrained on ImageNet) with projection layer
- **Tabular Encoder**: MLP for environmental sensor data
- **Fusion Module**: Concatenation + MLP projection
- **Transformer Decoder**: 4 layers, 8 heads for text generation
- **Classification Head**: Disease classification (4 classes)

## Installation

### Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/MultiModalLeafDiseaseDetection.git
cd MultiModalLeafDiseaseDetection
```

2. **Create virtual environment:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Install the package:**
```bash
pip install -e .
```

## Quick Start

### 1. Data Preparation

Download the DiaMOS Plant dataset and generate synthetic sensor data:

```bash
# Download DiaMOS dataset
python scripts/download_data.py

# Generate synthetic sensor data
python scripts/generate_synthetic_data.py

# Prepare dataset (splits, normalization, vocabulary)
python scripts/prepare_dataset.py
```

**Manual Download** (if automatic download fails):
1. Download DiaMOS Plant dataset from [official source]
2. Extract to `data/raw/`
3. Run `python scripts/generate_synthetic_data.py`

### 2. Training

Train the model with default configuration:

```bash
python scripts/train.py --config config/config.yaml
```

**Training options:**
```bash
python scripts/train.py \
    --config config/config.yaml \
    --epochs 100 \
    --batch-size 32 \
    --lr 1e-4 \
    --output-dir outputs/experiment1
```

**Resume training:**
```bash
python scripts/train.py \
    --config config/config.yaml \
    --checkpoint outputs/checkpoints/interrupted.pt
```

### 3. Evaluation

Evaluate a trained model:

```bash
python scripts/evaluate.py \
    --config config/config.yaml \
    --checkpoint outputs/checkpoints/best.pt \
    --output-dir outputs/evaluation
```

**Evaluation options:**
```bash
python scripts/evaluate.py \
    --checkpoint outputs/checkpoints/best.pt \
    --split test \
    --no-text-metrics  # Skip BLEU/ROUGE computation (faster)
```

### 4. Inference

Run prediction on a single image:

```bash
python scripts/inference.py \
    --config config/config.yaml \
    --checkpoint outputs/checkpoints/best.pt \
    --image path/to/leaf_image.jpg \
    --sensors "22.5,80.0,45.0"  # temperature,humidity,soil_moisture
```

**Example:**
```bash
python scripts/inference.py \
    --checkpoint outputs/checkpoints/best.pt \
    --image data/test/alternaria_001.jpg \
    --sensors "25.0,82.5,40.0" \
    --output outputs/prediction.png
```

## Configuration

Configuration files are located in `config/`:

- **`config.yaml`**: Main configuration (paths, data, device)
- **`model_config.yaml`**: Model architecture settings
- **`training_config.yaml`**: Training hyperparameters

### Key Configuration Options

**Model:**
```yaml
model:
  image_encoder:
    backbone: resnet50
    pretrained: true
    freeze_layers: 7
    output_dim: 512

  decoder:
    num_layers: 4
    num_heads: 8
    embed_dim: 512
```

**Training:**
```yaml
training:
  num_epochs: 100
  batch_size: 32
  gradient_clip: 1.0
  mixed_precision: true

  loss:
    alpha: 0.7  # Text generation weight
    beta: 0.3   # Classification weight
```

**Optimizer:**
```yaml
optimizer:
  type: adamw
  learning_rate: 1e-4
  weight_decay: 1e-5

  differential_lr:
    enabled: true
    image_encoder_lr: 1e-5
    other_lr: 1e-4
```

## Dataset

### DiaMOS Plant Dataset

- **Total samples**: 3,505 pear leaf images
- **Classes**: 4 (Healthy, Alternaria, Stemphylium, Marssonina)
- **Resolution**: Variable (resized to 224×224)
- **Split**: 70% train / 15% val / 15% test (stratified)

### Synthetic Sensor Data

Environmental sensor values are generated with disease-dependent distributions:

| Disease | Temperature (°C) | Humidity (%) | Soil Moisture (%) |
|---------|-----------------|--------------|-------------------|
| Healthy | 22.5 ± 2.0 | 65.0 ± 5.0 | 50.0 ± 8.0 |
| Alternaria | 25.0 ± 2.5 | 82.5 ± 5.0 | 40.0 ± 7.0 |
| Stemphylium | 21.0 ± 2.0 | 87.5 ± 5.0 | 50.0 ± 8.0 |
| Marssonina | 18.5 ± 2.0 | 77.5 ± 6.0 | 60.0 ± 8.0 |

## Project Structure

```
MultiModalLeafDiseaseDetection/
├── config/                    # Configuration files
│   ├── config.yaml
│   ├── model_config.yaml
│   └── training_config.yaml
├── data/                      # Dataset storage (gitignored)
│   ├── raw/
│   ├── processed/
│   └── splits/
├── src/
│   ├── data/                  # Data pipeline
│   │   ├── dataset.py
│   │   ├── dataloader.py
│   │   ├── transforms.py
│   │   ├── preprocessing.py
│   │   ├── download.py
│   │   └── synthetic_generator.py
│   ├── models/                # Model components
│   │   ├── multimodal_model.py
│   │   ├── image_encoder.py
│   │   ├── tabular_encoder.py
│   │   ├── fusion_module.py
│   │   ├── transformer_decoder.py
│   │   └── classification_head.py
│   ├── training/              # Training infrastructure
│   │   ├── trainer.py
│   │   ├── losses.py
│   │   ├── optimizer.py
│   │   ├── early_stopping.py
│   │   └── checkpointing.py
│   ├── evaluation/            # Evaluation metrics
│   │   ├── evaluator.py
│   │   ├── metrics.py
│   │   ├── bleu_scorer.py
│   │   ├── rouge_scorer.py
│   │   └── confusion_matrix.py
│   ├── templates/             # Text generation templates
│   │   ├── template_engine.py
│   │   ├── template_definitions.py
│   │   └── vocabulary.py
│   └── utils/                 # Utilities
│       ├── config_loader.py
│       ├── logging_utils.py
│       ├── device.py
│       ├── seed.py
│       └── visualization.py
├── scripts/                   # Executable scripts
│   ├── download_data.py
│   ├── generate_synthetic_data.py
│   ├── prepare_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
├── outputs/                   # Checkpoints, logs (gitignored)
│   ├── checkpoints/
│   ├── tensorboard/
│   └── evaluation/
├── requirements.txt
├── setup.py
└── README.md
```

## Performance

### Expected Metrics

**Classification:**
- Accuracy: >85%
- F1-Score (macro): >0.82
- Per-class F1: >0.80 for all classes

**Text Generation:**
- BLEU-4: >0.60 (template-based)
- ROUGE-L: >0.70
- Fluency: High (template-based generation)

### Training Time

- **GPU (NVIDIA RTX 3090)**: ~2-3 hours for 100 epochs
- **GPU (NVIDIA GTX 1080 Ti)**: ~4-5 hours for 100 epochs
- **CPU**: Not recommended (very slow)

## Usage Examples

### Python API

```python
import torch
from src.models.multimodal_model import MultiModalLeafDiseaseModel
from src.data.transforms import get_inference_transforms
from src.templates.vocabulary import Vocabulary
from PIL import Image

# Load model
model = MultiModalLeafDiseaseModel(vocab_size=300, num_classes=4)
checkpoint = torch.load('outputs/checkpoints/best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load and preprocess image
transform = get_inference_transforms(image_size=224)
image = Image.open('leaf_image.jpg').convert('RGB')
image_tensor = transform(image=np.array(image))['image'].unsqueeze(0)

# Prepare sensor data
sensor_tensor = torch.tensor([[22.5, 80.0, 45.0]]) / 100.0  # Normalized

# Run inference
with torch.no_grad():
    outputs = model(images=image_tensor, sensors=sensor_tensor)

    # Classification
    predicted_class = torch.argmax(outputs['class_logits'], dim=1)

    # Text generation
    vocabulary = Vocabulary.load_from_json('data/processed/vocabulary.json')
    generated_ids = model.text_decoder.generate(
        memory=outputs['fused_features'].unsqueeze(1),
        start_token_idx=vocabulary.get_token_index('<SOS>'),
        end_token_idx=vocabulary.get_token_index('<EOS>')
    )
    explanation = vocabulary.decode(generated_ids[0].tolist())

print(f"Predicted class: {predicted_class.item()}")
print(f"Explanation: {explanation}")
```

### Custom Dataset

To use your own dataset:

1. **Organize images:**
```
data/raw/
├── class_1/
│   ├── image001.jpg
│   └── ...
├── class_2/
│   └── ...
```

2. **Create metadata CSV:**
```csv
image_path,disease,temperature,humidity,soil_moisture,explanation
data/raw/class_1/image001.jpg,Healthy,22.5,65.0,50.0,"Healthy leaf..."
```

3. **Update config:**
```yaml
data:
  class_names: ['Class1', 'Class2', ...]
  num_classes: 4
```

4. **Run preprocessing:**
```bash
python scripts/prepare_dataset.py --metadata-path data/custom_metadata.csv
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

```bash
# Format code
black src/ scripts/

# Lint
pylint src/
```

### Adding New Features

1. Create feature branch: `git checkout -b feature/new-feature`
2. Implement changes
3. Add tests
4. Update documentation
5. Submit pull request

## Troubleshooting

### Common Issues

**1. CUDA out of memory:**
```yaml
# Reduce batch size in config.yaml
data:
  batch_size: 16  # or 8

# Enable gradient accumulation
training:
  gradient_accumulation_steps: 2
```

**2. Slow training:**
- Enable mixed precision: `training.mixed_precision: true`
- Freeze more image encoder layers: `model.image_encoder.freeze_layers: 10`
- Use smaller image size: `data.image_size: 224` → `192`

**3. Poor text generation quality:**
- Increase decoder layers: `model.decoder.num_layers: 6`
- Adjust generation temperature: `--temperature 0.7`
- Use beam search instead of sampling (requires implementation)

**4. Dataset download fails:**
- Download manually from official source
- Check internet connection
- Ensure sufficient disk space (>5GB)

## Citation

If you use this code in your research, please cite:

```bibtex
@software{multimodal_leaf_disease_2024,
  title = {Multi-Modal Leaf Disease Detection},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/MultiModalLeafDiseaseDetection}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- DiaMOS Plant dataset creators
- PyTorch team for the deep learning framework
- Pretrained ResNet-50 from torchvision
- Community contributors and testers

## Contact

- **Author**: Your Name
- **Email**: your.email@example.com
- **GitHub**: [@yourusername](https://github.com/yourusername)
- **Issues**: [GitHub Issues](https://github.com/yourusername/MultiModalLeafDiseaseDetection/issues)

## Roadmap

- [ ] Add LIME explainability
- [ ] Implement beam search for text generation
- [ ] Support for more plant species
- [ ] Mobile deployment (TensorFlow Lite)
- [ ] Web interface
- [ ] Real-time video inference
- [ ] Multi-language explanations

---

**Built with ❤️ for precision agriculture and plant health monitoring**
