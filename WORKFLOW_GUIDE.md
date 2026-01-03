# Complete End-to-End Workflow Guide

This guide explains the complete machine learning pipeline from training to production inference.

## Pipeline Overview

```
┌─────────────────┐
│  1. Data Prep   │  ← Generate synthetic data, create train/val/test splits
└────────┬────────┘
         │
┌────────▼────────┐
│  2. Training    │  ← Train the multi-modal model
└────────┬────────┘
         │
┌────────▼────────┐
│  3. Evaluation  │  ← Evaluate on test set, generate metrics
└────────┬────────┘
         │
┌────────▼────────┐
│  4. Inference   │  ← Run predictions on new data
└─────────────────┘
```

---

## 1. Data Preparation ✓

**Status**: Completed

**Files**:
- `data/processed/train.csv` - Training data (3,508 samples)
- `data/processed/val.csv` - Validation data (752 samples)
- `data/processed/test.csv` - Test data (1,778 samples)
- `data/processed/vocabulary.pkl` - Text vocabulary (690 tokens)

**Scripts used**:
```bash
python scripts/download_data.py
python scripts/generate_synthetic_data.py
python scripts/prepare_dataset.py
```

---

## 2. Model Training ✓

**Status**: Completed (1 epoch)

**Script**: `scripts/train_simple.py`

**Results**:
- Training Accuracy: 83.67%
- Validation Accuracy: 89.20%
- Checkpoint: `outputs/checkpoints/best_model.pt`

**To continue training**:
```bash
# Train for more epochs (currently only 1/20 completed)
python scripts/train_simple.py
```

**Model Architecture**:
- Image Encoder: ResNet-50 (pretrained)
- Tabular Encoder: MLP for sensor data
- Fusion Module: Concatenation + projection
- Text Decoder: Transformer (4 layers, 8 heads)
- Classification Head: Disease classification
- Total Parameters: ~42.8M

---

## 3. Model Evaluation ✓

**Status**: Completed on test set

**Script**: `scripts/evaluate_simple.py`

**Results**:
- Test Accuracy: 89.15%
- Macro F1-Score: 0.7252
- Best performing class: Healthy (F1: 0.9293)
- Challenging classes: Stemphylium (F1: 0.4242) - only 20 samples

**Output files**:
- `outputs/evaluation/results.txt` - Detailed metrics
- `outputs/evaluation/confusion_matrix.png` - Visualization
- `outputs/evaluation/predictions.csv` - Per-sample predictions

**To re-run evaluation**:
```bash
python scripts/evaluate_simple.py
```

---

## 4. Inference on New Data → Next Step

This is where you use the trained model in production or for validation.

### Option A: Single Image Inference

**Use Case**: Test the model on one image at a time

**Script**: `scripts/inference_simple.py`

**Example Usage**:
```bash
# Example 1: Predict on a healthy leaf
python scripts/inference_simple.py \
    --image data/raw/DiaMOS_Plant/Healthy/u1.jpg \
    --sensors "22.5,65.0,50.0" \
    --output outputs/prediction_healthy.png

# Example 2: Predict on an Alternaria-infected leaf
python scripts/inference_simple.py \
    --image data/raw/DiaMOS_Plant/Alternaria/u100.jpg \
    --sensors "25.0,82.5,40.0" \
    --output outputs/prediction_alternaria.png

# Example 3: Use GPU if available
python scripts/inference_simple.py \
    --image path/to/new_image.jpg \
    --sensors "23.0,75.0,45.0" \
    --device cuda
```

**Sensor Format**: `"temperature,humidity,soil_moisture"`
- Temperature: in °C (e.g., 22.5)
- Humidity: in % (e.g., 65.0)
- Soil Moisture: in % (e.g., 50.0)

**Output**:
- Predicted disease class
- Confidence score
- Class probabilities
- Generated text explanation
- Visualization (saved if --output specified)

---

### Option B: Batch Validation Pipeline

**Use Case**: Validate on multiple new images, production simulation

**Script**: `scripts/validate_pipeline.py`

**Example Usage**:
```bash
# Validate on a directory of healthy leaves
python scripts/validate_pipeline.py \
    --data-dir data/raw/DiaMOS_Plant/Healthy \
    --num-samples 10 \
    --generate-sensors \
    --output-dir outputs/validation_healthy

# Validate on Alternaria samples
python scripts/validate_pipeline.py \
    --data-dir data/raw/DiaMOS_Plant/Alternaria \
    --num-samples 20 \
    --output-dir outputs/validation_alternaria

# Process all images in a directory
python scripts/validate_pipeline.py \
    --data-dir path/to/new_data \
    --num-samples 100 \
    --output-dir outputs/production_validation
```

**Output**:
- `validation_results_YYYYMMDD_HHMMSS.csv` - All predictions
- `validation_report_YYYYMMDD_HHMMSS.txt` - Summary report
- `validation_summary.png` - Visual grid of predictions

---

## Complete End-to-End Example

Here's a complete workflow from new data to predictions:

### Scenario: You receive new leaf images from the field

**Step 1**: Organize your new images
```
new_data/
├── field_sample_001.jpg
├── field_sample_002.jpg
├── field_sample_003.jpg
└── ...
```

**Step 2**: Run batch validation
```bash
python scripts/validate_pipeline.py \
    --data-dir new_data \
    --num-samples 50 \
    --generate-sensors \
    --output-dir outputs/field_validation
```

**Step 3**: Review results
```
outputs/field_validation/
├── validation_results_20260103_120000.csv  ← All predictions
├── validation_report_20260103_120000.txt   ← Summary
└── validation_summary.png                   ← Visualization
```

**Step 4**: Examine CSV for detailed analysis
```python
import pandas as pd

results = pd.read_csv('outputs/field_validation/validation_results_*.csv')

# See prediction distribution
print(results['predicted_class'].value_counts())

# Filter high-confidence predictions
high_conf = results[results['confidence'] > 0.9]

# Find uncertain predictions (might need manual review)
uncertain = results[results['confidence'] < 0.7]
```

---

## Production Deployment Options

### Option 1: Python API (Flask/FastAPI)

Create a REST API for real-time predictions:

```python
# Example: api.py
from flask import Flask, request, jsonify
from inference_simple import load_model, predict

app = Flask(__name__)
model, vocab, classes = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    image = request.files['image']
    sensors = request.form['sensors']  # "temp,humid,soil"

    result = run_inference(image, sensors, model, vocab, classes)

    return jsonify({
        'class': result['predicted_class'],
        'confidence': result['confidence'],
        'explanation': result['explanation']
    })
```

### Option 2: Batch Processing Script

Process images from a database or file system periodically:

```python
# Example: batch_process.py
import schedule
import time

def process_new_images():
    # Scan for new images
    # Run inference
    # Save results to database
    pass

schedule.every().hour.do(process_new_images)

while True:
    schedule.run_pending()
    time.sleep(60)
```

### Option 3: Web Interface

Create a simple web UI using Gradio or Streamlit:

```python
# Example: app.py
import gradio as gr
from inference_simple import run_inference

def predict(image, temperature, humidity, soil_moisture):
    sensors = f"{temperature},{humidity},{soil_moisture}"
    result = run_inference(image, sensors)
    return result['predicted_class'], result['confidence'], result['explanation']

interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="filepath"),
        gr.Number(label="Temperature (°C)"),
        gr.Number(label="Humidity (%)"),
        gr.Number(label="Soil Moisture (%)")
    ],
    outputs=[
        gr.Text(label="Predicted Class"),
        gr.Number(label="Confidence"),
        gr.Text(label="Explanation")
    ]
)

interface.launch()
```

---

## Handling New Data Without Ground Truth

When you don't have labels for new data:

1. **Run inference** to get predictions
2. **Review confidence scores** - flag low confidence (<70%) for manual review
3. **Use explanations** to understand model reasoning
4. **Track predictions** over time to monitor model drift
5. **Collect feedback** from domain experts for continuous improvement

```bash
# Process unlabeled data
python scripts/validate_pipeline.py \
    --data-dir unlabeled_field_data \
    --output-dir outputs/unlabeled_predictions

# Review low confidence predictions
# Filter CSV: confidence < 0.7
# Send to experts for manual labeling
# Use labeled data to retrain/fine-tune model
```

---

## Integration with Real Sensor Data

If you have real sensor data in a CSV:

```csv
image_path,temperature,humidity,soil_moisture
field_001.jpg,23.5,68.0,45.5
field_002.jpg,24.0,72.5,42.0
...
```

Modify `validate_pipeline.py` to read from this CSV instead of generating synthetic data.

---

## Monitoring and Model Maintenance

### 1. Track Performance Over Time
```python
# Save predictions with timestamps
# Monitor: accuracy, confidence, class distribution
# Alert if metrics degrade
```

### 2. Detect Model Drift
```python
# Compare prediction distributions: training vs production
# If distribution shifts significantly → retrain
```

### 3. Active Learning
```python
# Identify samples where model is uncertain
# Get expert labels
# Add to training set
# Retrain periodically
```

---

## Next Steps Recommendations

Based on your current progress:

1. **Test inference on sample images** (try both scripts)
   ```bash
   python scripts/inference_simple.py --image data/raw/DiaMOS_Plant/Healthy/u1.jpg --sensors "22.5,65.0,50.0"
   ```

2. **Run batch validation** to see how the model performs on different classes
   ```bash
   python scripts/validate_pipeline.py --data-dir data/raw/DiaMOS_Plant/Healthy --num-samples 5
   ```

3. **Address class imbalance** before production deployment
   - Collect more Alternaria and Stemphylium samples
   - Use class weights in training
   - Consider oversampling minority classes

4. **Continue training** for remaining epochs (19 more)
   ```bash
   python scripts/train_simple.py
   ```

5. **Create a simple web demo** using Gradio for stakeholder demonstration

---

## Troubleshooting

### Issue: "Checkpoint not found"
- Ensure you've run training: `python scripts/train_simple.py`
- Check checkpoint exists: `outputs/checkpoints/best_model.pt`

### Issue: "Vocabulary not found"
- Run data preparation: `python scripts/prepare_dataset.py`
- Check file exists: `data/processed/vocabulary.pkl`

### Issue: "Low confidence predictions"
- Model only trained for 1 epoch
- Continue training for better performance
- Check if image quality is poor
- Verify sensor values are in correct range

### Issue: "Out of memory error"
- Reduce batch size in training script
- Use CPU instead of GPU: `--device cpu`
- Process fewer samples: `--num-samples 5`

---

## File Structure Reference

```
project/
├── scripts/
│   ├── train_simple.py          ← Training
│   ├── evaluate_simple.py       ← Evaluation
│   ├── inference_simple.py      ← Single image inference
│   └── validate_pipeline.py     ← Batch validation
├── outputs/
│   ├── checkpoints/
│   │   └── best_model.pt        ← Trained model
│   ├── evaluation/
│   │   ├── results.txt
│   │   ├── confusion_matrix.png
│   │   └── predictions.csv
│   └── validation/              ← New predictions go here
└── data/
    ├── processed/
    │   ├── train.csv
    │   ├── val.csv
    │   ├── test.csv
    │   └── vocabulary.pkl
    └── raw/
        └── DiaMOS_Plant/        ← Original images
```

---

## Quick Reference Commands

```bash
# Train model
python scripts/train_simple.py

# Evaluate model
python scripts/evaluate_simple.py

# Predict single image
python scripts/inference_simple.py \
    --image path/to/image.jpg \
    --sensors "temp,humid,soil"

# Validate multiple images
python scripts/validate_pipeline.py \
    --data-dir path/to/images \
    --num-samples 10

# Check results
cat outputs/evaluation/results.txt
ls outputs/validation/
```

---

**Last Updated**: 2026-01-03
**Model Version**: v1.0 (1 epoch)
**Test Accuracy**: 89.15%
