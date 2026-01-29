# Train Custom RF-DETR Object Detection Model

A comprehensive guide and implementation for training RF-DETR (Real-time DETR) object detection models on custom datasets.

## Overview

This project demonstrates how to train RF-DETR, a state-of-the-art real-time object detection model, on your own custom dataset. RF-DETR combines the accuracy of DETR-based models with the speed needed for real-time applications.

## Features

- 🚀 Train RF-DETR on custom datasets
- 📊 Support for COCO format annotations
- 🎯 Pre-trained model fine-tuning
- 🔍 Built-in inference and visualization
- ⚡ Optimized for GPU training

## Requirements

- Python >= 3.12.11
- NVIDIA GPU with CUDA support (recommended)
- 8GB+ VRAM (for training)

## Installation

This project uses `uv` for dependency management. To set up the environment:

```bash
# Clone the repository
git clone <your-repo-url>
cd YouTube-Train-Custom-RF-DETR

# Install dependencies with uv
uv sync

# Or install manually with pip
pip install rfdetr>=1.4.0.post0 supervision>=0.27.0 transformers==4.52.2 ipykernel ipywidgets
```

## Dataset Structure

Your dataset should follow the COCO format with this structure:

```
dataset/
├── train/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── _annotations.coco.json
├── valid/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── _annotations.coco.json
└── test/
    ├── image1.jpg
    ├── image2.jpg
    └── _annotations.coco.json
```

### COCO Annotations Format

Each `_annotations.coco.json` file should contain:
- `images`: List of image metadata
- `annotations`: Bounding box annotations with category IDs
- `categories`: Class names and IDs

## Usage

### 1. Prepare Your Dataset

Organize your images and annotations in the structure shown above. You can use tools like:
- [Roboflow](https://roboflow.com/) - Export in COCO JSON format
- [CVAT](https://www.cvat.ai/) - Annotation and export
- [LabelImg](https://github.com/heartexlabs/labelImg) - Manual annotation

### 2. Train the Model

Open the Jupyter notebook and follow the training pipeline:

```bash
jupyter notebook train.ipynb
```

The notebook covers:
1. Loading and visualizing your dataset
2. Configuring the RF-DETR model
3. Training with custom hyperparameters
4. Monitoring training progress
5. Evaluating model performance
6. Running inference on test images

### 3. Run Inference

After training, you can use your model for predictions:

```python
from rfdetr import RFDETRMedium
from PIL import Image

# Load your custom model
model = RFDETRMedium(resolution=640)
model.optimize_for_inference()

# Load and predict
image = Image.open("path/to/image.jpg")
detections = model.predict(image, threshold=0.5)
```

## Model Architecture

RF-DETR (Real-time DETR) features:
- Transformer-based detection
- End-to-end training (no NMS needed)
- Multiple resolution support (640, 800, 1024)
- Optimized for inference speed

## Training Tips

- **Batch Size**: Start with batch size 4-8 depending on GPU memory
- **Learning Rate**: Use 1e-4 as a starting point
- **Epochs**: 50-100 epochs typically sufficient for small datasets
- **Augmentation**: Enable data augmentation for better generalization
- **Resolution**: 640x640 balances speed and accuracy

## Visualization

The project uses [Supervision](https://github.com/roboflow/supervision) for visualization:
- Bounding box annotations
- Confidence scores
- Class labels
- Custom color palettes

## Common Issues

### CUDA Out of Memory
- Reduce batch size
- Lower resolution (e.g., 640 instead of 800)
- Use gradient accumulation

### Poor Performance
- Increase training epochs
- Add more training data
- Enable data augmentation
- Check annotation quality

## Dependencies

- `rfdetr` - RF-DETR model implementation
- `supervision` - Computer vision utilities
- `transformers` - Hugging Face transformers library
- `torch` - PyTorch deep learning framework
- `PIL/Pillow` - Image processing

## License

[Add your license here]

## Acknowledgments

- [RF-DETR](https://github.com/Pent/rfdetr) - Original RF-DETR implementation
- [Roboflow Supervision](https://github.com/roboflow/supervision) - CV utilities

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

[Add your contact information or links here]
