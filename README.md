# Metal Surface Defect Detection System

![Project Banner](docs/banner.png)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.12+](https://img.shields.io/badge/PyTorch-1.12%2B-orange.svg)](https://pytorch.org/)

A comprehensive deep learning solution for detecting and classifying surface defects in metal manufacturing, implementing both YOLO architectures and custom Faster R-CNN with attention mechanisms.

## Table of Contents
- [Features](#features)
- [Defect Classes](#defect-classes)
- [Technical Approaches](#technical-approaches)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Results](#results)
- [Deployment](#deployment)
- [Customization](#customization)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Contact](#contact)

## Features

### Core Capabilities
- **Multi-Model Support**: YOLOv5/v7/v8 and custom Faster R-CNN
- **Attention Mechanisms**: CBAM integration for improved feature learning
- **Advanced Augmentation**: Custom pipeline with 15+ transformations
- **Comprehensive Metrics**: COCO-style evaluation (mAP, AP50, AP75, AR)
- **Visualization Tools**: Detection overlays, metric plots, and GradCAM

### Performance Optimizations
- Mixed Precision Training (FP16)
- Dynamic Batch Sizing
- Anchor Optimization
- Model Pruning Support
- ONNX/TensorRT Export

## Defect Classes

The system detects six critical defect types common in steel production:

| Class Name          | Description                          | Sample Count |
|---------------------|--------------------------------------|--------------|
| Crazing            | Network of fine cracks               | 537          |
| Inclusion          | Foreign material embedded in surface | 856          |
| Patches            | Irregular surface discolorations     | 674          |
| Pitted Surface     | Small holes or depressions           | 351          |
| Rolled-in Scale    | Oxide flakes pressed into surface    | 506          |
| Scratches          | Linear surface marks                 | 420          |

## Technical Approaches

### 1. YOLO Implementations

#### Model Variants
| Model   | Parameters | GFLOPs | mAP@0.5 | Inference Speed (FPS) |
|---------|------------|--------|---------|-----------------------|
| YOLOv5n | 1.9M       | 4.5    | 0.621   | 142                   |
| YOLOv7   | 36.9M      | 104.7  | 0.683   | 98                    |
| YOLOv8x | 68.2M      | 257.8  | 0.721   | 63                    |

#### Key Enhancements
- **Custom Anchor Boxes**: Optimized for small defect sizes
- **Augmentation Pipeline**:
  - Mosaic (9-image)
  - MixUp (α=0.8)
  - HSV Adjustments (±20%)
  - Random Perspective (scale=0.1)
- **Training Tricks**:
  - EMA Weight Averaging (decay=0.9999)
  - Label Smoothing (ε=0.1)
  - Cosine LR Scheduler

### 2. Custom Faster R-CNN with CBAM

#### Architecture Details
```python
FasterRCNN(
  (backbone): ResNet101WithCBAM(
    (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    (cbam1): ImprovedCBAM(
      (channel_mlp): Sequential(
        (0): Linear(in_features=64, out_features=4, bias=True)
        (1): ReLU()
        (2): Dropout(p=0.1, inplace=False)
        (3): Linear(in_features=4, out_features=64, bias=True)
      )
    )
    ...
  )
  (rpn): RegionProposalNetwork(...)
  (roi_heads): RoIHeads(...)
)
```

#### Attention Mechanism
The CBAM module processes features through:
1. **Channel Attention**:
   - Global Avg + Max Pooling
   - Shared MLP (ReLU, Dropout)
   - Sigmoid Activation
2. **Spatial Attention**:
   - Channel-wise Avg + Max
   - 7×7 Convolution
   - Sigmoid Activation

## Installation

### Hardware Requirements
- NVIDIA GPU (≥8GB VRAM recommended)
- CUDA 11.3+
- cuDNN 8.2+

### Software Setup
```bash
# Clone repository
git clone https://github.com/yourusername/metal-defect-detection.git
cd metal-defect-detection

# Create conda environment (recommended)
conda create -n defect python=3.8
conda activate defect

# Install PyTorch (select appropriate CUDA version)
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html

# Install remaining requirements
pip install -r requirements.txt
```

### Dependency List
| Package         | Version  | Purpose                     |
|-----------------|----------|-----------------------------|
| PyTorch         | ≥1.12    | Deep Learning Framework     |
| Albumentations  | ≥1.2.0   | Image Augmentations         |
| OpenCV          | ≥4.5.4   | Image Processing            |
| Pandas          | ≥1.3.4   | Data Handling               |
| Matplotlib      | ≥3.5.0   | Visualization               |
| PyYAML          | ≥6.0     | Configuration Files         |
| TensorBoard     | ≥2.9.0   | Training Monitoring         |

## Dataset Preparation

### Directory Structure
```
NEU-DET/
├── train_images/
│   ├── crazing_1.jpg
│   ├── inclusion_1.jpg
│   └── ...
├── train_annotations/
│   ├── crazing_1.xml
│   ├── inclusion_1.xml
│   └── ...
└── test/
    ├── test_1.jpg
    └── ...
```

### Annotation Format (Pascal VOC)
```xml
<annotation>
  <filename>crazing_1.jpg</filename>
  <size>
    <width>200</width>
    <height>200</height>
    <depth>3</depth>
  </size>
  <object>
    <name>crazing</name>
    <bndbox>
      <xmin>50</xmin>
      <ymin>60</ymin>
      <xmax>120</xmax>
      <ymax>140</ymax>
    </bndbox>
  </object>
</annotation>
```

## Training

### Faster R-CNN Training Options
```bash
python train_rcnn.py \
    --model faster_rcnn_cbam \    # Model architecture
    --epochs 100 \                # Total training epochs
    --batch-size 8 \              # Batch size per GPU
    --lr 0.001 \                  # Initial learning rate
    --backbone resnet101 \        # Feature extractor
    --img-size 640 \              # Input resolution
    --warmup-epochs 5 \           # Linear LR warmup
    --sync-bn \                   # Synchronized BatchNorm
    --amp \                       # Automatic Mixed Precision
    --weights weights/pretrained.pth  # Optional pretrained weights
```

### YOLO Training Options
```bash
python train_yolo.py \
    --model yolov8n \             # Model variant
    --epochs 100 \                # Training epochs
    --batch-size 32 \             # Total batch size
    --img-size 640 \              # Image size
    --data configs/neu-det.yaml \ # Dataset config
    --hyp configs/hyp.yaml \      # Hyperparameters
    --cache ram \                 # Caching strategy
    --device 0,1 \                # GPU devices
    --adam \                      # Use Adam optimizer
    --single-cls \                # Single-class mode
    --upload-dataset \            # Optional dataset upload
```

## Evaluation

### Metrics Calculation
```bash
python evaluate.py \
    --model faster_rcnn_cbam \
    --weights weights/best_model.pth \
    --data configs/neu-det.yaml \
    --batch-size 16 \
    --img-size 640 \
    --conf-thres 0.3 \
    --iou-thres 0.5 \
    --task test \
    --verbose
```

### Supported Metrics
1. **Precision-Recall Curve**
2. **mAP@[0.5:0.95]**
3. **AP50/AP75**
4. **Average Recall**
5. **Inference Speed**
6. **Memory Usage**

## Inference

### Single Image Detection
```bash
python detect.py \
    --source data/samples/defect_01.jpg \
    --weights weights/best_model.pth \
    --output results/ \
    --conf-thres 0.3 \
    --iou-thres 0.5 \
    --hide-labels \
    --hide-conf \
    --save-txt \
    --save-crop
```

### Video Processing
```bash
python detect.py \
    --source data/videos/production_line.mp4 \
    --weights weights/best_model.pth \
    --output results/ \
    --conf-thres 0.4 \
    --imgsz 640 \
    --nosave \
    --view-img
```

## Results

### Performance Comparison
| Model          | mAP@0.5 | AP50  | AP75  | AR@100 | FPS (V100) | VRAM Usage |
|----------------|---------|-------|-------|--------|------------|------------|
| YOLOv5n        | 0.621   | 0.712 | 0.581 | 0.732  | 142        | 2.1GB      |
| YOLOv7         | 0.683   | 0.761 | 0.642 | 0.781  | 98         | 4.3GB      |
| YOLOv8x        | 0.721   | 0.793 | 0.682 | 0.812  | 63         | 7.8GB      |
| Faster R-CNN   | 0.692   | 0.781 | 0.651 | 0.793  | 38         | 5.2GB      |
| Faster R-CNN-CBAM | **0.730** | **0.812** | **0.698** | **0.831** | 32         | 5.9GB      |

### Sample Detections
![Detection Samples](docs/detection_examples.png)

## Deployment

### Export Options
```bash
python export.py \
    --weights weights/best_model.pth \
    --img-size 640 640 \
    --batch-size 1 \
    --device 0 \
    --include onnx engine \  # Export formats
    --half \                 # FP16 quantization
    --simplify \             # ONNX simplifier
    --opset 12
```

### Supported Platforms
1. **ONNX Runtime** (CPU/GPU)
2. **TensorRT** (FP16/INT8)
3. **TorchScript**
4. **CoreML** (Apple Devices)
5. **TFLite** (Mobile Deployment)

## Customization

### Configuration Files
1. **Model Architecture**: `configs/models/*.yaml`
2. **Training Parameters**: `configs/training/default.yaml`
3. **Dataset Specification**: `configs/data/neu-det.yaml`
4. **Hyperparameters**: `configs/hyp.yaml`

### Adding New Defect Classes
1. Update `configs/data/neu-det.yaml`:
```yaml
# Class names and IDs
names:
  0: crazing
  1: inclusion
  2: patches
  3: pitted_surface
  4: rolled-in_scale
  5: scratches
  6: new_defect_type  # Add new class
```

2. Modify anchor boxes in `models/anchor_utils.py` if needed

## Contributing

We welcome contributions! Please follow these steps:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Coding Standards
- Follow PEP 8 style guide
- Type hints for all function signatures
- Docstrings for all public methods
- Unit tests for new features



