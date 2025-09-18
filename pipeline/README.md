# PlotQA Chart Data Extraction Pipeline

A complete end-to-end pipeline for extracting structured data from chart and graph images, based on the PlotQA paper ("PlotQA: Reasoning over Scientific Plots" by Nitesh Methani et al., 2020).

## Overview

This pipeline takes chart images (bar charts, line graphs, scatter plots) as input and outputs structured JSON data containing:
- Chart type and title
- Axis labels and tick values  
- Data series with extracted values
- Legend information

**Key Features:**
- **Modernized Implementation**: Uses Detectron2 (PyTorch-based) instead of original Detectron (Caffe2-based)
- **Original Pipeline Compatibility**: Maintains compatibility with original PlotQA dataset structure and formats
- **Enhanced Output**: Provides both CSV (original format) and JSON (structured) outputs

The pipeline consists of three main stages:
1. **VED (Visual Element Detection)**: Uses Detectron2 Faster R-CNN with FPN to detect chart elements
2. **OCR (Optical Character Recognition)**: Extracts text from detected regions using Tesseract
3. **SIE (Structural Information Extraction)**: Builds structured data from detections and OCR results

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for training)
- Tesseract OCR

### Setup
Run the setup script to install all dependencies:

```bash
chmod +x setup.sh
./setup.sh
```

Or install manually:

```bash
# Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder detectron2==detectron2-0.6+18f6958pt2.8.0cu128

# Computer vision and OCR
pip install opencv-python pillow pyocr pytesseract

# Data processing  
pip install pandas numpy scipy matplotlib tqdm click pyyaml
```

### Dataset Setup

1. Download the PlotQA dataset from: https://drive.google.com/drive/folders/15bWhzXxAN4WsXn4p37t_GYABb1F52nQw?usp=sharing

2. Extract to `~/data/plotqa/` with structure (matches original PlotQA pipeline):
```
~/data/plotqa/
├── TRAIN/          # Training images
├── VAL/            # Validation images  
├── TEST/           # Test images
└── annotations/    # COCO-style annotations
    ├── train_50k_annotations.json      # Main training set (plotqa_train1)
    ├── train_50k_1l_annotations.json   # Alternative training set (plotqa_train2)
    ├── train_1l_end_annotations.json   # Alternative training set (plotqa_train3)
    ├── val_annotations.json            # Validation set
    └── test_annotations.json           # Test set
```

3. (Optional) Download pre-trained weights from: https://drive.google.com/drive/folders/1P00jD-WFg_RBissIPmuWEWct3xoM3mgU?usp=sharing

## Usage

### Quick Start - End-to-End Extraction

Extract data from a single chart image:

```bash
python extract_data.py \
    --image_path chart.png \
    --model_path ./models/ved/model_final.pth \
    --output result.json
```

Process multiple images:

```bash
python extract_data.py \
    --image_dir ./charts/ \
    --model_path ./models/ved/model_final.pth \
    --output_dir ./results/
```

### Training Your Own Model

Train the VED (Visual Element Detection) model (matches original PlotQA pipeline):

```bash
# Train on main 50k dataset (default)
python train_ved.py \
    --data_dir ~/data/plotqa/ \
    --output_dir ./models/ved/ \
    --batch_size 4 \
    --max_iter 40000

# Train on alternative datasets
python train_ved.py \
    --data_dir ~/data/plotqa/ \
    --output_dir ./models/ved/ \
    --train_dataset plotqa_train2 \
    --batch_size 4 \
    --max_iter 40000
```

Training parameters:
- `--train_dataset`: Training dataset (plotqa_train1/plotqa_train2/plotqa_train3, default: plotqa_train1)
- `--batch_size`: Batch size (default: 4)
- `--learning_rate`: Learning rate (default: 0.001)  
- `--max_iter`: Training iterations (default: 40000)
- `--num_workers`: Data loader workers (default: 4)

### Step-by-Step Processing

For more control, run each stage separately:

#### 1. Generate Detections

```bash
# Single image
python generate_detections.py \
    --model_path ./models/ved/model_final.pth \
    --image_path chart.png \
    --output detections.txt

# Batch processing
python generate_detections.py \
    --model_path ./models/ved/model_final.pth \
    --image_dir ./images/ \
    --output ./detections/ \
    --output_format json
```

#### 2. OCR and Structural Extraction

```bash
python ocr_and_sie.py \
    --image_dir ./images/ \
    --detections_dir ./detections/ \
    --output_dir ./extracted/
```

## Output Format

The pipeline outputs JSON with the following structure:

```json
{
  "chart_type": "bar",
  "title": "Sales by Quarter",
  "x_axis": {
    "label": "Quarter",
    "ticks": ["Q1", "Q2", "Q3", "Q4"],
    "values": ["Q1", "Q2", "Q3", "Q4"]
  },
  "y_axis": {
    "label": "Sales ($M)",
    "ticks": ["0", "10", "20", "30"],
    "values": [0, 10, 20, 30]
  },
  "data_series": [
    {
      "name": "Sales",
      "type": "bar",
      "x_values": ["Q1", "Q2", "Q3", "Q4"],
      "y_values": [15.2, 23.1, 18.7, 28.3]
    }
  ],
  "legend": {
    "labels": ["Sales"],
    "position": "right"
  },
  "metadata": {
    "image_path": "chart.png",
    "image_dimensions": [800, 600],
    "total_detections": 25,
    "text_extractions": 12,
    "confidence_threshold": 0.5
  }
}
```

## Supported Chart Types

- **Bar Charts**: Vertical and horizontal bars
- **Line Graphs**: Single and multi-series lines  
- **Scatter Plots**: Point-based data visualization
- **Mixed Charts**: Combinations of the above

## Detection Categories

The VED model detects the following chart elements (matching original PlotQA categories):

| Category | Description |
|----------|-------------|
| `bar` | Bar chart bars |
| `dot_line` | Dotted line elements |
| `legend_label` | Legend item labels |
| `line` | Line chart lines |
| `preview` | Legend preview boxes |
| `title` | Chart title |
| `xlabel` | X-axis label |
| `xticklabel` | X-axis tick labels |
| `ylabel` | Y-axis label |
| `yticklabel` | Y-axis tick labels |

## Configuration

### Model Parameters

- **Confidence Threshold**: Minimum detection confidence (default: 0.5)
- **NMS Threshold**: Non-maximum suppression threshold  
- **Input Size**: Image resizing for detection (default: 800x1333)

### OCR Parameters

- **Tesseract Config**: Character whitelist and recognition mode
- **Text Padding**: Padding around text bounding boxes (default: 5px)
- **Language**: OCR language model (default: English)

## Performance Tips

1. **GPU Usage**: Use CUDA-enabled GPU for faster training and inference
2. **Batch Size**: Adjust based on GPU memory (4-8 for most GPUs)
3. **Image Quality**: Higher resolution images generally produce better results
4. **Preprocessing**: Ensure charts are clearly visible with good contrast

## Troubleshooting

### Common Issues

**"No module named 'detectron2'"**
- Install Detectron2 with CUDA support: `pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html`

**"Tesseract not found"**  
- Install Tesseract OCR and add to PATH
- Ubuntu: `sudo apt-get install tesseract-ocr`
- Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki

**"CUDA out of memory"**
- Reduce batch size in training
- Use smaller input image sizes
- Enable gradient checkpointing

**Poor extraction accuracy**
- Check detection confidence threshold
- Verify image quality and resolution  
- Retrain model with more similar data

### Debug Mode

Enable verbose logging for debugging:

```bash
python extract_data.py --image_path chart.png --model_path model.pth --verbose
```

## File Structure

```
pipeline/
├── setup.sh                 # Installation script
├── train_ved.py             # VED model training
├── generate_detections.py   # Detection generation  
├── ocr_and_sie.py          # OCR and structural extraction
├── extract_data.py         # End-to-end pipeline
└── README.md               # This file

models/
└── ved/                    # Trained model weights

data/
└── plotqa/                 # PlotQA dataset

output/
├── detections/             # Detection results
└── extracted/              # Final extracted data
```

## API Reference

### ChartDataExtractor

Main pipeline class for end-to-end extraction:

```python
from extract_data import ChartDataExtractor

extractor = ChartDataExtractor(
    model_path="./models/ved/model_final.pth",
    confidence_threshold=0.5
)

result = extractor.extract_from_image("chart.png")
```

### PlotQADetector

Visual element detection:

```python
from generate_detections import PlotQADetector

detector = PlotQADetector(model_path="model.pth")
detections = detector.detect_single_image("chart.png")
```

### OCRProcessor  

Text extraction:

```python
from ocr_and_sie import OCRProcessor

ocr = OCRProcessor()
text = ocr.extract_text(image, bbox)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable  
5. Submit a pull request

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Citation

If you use this pipeline in your research, please cite the original PlotQA paper:

```bibtex
@inproceedings{methani2020plotqa,
  title={PlotQA: Reasoning over Scientific Plots},
  author={Methani, Nitesh and Ganguly, Pritha and Khapra, Mitesh M and Kumar, Pratyush},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={1527--1536},
  year={2020}
}
```

## Original PlotQA Pipeline Compatibility

This implementation maintains compatibility with the original PlotQA pipeline while modernizing the underlying technology:

### What's Preserved from Original:
- **Dataset Structure**: Uses exact same directory structure and annotation files (train_50k_annotations.json, etc.)
- **Detection Format**: Outputs detections in original format: `CLASS_LABEL CLASS_CONFIDENCE XMIN YMIN XMAX YMAX`
- **Category Labels**: Uses same visual element categories (axis, tick, tick_label, etc.)
- **CSV Output**: Maintains original semi-structured CSV format for compatibility
- **Pipeline Stages**: Follows same VED → OCR → SIE workflow

### What's Modernized:
- **Detectron2**: Uses PyTorch-based Detectron2 instead of Caffe2-based Detectron
- **Python 3**: Updated from Python 2 to Python 3.8+
- **Enhanced OCR**: Improved OCR processing with better error handling
- **JSON Output**: Added structured JSON output alongside original CSV
- **Better Documentation**: Comprehensive usage examples and API documentation

### Migration from Original:
If you have an existing PlotQA setup, you can:
1. Use the same dataset directory structure
2. Replace the VED training with `train_ved.py` 
3. Replace detection generation with `generate_detections.py`
4. Replace OCR/SIE with `ocr_and_sie.py`
5. Get both original CSV and new JSON outputs

## Testing

Run the test suite to validate pipeline components:

```bash
cd pipeline
python test_pipeline.py
```

## Acknowledgments

- Facebook AI Research for Detectron2
- The PlotQA dataset authors (Methani et al., 2020)
- Original PlotQA pipeline contributors
- Tesseract OCR team
- PyTorch community
