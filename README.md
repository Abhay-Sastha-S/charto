# PlotQA Chart Data Extraction Pipeline

A complete end-to-end pipeline for extracting structured data from chart and graph images, based on the PlotQA paper ("PlotQA: Reasoning over Scientific Plots" by Nitesh Methani et al., 2020).

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder detectron2==detectron2-0.6+18f6958pt2.8.0cu128

# Download model weights (see Installation section)
# Then run:
python process_chart.py --image your_chart.png --model models/ved/model_final.pkl --use-caffe2 --confidence 0.05
```

## Overview

This pipeline takes chart images (bar charts, line graphs, scatter plots) as input and outputs structured JSON data containing:
- Chart type and title
- Axis labels and tick values  
- Data series with extracted values
- Legend information
- **Visual element values** (heights, lengths, positions)

**Key Features:**
- **Caffe2 Compatibility**: Full compatibility with original PlotQA Caffe2 models through specialized detectors
- **Dual Architecture Support**: Both Caffe2-compatible and exact Caffe2 architecture replication
- **Enhanced Output**: Provides both CSV (original format) and JSON (structured) outputs
- **Visual Value Calculation**: Calculates numerical values for visual elements (bars, lines, dots)
- **Debug Mode**: Comprehensive debugging and visualization tools

The pipeline consists of four main stages:
1. **VED (Visual Element Detection)**: Uses specialized Caffe2-compatible detectors for original PlotQA models
2. **OCR (Optical Character Recognition)**: Extracts text from detected regions using Tesseract
3. **SIE (Structural Information Extraction)**: Builds structured data from detections and OCR results
4. **Visual Value Calculation**: Computes numerical values for visual elements using coordinate-based scaling

## Installation

### Prerequisites
- Python 3.8+ (tested with Python 3.12.11)
- CUDA-compatible GPU (recommended for inference)
- Tesseract OCR

### Installation Options

#### Option 1: Latest Versions (Recommended)
```bash
pip install -r requirements.txt
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder detectron2==detectron2-0.6+18f6958pt2.8.0cu128
```

#### Option 2: Exact Environment Replication
```bash
pip install -r requirements-pinned.txt
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder detectron2==detectron2-0.6+18f6958pt2.8.0cu128
```

### Download Required Files

#### Model Weights (Required)
Download from: https://drive.google.com/drive/folders/1P00jD-WFg_RBissIPmuWEWct3xoM3mgU?usp=sharing

Required files:
- `model_final.pkl` (~100MB+)
- `net.pbtxt`
- `param_init_net.pbtxt`

Place in: `models/ved/`

#### Dataset (Optional - for testing)
Download from: https://drive.google.com/drive/folders/15bWhzXxAN4WsXn4p37t_GYABb1F52nQw?usp=sharing

Place in: `data/plotqa/`

## Usage

### Basic Usage

```bash
# Basic chart processing
python process_chart.py --image your_chart.png --model models/ved/model_final.pkl --use-caffe2

# Visual values calculation with debug
python calculate_visual_values.py --image your_chart.png --model models/ved/model_final.pkl --confidence 0.05 --use-caffe2 --debug

# With KMP fix for Windows
$env:KMP_DUPLICATE_LIB_OK="TRUE"; python calculate_visual_values.py --image your_chart.png --model models/ved/model_final.pkl --confidence 0.05 --use-caffe2 --debug
```

### Test with Sample Data

```bash
# Test with PlotQA dataset (if downloaded)
python process_chart.py --image data/plotqa/VAL/png/18458.png --model models/ved/model_final.pkl --use-caffe2

# Test visual values calculation
python calculate_visual_values.py --image data/plotqa/VAL/png/18458.png --model models/ved/model_final.pkl --confidence 0.05 --use-caffe2 --debug
```

### Advanced Usage

#### Command Line Arguments

**Common Arguments:**
- `--image`: Path to input chart image (required)
- `--model`: Path to trained VED model weights (required)
- `--output`: Output directory (default: "results")
- `--confidence`: Detection confidence threshold (default: 0.3)
- `--debug`: Enable debug mode for detailed analysis
- `--verbose`: Enable verbose logging

**Detector Selection:**
- `--use-caffe2`: Use Caffe2-compatible detector (recommended for original models)
- `--use-exact-caffe2`: Use exact Caffe2 architecture replication (most compatible)
- `--use-detectron2`: Use Detectron2 detector (modern PyTorch-based)

#### Output Files

**Standard Output:**
- `{image_name}.csv` - Original PlotQA format
- `{image_name}.json` - Structured JSON format
- `{image_name}_metadata.json` - Processing metadata

**Visual Values Output (calculate_visual_values.py):**
- `{image_name}_visual_values.json` - Visual element values with coordinates and scaling

**Debug Mode Output:**
- `temp/detection_summary.json` - Detailed detection information
- `temp/detections_visualized.png` - Image with bounding boxes
- `temp/ocr_debug.json` - OCR results with extended bounding boxes
- `temp/ocr_with_extended_bboxes.png` - OCR visualization
- `temp/debug_crops/` - Individual cropped regions

### Caffe2 Detector Architecture Options

The pipeline supports multiple detector architectures for maximum compatibility:

#### Caffe2-Compatible Detector (`caffe2_compatible_detector.py`)
- **Purpose**: Provides compatibility with original PlotQA Caffe2 models
- **Features**: 
  - Loads original `.pkl` model files
  - Maintains original detection format
  - Optimized for PlotQA dataset models
- **Usage**: `--use-caffe2` flag

#### Exact Caffe2 Architecture Replication (`exact_caffe2_detector.py`)
- **Purpose**: Exact replication of original Caffe2 architecture
- **Features**:
  - Pixel-perfect compatibility with original models
  - Handles edge cases in original implementation
  - Best for production use with original PlotQA models
- **Usage**: `--use-exact-caffe2` flag (recommended)

#### Detectron2 Detector (`generate_detections.py`)
- **Purpose**: Modern PyTorch-based detection
- **Features**:
  - Faster inference
  - Better GPU utilization
  - Modern PyTorch ecosystem integration
- **Usage**: `--use-detectron2` flag or omit Caffe2 flags

## Caffe2 Detector Architecture Details

### Why Caffe2 Compatibility?

The original PlotQA models were trained using Caffe2, and while Detectron2 provides modern PyTorch-based detection, there can be subtle differences in:
- Model loading and initialization
- Preprocessing pipelines
- Post-processing steps
- Numerical precision

Our Caffe2-compatible detectors ensure pixel-perfect compatibility with original PlotQA models.

### Architecture Comparison

| Feature | Caffe2-Compatible | Exact Caffe2 | Detectron2 |
|---------|------------------|--------------|------------|
| **Model Loading** | Original `.pkl` files | Original `.pkl` files | Converted `.pth` files |
| **Preprocessing** | Original pipeline | Exact original pipeline | Modern pipeline |
| **Post-processing** | Original NMS | Exact original NMS | Modern NMS |
| **Compatibility** | High | Perfect | Good |
| **Performance** | Good | Good | Best |
| **Maintenance** | Medium | High | Low |

### Technical Implementation

#### Caffe2-Compatible Detector (`caffe2_compatible_detector.py`)
```python
class Caffe2CompatibleDetector:
    def __init__(self, model_path, confidence_threshold=0.1):
        # Load original Caffe2 model
        self.model = self._load_caffe2_model(model_path)
        # Configure preprocessing to match original
        self.preprocessor = self._setup_preprocessing()
        # Configure post-processing
        self.postprocessor = self._setup_postprocessing()
    
    def detect_single_image(self, image_path):
        # Resize to 650x650 (original PlotQA size)
        resized_image = self._resize_image(image_path, (650, 650))
        # Run inference
        detections = self._run_inference(resized_image)
        # Apply confidence filtering
        filtered_detections = self._filter_detections(detections)
        return filtered_detections, resized_image, original_dimensions
```

#### Exact Caffe2 Architecture Replication (`exact_caffe2_detector.py`)
```python
class ExactCaffe2Detector:
    def __init__(self, model_path, confidence_threshold=0.1):
        # Load with exact Caffe2 architecture
        self.model = self._load_exact_caffe2_model(model_path)
        # Replicate exact preprocessing steps
        self.preprocessor = self._replicate_caffe2_preprocessing()
        # Replicate exact post-processing
        self.postprocessor = self._replicate_caffe2_postprocessing()
    
    def detect_single_image(self, image_path):
        # Exact same preprocessing as original Caffe2
        processed_image = self._exact_caffe2_preprocess(image_path)
        # Run with exact same inference pipeline
        detections = self._exact_caffe2_inference(processed_image)
        # Apply exact same post-processing
        final_detections = self._exact_caffe2_postprocess(detections)
        return final_detections, processed_image, original_dimensions
```

### Model File Compatibility

The pipeline supports both original Caffe2 model formats:

- **`.pkl` files**: Original Caffe2 pickle files (recommended)
- **`.pth` files**: PyTorch state dict files (for Detectron2)

For best compatibility with original PlotQA models, use `.pkl` files with Caffe2-compatible detectors.

### Advanced Usage

#### Command Line Arguments

**Common Arguments:**
- `--image`: Path to input chart image (required)
- `--model`: Path to trained VED model weights (required)
- `--output`: Output directory (default: "results")
- `--confidence`: Detection confidence threshold (default: 0.3)
- `--debug`: Enable debug mode for detailed analysis
- `--verbose`: Enable verbose logging

**Detector Selection:**
- `--use-caffe2`: Use Caffe2-compatible detector (recommended for original models)
- `--use-exact-caffe2`: Use exact Caffe2 architecture replication (most compatible)
- `--use-detectron2`: Use Detectron2 detector (modern PyTorch-based)

#### Debug Mode Features

When using `--debug` flag, the pipeline generates additional files in a `temp/` directory:

- `detection_summary.json` - Detailed detection information with coordinates
- `detections_visualized.png` - Image with bounding boxes overlaid
- `ocr_debug.json` - OCR results with extended bounding boxes
- `ocr_with_extended_bboxes.png` - Visualization of OCR processing
- `debug_crops/` - Individual cropped regions for each detected element

#### Step-by-Step Processing

For advanced users who need more control, individual components can be used separately:

```bash
# 1. Generate detections only
python generate_detections.py \
    --model_path ./models/ved/model_final.pkl \
    --image_path chart.png \
    --output detections.txt

# 2. OCR and structural extraction
python ocr_and_sie.py \
    --image_dir ./images/ \
    --detections_dir ./detections/ \
    --output_dir ./extracted/
```

## Output Formats

### Standard JSON Output (`process_chart.py`)

The pipeline outputs structured JSON with the following format:

```json
{
  "chart_type": "bar",
  "title": "Sales by Quarter",
  "x_axis": {
    "label": "Quarter",
    "type": "categorical"
  },
  "y_axis": {
    "label": "Sales ($M)",
    "type": "numeric"
  },
  "data_series": [
    {
      "name": "Sales",
      "type": "bar",
      "data": [
        {"x": "Q1", "y": 15.2},
        {"x": "Q2", "y": 23.1},
        {"x": "Q3", "y": 18.7},
        {"x": "Q4", "y": 28.3}
      ]
    }
  ],
  "metadata": {
    "source_image": "chart.png",
    "extraction_method": "PlotQA Pipeline",
    "total_series": 1
  }
}
```

### Visual Values Output (`calculate_visual_values.py`)

When using `calculate_visual_values.py`, additional visual value data is included:

```json
{
  "chart_orientation": {
    "isHbar": false,
    "isSinglePlot": true
  },
  "visual_elements_with_values": [
    {
      "pred_class": "bar",
      "bbox": [100, 200, 150, 400],
      "confidence": 0.95,
      "x_value": "Q1",
      "y_value": 15.2,
      "ocr_text": null
    },
    {
      "pred_class": "xticklabel",
      "bbox": [120, 450, 140, 470],
      "confidence": 0.88,
      "ocr_text": "Q1"
    }
  ],
  "calculation_method": "find_visual_values_algorithm"
}
```

### CSV Output (Original PlotQA Format)

The pipeline also generates CSV files in the original PlotQA format for compatibility:

```csv
title,xlabel,ylabel,Sales
Sales by Quarter,Quarter,Sales ($M),15.2
Sales by Quarter,Quarter,Sales ($M),23.1
Sales by Quarter,Quarter,Sales ($M),18.7
Sales by Quarter,Quarter,Sales ($M),28.3
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

Enable verbose logging and detailed analysis:

```bash
# Basic debug mode
python process_chart.py --image chart.png --model models/ved/model_final.pkl --use-caffe2 --debug

# Visual values with debug
python calculate_visual_values.py --image chart.png --model models/ved/model_final.pkl --use-caffe2 --debug --verbose
```

## File Structure

```
charto/                                # Main project directory
├── process_chart.py                   # Main chart processing script
├── calculate_visual_values.py         # Visual values calculation script
├── caffe2_compatible_detector.py      # Caffe2-compatible detector
├── exact_caffe2_detector.py           # Exact Caffe2 architecture replication
├── generate_detections.py             # Detectron2-based detection
├── ocr_and_sie.py                    # OCR and structural extraction
├── utils.py                          # Utility functions
├── bbox_conversion.py                # Bounding box conversion utilities
├── upscale_boxes.py                  # Box upscaling functionality
├── requirements.txt                  # Complete dependencies list (latest versions)
├── requirements-pinned.txt           # Exact versions from working environment
├── setup.sh                          # Installation script
├── README.md                         # This file
├── models/                           # Trained model weights
│   └── ved/
│       ├── model_final.pkl           # Caffe2 model file
│       ├── model_iter19999.pkl       # Training checkpoint
│       ├── net.pbtxt                 # Network architecture
│       └── param_init_net.pbtxt      # Parameter initialization
├── data/                             # PlotQA dataset
│   └── plotqa/
│       ├── TRAIN/                    # Training images
│       ├── VAL/                      # Validation images
│       ├── TEST/                     # Test images
│       └── annotations/              # COCO-style annotations
├── bar_results/                      # Example results
└── misc/                             # Additional utilities and documentation
    ├── codes/                        # Legacy code files
    ├── docs/                         # Documentation
    └── dataset_catalog.py            # Dataset utilities
```

## API Reference

### PlotQAProcessor

Main pipeline class for chart processing:

```python
from process_chart import PlotQAProcessor

# Initialize with Caffe2-compatible detector
processor = PlotQAProcessor(
    model_path="./models/ved/model_final.pkl",
    confidence_threshold=0.3,
    use_caffe2=True,
    use_exact_caffe2=True,
    debug=False
)

# Process a single image
results = processor.process_image("chart.png", "output_dir")
```

### VisualValuesCalculator

Extended processor with visual value calculations:

```python
from calculate_visual_values import VisualValuesCalculator

# Initialize calculator
calculator = VisualValuesCalculator(
    model_path="./models/ved/model_final.pkl",
    confidence_threshold=0.3,
    use_caffe2=True,
    use_exact_caffe2=True,
    debug=True  # Enable for full visual value calculation
)

# Process with visual values
results = calculator.process_with_values("chart.png", "output_dir")
```

### Caffe2-Compatible Detector

Direct access to Caffe2-compatible detection:

```python
from caffe2_compatible_detector import Caffe2CompatibleDetector

detector = Caffe2CompatibleDetector(
    model_path="./models/ved/model_final.pkl",
    confidence_threshold=0.3
)

detections, resized_image, original_dimensions = detector.detect_single_image("chart.png")
```

### Exact Caffe2 Detector

Exact Caffe2 architecture replication:

```python
from exact_caffe2_detector import ExactCaffe2Detector

detector = ExactCaffe2Detector(
    model_path="./models/ved/model_final.pkl",
    confidence_threshold=0.3
)

detections, resized_image, original_dimensions = detector.detect_single_image("chart.png")
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
2. Use the same `.pkl` model files with Caffe2-compatible detectors
3. Replace detection generation with `process_chart.py` or `calculate_visual_values.py`
4. Get both original CSV and new JSON outputs
5. Access visual element values with coordinate-based calculations

## Troubleshooting

### Common Issues

1. **"No module named 'detectron2'"**
   - Install Detectron2: `pip install --extra-index-url https://miropsota.github.io/torch_packages_builder detectron2==detectron2-0.6+18f6958pt2.8.0cu128`

2. **"Model file not found" or "Could not load model"**
   - Ensure `model_final.pkl` is in `models/ved/` directory
   - Check file size: `ls -lh models/ved/model_final.pkl` (should be ~100MB+)
   - Re-download from: https://drive.google.com/drive/folders/1P00jD-WFg_RBissIPmuWEWct3xoM3mgU?usp=sharing

3. **"Tesseract not found"**
   - Install Tesseract OCR and add to PATH
   - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki

4. **"CUDA out of memory"**
   - Use CPU version: Install PyTorch without CUDA
   - Reduce batch size in processing

5. **KMP_DUPLICATE_LIB_OK error (Windows)**
   ```bash
   $env:KMP_DUPLICATE_LIB_OK="TRUE"
   python process_chart.py --image chart.png --model models/ved/model_final.pkl --use-caffe2
   ```

6. **"No visual elements detected"**
   - Check image quality and resolution
   - Try different confidence thresholds: `--confidence 0.1` or `--confidence 0.05`
   - Ensure image is a chart/graph (not a photo or other image type)

## Testing

Test the pipeline with sample images:

```bash
# Test basic processing
python process_chart.py --image test_chart.png --model models/ved/model_final.pkl --use-caffe2

# Test visual values calculation
python calculate_visual_values.py --image test_chart.png --model models/ved/model_final.pkl --use-caffe2 --debug

# Test different detector architectures
python process_chart.py --image test_chart.png --model models/ved/model_final.pkl --use-exact-caffe2
python process_chart.py --image test_chart.png --model models/ved/model_final.pkl --use-detectron2

#if error

$env:KMP_DUPLICATE_LIB_OK="TRUE"; python calculate_visual_values.py --image test_chart.png --model models/ved/model_final.pkl --confidence 0.05 --use-caffe2 --debug --verbose 

$env:KMP_DUPLICATE_LIB_OK="TRUE"; python process_chart.py --image test_chart.png --model models/ved/model_final.pkl --confidence 0.05 --use-caffe2 --debug --verbose 

```

## Quick Reference

### Essential Commands
```bash
# Install dependencies
pip install -r requirements.txt
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder detectron2==detectron2-0.6+18f6958pt2.8.0cu128

# Basic processing
python process_chart.py --image chart.png --model models/ved/model_final.pkl --use-caffe2

# Visual values with debug
python calculate_visual_values.py --image chart.png --model models/ved/model_final.pkl --use-caffe2 --debug

# Windows KMP fix
$env:KMP_DUPLICATE_LIB_OK="TRUE"; python process_chart.py --image chart.png --model models/ved/model_final.pkl --use-caffe2
```

### Download Links
- **Model Weights**: https://drive.google.com/drive/folders/1P00jD-WFg_RBissIPmuWEWct3xoM3mgU?usp=sharing
- **Dataset**: https://drive.google.com/drive/folders/15bWhzXxAN4WsXn4p37t_GYABb1F52nQw?usp=sharing
- **Tesseract (Windows)**: https://github.com/UB-Mannheim/tesseract/wiki

## Acknowledgments

- Facebook AI Research for Detectron2
- The PlotQA dataset authors (Methani et al., 2020)
- Original PlotQA pipeline contributors
- Tesseract OCR team
- PyTorch community
