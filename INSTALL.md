# Installation Guide

This guide provides step-by-step instructions for installing the PlotQA Chart Data Extraction Pipeline.

## Quick Start

### 1. Clone or Download the Project
```bash
git clone <repository-url>
cd charto
```

### 2. Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

#### Option A: Latest Versions (Recommended)
```bash
pip install -r requirements.txt
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder detectron2==detectron2-0.6+18f6958pt2.8.0cu128
```

#### Option B: Exact Environment Replication
```bash
pip install -r requirements-pinned.txt
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder detectron2==detectron2-0.6+18f6958pt2.8.0cu128
```

### 4. Install Tesseract OCR

#### Windows
1. Download from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install and add to PATH
3. Verify: `tesseract --version`

#### macOS
```bash
brew install tesseract
```

#### Ubuntu/Debian
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-eng
```

#### CentOS/RHEL
```bash
sudo yum install tesseract tesseract-langpack-eng
```

### 5. Download Dataset (Optional)
```bash
# Download PlotQA dataset from:
# https://drive.google.com/drive/folders/15bWhzXxAN4WsXn4p37t_GYABb1F52nQw?usp=sharing

# Extract to: data/plotqa/
```

### 6. Download Pre-trained Models (Required for inference)

The pipeline requires pre-trained VED (Visual Element Detection) model weights to work. Download the following files:

#### Model Files to Download:
- **model_final.pkl** - Main trained model (required)
- **model_iter19999.pkl** - Training checkpoint (optional)
- **net.pbtxt** - Network architecture definition (required)
- **param_init_net.pbtxt** - Parameter initialization (required)

#### Download Links:
1. **Primary Download**: https://drive.google.com/drive/folders/1P00jD-WFg_RBissIPmuWEWct3xoM3mgU?usp=sharing
2. **Alternative Download**: https://drive.google.com/drive/folders/15bWhzXxAN4WsXn4p37t_GYABb1F52nQw?usp=sharing

#### Installation:
```bash
# Download from:
#https://drive.google.com/drive/folders/1P00jD-WFg_RBissIPmuWEWct3xoM3mgU?usp=sharing

# Download and place files in: models/ved/
# The structure should be:
# models/ved/
# ├── model_final.pkl
# ├── model_iter19999.pkl
# ├── net.pbtxt
# └── param_init_net.pbtxt
```

**Note**: The `model_final.pkl` file is essential for running the pipeline. Without it, the scripts will fail with model loading errors.

## Verification

### 1. Check Model Files
```bash
# Verify model files are downloaded
ls models/ved/
# Should show: model_final.pkl, net.pbtxt, param_init_net.pbtxt

# Check file sizes (model_final.pkl should be ~100MB+)
ls -lh models/ved/model_final.pkl
```

### 2. Test Basic Functionality
```bash
# Test basic functionality
python process_chart.py --help

# Test visual values calculation
python calculate_visual_values.py --help
```

### 3. Test with Sample Image
```bash
# Test with sample image (requires both models and data)
python process_chart.py --image data/plotqa/VAL/png/18458.png --model models/ved/model_final.pkl --use-caffe2

# Test visual values calculation
python calculate_visual_values.py --image data/plotqa/VAL/png/18458.png --model models/ved/model_final.pkl --confidence 0.05 --use-caffe2 --debug
```

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

### GPU Support

For GPU acceleration, install PyTorch with CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## Next Steps

1. Read the main [README.md](README.md) for usage instructions
2. Try the example commands in the README
3. Process your own chart images

## Quick Reference

### Essential Download Links
- **Model Weights**: https://drive.google.com/drive/folders/1P00jD-WFg_RBissIPmuWEWct3xoM3mgU?usp=sharing
- **Dataset**: https://drive.google.com/drive/folders/15bWhzXxAN4WsXn4p37t_GYABb1F52nQw?usp=sharing
- **Tesseract (Windows)**: https://github.com/UB-Mannheim/tesseract/wiki

### Required Files Checklist
- [ ] `models/ved/model_final.pkl` (~100MB+)
- [ ] `models/ved/net.pbtxt`
- [ ] `models/ved/param_init_net.pbtxt`
- [ ] Tesseract OCR installed and in PATH
- [ ] Python dependencies installed

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify all dependencies are installed correctly
3. Check that Tesseract is in your PATH
4. Ensure you have the required model files
5. Verify model file sizes and integrity
