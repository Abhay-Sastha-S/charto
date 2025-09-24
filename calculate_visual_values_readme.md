# Visual Values Calculator - Detailed Documentation

## Overview

The `calculate_visual_values.py` script is an advanced extension of the PlotQA pipeline that calculates precise numerical values for visual elements in chart images. It integrates the standard PlotQA detection pipeline with sophisticated visual value calculation algorithms to extract quantitative data from bar charts, line graphs, and scatter plots.

## Table of Contents

1. [Introduction](#introduction)
2. [Architecture and Components](#architecture-and-components)
3. [Detailed Processing Pipeline](#detailed-processing-pipeline)
4. [Chart Type-Specific Processing](#chart-type-specific-processing)
5. [Visual Value Calculation Methods](#visual-value-calculation-methods)
6. [Coordinate System and Scaling](#coordinate-system-and-scaling)
7. [OCR Text Processing](#ocr-text-processing)
8. [Error Handling and Edge Cases](#error-handling-and-edge-cases)
9. [Output Formats](#output-formats)
10. [Usage Examples](#usage-examples)
11. [Troubleshooting](#troubleshooting)

## Introduction

The Visual Values Calculator extends the standard PlotQA pipeline by adding coordinate-based scaling and numerical value extraction for visual elements. Unlike the basic pipeline that only extracts text and structure, this calculator determines the actual numerical values represented by visual elements like bar heights, line positions, and dot coordinates.

### Key Features

- **Coordinate-Based Scaling**: Uses pixel-to-value mapping based on axis tick labels
- **Multi-Chart Support**: Handles bar charts, line graphs, and scatter plots
- **Orientation Detection**: Automatically detects horizontal vs vertical chart orientations
- **Precision Calculation**: Provides accurate numerical values for visual elements
- **Debug Visualization**: Generates detailed debug information for analysis

## Architecture and Components

### Core Classes

#### `VisualValuesCalculator`
The main class that orchestrates the entire visual value calculation process.

```python
class VisualValuesCalculator:
    def __init__(self, model_path, confidence_threshold=0.1, use_caffe2=True, use_exact_caffe2=True, debug=False):
        # Initialize with PlotQA processor
        self.processor = PlotQAProcessor(...)
        self.debug = debug
```

**Key Methods:**
- `process_with_values()`: Main processing pipeline
- `_calculate_visual_values()`: Core visual value calculation
- `_find_visual_values()`: Adapted algorithm from original find_visual_values.py
- `_find_first_coord()`: Associates visual elements with axis labels
- `_handle_negative_visuals()`: Handles negative value visualizations

### Integration with PlotQA Pipeline

The calculator integrates with the standard PlotQA pipeline through:

1. **VED (Visual Element Detection)**: Uses Caffe2-compatible or exact Caffe2 detectors
2. **OCR (Optical Character Recognition)**: Extracts text from detected regions
3. **SIE (Structural Information Extraction)**: Builds structured data
4. **Visual Value Calculation**: **NEW** - Calculates numerical values for visual elements

## Detailed Processing Pipeline

### Stage 1: Image Preprocessing and Detection

#### 1.1 Image Loading and Resizing
```python
# Image is loaded and resized to 650x650 (standard PlotQA size)
resized_image = self._resize_image(image_path, (650, 650))
original_dimensions = get_original_dimensions(image_path)
```

#### 1.2 Visual Element Detection
The VED model detects the following elements:
- **Visual Elements**: `bar`, `line`, `dot_line`
- **Text Elements**: `title`, `xlabel`, `ylabel`, `xticklabel`, `yticklabel`
- **Legend Elements**: `legend_label`, `preview`

#### 1.3 Detection Filtering
```python
# Filter detections by confidence threshold
filtered_detections = [d for d in detections if d[1] >= confidence_threshold]
```

### Stage 2: OCR and Text Extraction

#### 2.1 Text Region Processing
For each detected text element:
1. **Bounding Box Extension**: Extends bounding boxes by 5-10 pixels for better OCR
2. **Image Cropping**: Crops text regions from the main image
3. **OCR Processing**: Uses Tesseract OCR with optimized settings
4. **Text Cleaning**: Applies OCR error corrections

#### 2.2 OCR Configuration
```python
# Tesseract configuration for chart text
config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,-+eE'
```

### Stage 3: Chart Orientation Detection

#### 3.1 Orientation Analysis
The system automatically detects chart orientation:

```python
# Determine if chart is horizontal bar chart
isHbar = self._detect_horizontal_orientation(detections)

# Determine if chart is single plot
isSinglePlot = self._detect_single_plot(detections)
```

#### 3.2 Axis Label Association
- **Vertical Charts**: X-axis labels (categories), Y-axis labels (values)
- **Horizontal Charts**: Y-axis labels (categories), X-axis labels (values)

### Stage 4: Visual Value Calculation

This is the core innovation of the calculator. The process involves:

#### 4.1 Scale Calculation
```python
# Find two valid tick labels for scale calculation
valid_tick_pair = self._find_valid_tick_pair(ticklabels)

# Calculate pixel-to-value scale
pixel_difference = abs(c_y2 - c_y1)  # or abs(c_x2 - c_x1) for horizontal
value_difference = abs(float(tick1_text) - float(tick2_text))
scale = value_difference / pixel_difference
```

#### 4.2 Visual Element Processing
For each visual element (bar, line, dot):

```python
# Calculate the value based on position and scale
if element_type == "bar":
    if isHbar:
        compare_with = abs(bbox[2] - start)  # bar length
    else:
        compare_with = abs(bbox[1] - start)  # bar height
elif element_type == "dot_line":
    cx, cy = find_center(bbox)
    compare_with = abs(cy - start)
elif element_type == "line":
    slope = find_slope(image, bbox)
    # Handle line slope for value calculation
    if slope == "positive":
        compare_with = abs(y2 - start)
    else:
        compare_with = abs(y1 - start)

value = compare_with * scale
```

## Chart Type-Specific Processing

### Bar Charts

#### Vertical Bar Charts
**Detection Process:**
1. **Bar Detection**: Identifies rectangular bar elements
2. **X-axis Association**: Associates bars with category labels (xticklabels)
3. **Y-axis Scaling**: Uses yticklabels to establish value scale
4. **Height Calculation**: Measures bar height in pixels and converts to values

**Value Calculation:**
```python
# For vertical bars
bar_height_pixels = abs(bar_bbox[1] - y_axis_start)
bar_value = bar_height_pixels * scale_factor
```

#### Horizontal Bar Charts
**Detection Process:**
1. **Bar Detection**: Identifies horizontal rectangular elements
2. **Y-axis Association**: Associates bars with category labels (yticklabels)
3. **X-axis Scaling**: Uses xticklabels to establish value scale
4. **Length Calculation**: Measures bar length in pixels and converts to values

**Value Calculation:**
```python
# For horizontal bars
bar_length_pixels = abs(bar_bbox[2] - x_axis_start)
bar_value = bar_length_pixels * scale_factor
```

### Line Graphs

#### Line Detection and Processing
**Detection Process:**
1. **Line Detection**: Identifies line segments in the image
2. **Point Association**: Associates line endpoints with axis values
3. **Slope Analysis**: Determines line direction (positive/negative slope)
4. **Value Calculation**: Calculates values for line endpoints

**Slope Detection:**
```python
def find_slope(image, bbox):
    x1, y1, x2, y2 = bbox
    # Sample points along the line to determine slope
    if y2 < y1:  # Line goes up (positive slope)
        return "positive"
    else:  # Line goes down (negative slope)
        return "negative"
```

**Value Calculation for Lines:**
```python
# For line endpoints
if slope == "positive":
    endpoint_value = abs(y2 - start) * scale
else:
    endpoint_value = abs(y1 - start) * scale
```

#### Multi-Series Line Graphs
**Processing Steps:**
1. **Line Separation**: Distinguishes between different line series
2. **Color Analysis**: Uses color information to group related line segments
3. **Series Association**: Associates lines with legend labels
4. **Value Calculation**: Calculates values for each line series independently

### Scatter Plots (Dot-Line Elements)

#### Dot Detection and Processing
**Detection Process:**
1. **Dot Detection**: Identifies circular or point-like elements
2. **Coordinate Mapping**: Maps dot positions to axis values
3. **Value Calculation**: Calculates both X and Y values for each dot

**Coordinate Calculation:**
```python
# For scatter plot dots
dot_center_x, dot_center_y = find_center(dot_bbox)
x_value = abs(dot_center_x - x_axis_start) * x_scale
y_value = abs(dot_center_y - y_axis_start) * y_scale
```

#### Multi-Series Scatter Plots
**Processing Steps:**
1. **Dot Grouping**: Groups dots by color or proximity
2. **Series Identification**: Associates dot groups with legend labels
3. **Value Calculation**: Calculates values for each dot group

## Visual Value Calculation Methods

### Method 1: Coordinate-Based Scaling

This is the primary method used by the calculator:

#### Scale Factor Calculation
```python
def calculate_scale_factor(tick1, tick2, isHbar):
    # Get tick label positions
    c_x1, c_y1 = find_center(tick1['bbox'])
    c_x2, c_y2 = find_center(tick2['bbox'])
    
    # Calculate pixel difference
    if isHbar:
        pixel_diff = abs(c_x2 - c_x1)
    else:
        pixel_diff = abs(c_y2 - c_y1)
    
    # Calculate value difference
    tick1_value = float(clean_ocr_text(tick1['ocr_text']))
    tick2_value = float(clean_ocr_text(tick2['ocr_text']))
    value_diff = abs(tick2_value - tick1_value)
    
    # Calculate scale factor
    scale = value_diff / pixel_diff
    return scale
```

#### Visual Element Value Calculation
```python
def calculate_visual_value(element, scale, start_point, isHbar):
    bbox = element['bbox']
    
    if element['pred_class'] == 'bar':
        if isHbar:
            pixel_value = abs(bbox[2] - start_point)  # bar length
        else:
            pixel_value = abs(bbox[1] - start_point)  # bar height
    elif element['pred_class'] == 'dot_line':
        cx, cy = find_center(bbox)
        if isHbar:
            pixel_value = abs(cx - start_point)
        else:
            pixel_value = abs(cy - start_point)
    elif element['pred_class'] == 'line':
        # Handle line slope for proper value calculation
        slope = find_slope(image, bbox)
        if slope == "positive":
            pixel_value = abs(bbox[3] - start_point)  # y2
        else:
            pixel_value = abs(bbox[1] - start_point)  # y1
    
    return pixel_value * scale
```

### Method 2: Axis Label Association

#### First Coordinate Association
```python
def associate_visual_with_labels(visual_elements, ticklabels, isHbar):
    for visual in visual_elements:
        min_distance = float('inf')
        closest_label = None
        
        for label in ticklabels:
            distance = calculate_distance(visual['bbox'], label['bbox'])
            if distance < min_distance:
                min_distance = distance
                closest_label = label
        
        if closest_label:
            if isHbar:
                visual['y_value'] = closest_label['ocr_text']
            else:
                visual['x_value'] = closest_label['ocr_text']
    
    return visual_elements
```

#### Distance Calculation
```python
def calculate_distance(visual_bbox, label_bbox):
    vx1, vy1, vx2, vy2 = visual_bbox
    lx1, ly1, lx2, ly2 = label_bbox
    
    # Calculate center points
    v_center = ((vx1 + vx2) / 2, (vy1 + vy2) / 2)
    l_center = ((lx1 + lx2) / 2, (ly1 + ly2) / 2)
    
    # Calculate Euclidean distance
    distance = math.sqrt((v_center[0] - l_center[0])**2 + (v_center[1] - l_center[1])**2)
    return distance
```

### Method 3: Negative Value Handling

#### Negative Visual Detection
```python
def detect_negative_visuals(visual_elements, axis_start, isHbar):
    negative_visuals = []
    
    for element in visual_elements:
        bbox = element['bbox']
        
        if isHbar:
            # For horizontal bars, check if bar extends left of axis
            if bbox[0] < axis_start:
                element['isNegative'] = True
                negative_visuals.append(element)
        else:
            # For vertical bars, check if bar extends below axis
            if bbox[1] > axis_start:
                element['isNegative'] = True
                negative_visuals.append(element)
    
    return negative_visuals
```

#### Negative Value Processing
```python
def handle_negative_visuals(negative_visuals, isHbar):
    for element in negative_visuals:
        if isHbar:
            element['x_value'] = 0.0  # Set to zero for negative values
        else:
            element['y_value'] = 0.0  # Set to zero for negative values
    return negative_visuals
```

## Coordinate System and Scaling

### Image Coordinate System

The calculator uses a standard image coordinate system:
- **Origin (0,0)**: Top-left corner of the image
- **X-axis**: Horizontal axis (left to right)
- **Y-axis**: Vertical axis (top to bottom)
- **Bounding Boxes**: Format [x1, y1, x2, y2] where (x1,y1) is top-left, (x2,y2) is bottom-right

### Scaling Methodology

#### 1. Reference Point Selection
```python
# Find axis start point for scaling
if isHbar:
    # For horizontal bars, use y-axis start
    axis_start = yticklabels[0]['bbox'][2] + 9  # Right edge + padding
else:
    # For vertical bars, use x-axis start
    axis_start = xticklabels[0]['bbox'][1] - 9  # Top edge - padding
```

#### 2. Scale Factor Calculation
```python
def calculate_scale_factor(tick1, tick2, isHbar):
    # Get center points of tick labels
    c1 = find_center(tick1['bbox'])
    c2 = find_center(tick2['bbox'])
    
    # Calculate pixel distance
    if isHbar:
        pixel_distance = abs(c2[0] - c1[0])
    else:
        pixel_distance = abs(c2[1] - c1[1])
    
    # Calculate value distance
    value1 = float(clean_ocr_text(tick1['ocr_text']))
    value2 = float(clean_ocr_text(tick2['ocr_text']))
    value_distance = abs(value2 - value1)
    
    # Return scale factor (value per pixel)
    return value_distance / pixel_distance
```

#### 3. Value Calculation
```python
def calculate_element_value(element, scale_factor, axis_start, isHbar):
    bbox = element['bbox']
    
    if element['pred_class'] == 'bar':
        if isHbar:
            # Horizontal bar: measure length from axis start
            pixel_value = abs(bbox[2] - axis_start)
        else:
            # Vertical bar: measure height from axis start
            pixel_value = abs(bbox[1] - axis_start)
    
    elif element['pred_class'] == 'dot_line':
        # Dot: measure distance from axis start
        center = find_center(bbox)
        if isHbar:
            pixel_value = abs(center[0] - axis_start)
        else:
            pixel_value = abs(center[1] - axis_start)
    
    elif element['pred_class'] == 'line':
        # Line: handle slope and measure appropriate endpoint
        slope = find_slope(image, bbox)
        if slope == "positive":
            pixel_value = abs(bbox[3] - axis_start)  # Use y2
        else:
            pixel_value = abs(bbox[1] - axis_start)  # Use y1
    
    # Convert pixel value to actual value
    return pixel_value * scale_factor
```

## OCR Text Processing

### OCR Configuration

The calculator uses Tesseract OCR with optimized settings for chart text:

```python
# Tesseract configuration
config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,-+eE'

# OCR processing
text = pytesseract.image_to_string(cropped_image, config=config)
```

### Text Cleaning and Correction

#### Numeric Text Validation
```python
def validate_numeric_text(text):
    try:
        # Try to convert to float
        value = float(clean_ocr_text(text))
        return True, value
    except ValueError:
        return False, None
```

### Text Region Processing

#### Bounding Box Extension
```python
def extend_text_bbox(bbox, padding=5):
    x1, y1, x2, y2 = bbox
    return [x1 - padding, y1 - padding, x2 + padding, y2 + padding]
```

#### Image Cropping and Preprocessing
```python
def crop_and_preprocess_text_region(image, bbox):
    # Crop the text region
    x1, y1, x2, y2 = bbox
    cropped = image[y1:y2, x1:x2]
    
    # Apply preprocessing for better OCR
    # Convert to grayscale
    if len(cropped.shape) == 3:
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding
    _, cropped = cv2.threshold(cropped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return cropped
```

## Error Handling and Edge Cases

### Common Error Scenarios

#### 1. No Valid Tick Labels Found
```python
if valid_tick_pair is None:
    return "Skip, valid numeric tick labels not found"
```

**Handling:**
- Try different confidence thresholds
- Use alternative tick label detection methods
- Fall back to simplified value calculation

#### 2. Scale Calculation Errors
```python
try:
    value_difference = abs(float(tick1_text) - float(tick2_text))
    scale = value_difference / pixel_difference
except (ValueError, ZeroDivisionError) as e:
    logger.error(f"Error calculating scale: {e}")
    return f"Skip, scale calculation failed: {e}"
```

**Handling:**
- Validate tick label text before conversion
- Check for zero pixel differences
- Use alternative scale calculation methods

#### 3. No Visual Elements Detected
```python
visual_data = [dd for dd in image_data if dd["pred_class"] in ["bar", "dot_line", "line"]]
if len(visual_data) == 0:
    return -1
```

**Handling:**
- Lower confidence threshold
- Check image quality and resolution
- Verify chart type detection

### Edge Case Handling

#### 1. Negative Values
```python
def handle_negative_visuals(negative_visuals, isHbar):
    for element in negative_visuals:
        if isHbar:
            element['x_value'] = 0.0
        else:
            element['y_value'] = 0.0
    return negative_visuals
```

#### 2. Overlapping Elements
```python
def resolve_overlapping_elements(elements):
    # Sort by confidence
    elements.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Remove overlapping elements with lower confidence
    filtered_elements = []
    for element in elements:
        if not any(overlaps(element, existing) for existing in filtered_elements):
            filtered_elements.append(element)
    
    return filtered_elements
```

#### 3. Missing Axis Labels
```python
def handle_missing_axis_labels(detections):
    # Try to infer axis labels from visual elements
    # Use heuristics based on element positions
    pass
```

## Output Formats

### Standard JSON Output

The calculator generates structured JSON output with visual values:

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
  "calculation_method": "find_visual_values_algorithm",
  "scale_factor": 0.5,
  "axis_start_point": 450
}
```

### Enhanced JSON Output (Debug Mode)

When debug mode is enabled, additional information is included:

```json
{
  "chart_orientation": {
    "isHbar": false,
    "isSinglePlot": true
  },
  "visual_elements_with_values": [...],
  "calculation_method": "find_visual_values_algorithm",
  "debug_info": {
    "scale_calculation": {
      "tick1": {"text": "0", "bbox": [50, 400, 70, 420]},
      "tick2": {"text": "10", "bbox": [50, 350, 70, 370]},
      "pixel_difference": 50,
      "value_difference": 10,
      "scale_factor": 0.2
    },
    "axis_detection": {
      "x_axis_start": 450,
      "y_axis_start": 50,
      "detected_orientation": "vertical"
    },
    "processing_steps": [
      "Image loaded and resized to 650x650",
      "Visual elements detected: 4 bars, 0 lines, 0 dots",
      "Text elements detected: 4 xticklabels, 4 yticklabels",
      "Scale factor calculated: 0.2",
      "Visual values calculated for 4 elements"
    ]
  }
}
```

### CSV Output (Compatible with Original PlotQA)

The calculator also generates CSV output in the original PlotQA format:

```csv
title,xlabel,ylabel,Sales
Sales by Quarter,Quarter,Sales ($M),15.2
Sales by Quarter,Quarter,Sales ($M),23.1
Sales by Quarter,Quarter,Sales ($M),18.7
Sales by Quarter,Quarter,Sales ($M),28.3
```

## Usage Examples

### Basic Usage

```bash
# Basic visual values calculation
python calculate_visual_values.py --image chart.png --model models/ved/model_final.pkl --use-caffe2

# With debug mode for detailed analysis
python calculate_visual_values.py --image chart.png --model models/ved/model_final.pkl --use-caffe2 --debug

# Custom confidence threshold
python calculate_visual_values.py --image chart.png --model models/ved/model_final.pkl --confidence 0.05 --use-caffe2 --debug
```

### Advanced Usage

```bash
# Use exact Caffe2 architecture (most compatible)
python calculate_visual_values.py --image chart.png --model models/ved/model_final.pkl --use-exact-caffe2 --debug

# Use Detectron2 detector (modern PyTorch)
python calculate_visual_values.py --image chart.png --model models/ved/model_final.pkl --use-detectron2 --debug

# Custom output directory
python calculate_visual_values.py --image chart.png --model models/ved/model_final.pkl --output my_results/ --use-caffe2 --debug
```

### Programmatic Usage

```python
from calculate_visual_values import VisualValuesCalculator

# Initialize calculator
calculator = VisualValuesCalculator(
    model_path="models/ved/model_final.pkl",
    confidence_threshold=0.05,
    use_caffe2=True,
    use_exact_caffe2=True,
    debug=True
)

# Process image with visual values
results = calculator.process_with_values("chart.png", "output_dir")

# Access results
print(f"Visual values file: {results['visual_values_file']}")
print(f"Elements processed: {results['visual_elements_count']}")
```

### Batch Processing

```python
import os
from calculate_visual_values import VisualValuesCalculator

# Initialize calculator
calculator = VisualValuesCalculator(
    model_path="models/ved/model_final.pkl",
    confidence_threshold=0.05,
    use_caffe2=True,
    debug=True
)

# Process multiple images
image_dir = "charts/"
output_dir = "results/"

for image_file in os.listdir(image_dir):
    if image_file.endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(image_dir, image_file)
        results = calculator.process_with_values(image_path, output_dir)
        print(f"Processed {image_file}: {results['visual_elements_count']} elements")
```

## Troubleshooting

### Common Issues and Solutions

#### 1. "No visual elements detected"
**Cause**: Low confidence threshold or poor image quality
**Solution**: 
```bash
# Lower confidence threshold
python calculate_visual_values.py --image chart.png --model models/ved/model_final.pkl --confidence 0.05 --use-caffe2 --debug
```

#### 2. "Scale calculation failed"
**Cause**: Invalid tick labels or OCR errors
**Solution**:
- Check image quality and resolution
- Verify tick labels are clearly visible
- Try different OCR preprocessing

#### 3. "Valid numeric tick labels not found"
**Cause**: OCR failed to extract numeric values from tick labels
**Solution**:
- Improve image quality
- Check tick label visibility
- Use debug mode to inspect OCR results

#### 4. "CUDA out of memory"
**Cause**: GPU memory insufficient for model inference
**Solution**:
```bash
# Use CPU version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### 5. "KMP_DUPLICATE_LIB_OK error" (Windows)
**Cause**: Intel MKL library conflict
**Solution**:
```bash
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python calculate_visual_values.py --image chart.png --model models/ved/model_final.pkl --use-caffe2 --debug
```

### Debug Mode Analysis

When using debug mode, the calculator generates detailed debug information:

#### Debug Files Generated
- `temp/detection_summary.json`: Detailed detection information
- `temp/detections_visualized.png`: Image with bounding boxes
- `temp/ocr_debug.json`: OCR results with extended bounding boxes
- `temp/ocr_with_extended_bboxes.png`: OCR visualization
- `temp/debug_crops/`: Individual cropped regions

#### Debug Information Analysis
```python
# Load debug information
with open('temp/detection_summary.json', 'r') as f:
    debug_info = json.load(f)

# Analyze detection results
print(f"Total detections: {len(debug_info['detections'])}")
print(f"Visual elements: {len([d for d in debug_info['detections'] if d['class'] in ['bar', 'line', 'dot_line']])}")
print(f"Text elements: {len([d for d in debug_info['detections'] if d['class'] in ['xticklabel', 'yticklabel']])}")

# Analyze OCR results
with open('temp/ocr_debug.json', 'r') as f:
    ocr_info = json.load(f)

print(f"OCR success rate: {ocr_info['success_rate']}")
print(f"Common OCR errors: {ocr_info['common_errors']}")
```

### Performance Optimization

#### 1. GPU Memory Optimization
```python
# Use smaller batch sizes
calculator = VisualValuesCalculator(
    model_path="models/ved/model_final.pkl",
    confidence_threshold=0.05,
    use_caffe2=True,
    debug=False  # Disable debug for better performance
)
```

#### 2. Image Preprocessing Optimization
```python
# Resize images to optimal size
def optimize_image_size(image_path, target_size=(650, 650)):
    image = cv2.imread(image_path)
    resized = cv2.resize(image, target_size)
    return resized
```

#### 3. OCR Optimization
```python
# Use optimized OCR settings
config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.,-+eE'
```

### Best Practices

#### 1. Image Quality
- Use high-resolution images (at least 800x600)
- Ensure good contrast between text and background
- Avoid blurry or distorted images

#### 2. Chart Type Selection
- Use appropriate confidence thresholds for different chart types
- Bar charts: 0.05-0.1
- Line graphs: 0.1-0.2
- Scatter plots: 0.05-0.1

#### 3. Model Selection
- Use Caffe2-compatible detectors for original PlotQA models
- Use exact Caffe2 architecture for maximum compatibility
- Use Detectron2 for modern PyTorch workflows

#### 4. Error Handling
- Always use debug mode for initial testing
- Check debug outputs for processing issues
- Validate results against known values

This comprehensive documentation provides detailed information about the `calculate_visual_values.py` script, its processing pipeline, and methods for each chart type. The calculator represents a significant advancement over basic chart processing by providing precise numerical values for visual elements through coordinate-based scaling and sophisticated OCR processing.
