#!/usr/bin/env python3
"""
Simple test script to validate the PlotQA pipeline components.
This script tests the pipeline without requiring the full dataset.
"""
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

import os
import sys
import json
import tempfile
import numpy as np
from pathlib import Path

# Add pipeline directory to path
sys.path.append(os.path.dirname(__file__))

def create_sample_detection_file(output_path):
    """Create a sample detection file in PlotQA format for testing"""
    sample_detections = [
        "bar 0.94 140 200 170 300",
        "bar 0.91 190 150 220 300", 
        "bar 0.89 240 180 270 300",
        "xticklabel 0.88 150 350 180 370",
        "xticklabel 0.85 200 350 230 370",
        "yticklabel 0.83 50 200 80 220",
        "xlabel 0.90 200 380 300 400",
        "ylabel 0.85 20 100 60 300",
        "title 0.87 200 50 400 80",
        "legend_label 0.80 450 150 500 170",
    ]
    
    with open(output_path, 'w') as f:
        for detection in sample_detections:
            f.write(detection + '\n')

def create_sample_image(output_path, width=600, height=400):
    """Create a simple sample chart image for testing"""
    try:
        import cv2
        
        # Create a simple bar chart-like image
        img = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background
        
        # Draw axes
        cv2.line(img, (100, 320), (500, 320), (0, 0, 0), 2)  # X-axis
        cv2.line(img, (100, 100), (100, 320), (0, 0, 0), 2)  # Y-axis
        
        # Draw bars
        cv2.rectangle(img, (140, 200), (170, 320), (0, 100, 200), -1)  # Bar 1
        cv2.rectangle(img, (190, 150), (220, 320), (0, 150, 100), -1)  # Bar 2
        cv2.rectangle(img, (240, 180), (270, 320), (200, 0, 100), -1)  # Bar 3
        
        # Add some text (will be detected by OCR)
        cv2.putText(img, "Sample Chart", (200, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(img, "X-Axis", (280, 395), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(img, "2020", (150, 365), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(img, "2021", (200, 365), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(img, "2022", (250, 365), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        cv2.imwrite(output_path, img)
        return True
        
    except ImportError:
        print("Warning: OpenCV not available, skipping sample image creation")
        return False

def test_ocr_and_sie():
    """Test the OCR and SIE components"""
    print("Testing OCR and SIE components...")
    
    try:
        from ocr_and_sie import load_detections, OCRProcessor, StructuralExtractor, ChartElement
        
        # Create temporary files
        with tempfile.TemporaryDirectory() as temp_dir:
            detection_file = os.path.join(temp_dir, "test_detections.txt")
            image_file = os.path.join(temp_dir, "test_image.png")
            
            # Create sample data
            create_sample_detection_file(detection_file)
            image_created = create_sample_image(image_file)
            
            # Test detection loading
            detections = load_detections(detection_file)
            print(f"✓ Loaded {len(detections)} detections")
            
            # Test structural extraction
            extractor = StructuralExtractor()
            for detection in detections:
                extractor.add_element(detection)
            
            extractor.detect_chart_type()
            print(f"✓ Detected chart type: {extractor.chart_type}")
            
            if image_created:
                import cv2
                image = cv2.imread(image_file)
                extractor.extract_axes_info(image.shape)
                extractor.extract_data_elements(image.shape)
                extractor.extract_title_and_legend()
                
                result = extractor.to_dict()
                print(f"✓ Extracted structured data with {len(result.get('data_series', []))} data series")
            
            return True
            
    except Exception as e:
        print(f"✗ OCR and SIE test failed: {e}")
        return False

def test_detection_format():
    """Test detection file format parsing"""
    print("Testing detection format parsing...")
    
    try:
        from ocr_and_sie import load_detections
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            # Write sample detections in PlotQA format
            f.write("bar 0.950000 100.00 300.00 500.00 320.00\n")
            f.write("xticklabel 0.880000 150.00 350.00 180.00 370.00\n")
            f.write("title 0.940000 140.00 200.00 170.00 300.00\n")
            temp_file = f.name
        
        try:
            detections = load_detections(temp_file)
            
            # Verify parsing
            assert len(detections) == 3, f"Expected 3 detections, got {len(detections)}"
            
            # Check first detection
            first = detections[0]
            assert first.class_name == "bar", f"Expected 'bar', got '{first.class_name}'"
            assert abs(first.confidence - 0.95) < 0.001, f"Expected 0.95, got {first.confidence}"
            assert first.bbox == [100.0, 300.0, 500.0, 320.0], f"Unexpected bbox: {first.bbox}"
            
            print("✓ Detection format parsing works correctly")
            return True
            
        finally:
            os.unlink(temp_file)
            
    except Exception as e:
        print(f"✗ Detection format test failed: {e}")
        return False

def test_json_output_format():
    """Test JSON output format"""
    print("Testing JSON output format...")
    
    try:
        from ocr_and_sie import StructuralExtractor, ChartElement
        
        extractor = StructuralExtractor()
        
        # Add sample elements
        elements = [
            ChartElement("title", 0.9, [200, 50, 400, 80], "Sample Chart"),
            ChartElement("xlabel", 0.85, [280, 380, 350, 400], "Years"),
            ChartElement("bar", 0.94, [140, 200, 170, 320]),
            ChartElement("bar", 0.91, [190, 150, 220, 320]),
        ]
        
        for element in elements:
            extractor.add_element(element)
        
        extractor.detect_chart_type()
        result = extractor.to_dict()
        
        # Verify JSON structure
        required_keys = ["chart_type", "title", "x_axis", "y_axis", "data_series", "legend"]
        for key in required_keys:
            assert key in result, f"Missing required key: {key}"
        
        # Verify axis structure
        for axis in ["x_axis", "y_axis"]:
            axis_data = result[axis]
            assert "label" in axis_data, f"Missing label in {axis}"
            assert "ticks" in axis_data, f"Missing ticks in {axis}"
            assert "values" in axis_data, f"Missing values in {axis}"
        
        print("✓ JSON output format is correct")
        return True
        
    except Exception as e:
        print(f"✗ JSON output format test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("PlotQA Pipeline Test Suite")
    print("=" * 50)
    
    tests = [
        test_detection_format,
        test_json_output_format,
        test_ocr_and_sie,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! Pipeline components are working correctly.")
        return 0
    else:
        print("✗ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
