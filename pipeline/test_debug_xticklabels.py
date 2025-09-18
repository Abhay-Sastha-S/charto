#!/usr/bin/env python3
"""
Test script to demonstrate debug functionality for xticklabel OCR
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ocr_and_sie import OCRProcessor
from PIL import Image
import json

def test_debug_xticklabel_ocr():
    """Test xticklabel OCR with debug mode enabled"""
    
    # Load the chart image
    image_path = "test_chart.png"
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found")
        return
    
    # Load the debug data to get the bounding boxes
    debug_path = "temp/ocr_debug.json"
    if not os.path.exists(debug_path):
        print(f"Error: {debug_path} not found")
        return
    
    with open(debug_path, 'r') as f:
        debug_data = json.load(f)
    
    # Load the image
    image = Image.open(image_path)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    # Initialize OCR processor with debug mode enabled
    ocr_processor = OCRProcessor(debug=True, debug_dir="temp/debug_xticklabels")
    
    print("Testing xticklabel OCR with debug mode...")
    print("=" * 50)
    print(f"Debug images will be saved to: temp/debug_xticklabels/")
    
    # Test each xticklabel
    xticklabels = [item for item in debug_data['ocr_results'] if item['class'] == 'xticklabel']
    
    for i, xticklabel in enumerate(xticklabels):
        print(f"\nProcessing Xticklabel {i+1}:")
        print(f"  Confidence: {xticklabel['confidence']}")
        print(f"  Bbox: {xticklabel['extended_bbox_650']}")
        print(f"  Previous OCR result: '{xticklabel['ocr_text']}'")
        
        # Test with debug mode enabled
        try:
            debug_id = f"test_{i+1}"
            new_text = ocr_processor.extract_text(
                image, 
                xticklabel['extended_bbox_650'], 
                role='xticklabel', 
                isHbar=False,
                debug_id=debug_id
            )
            print(f"  New OCR result: '{new_text}'")
            
            if new_text != xticklabel['ocr_text']:
                print(f"  ✓ IMPROVEMENT: '{xticklabel['ocr_text']}' -> '{new_text}'")
            else:
                print(f"  - No change")
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print("\n" + "=" * 50)
    print("Debug test completed!")
    print("Check the temp/debug_xticklabels/ directory for saved images:")
    print("  - *_original.png: Original cropped image")
    print("  - *_resized.png: Resized image")
    print("  - *_preprocessed.png: Preprocessed grayscale image")
    print("  - *_info.txt: OCR processing details and result")

if __name__ == "__main__":
    test_debug_xticklabel_ocr()
