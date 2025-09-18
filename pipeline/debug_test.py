#!/usr/bin/env python3
"""
Debug test for the OCR and SIE pipeline
"""

import tempfile
import os
import shutil
from ocr_and_sie import run_original_plotqa_pipeline
from generate_detections import PlotQADetector, save_detections_text_format

def debug_pipeline():
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        png_dir = os.path.join(temp_dir, 'images')
        detections_dir = os.path.join(temp_dir, 'detections') 
        csv_dir = os.path.join(temp_dir, 'csv')
        
        os.makedirs(png_dir)
        os.makedirs(detections_dir)
        os.makedirs(csv_dir)
        
        # Copy test files
        shutil.copy('test_chart.png', os.path.join(png_dir, '1.png'))
        
        # Create a proper detection file
        detector = PlotQADetector('models/ved/model_final.pkl', confidence_threshold=0.3)  # Lower threshold
        detections = detector.detect_single_image('test_chart.png')
        save_detections_text_format(detections, os.path.join(detections_dir, '1.txt'))
        
        print('Files created:')
        print(f'PNG: {os.path.exists(os.path.join(png_dir, "1.png"))}')
        print(f'Detection: {os.path.exists(os.path.join(detections_dir, "1.txt"))}')
        
        # Check detection file content
        with open(os.path.join(detections_dir, '1.txt'), 'r', encoding='utf-8', errors='ignore') as f:
            detection_content = f.read()
            print(f'Detection file length: {len(detection_content)}')
            print('First 200 chars:', repr(detection_content[:200]))
        
        # Run pipeline with debug
        try:
            run_original_plotqa_pipeline(png_dir, detections_dir, csv_dir, MIN_CLASS_CONFIDENCE=0.3)
            print('Pipeline completed')
            
            # Check output
            csv_files = os.listdir(csv_dir)
            print(f'CSV files created: {csv_files}')
            
            if csv_files:
                with open(os.path.join(csv_dir, csv_files[0]), 'r') as f:
                    content = f.read()
                    print(f'CSV content length: {len(content)}')
                    print('First 500 chars:')
                    print(content[:500])
            
        except Exception as e:
            import traceback
            print(f'Error: {e}')
            print(traceback.format_exc())

if __name__ == "__main__":
    debug_pipeline()
