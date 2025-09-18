#!/usr/bin/env python3
"""
Test script to demonstrate single image processing with the PlotQA pipeline
Creates a simple test chart and processes it through the complete pipeline.
"""
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def create_test_chart(output_path="test_chart.png"):
    """Create a simple test bar chart"""
    
    # Sample data
    categories = ['Category A', 'Category B', 'Category C', 'Category D']
    values = [23, 45, 56, 78]
    
    # Create figure
    plt.figure(figsize=(8, 6))
    bars = plt.bar(categories, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    
    # Add labels and title
    plt.title('Sample Bar Chart for PlotQA Testing')
    plt.xlabel('Categories')
    plt.ylabel('Values')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                str(value), ha='center', va='bottom')
    
    # Add grid for better readability
    plt.grid(axis='y', alpha=0.3)
    
    # Save chart
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Test chart created: {output_path}")
    return output_path

def main():
    """Main test function"""
    print("PlotQA Single Image Processing Test")
    print("=" * 40)
    
    # Create test chart
    test_image = create_test_chart()
    
    # Check if model exists
    model_path = "models/ved/model_final.pkl"
    if not os.path.exists(model_path):
        print(f"Warning: Model file not found at {model_path}")
        print("Please ensure you have trained a VED model or update the path.")
        print("\nTo train a model, run:")
        print("python train_ved.py --dataset data/plotqa --output models/ved/")
        return 1
    
    # Show processing command
    print(f"\nTo process the test image, run:")
    print(f"python process_chart.py --image {test_image} --model {model_path}")
    print(f"\nOr with custom output directory:")
    print(f"python process_chart.py --image {test_image} --model {model_path} --output test_results/")
    
    # Try to run processing if model exists
    try:
        from process_chart import PlotQAProcessor
        
        print(f"\nAttempting to process {test_image}...")
        processor = PlotQAProcessor(model_path=model_path)
        results = processor.process_image(test_image, "test_results")
        
        print("\nProcessing completed successfully!")
        print("Check the 'test_results' directory for output files.")
        
    except FileNotFoundError as e:
        print(f"\nModel not found: {e}")
        print("Please train a VED model first using train_ved.py")
    except Exception as e:
        print(f"\nProcessing failed: {e}")
        print("This might be expected if dependencies are missing or model is not trained.")
    
    return 0

if __name__ == "__main__":
    exit(main())
