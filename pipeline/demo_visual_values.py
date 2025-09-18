#!/usr/bin/env python3
"""
Demonstration script for the Visual Values Calculator
Shows how to use the integrated find_visual_values logic with process_chart.py
"""

import os
import sys
import json
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

def demo_visual_values_calculation():
    """Demonstrate the visual values calculation functionality"""
    
    print("Visual Values Calculator Demo")
    print("=" * 50)
    print()
    
    # Check if we have a test image
    test_image = "test_chart.png"
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        print("Please ensure you have a test chart image in the pipeline directory.")
        return False
    
    # Check if we have a model
    model_path = "models/ved/model_final.pkl"
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please ensure you have the trained VED model in the models/ved/ directory.")
        return False
    
    print(f"✅ Found test image: {test_image}")
    print(f"✅ Found model: {model_path}")
    print()
    
    # Import the calculator
    try:
        from calculate_visual_values import VisualValuesCalculator
        print("✅ Successfully imported VisualValuesCalculator")
    except ImportError as e:
        print(f"❌ Failed to import VisualValuesCalculator: {e}")
        return False
    
    # Initialize the calculator
    try:
        calculator = VisualValuesCalculator(
            model_path=model_path,
            confidence_threshold=0.3,
            debug=True  # Enable debug mode for full functionality
        )
        print("✅ Successfully initialized calculator with debug mode")
    except Exception as e:
        print(f"❌ Failed to initialize calculator: {e}")
        return False
    
    # Process the image
    try:
        print(f"\n🔄 Processing {test_image}...")
        results = calculator.process_with_values(test_image, "demo_results")
        print("✅ Processing completed successfully!")
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        return False
    
    # Display results
    print(f"\n📊 Results Summary:")
    print(f"   Output Directory: demo_results/")
    
    if "files_created" in results:
        print(f"   Files Created:")
        for file_path in results["files_created"]:
            print(f"     - {Path(file_path).name}")
    
    if "visual_values_file" in results:
        print(f"   Visual Values File: {Path(results['visual_values_file']).name}")
        
        # Try to load and display some visual values
        try:
            with open(results['visual_values_file'], 'r') as f:
                visual_data = json.load(f)
            
            if "visual_elements_with_values" in visual_data:
                visual_elements = visual_data["visual_elements_with_values"]
                chart_elements = [e for e in visual_elements if e.get("pred_class") in ["bar", "dot_line", "line"]]
                
                print(f"   Visual Elements Found: {len(chart_elements)}")
                
                for i, element in enumerate(chart_elements[:3]):  # Show first 3
                    print(f"     Element {i+1}:")
                    print(f"       Type: {element.get('pred_class', 'unknown')}")
                    if 'x_value' in element:
                        print(f"       X Value: {element['x_value']}")
                    if 'y_value' in element:
                        print(f"       Y Value: {element['y_value']}")
                
                if len(chart_elements) > 3:
                    print(f"     ... and {len(chart_elements) - 3} more elements")
            
        except Exception as e:
            print(f"   ⚠️  Could not parse visual values: {e}")
    
    if "visual_values_error" in results:
        print(f"   ⚠️  Warning: {results['visual_values_error']}")
    
    print(f"\n🎉 Demo completed successfully!")
    print(f"\nTo run the calculator on your own images:")
    print(f"python calculate_visual_values.py --image your_chart.png --model {model_path} --debug")
    
    return True

def show_usage_examples():
    """Show usage examples for the visual values calculator"""
    
    print("\n" + "=" * 60)
    print("Usage Examples")
    print("=" * 60)
    
    examples = [
        {
            "title": "Basic Usage (Default Caffe2)",
            "command": "python calculate_visual_values.py --image chart.png --model models/ved/model_final.pth",
            "description": "Process a chart image and calculate visual values using Caffe2 detector (recommended for original PlotQA models)"
        },
        {
            "title": "Debug Mode (Recommended)",
            "command": "python calculate_visual_values.py --image chart.png --model models/ved/model_final.pth --debug",
            "description": "Enable debug mode for detailed coordinate-based value calculations"
        },
        {
            "title": "Custom Output Directory",
            "command": "python calculate_visual_values.py --image chart.png --model models/ved/model_final.pth --output my_results/",
            "description": "Save results to a custom directory"
        },
        {
            "title": "Custom Confidence Threshold",
            "command": "python calculate_visual_values.py --image chart.png --model models/ved/model_final.pth --confidence 0.7",
            "description": "Use higher confidence threshold for more precise detections"
        },
        {
            "title": "Use Detectron2 Instead of Caffe2",
            "command": "python calculate_visual_values.py --image chart.png --model models/ved/model_final.pth --use-detectron2",
            "description": "Use Detectron2 detector instead of default Caffe2 (for newer models)"
        },
        {
            "title": "Verbose Output",
            "command": "python calculate_visual_values.py --image chart.png --model models/ved/model_final.pth --debug --verbose",
            "description": "Enable detailed logging for debugging"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['title']}")
        print(f"   {example['description']}")
        print(f"   Command: {example['command']}")
    
    print(f"\n{'=' * 60}")
    print("Key Features")
    print("=" * 60)
    
    features = [
        "🔍 Visual Element Detection (bars, lines, dot-lines)",
        "📝 OCR for tick labels and axis labels", 
        "📏 Coordinate-based value calculation using scaling",
        "🎯 Handles horizontal and vertical bar charts",
        "📊 Supports line plots and dot-line plots",
        "🔧 OCR text cleaning and error correction",
        "🐛 Debug mode with detailed intermediate results",
        "📄 JSON output with calculated values"
    ]
    
    for feature in features:
        print(f"   {feature}")

def main():
    """Main demonstration function"""
    
    # Run the demo
    success = demo_visual_values_calculation()
    
    # Show usage examples
    show_usage_examples()
    
    if success:
        return 0
    else:
        print(f"\n❌ Demo failed. Please check the requirements and try again.")
        return 1

if __name__ == "__main__":
    exit(main())
