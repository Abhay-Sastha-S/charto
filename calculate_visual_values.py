#!/usr/bin/env python3
"""
Visual Values Calculator for PlotQA Charts
Integrates find_visual_values.py logic with process_chart.py pipeline to calculate numerical values
for visual elements (bars, lines, dot-lines) in chart images.

Usage:
    python calculate_visual_values.py --image chart.png --model models/ved/model_final.pth
    python calculate_visual_values.py --image chart.png --model models/ved/model_final.pth --output results/
    python calculate_visual_values.py --image chart.png --model models/ved/model_final.pth --confidence 0.7

This script extends the PlotQA pipeline to include visual value calculations similar to the original
find_visual_values.py implementation.
"""

import os
import sys
import json
import argparse
import logging
import copy
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

# Import process_chart components
from process_chart import PlotQAProcessor

# Import local utilities
from utils import find_center, find_Distance, list_subtraction, find_slope

# Import the exact pipeline from process_chart.py
from ocr_and_sie import run_original_plotqa_pipeline
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VisualValuesCalculator:
    """Extended PlotQA processor that calculates visual values for chart elements"""
    
    def __init__(self, model_path, confidence_threshold=0.1, use_caffe2=True, use_exact_caffe2=True, debug=False):
        """
        Initialize the visual values calculator
        
        Args:
            model_path: Path to trained VED model weights
            confidence_threshold: Minimum confidence for detections
            use_caffe2: Whether to use Caffe2 compatible detector (recommended for original models)
            use_exact_caffe2: Whether to use exact Caffe2 architecture replication
            debug: Whether to save debug information
        """
        self.processor = PlotQAProcessor(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            use_caffe2=use_caffe2,
            use_exact_caffe2=use_exact_caffe2,
            debug=debug
        )
        self.debug = debug
        
    def process_with_values(self, image_path, output_dir="results"):
        """
        Process chart image and calculate visual values for elements
        
        Args:
            image_path: Path to input chart image
            output_dir: Directory to save results
            
        Returns:
            Dictionary with processing results including calculated values
        """
        logger.info(f"Processing image with visual value calculations: {image_path}")
        
        # First run the standard PlotQA pipeline
        results = self.processor.process_image(image_path, output_dir)
        
        # Load the generated JSON results to get detection data
        if 'json_file' in results:
            json_path = results['json_file']
            
            # Calculate visual values using the detection data
            enhanced_results = self._calculate_visual_values(image_path, json_path, output_dir)
            
            # Merge results
            results.update(enhanced_results)
            
        return results
    
    def _run_ved_detection(self, image_path, detections_dir, image_name):
        """Run VED detection using the exact method from process_chart.py"""
        logger.info("Running VED detection...")
        
        # Use the detector from the processor
        detections, resized_image, original_dimensions = self.processor.detector.detect_single_image(image_path)
        
        # Save detections in the format expected by the pipeline
        detections_path = os.path.join(detections_dir, f"{image_name}.txt")
        with open(detections_path, 'w') as f:
            for detection in detections:
                class_name = detection[0]
                confidence = detection[1]
                xmin, ymin, xmax, ymax = detection[2:6]
                f.write(f"{class_name} {confidence:.6f} {xmin:.6f} {ymin:.6f} {xmax:.6f} {ymax:.6f}\n")
        
        # Convert resized image to PIL format
        if isinstance(resized_image, np.ndarray):
            resized_image = Image.fromarray(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
        
        return detections_path, resized_image, original_dimensions
    
    def _run_ocr_sie_pipeline(self, png_dir, detections_dir, csv_dir, resized_image, original_dimensions):
        """Run OCR and SIE using the exact method from process_chart.py"""
        logger.info("Running OCR and SIE pipeline...")
        
        # Use the original PlotQA pipeline with lower threshold for tick labels
        run_original_plotqa_pipeline(
            png_dir=png_dir,
            detections_dir=detections_dir,
            csv_dir=csv_dir,
            MIN_CLASS_CONFIDENCE=self.processor.confidence_threshold,
            MIN_TICKLABEL_CONFIDENCE=0.05,  # Lower threshold for tick labels
            debug=self.debug  # Pass debug flag to OCR pipeline
        )
        
        logger.info("OCR and SIE completed")
    
    def _csv_to_json_with_visual_values(self, csv_path, json_path, image_path, png_dir, detections_dir):
        """Convert CSV to JSON with visual values using the exact method from process_chart.py"""
        logger.info("Converting CSV to JSON with visual values...")
        
        # Load the CSV data
        import pandas as pd
        df = pd.read_csv(csv_path)
        
        # Extract metadata
        metadata_cols = ['xlabel', 'ylabel', 'title', 'legend orientation', 'plot_type']
        data_cols = [col for col in df.columns if col not in metadata_cols]
        
        # Determine chart type
        if 'plot_type' in df.columns and not df.empty:
            chart_type = df['plot_type'].iloc[0]
        else:
            # Use heuristic detection
            if any("bar" in str(col).lower() for col in df.columns):
                chart_type = "bar"
            elif any("line" in str(col).lower() for col in df.columns):
                chart_type = "line"
            else:
                chart_type = "line"  # Default for scatter plots
        
        # Extract data series
        data_series = []
        for col in data_cols:
            if col in df.columns:
                series_data = []
                for _, row in df.iterrows():
                    if pd.notna(row[col]) and row[col] != '':
                        series_data.append({
                            "x": str(row.get('xlabel', '')),
                            "y": float(row[col]) if isinstance(row[col], (int, float)) else row[col]
                        })
                
                if series_data:
                    data_series.append({
                        "name": col,
                        "type": chart_type,
                        "data": series_data
                    })
        
        # Create the JSON structure
        result = {
            "chart_type": chart_type,
            "title": df['title'].iloc[0] if 'title' in df.columns and not df.empty else "",
            "x_axis": {
                "label": df['xlabel'].iloc[0] if 'xlabel' in df.columns and not df.empty else "",
                "type": "categorical"
            },
            "y_axis": {
                "label": df['ylabel'].iloc[0] if 'ylabel' in df.columns and not df.empty else "",
                "type": "numeric"
            },
            "data_series": data_series,
            "metadata": {
                "source_image": str(image_path),
                "extraction_method": "PlotQA Pipeline with Visual Values",
                "total_series": len(data_series)
            }
        }
        
        # Save the JSON
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"JSON with visual values saved to: {json_path}")
    
    def _calculate_visual_values(self, image_path, json_path, output_dir):
        """
        Calculate visual values using the exact method from process_chart.py
        
        Args:
            image_path: Original image path
            json_path: Path to JSON results from PlotQA pipeline
            output_dir: Output directory
            
        Returns:
            Dictionary with enhanced results including visual values
        """
        logger.info("Calculating visual values using exact process_chart.py method...")
        
        try:
            # Use the exact method from process_chart.py by running the full pipeline
            # This ensures proper handling of scatter plots and dot_line elements
            
            # Create temporary directories for the pipeline
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                png_dir = os.path.join(temp_dir, "png")
                detections_dir = os.path.join(temp_dir, "detections")
                csv_dir = os.path.join(temp_dir, "csv")
                
                os.makedirs(png_dir, exist_ok=True)
                os.makedirs(detections_dir, exist_ok=True)
                os.makedirs(csv_dir, exist_ok=True)
                
                # Copy image to png_dir
                image_name = Path(image_path).stem
                temp_image_path = os.path.join(png_dir, f"{image_name}.png")
                shutil.copy2(image_path, temp_image_path)
                
                # Run VED detection
                detections_path, resized_image, original_dimensions = self._run_ved_detection(image_path, detections_dir, image_name)
                
                # Run the exact OCR/SIE pipeline from process_chart.py
                self._run_ocr_sie_pipeline(png_dir, detections_dir, csv_dir, resized_image, original_dimensions)
                
                # Load the results
                csv_path = os.path.join(csv_dir, f"{image_name}.csv")
                if os.path.exists(csv_path):
                    # Convert CSV to JSON with visual values
                    json_output = os.path.join(output_dir, f"{image_name}_visual_values.json")
                    self._csv_to_json_with_visual_values(csv_path, json_output, image_path, png_dir, detections_dir)
                    
                    logger.info(f"Visual values saved to: {json_output}")
                    
                    return {
                        "visual_values_file": json_output,
                        "calculation_method": "process_chart_pipeline"
                    }
                else:
                    raise Exception("No CSV output generated from pipeline")
                    
        except Exception as e:
            logger.error(f"Error calculating visual values: {e}")
            return {
                "error": f"Visual value calculation failed: {str(e)}",
                "calculation_method": "failed"
            }
    
    def _calculate_from_debug_data(self, image, detection_file, ocr_file, output_dir):
        """Calculate values using debug detection and OCR data"""
        logger.info("Calculating visual values from debug data...")
        
        try:
            # Load debug data
            with open(detection_file, 'r') as f:
                detection_data = json.load(f)
            
            with open(ocr_file, 'r') as f:
                ocr_data = json.load(f)
            
            # Convert debug data to image_data format expected by find_visual_values
            image_data = self._convert_debug_to_image_data(detection_data, ocr_data)
            
            # Determine chart orientation
            isHbar = ocr_data.get("chart_orientation", {}).get("isHbar", False)
            isSinglePlot = ocr_data.get("chart_orientation", {}).get("isSinglePlot", True)
            
            # Calculate visual values using the original algorithm
            enhanced_image_data = self._find_visual_values(image, image_data, isHbar, isSinglePlot)
            
            if enhanced_image_data == -1:
                return {"visual_values_error": "No visual elements found"}
            elif isinstance(enhanced_image_data, str):
                return {"visual_values_error": enhanced_image_data}
            
            # Save enhanced results
            output_path = os.path.join(output_dir, f"{Path(image.filename).stem}_visual_values.json")
            with open(output_path, 'w') as f:
                json.dump({
                    "chart_orientation": {"isHbar": isHbar, "isSinglePlot": isSinglePlot},
                    "visual_elements_with_values": enhanced_image_data,
                    "calculation_method": "find_visual_values_algorithm"
                }, f, indent=2)
            
            logger.info(f"Visual values saved to: {output_path}")
            
            return {
                "visual_values_file": output_path,
                "visual_elements_count": len([d for d in enhanced_image_data if d.get("pred_class") in ["bar", "dot_line", "line"]]),
                "chart_orientation": {"isHbar": isHbar, "isSinglePlot": isSinglePlot}
            }
            
        except Exception as e:
            logger.error(f"Error in debug data calculation: {e}")
            return {"visual_values_error": str(e)}
    
    def _convert_debug_to_image_data(self, detection_data, ocr_data):
        """Convert debug data format to image_data format for find_visual_values"""
        image_data = []
        
        # Process OCR results (these have text)
        for ocr_result in ocr_data.get("ocr_results", []):
            element = {
                "pred_class": ocr_result["class"],
                "bbox": ocr_result["extended_bbox_650"],  # Use extended bbox for better text extraction
                "ocr_text": ocr_result["ocr_text"],
                "confidence": ocr_result["confidence"]
            }
            image_data.append(element)
        
        # Process visual elements (bars, lines, etc.) from detection data
        for detection in detection_data.get("detection_details", []):
            if detection["class"] in ["bar", "dot_line", "line"]:
                element = {
                    "pred_class": detection["class"],
                    "bbox": detection["bbox_650x650"],  # Use 650x650 coordinates
                    "confidence": detection["confidence"]
                }
                image_data.append(element)
        
        return image_data
    
    def _calculate_from_chart_data(self, image, chart_data, output_dir):
        """Calculate values from existing chart data (fallback method)"""
        logger.info("Calculating visual values from chart data (limited functionality)...")
        
        # This is a simplified fallback - the full calculation requires detection coordinates
        # which are not preserved in the final JSON output
        
        visual_elements = []
        if 'data_series' in chart_data:
            for series in chart_data['data_series']:
                for data_point in series.get('data', []):
                    visual_elements.append({
                        "series_name": series['name'],
                        "x_value": data_point.get('x'),
                        "y_value": data_point.get('y'),
                        "chart_type": series.get('type', 'unknown')
                    })
        
        # Save simplified results
        output_path = os.path.join(output_dir, f"{Path(image.filename).stem}_visual_values_simplified.json")
        with open(output_path, 'w') as f:
            json.dump({
                "visual_elements": visual_elements,
                "calculation_method": "simplified_from_chart_data",
                "note": "Limited functionality - enable debug mode for full visual value calculation"
            }, f, indent=2)
        
        return {
            "visual_values_file": output_path,
            "visual_elements_count": len(visual_elements),
            "calculation_method": "simplified"
        }
    
    def _find_visual_values(self, image, image_data, isHbar, isSinglePlot):
        """
        Adapted version of find_visual_values from misc/codes/find_visual_values.py
        """
        logger.info(f"Running visual values calculation (isHbar: {isHbar}, isSinglePlot: {isSinglePlot})")
        
        # Associate the bar with the x-label (if vertical bar) or y-label (if Hbar)
        if isHbar:
            ticklabel = [dd for dd in image_data if dd["pred_class"] == "yticklabel"]
        else:
            ticklabel = [dd for dd in image_data if dd["pred_class"] == "xticklabel"]
            ticklabel = sorted(ticklabel, key=lambda x: x['bbox'][0])

        visual_data = [dd for dd in image_data if dd["pred_class"] in ["bar", "dot_line", "line"]]
        visual_data = sorted(visual_data, key=lambda x: x['bbox'][0])

        if len(visual_data) == 0:
            return -1

        image_data = list_subtraction(image_data, visual_data)
        visual_data = self._find_first_coord(visual_data, isHbar, ticklabel)
        image_data = image_data + visual_data

        # Associate the bar with the y-label (if vertical bar) or x-label (if Hbar)
        if isHbar:
            ticklabel = [dd for dd in image_data if dd["pred_class"] == "xticklabel"]
            ticklabel = sorted(ticklabel, key=lambda x: x['bbox'][0])
            yticks = [dd for dd in image_data if dd["pred_class"] == "yticklabel"]
            if len(yticks) > 0:
                start = yticks[0]['bbox'][2] + 9  # added 9 so that the start starts from the center of the major tick
            else:
                logger.warning("No yticklabels found for horizontal bar chart")
                return "Skip, yticklabels are not detected"
        else:
            ticklabel = [dd for dd in image_data if dd["pred_class"] == "yticklabel"]
            ticklabel = sorted(ticklabel, key=lambda x: x['bbox'][1])
            xticks = [dd for dd in image_data if dd["pred_class"] == "xticklabel"]
            if len(xticks) > 0:
                start = xticks[0]['bbox'][1] - 9  # added 9 so that the start starts from the center of the major tick
            else:
                logger.warning("No xticklabels found for vertical bar chart")
                return "Skip, xticklabels are not detected"

        # Find two valid tick labels for scale calculation
        valid_tick_pair = None
        for t_i in range(len(ticklabel)-1):
            tick1 = ticklabel[t_i]
            tick2 = ticklabel[t_i + 1]

            # Clean OCR text
            tick1_text = self._clean_ocr_text(tick1['ocr_text'])
            tick2_text = self._clean_ocr_text(tick2['ocr_text'])

            if len(tick1_text) > 0 and len(tick2_text) > 0 and tick2_text != tick1_text:
                try:
                    # Try to convert to float to validate
                    float(tick1_text)
                    float(tick2_text)
                    valid_tick_pair = (tick1, tick2, tick1_text, tick2_text)
                    break
                except ValueError:
                    continue

        if valid_tick_pair is None:
            return "Skip, valid numeric tick labels not found"

        tick1, tick2, tick1_text, tick2_text = valid_tick_pair

        # Calculating pixel difference from tick-label's center
        c_x1, c_y1 = find_center(tick1['bbox'])
        c_x2, c_y2 = find_center(tick2['bbox'])

        if isHbar:
            pixel_difference = abs(c_x2 - c_x1)
        else:
            pixel_difference = abs(c_y2 - c_y1)

        try:
            value_difference = abs(float(tick1_text) - float(tick2_text))
            scale = value_difference / pixel_difference
            
            logger.info(f"Scale calculation: value_diff={value_difference}, pixel_diff={pixel_difference}, scale={scale}")
        except (ValueError, ZeroDivisionError) as e:
            logger.error(f"Error calculating scale: {e}")
            return f"Skip, scale calculation failed: {e}"

        # Get visual elements and handle negative visuals
        visual_data = [dd for dd in image_data if dd["pred_class"] in ["bar", "dot_line", "line"] and "isNegative" not in dd.keys()]
        negative_visuals = [dd for dd in image_data if dd["pred_class"] in ["bar", "dot_line", "line"] and "isNegative" in dd.keys()]

        image_data = list_subtraction(image_data, visual_data)
        image_data = list_subtraction(image_data, negative_visuals)

        negative_visuals = self._handle_negative_visuals(negative_visuals, isHbar)

        if not isHbar:
            visual_data = sorted(visual_data, key=lambda x: x['bbox'][0])

        # Calculate second coordinate (value) for each visual element
        for bidx in range(len(visual_data)):
            if visual_data[bidx]["pred_class"] == "bar":
                if isHbar:
                    compare_with = abs(visual_data[bidx]['bbox'][2] - start)  # length of the bar
                else:
                    compare_with = abs(visual_data[bidx]['bbox'][1] - start)  # height of the bar
            else:
                if visual_data[bidx]["pred_class"] == "dot_line":
                    # center of the dot-line
                    cx, cy = find_center(visual_data[bidx]['bbox'])
                    compare_with = abs(cy - start)
                elif visual_data[bidx]["pred_class"] == "line":
                    slope = find_slope(image, visual_data[bidx]['bbox'])
                    x1, y1, x2, y2 = visual_data[bidx]['bbox']
                    if slope == "positive":
                        compare_with = abs(y2 - start)
                    else:
                        compare_with = abs(y1 - start)

            value = compare_with * scale

            if isHbar:
                visual_data[bidx]["x_value"] = value
            else:
                visual_data[bidx]["y_value"] = value

        # Handle line plot end point
        if len(visual_data) > 0 and visual_data[-1]["pred_class"] == "line":
            slope = find_slope(image, visual_data[-1]['bbox'])
            x1, y1, x2, y2 = visual_data[-1]['bbox']
            if slope == "positive":
                compare_with = abs(y1 - start)
            else:
                compare_with = abs(y2 - start)
            value = compare_with * scale
            visual_data[-1]["y_value"] = value

        image_data = image_data + visual_data + negative_visuals

        return image_data
    
    def _find_first_coord(self, visual_data, isHbar, ticklabel):
        """Find first coordinate for visual elements (adapted from find_visual_values.py)"""
        for bidx in range(len(visual_data)):
            x1, y1, x2, y2 = visual_data[bidx]["bbox"]
            minDistance = 1e10
            b_lbl_idx = -1
            for tidx in range(len(ticklabel)):
                a1, b1, a2, b2 = ticklabel[tidx]["bbox"]
                ax, by = find_center([a1, b1, a2, b2])
                if isHbar:
                    visual_point = [x1, y2]
                    lbl_point = [a2, b2]
                else:
                    visual_point = [x1, y2]  # Take x1,y2 instead of x2,y2
                    lbl_point = [ax, b1]     # Take ax, b1 instead of a2,b1
                d = find_Distance(lbl_point, visual_point)
                if d < minDistance:
                    b_lbl_idx = tidx
                    minDistance = d

            if b_lbl_idx >= 0:
                if isHbar:
                    visual_data[bidx]["y_value"] = ticklabel[b_lbl_idx]["ocr_text"]
                else:
                    visual_data[bidx]["x_value"] = ticklabel[b_lbl_idx]["ocr_text"]

        # Handle the last bbox for line plot
        if len(visual_data) > 0 and visual_data[0]["pred_class"] == "line":
            _visual_data = copy.deepcopy(visual_data)
            visual_data.append(visual_data[-1])
            if len(ticklabel) > 0:
                visual_data[-1]["x_value"] = ticklabel[-1]["ocr_text"]
            _visual_data.append(visual_data[-1])
            return _visual_data

        return visual_data
    
    def _handle_negative_visuals(self, negative_visuals, isHbar):
        """Handle negative visual elements"""
        for dd in negative_visuals:
            assert dd["isNegative"] == True
            if isHbar:
                dd["x_value"] = 0.0
            else:
                dd["y_value"] = 0.0
        return negative_visuals
    
    def _clean_ocr_text(self, text):
        """Clean OCR text for numerical parsing (adapted from find_visual_values.py)"""
        if not text:
            return ""
        
        # Apply OCR corrections
        cleaned = text.replace(" ", "").replace("C", "0").replace("+", "e+").replace("ee+", "e+")
        cleaned = cleaned.replace("O", "0").replace("o", "0").replace("B", "8")
        
        # Remove trailing dash
        if cleaned.endswith("-"):
            cleaned = cleaned[:-1]
        
        # Handle specific OCR errors
        if "84-" in cleaned:
            cleaned = cleaned.replace("84-", "e+")
        if "91-" in cleaned:
            cleaned = cleaned.replace("91-", "e+")
        
        return cleaned


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Calculate visual values for chart elements using PlotQA pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with visual value calculation (uses Caffe2 by default)
  python calculate_visual_values.py --image chart.png --model models/ved/model_final.pth
  
  # Enable debug mode for detailed visual value calculation
  python calculate_visual_values.py --image chart.png --model models/ved/model_final.pth --debug
  
  # Use Detectron2 detector instead of Caffe2
  python calculate_visual_values.py --image chart.png --model models/ved/model_final.pth --use-detectron2
  
  # Custom output directory and confidence
  python calculate_visual_values.py --image chart.png --model models/ved/model_final.pth --output my_results/ --confidence 0.7
        """
    )
    
    # Required arguments
    parser.add_argument("--image", type=str, required=True,
                        help="Path to input chart image")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained VED model weights (.pth or .pkl file)")
    
    # Optional arguments
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory for results (default: results)")
    parser.add_argument("--confidence", type=float, default=0.3,
                        help="Confidence threshold for detections (default: 0.3)")
    parser.add_argument("--use-caffe2", action="store_true", default=True,
                        help="Use Caffe2 compatible detector (default: True, recommended for original models)")
    parser.add_argument("--use-exact-caffe2", action="store_true", default=True,
                        help="Use exact Caffe2 architecture replication (default: True)")
    parser.add_argument("--use-detectron2", action="store_true",
                        help="Use Detectron2 detector instead of Caffe2 (overrides --use-caffe2)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode for detailed visual value calculation")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate arguments
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return 1
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return 1
    
    try:
        # Determine detector type
        use_caffe2 = not args.use_detectron2  # Use Caffe2 unless explicitly overridden
        use_exact_caffe2 = args.use_exact_caffe2 and use_caffe2
        
        # Initialize calculator
        calculator = VisualValuesCalculator(
            model_path=args.model,
            confidence_threshold=args.confidence,
            use_caffe2=use_caffe2,
            use_exact_caffe2=use_exact_caffe2,
            debug=args.debug
        )
        
        # Process image with visual value calculations
        results = calculator.process_with_values(args.image, args.output)
        
        # Print summary
        print(f"\n{'='*60}")
        print("Visual Values Calculation Complete!")
        print(f"{'='*60}")
        print(f"Input Image: {args.image}")
        print(f"Output Directory: {args.output}")
        print(f"Files Created:")
        for file_path in results.get("files_created", []):
            print(f"  - {file_path}")
        
        if "visual_values_file" in results:
            print(f"Visual Values: {results['visual_values_file']}")
            if "visual_elements_count" in results:
                print(f"Visual Elements Processed: {results['visual_elements_count']}")
        
        if "visual_values_error" in results:
            print(f"Warning: {results['visual_values_error']}")
        
        if not args.debug:
            print("\nNote: Enable --debug for full visual value calculation with coordinate-based scaling")
        
        return 0
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        print(f"\nError: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
