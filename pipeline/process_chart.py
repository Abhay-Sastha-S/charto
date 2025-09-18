#!/usr/bin/env python3
"""
Complete PlotQA Chart Processing Script
Runs the full VED → OCR → SIE pipeline on a single chart image.

Usage:
    python process_chart.py --image chart.png --model models/ved/model_final.pth
    python process_chart.py --image chart.png --model models/ved/model_final.pth --output results/
    python process_chart.py --image chart.png --model models/ved/model_final.pth --confidence 0.7

Debugging:
    $env:KMP_DUPLICATE_LIB_OK="TRUE",
    then run again
"""
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
import pyocr
pyocr.tesseract.TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

import os
import sys
import json
import tempfile
import argparse
import logging
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image

# Import our pipeline components
from generate_detections import PlotQADetector, save_detections_text_format
from caffe2_compatible_detector import Caffe2CompatibleDetector
from exact_caffe2_detector import ExactCaffe2Detector
from ocr_and_sie import run_original_plotqa_pipeline

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PlotQAProcessor:
    """Complete PlotQA pipeline processor for single images"""
    
    def __init__(self, model_path, confidence_threshold=0.1, use_caffe2=True, use_exact_caffe2=True, debug=False):
        """
        Initialize the PlotQA processor
        
        Args:
            model_path: Path to trained VED model weights
            confidence_threshold: Minimum confidence for detections
            use_caffe2: Whether to use Caffe2 compatible detector (recommended for original models)
            use_exact_caffe2: Whether to use exact Caffe2 architecture replication
            debug: Whether to save intermediary results for debugging
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.use_caffe2 = use_caffe2
        self.use_exact_caffe2 = use_exact_caffe2
        self.debug = debug
        
        # Create temp directory for debug outputs
        if self.debug:
            self.temp_dir = "temp"
            os.makedirs(self.temp_dir, exist_ok=True)
            logger.info(f"Debug mode enabled. Intermediary results will be saved to: {self.temp_dir}")
        
        # Validate model path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        logger.info(f"Initializing PlotQA processor with model: {model_path}")
        logger.info(f"Using confidence threshold: {confidence_threshold}")
        
        # Initialize VED detector
        if use_caffe2:
            if use_exact_caffe2:
                self.detector = ExactCaffe2Detector(
                    model_path=model_path,
                    confidence_threshold=confidence_threshold
                )
                logger.info("Using exact Caffe2 architecture replication for original PlotQA models")
            else:
                self.detector = Caffe2CompatibleDetector(
                    model_path=model_path,
                    confidence_threshold=confidence_threshold
                )
                logger.info("Using Caffe2 compatible detector for original PlotQA models")
        else:
            self.detector = PlotQADetector(
                model_path=model_path,
                confidence_threshold=confidence_threshold
            )
            logger.info("Using Detectron2 detector")
    
    def process_image(self, image_path, output_dir="results"):
        """
        Process a single chart image through the complete pipeline
        
        Args:
            image_path: Path to input chart image
            output_dir: Directory to save results
            
        Returns:
            Dictionary with paths to generated files
        """
        logger.info(f"Processing image: {image_path}")
        
        # Validate input
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get image name for file naming
        image_name = Path(image_path).stem
        
        # Create temporary directories for intermediate files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_png_dir = os.path.join(temp_dir, "images")
            temp_detections_dir = os.path.join(temp_dir, "detections")
            temp_csv_dir = os.path.join(temp_dir, "csv")
            
            os.makedirs(temp_png_dir, exist_ok=True)
            os.makedirs(temp_detections_dir, exist_ok=True)
            os.makedirs(temp_csv_dir, exist_ok=True)
            
            try:
                # Step 1: Prepare image
                logger.info("Step 1: Preparing image...")
                prepared_image_path = self._prepare_image(image_path, temp_png_dir, image_name)
                
                # Step 2: Visual Element Detection (VED)
                logger.info("Step 2: Running Visual Element Detection...")
                detections_path, resized_image, original_dimensions = self._run_ved(prepared_image_path, temp_detections_dir, image_name)
                
                # Step 3: OCR and Structural Information Extraction
                logger.info("Step 3: Running OCR and SIE...")
                self._run_ocr_sie(temp_png_dir, temp_detections_dir, temp_csv_dir, resized_image, original_dimensions)
                
                # Step 4: Collect and format results
                logger.info("Step 4: Collecting results...")
                results = self._collect_results(temp_csv_dir, output_dir, image_name, image_path)
                
                logger.info(f"Processing complete! Results saved to: {output_dir}")
                return results
                
            except Exception as e:
                logger.error(f"Error during processing: {e}")
                raise
    
    def _prepare_image(self, image_path, temp_png_dir, image_name):
        """Prepare image for processing (convert to PNG with numeric name)"""
        # Load and validate image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # The original pipeline expects images with numeric names
        # We'll use a simple numeric ID (1) for single image processing
        numeric_name = "1"
        output_path = os.path.join(temp_png_dir, f"{numeric_name}.png")
        
        # Save as PNG
        cv2.imwrite(output_path, image)
        logger.info(f"Image prepared: {output_path}")
        
        return output_path
    
    def _run_ved(self, image_path, detections_dir, image_name):
        """Run Visual Element Detection"""
        logger.info("Running VED model inference...")
        
        # Run detection
        detections, resized_image, original_dimensions = self.detector.detect_single_image(image_path)
        
        if not detections:
            logger.warning("No visual elements detected in image")
            return None
        
        logger.info(f"Detected {len(detections)} visual elements")
        
        # Debug: Save detailed detection information
        if self.debug:
            self._save_detection_debug_info(detections, image_path, original_dimensions)
        
        # Save detections in PlotQA format (numeric name expected)
        detections_path = os.path.join(detections_dir, "1.txt")
        save_detections_text_format(detections, detections_path)
        
        logger.info(f"Detections saved: {detections_path}")
        return detections_path, resized_image, original_dimensions
    
    def _run_ocr_sie(self, png_dir, detections_dir, csv_dir, resized_image, original_dimensions):
        """Run OCR and Structural Information Extraction"""
        logger.info("Running OCR and SIE...")
        
        # Debug: Save OCR debug information
        if self.debug:
            self._save_ocr_debug_info(png_dir, detections_dir, resized_image, original_dimensions)
        
        # Use the original PlotQA pipeline with lower threshold for tick labels
        run_original_plotqa_pipeline(
            png_dir=png_dir,
            detections_dir=detections_dir,
            csv_dir=csv_dir,
            MIN_CLASS_CONFIDENCE=self.confidence_threshold,
            MIN_TICKLABEL_CONFIDENCE=0.05,  # Lower threshold for tick labels
            debug=self.debug  # Pass debug flag to OCR pipeline
        )
        
        logger.info("OCR and SIE completed")
    
    def _collect_results(self, temp_csv_dir, output_dir, image_name, original_image_path):
        """Collect and format final results"""
        results = {
            "image_path": str(original_image_path),
            "output_directory": str(output_dir),
            "files_created": []
        }
        
        # Look for generated CSV file (should be 1.csv)
        csv_source = os.path.join(temp_csv_dir, "1.csv")
        
        if os.path.exists(csv_source):
            # Copy CSV to output directory with proper name
            csv_output = os.path.join(output_dir, f"{image_name}.csv")
            shutil.copy2(csv_source, csv_output)
            results["files_created"].append(csv_output)
            results["csv_file"] = csv_output
            
            # Convert CSV to structured JSON
            json_output = os.path.join(output_dir, f"{image_name}.json")
            self._csv_to_json(csv_output, json_output, original_image_path)
            results["files_created"].append(json_output)
            results["json_file"] = json_output
            
            logger.info(f"Results saved:")
            logger.info(f"  CSV: {csv_output}")
            logger.info(f"  JSON: {json_output}")
        else:
            logger.warning("No CSV output generated - processing may have failed")
            results["error"] = "No output generated"
        
        # Save processing metadata
        metadata_output = os.path.join(output_dir, f"{image_name}_metadata.json")
        with open(metadata_output, 'w') as f:
            json.dump(results, f, indent=2)
        results["files_created"].append(metadata_output)
        results["metadata_file"] = metadata_output
        
        return results
    
    def _csv_to_json(self, csv_path, json_path, image_path):
        """Convert PlotQA CSV output to structured JSON"""
        try:
            import pandas as pd
            
            # Read CSV
            df = pd.read_csv(csv_path)
            
            if df.empty:
                logger.warning("CSV file is empty")
                structured_data = {"error": "No data extracted"}
            else:
                # Note: OCR corrections are handled by the original PlotQA OCR pipeline
                # No additional hard-coded corrections needed
                
                # Extract metadata from CSV
                title = df['title'].iloc[0] if 'title' in df.columns and not df.empty else ""
                xlabel = df['xlabel'].iloc[0] if 'xlabel' in df.columns and not df.empty else ""
                ylabel = df['ylabel'].iloc[0] if 'ylabel' in df.columns and not df.empty else ""
                
                # Get the main axis column (first column that's not metadata)
                metadata_cols = ['title', 'xlabel', 'ylabel', 'legend orientation']
                data_cols = [col for col in df.columns if col not in metadata_cols]
                
                # Determine chart type based on columns and data
                chart_type = "unknown"
                
                # Check for bar chart indicators
                if any("bar" in str(col).lower() for col in df.columns):
                    chart_type = "bar"
                # Check for line chart indicators  
                elif any("line" in str(col).lower() for col in df.columns):
                    chart_type = "line"
                # Check data patterns to infer chart type
                elif len(data_cols) > 1:
                    # If we have multiple data columns, likely a bar chart
                    chart_type = "bar"
                elif len(data_cols) == 1:
                    # Single data column could be bar or line
                    # Check if values are discrete (likely bar) or continuous (likely line)
                    try:
                        value_col = data_cols[0]
                        if value_col in df.columns:
                            values = pd.to_numeric(df[value_col], errors='coerce').dropna()
                            if len(values) > 0:
                                # If values are mostly integers and small range, likely bar chart
                                if all(v == int(v) for v in values) and len(set(values)) <= 10:
                                    chart_type = "bar"
                                else:
                                    chart_type = "line"
                    except:
                        chart_type = "bar"  # Default to bar for single series
                
                # Extract data series
                data_series = []
                
                if data_cols:
                    axis_col = data_cols[0]  # First data column is usually the axis
                    value_cols = data_cols[1:]  # Remaining columns are values
                    
                    for col in value_cols:
                        if col not in metadata_cols:
                            series_data = []
                            for idx, row in df.iterrows():
                                if pd.notna(row[col]) and pd.notna(row[axis_col]):
                                    series_data.append({
                                        "x": str(row[axis_col]),
                                        "y": row[col] if pd.notna(row[col]) else 0
                                    })
                            
                            if series_data:  # Only add non-empty series
                                data_series.append({
                                    "name": col,
                                    "type": chart_type,
                                    "data": series_data
                                })
                
                # Create structured JSON
                structured_data = {
                    "chart_type": chart_type,
                    "title": title,
                    "x_axis": {
                        "label": xlabel,
                        "type": "categorical"
                    },
                    "y_axis": {
                        "label": ylabel,
                        "type": "numeric"
                    },
                    "data_series": data_series,
                    "metadata": {
                        "source_image": str(image_path),
                        "extraction_method": "PlotQA Pipeline",
                        "total_series": len(data_series)
                    }
                }
            
            # Save JSON
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(structured_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Structured JSON created: {json_path}")
            
        except Exception as e:
            logger.error(f"Error converting CSV to JSON: {e}")
            # Create error JSON
            error_data = {
                "error": f"Failed to convert CSV to JSON: {str(e)}",
                "source_image": str(image_path)
            }
            with open(json_path, 'w') as f:
                json.dump(error_data, f, indent=2)
    
    def _save_detection_debug_info(self, detections, image_path, original_dimensions):
        """Save detailed detection information for debugging"""
        import json
        from PIL import Image, ImageDraw, ImageFont
        
        # Load original image
        image = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(image)
        
        # Color mapping for different classes
        colors = {
            'bar': 'red',
            'xticklabel': 'blue', 
            'yticklabel': 'green',
            'xlabel': 'orange',
            'ylabel': 'purple',
            'title': 'yellow',
            'legend_label': 'cyan'
        }
        
        # Create detection summary
        detection_summary = {
            "total_detections": len(detections),
            "detections_by_class": {},
            "detection_details": []
        }
        
        # Get scaling factors for upscaling from 650x650 to original image size
        original_width, original_height = original_dimensions
        x_scale = original_width / 650.0
        y_scale = original_height / 650.0
        
        # Group by class and create detailed info
        for i, detection in enumerate(detections):
            class_name, confidence, x1, y1, x2, y2 = detection
            
            # Upscale bbox for visualization on original image
            upscaled_x1 = x1 * x_scale
            upscaled_y1 = y1 * y_scale
            upscaled_x2 = x2 * x_scale
            upscaled_y2 = y2 * y_scale
            
            # Count by class
            if class_name not in detection_summary["detections_by_class"]:
                detection_summary["detections_by_class"][class_name] = 0
            detection_summary["detections_by_class"][class_name] += 1
            
            # Add detailed info (store both 650x650 and upscaled coordinates)
            detection_info = {
                "id": i,
                "class": class_name,
                "confidence": confidence,
                "bbox_650x650": [x1, y1, x2, y2],
                "bbox_upscaled": [upscaled_x1, upscaled_y1, upscaled_x2, upscaled_y2],
                "width": x2 - x1,
                "height": y2 - y1
            }
            detection_summary["detection_details"].append(detection_info)
            
            # Draw bounding box (upscaled for visualization)
            color = colors.get(class_name, 'red')
            draw.rectangle([upscaled_x1, upscaled_y1, upscaled_x2, upscaled_y2], outline=color, width=2)
            
            # Add label
            label = f"{class_name} ({confidence:.3f})"
            try:
                font = ImageFont.truetype("arial.ttf", 12)
            except:
                font = ImageFont.load_default()
            
            # Draw text background
            text_bbox = draw.textbbox((x1, y1-20), label, font=font)
            draw.rectangle(text_bbox, fill=color)
            draw.text((x1, y1-20), label, fill='white', font=font)
        
        # Save detection summary JSON
        with open(os.path.join(self.temp_dir, "detection_summary.json"), 'w') as f:
            json.dump(detection_summary, f, indent=2)
        
        # Save visualization image
        image.save(os.path.join(self.temp_dir, "detections_visualized.png"))
        
        logger.info(f"Debug: Detection info saved to {self.temp_dir}/detection_summary.json")
        logger.info(f"Debug: Detection visualization saved to {self.temp_dir}/detections_visualized.png")
    
    def _save_ocr_debug_info(self, png_dir, detections_dir, resized_image, original_dimensions):
        """Save OCR debug information including extended bounding boxes"""
        import json
        import cv2
        from PIL import Image, ImageDraw, ImageFont
        from ocr_and_sie import OCRProcessor, find_isHbar, preprocess_detections
        from upscale_boxes import upscale_boxes
        
        # Use the 650x650 resized image for OCR processing
        image = Image.fromarray(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)
        
        # Create a copy for visualization (upscaled to original size)
        original_width, original_height = original_dimensions
        visualization_image = image.resize((original_width, original_height), Image.Resampling.LANCZOS)
        vis_draw = ImageDraw.Draw(visualization_image)
        
        # Load detections
        detections_path = os.path.join(detections_dir, "1.txt")
        with open(detections_path, 'r') as f:
            lines = f.read().split("\n")[:-1]
        
        lines = preprocess_detections(lines)
        isHbar, isSinglePlot = find_isHbar(lines)
        
        # Color mapping
        colors = {
            'bar': 'red',
            'xticklabel': 'blue', 
            'yticklabel': 'green',
            'xlabel': 'orange',
            'ylabel': 'purple',
            'title': 'yellow',
            'legend_label': 'cyan'
        }
        
        # OCR debug info
        ocr_debug = {
            "chart_orientation": {"isHbar": isHbar, "isSinglePlot": isSinglePlot},
            "ocr_results": [],
            "class_summary": {}
        }
        
        ocr_processor = OCRProcessor(debug=True, debug_dir=os.path.join(self.temp_dir, "debug_crops"))
        img_width, img_height = image.size  # Should be 650x650
        
        # Get scaling factors for visualization
        x_scale = original_width / 650.0
        y_scale = original_height / 650.0
        
        # Process each detection
        for i, line in enumerate(lines):
            parts = line.split()
            if len(parts) < 6:
                continue
                
            class_name, score = parts[0], float(parts[1])
            x1, y1, x2, y2 = [float(x) for x in parts[2:6]]  # These are in 650x650 coordinates
            
            # Skip non-text elements for OCR
            if class_name not in ["title", "xlabel", "ylabel", "xticklabel", "yticklabel", "legend_label"]:
                continue
            
            # Apply confidence filtering
            if class_name in ['xticklabel', 'yticklabel']:
                if score < 0.05:
                    continue
            else:
                if score < self.confidence_threshold:
                    continue
            
            # Get original bbox (in 650x650 coordinates)
            orig_bbox_650 = [x1, y1, x2, y2]
            
            # Apply percentage-based padding for better scaling (in 650x650 coordinates)
            # Calculate bbox dimensions for percentage-based padding
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            
            if class_name == 'xticklabel':
                # 15% padding horizontally, 20% vertically for tick labels
                width_pad = bbox_width * 0.30
                height_pad = bbox_height * 0.30
                ex_x1 = max(0, x1 - width_pad)
                ex_y1 = max(0, y1 - height_pad)
                ex_x2 = x2 + width_pad
                ex_y2 = y2 + height_pad
                width_extension = width_pad
                height_extension = height_pad
            elif class_name == 'yticklabel':
                # 20% padding horizontally, 15% vertically for tick labels
                width_pad = bbox_width * 0.30
                height_pad = bbox_height * 0.30
                ex_x1 = max(0, x1 - width_pad)
                ex_y1 = max(0, y1 - height_pad)
                ex_x2 = x2 + width_pad
                ex_y2 = y2 + height_pad
                width_extension = width_pad
                height_extension = height_pad
            elif class_name == 'title':
                # 10% padding in all directions for title
                width_pad = bbox_width * 0.10
                height_pad = bbox_height * 0.10
                ex_x1 = max(0, x1 - width_pad)
                ex_y1 = max(0, y1 - height_pad)
                ex_x2 = x2 + width_pad
                ex_y2 = y2 + height_pad
                width_extension = width_pad
                height_extension = height_pad
            elif class_name == 'xlabel':
                # 12% padding horizontally, 10% vertically for x-axis label
                width_pad = bbox_width * 0.15
                height_pad = bbox_height * 0.15
                ex_x1 = max(0, x1 - width_pad)
                ex_y1 = max(0, y1 - height_pad)
                ex_x2 = x2 + width_pad
                ex_y2 = y2 + height_pad
                width_extension = width_pad
                height_extension = height_pad
            elif class_name == 'ylabel':
                # 10% padding horizontally, 12% vertically for y-axis label
                width_pad = bbox_width * 0.15
                height_pad = bbox_height * 0.15
                ex_x1 = max(0, x1 - width_pad)
                ex_y1 = max(0, y1 - height_pad)
                ex_x2 = x2 + width_pad
                ex_y2 = y2 + height_pad
                width_extension = width_pad
                height_extension = height_pad
            elif class_name == 'legend_label':
                # 8% padding in all directions for legend labels
                width_pad = bbox_width * 0.10
                height_pad = bbox_height * 0.10
                ex_x1 = max(0, x1 - width_pad)
                ex_y1 = max(0, y1 - height_pad)
                ex_x2 = x2 + width_pad
                ex_y2 = y2 + height_pad
                width_extension = width_pad
                height_extension = height_pad
            else:
                # No padding for unknown elements
                ex_x1, ex_y1, ex_x2, ex_y2 = x1, y1, x2, y2
                width_extension = 0.1
                height_extension = 0.1
            
            extended_bbox_650 = [ex_x1, ex_y1, ex_x2, ex_y2]
            
            # Debug: Log the coordinates being passed to OCR
            if self.debug and class_name == 'xticklabel':
                logger.info(f"OCR Debug - {class_name}:")
                logger.info(f"  Original bbox 650: {orig_bbox_650}")
                logger.info(f"  Extended bbox 650: {extended_bbox_650}")
                logger.info(f"  Extension: {width_extension}x{height_extension}")
            
            # Perform OCR using 650x650 image and coordinates
            text = ocr_processor.extract_text(image, extended_bbox_650, role=class_name, isHbar=isHbar)
            
            # Upscale coordinates for visualization
            orig_bbox_upscaled = [x1 * x_scale, y1 * y_scale, x2 * x_scale, y2 * y_scale]
            extended_bbox_upscaled = [ex_x1 * x_scale, ex_y1 * y_scale, ex_x2 * x_scale, ex_y2 * y_scale]
            
            # Store OCR result
            ocr_result = {
                "class": class_name,
                "confidence": score,
                "original_bbox_650": orig_bbox_650,
                "extended_bbox_650": extended_bbox_650,
                "original_bbox_upscaled": orig_bbox_upscaled,
                "extended_bbox_upscaled": extended_bbox_upscaled,
                "ocr_text": text,
                "bbox_extension": {
                    "width_extension": width_extension,
                    "height_extension": height_extension,
                    "extension_type": "percentage_based_adaptive",
                    "original_bbox_size": {"width": bbox_width, "height": bbox_height}
                }
            }
            ocr_debug["ocr_results"].append(ocr_result)
            
            # Update class summary
            if class_name not in ocr_debug["class_summary"]:
                ocr_debug["class_summary"][class_name] = {"count": 0, "texts": []}
            ocr_debug["class_summary"][class_name]["count"] += 1
            ocr_debug["class_summary"][class_name]["texts"].append(text)
            
            # Draw visualization on upscaled image
            color = colors.get(class_name, 'red')
            
            # Draw original bbox (thin line) - upscaled coordinates
            vis_draw.rectangle(orig_bbox_upscaled, outline=color, width=1)
            
            # Draw extended bbox (thick line) - upscaled coordinates
            vis_draw.rectangle(extended_bbox_upscaled, outline=color, width=3)
            
            # Add label
            label = f"{class_name}: '{text}'"
            try:
                font = ImageFont.truetype("arial.ttf", 10)
            except:
                font = ImageFont.load_default()
            
            # Draw text background - upscaled coordinates
            text_bbox = vis_draw.textbbox((orig_bbox_upscaled[0], orig_bbox_upscaled[1]-15), label, font=font)
            vis_draw.rectangle(text_bbox, fill=color)
            vis_draw.text((orig_bbox_upscaled[0], orig_bbox_upscaled[1]-15), label, fill='white', font=font)
        
        # Save OCR debug JSON
        with open(os.path.join(self.temp_dir, "ocr_debug.json"), 'w') as f:
            json.dump(ocr_debug, f, indent=2)
        
        # Save OCR visualization (upscaled to original image size)
        visualization_image.save(os.path.join(self.temp_dir, "ocr_with_extended_bboxes.png"))
        
        logger.info(f"Debug: OCR info saved to {self.temp_dir}/ocr_debug.json")
        logger.info(f"Debug: OCR visualization saved to {self.temp_dir}/ocr_with_extended_bboxes.png")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Process a chart image through the complete PlotQA pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python process_chart.py --image chart.png --model models/ved/model_final.pth
  
  # Custom output directory
  python process_chart.py --image chart.png --model models/ved/model_final.pth --output my_results/
  
  # Custom confidence threshold
  python process_chart.py --image chart.png --model models/ved/model_final.pth --confidence 0.7
  
  # Verbose output
  python process_chart.py --image chart.png --model models/ved/model_final.pth --verbose
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
    parser.add_argument("--use-caffe2", action="store_true",
                        help="Use Caffe2 compatible detector (recommended for original models)")
    parser.add_argument("--use-exact-caffe2", action="store_true",
                        help="Use exact Caffe2 architecture replication")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode to save intermediary results to temp/ folder")
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
        # Initialize processor
        processor = PlotQAProcessor(
            model_path=args.model,
            confidence_threshold=args.confidence,
            use_caffe2=args.use_caffe2,
            use_exact_caffe2=args.use_exact_caffe2,
            debug=args.debug
        )
        
        # Process image
        results = processor.process_image(args.image, args.output)
        
        # Print summary
        print(f"\n{'='*50}")
        print("PlotQA Processing Complete!")
        print(f"{'='*50}")
        print(f"Input Image: {args.image}")
        print(f"Output Directory: {args.output}")
        print(f"Files Created:")
        for file_path in results.get("files_created", []):
            print(f"  - {file_path}")
        
        if "error" in results:
            print(f"Warning: {results['error']}")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        print(f"\nError: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
