#!/usr/bin/env python3
"""
PlotQA End-to-End Data Extraction Script
Complete pipeline for extracting structured data from chart images.

This script combines:
1. Visual Element Detection (VED) using trained Detectron2 model
2. Optical Character Recognition (OCR) on detected text regions  
3. Structural Information Extraction (SIE) to build JSON output

Usage: python extract_data.py --image_path chart.png --model_path ./models/ved/model_final.pth --output result.json
"""

import argparse
import os
import json
import tempfile
import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Import our pipeline components
from generate_detections import PlotQADetector, save_detections_json_format
from ocr_and_sie import (OCRProcessor, StructuralExtractor, ChartElement, 
                         run_original_plotqa_pipeline, find_isHbar, 
                         preprocess_detections, find_visual_values)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChartDataExtractor:
    """End-to-end chart data extraction pipeline"""
    
    def __init__(self, model_path, confidence_threshold=0.5, config_path=None):
        """
        Initialize the extraction pipeline
        
        Args:
            model_path: Path to trained VED model weights
            confidence_threshold: Minimum confidence for detections
            config_path: Optional path to model config file
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        
        # Initialize components
        logger.info("Initializing VED detector...")
        self.detector = PlotQADetector(
            model_path=model_path,
            config_path=config_path,
            confidence_threshold=confidence_threshold
        )
        
        logger.info("Initializing OCR processor...")
        self.ocr_processor = OCRProcessor()
        
        logger.info("Pipeline initialized successfully")
    
    def extract_from_image(self, image_path, output_path=None):
        """
        Extract structured data from a single chart image
        
        Args:
            image_path: Path to input chart image
            output_path: Optional path to save JSON output
            
        Returns:
            Dictionary containing extracted chart data
        """
        logger.info(f"Processing image: {image_path}")
        
        # Validate input
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        try:
            # Step 1: Visual Element Detection
            logger.info("Step 1: Running visual element detection...")
            raw_detections = self.detector.detect_single_image(image_path)
            
            if not raw_detections:
                logger.warning("No visual elements detected in image")
                return {"error": "No chart elements found", "chart_type": "unknown"}
            
            logger.info(f"Detected {len(raw_detections)} visual elements")
            
            # Convert to ChartElement objects
            elements = []
            for detection in raw_detections:
                element = ChartElement(
                    class_name=detection[0],
                    confidence=detection[1],
                    bbox=detection[2:6]
                )
                elements.append(element)
            
            # Step 2: OCR on text elements with original PlotQA approach
            logger.info("Step 2: Applying OCR to text elements...")
            
            # Convert detections to lines format for isHbar detection
            lines = []
            for detection in raw_detections:
                class_name, confidence = detection[0], detection[1]
                x1, y1, x2, y2 = detection[2:6]
                lines.append(f"{class_name} {confidence} {x1} {y1} {x2} {y2}")
            
            lines = preprocess_detections(lines)
            isHbar, isSinglePlot = find_isHbar(lines)
            
            # Load image as PIL for original OCR processor
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            text_classes = ["title", "xlabel", "ylabel", "xticklabel", "yticklabel", "legend_label"]
            ocr_count = 0
            
            for element in elements:
                if element.class_name in text_classes:
                    try:
                        # Use enhanced OCR with role-specific processing
                        text = self.ocr_processor.extract_text(pil_image, element.bbox, 
                                                              role=element.class_name, 
                                                              isHbar=isHbar)
                        element.text = text
                        if text.strip():
                            ocr_count += 1
                    except Exception as e:
                        logger.warning(f"OCR failed for element {element.class_name}: {e}")
                        element.text = ""
            
            logger.info(f"Successfully extracted text from {ocr_count} elements")
            
            # Step 3: Structural Information Extraction
            logger.info("Step 3: Extracting structural information...")
            extractor = StructuralExtractor()
            
            # Add all elements to extractor
            for element in elements:
                extractor.add_element(element)
            
            # Perform structural extraction
            extractor.detect_chart_type()
            extractor.extract_axes_info(image.shape)
            extractor.extract_data_elements(image.shape)
            extractor.extract_title_and_legend()
            
            # Convert to structured data
            result = extractor.to_dict()
            
            # Add metadata
            result["metadata"] = {
                "image_path": str(image_path),
                "image_dimensions": [image.shape[1], image.shape[0]],  # [width, height]
                "total_detections": len(elements),
                "text_extractions": ocr_count,
                "confidence_threshold": self.confidence_threshold
            }
            
            logger.info(f"Extraction complete. Chart type: {result['chart_type']}")
            
            # Save output if path provided
            if output_path:
                self._save_result(result, output_path)
                logger.info(f"Results saved to: {output_path}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during extraction: {e}")
            return {
                "error": str(e),
                "chart_type": "unknown",
                "metadata": {
                    "image_path": str(image_path),
                    "confidence_threshold": self.confidence_threshold
                }
            }
    
    def extract_batch(self, image_dir, output_dir, file_pattern="*.png"):
        """
        Extract data from multiple images in a directory
        
        Args:
            image_dir: Directory containing chart images
            output_dir: Directory to save extraction results
            file_pattern: File pattern to match (default: "*.png")
            
        Returns:
            Dictionary mapping image paths to extraction results
        """
        logger.info(f"Processing batch from directory: {image_dir}")
        
        # Find images
        image_paths = list(Path(image_dir).glob(file_pattern))
        
        # Also try common image extensions
        for ext in ['*.jpg', '*.jpeg', '*.bmp', '*.tiff']:
            image_paths.extend(Path(image_dir).glob(ext))
        
        if not image_paths:
            raise ValueError(f"No images found in {image_dir}")
        
        logger.info(f"Found {len(image_paths)} images to process")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each image
        results = {}
        successful = 0
        
        for image_path in image_paths:
            try:
                # Generate output path
                output_name = image_path.stem + "_extracted.json"
                output_path = os.path.join(output_dir, output_name)
                
                # Extract data
                result = self.extract_from_image(str(image_path), output_path)
                results[str(image_path)] = result
                
                if "error" not in result:
                    successful += 1
                
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                results[str(image_path)] = {"error": str(e)}
        
        logger.info(f"Batch processing complete. {successful}/{len(image_paths)} images processed successfully")
        
        # Save batch summary
        summary_path = os.path.join(output_dir, "batch_summary.json")
        summary = {
            "total_images": len(image_paths),
            "successful": successful,
            "failed": len(image_paths) - successful,
            "results": results
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Batch summary saved to: {summary_path}")
        
        return results
    
    def _save_result(self, result, output_path):
        """Save extraction result to file"""
        output_path = Path(output_path)
        
        if output_path.suffix.lower() == '.json':
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
        else:
            # Default to JSON format
            with open(output_path.with_suffix('.json'), 'w') as f:
                json.dump(result, f, indent=2)

def validate_model_path(model_path):
    """Validate that model path exists and is accessible"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Check if it's a valid PyTorch model file
    if not model_path.endswith(('.pth', '.pkl')):
        logger.warning(f"Model file {model_path} doesn't have expected extension (.pth or .pkl)")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Extract structured data from chart images using PlotQA pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract from single image
  python extract_data.py --image_path chart.png --model_path ./models/ved/model_final.pth --output result.json
  
  # Extract from directory of images  
  python extract_data.py --image_dir ./charts/ --model_path ./models/ved/model_final.pth --output_dir ./results/
  
  # Use custom confidence threshold
  python extract_data.py --image_path chart.png --model_path ./models/ved/model_final.pth --confidence 0.7
        """
    )
    
    # Required arguments
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained VED model weights (.pth file)")
    
    # Input specification (either single image or directory)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image_path", type=str,
                       help="Path to single input chart image")
    group.add_argument("--image_dir", type=str,
                       help="Directory containing chart images")
    
    # Output specification
    parser.add_argument("--output", type=str,
                        help="Output file path (for single image) or directory (for batch)")
    parser.add_argument("--output_dir", type=str,
                        help="Output directory (for batch processing)")
    
    # Model parameters
    parser.add_argument("--config_path", type=str,
                        help="Path to model config file (optional)")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Confidence threshold for detections (default: 0.5)")
    
    # Processing options
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate arguments
    validate_model_path(args.model_path)
    
    if args.image_path and not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image not found: {args.image_path}")
    
    if args.image_dir and not os.path.exists(args.image_dir):
        raise FileNotFoundError(f"Directory not found: {args.image_dir}")
    
    # Initialize extractor
    try:
        extractor = ChartDataExtractor(
            model_path=args.model_path,
            config_path=args.config_path,
            confidence_threshold=args.confidence
        )
    except Exception as e:
        logger.error(f"Failed to initialize extractor: {e}")
        return 1
    
    # Process input
    try:
        if args.image_path:
            # Single image processing
            output_path = args.output
            if not output_path:
                # Generate default output path
                image_path = Path(args.image_path)
                output_path = image_path.parent / f"{image_path.stem}_extracted.json"
            
            result = extractor.extract_from_image(args.image_path, output_path)
            
            # Print summary
            if "error" in result:
                logger.error(f"Extraction failed: {result['error']}")
                return 1
            else:
                print(f"\nExtraction Summary:")
                print(f"Chart Type: {result['chart_type']}")
                print(f"Title: {result.get('title', 'N/A')}")
                print(f"Data Series: {len(result.get('data_series', []))}")
                print(f"Output: {output_path}")
                
        else:
            # Batch processing
            output_dir = args.output_dir or args.output
            if not output_dir:
                output_dir = "./extracted_data"
            
            results = extractor.extract_batch(args.image_dir, output_dir)
            
            # Print summary
            successful = sum(1 for r in results.values() if "error" not in r)
            print(f"\nBatch Processing Summary:")
            print(f"Total Images: {len(results)}")
            print(f"Successful: {successful}")
            print(f"Failed: {len(results) - successful}")
            print(f"Output Directory: {output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
