#!/usr/bin/env python3
"""
PlotQA Detection Generation Script
Generates object detections from trained VED model for chart images.
Outputs detections in the format required by OCR and SIE stages.
"""

import argparse
import os
import json
import logging
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.logger import setup_logger

# PlotQA categories - EXACT mapping from original generate_detections_for_fasterrcnn.py
PLOTQA_CATEGORIES = [
    {"id": 1, "name": "bar", "supercategory": "visual"},
    {"id": 2, "name": "dot_line", "supercategory": "visual"},
    {"id": 3, "name": "legend_label", "supercategory": "text"},
    {"id": 4, "name": "line", "supercategory": "visual"},
    {"id": 5, "name": "preview", "supercategory": "visual"},
    {"id": 6, "name": "title", "supercategory": "text"},
    {"id": 7, "name": "xlabel", "supercategory": "text"},
    {"id": 8, "name": "xticklabel", "supercategory": "text"},
    {"id": 9, "name": "ylabel", "supercategory": "text"},
    {"id": 10, "name": "yticklabel", "supercategory": "text"},
]

# Create category id to name mapping
CATEGORY_ID_TO_NAME = {cat["id"]: cat["name"] for cat in PLOTQA_CATEGORIES}

class PlotQADetector:
    """PlotQA visual element detector using trained Detectron2 model"""
    
    def __init__(self, model_path, config_path=None, confidence_threshold=0.5):
        """
        Initialize the detector
        
        Args:
            model_path: Path to trained model weights
            config_path: Path to model config (optional, will use default if None)
            confidence_threshold: Minimum confidence for detections
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(__name__)
        
        # Setup configuration
        self.cfg = self._setup_config(config_path)
        
        # Create predictor
        self.predictor = DefaultPredictor(self.cfg)
        
        self.logger.info(f"Loaded model from {model_path}")
        self.logger.info(f"Using confidence threshold: {confidence_threshold}")
    
    def _setup_config(self, config_path):
        """Setup Detectron2 configuration"""
        cfg = get_cfg()
        
        if config_path and os.path.exists(config_path):
            # Load custom config if provided
            cfg.merge_from_file(config_path)
        else:
            # Use default Faster R-CNN config
            from detectron2 import model_zoo
            cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
        
        # Set model weights and parameters
        cfg.MODEL.WEIGHTS = self.model_path
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(PLOTQA_CATEGORIES)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.confidence_threshold
        
        return cfg
    
    def detect_single_image(self, image_path):
        """
        Run detection on a single image
        
        Args:
            image_path: Path to input image
            
        Returns:
            List of detections in format: [class_name, confidence, xmin, ymin, xmax, ymax]
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Run inference
        outputs = self.predictor(image)
        
        # Extract predictions
        instances = outputs["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()
        
        # Convert to required format
        detections = []
        for box, score, class_id in zip(boxes, scores, classes):
            class_name = CATEGORY_ID_TO_NAME.get(class_id + 1, f"unknown_{class_id}")  # +1 because categories start from 1
            xmin, ymin, xmax, ymax = box
            
            detection = [class_name, float(score), float(xmin), float(ymin), float(xmax), float(ymax)]
            detections.append(detection)
        
        return detections
    
    def detect_batch(self, image_paths):
        """
        Run detection on multiple images
        
        Args:
            image_paths: List of image paths
            
        Returns:
            Dictionary mapping image paths to detections
        """
        results = {}
        
        for image_path in tqdm(image_paths, desc="Processing images"):
            try:
                detections = self.detect_single_image(image_path)
                results[image_path] = detections
            except Exception as e:
                self.logger.error(f"Error processing {image_path}: {e}")
                results[image_path] = []
        
        return results

def save_detections_text_format(detections, output_path):
    """
    Save detections in text format (PlotQA pipeline format)
    Format: CLASS_LABEL CLASS_CONFIDENCE XMIN YMIN XMAX YMAX
    This matches the original generate_detections_for_fasterrcnn.py output format
    """
    with open(output_path, 'w') as f:
        for detection in detections:
            # Ensure format matches original: CLASS_LABEL CLASS_CONFIDENCE XMIN YMIN XMAX YMAX
            class_name = detection[0]
            confidence = f"{detection[1]:.6f}"  # Match original precision
            bbox = [f"{coord:.2f}" for coord in detection[2:6]]  # Format coordinates
            line = f"{class_name} {confidence} {' '.join(bbox)}"
            f.write(line + '\n')

def save_detections_json_format(detections, output_path):
    """Save detections in JSON format"""
    detection_dicts = []
    for detection in detections:
        detection_dict = {
            "class": detection[0],
            "confidence": detection[1],
            "bbox": [detection[2], detection[3], detection[4], detection[5]]
        }
        detection_dicts.append(detection_dict)
    
    with open(output_path, 'w') as f:
        json.dump(detection_dicts, f, indent=2)

def process_directory(detector, input_dir, output_dir, output_format="text"):
    """Process all images in a directory"""
    
    # Find all image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(Path(input_dir).glob(f"*{ext}"))
        image_paths.extend(Path(input_dir).glob(f"*{ext.upper()}"))
    
    if not image_paths:
        raise ValueError(f"No images found in {input_dir}")
    
    print(f"Found {len(image_paths)} images")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process images
    for image_path in tqdm(image_paths, desc="Processing images"):
        try:
            detections = detector.detect_single_image(str(image_path))
            
            # Save detections
            output_name = image_path.stem
            if output_format == "json":
                output_file = os.path.join(output_dir, f"{output_name}_detections.json")
                save_detections_json_format(detections, output_file)
            else:
                output_file = os.path.join(output_dir, f"{output_name}_detections.txt")
                save_detections_text_format(detections, output_file)
                
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate detections from PlotQA VED model")
    
    # Required arguments
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model weights (.pth file)")
    
    # Input specification (either single image or directory)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image_path", type=str,
                       help="Path to single input image")
    group.add_argument("--image_dir", type=str,
                       help="Directory containing input images")
    
    # Output specification
    parser.add_argument("--output", type=str, required=True,
                        help="Output file path (for single image) or directory (for batch)")
    parser.add_argument("--output_format", type=str, choices=["text", "json"], default="text",
                        help="Output format: text (PlotQA format) or json (default: text)")
    
    # Model parameters
    parser.add_argument("--config_path", type=str,
                        help="Path to model config file (optional)")
    parser.add_argument("--confidence_threshold", type=float, default=0.5,
                        help="Confidence threshold for detections (default: 0.5)")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logger()
    logger = logging.getLogger(__name__)
    
    # Validate model path
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    
    # Initialize detector
    detector = PlotQADetector(
        model_path=args.model_path,
        config_path=args.config_path,
        confidence_threshold=args.confidence_threshold
    )
    
    if args.image_path:
        # Single image processing
        logger.info(f"Processing single image: {args.image_path}")
        
        detections = detector.detect_single_image(args.image_path)
        
        # Save results
        if args.output_format == "json":
            save_detections_json_format(detections, args.output)
        else:
            save_detections_text_format(detections, args.output)
        
        logger.info(f"Found {len(detections)} detections")
        logger.info(f"Results saved to: {args.output}")
        
    else:
        # Directory processing
        logger.info(f"Processing directory: {args.image_dir}")
        
        process_directory(detector, args.image_dir, args.output, args.output_format)
        
        logger.info(f"Results saved to: {args.output}")

if __name__ == "__main__":
    main()
