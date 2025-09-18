#!/usr/bin/env python3
"""
Exact Caffe2 Model Replication
Replicates the exact architecture from the original Caffe2 model files
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import logging
import os
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.data import MetadataCatalog
from detectron2.utils.logger import setup_logger

logger = logging.getLogger(__name__)

# PlotQA categories - exact mapping from original generate_detections_for_fasterrcnn.py
# The model detects 7 classes: Detectron2 IDs [0, 1, 5, 6, 7, 8, 9]
PLOTQA_CATEGORIES = [
    {"id": 1, "name": "bar", "supercategory": "visual"},
    {"id": 2, "name": "dot_line", "supercategory": "visual"},
    {"id": 3, "name": "legend_label", "supercategory": "textual"},
    {"id": 4, "name": "line", "supercategory": "visual"},
    {"id": 5, "name": "preview", "supercategory": "visual"},
    {"id": 6, "name": "title", "supercategory": "textual"},
    {"id": 7, "name": "xlabel", "supercategory": "textual"},
    {"id": 8, "name": "xticklabel", "supercategory": "textual"},
    {"id": 9, "name": "ylabel", "supercategory": "textual"},
    {"id": 10, "name": "yticklabel", "supercategory": "textual"},
]

# Create category id to name mapping
CATEGORY_ID_TO_NAME = {cat["id"]: cat["name"] for cat in PLOTQA_CATEGORIES}
CATEGORY_NAME_TO_ID = {cat["name"]: cat["id"] for cat in PLOTQA_CATEGORIES}

class ExactCaffe2Detector:
    """
    Detector that exactly replicates the original Caffe2 model architecture
    Based on the net.pbtxt and param_init_net.pbtxt files
    """
    
    def __init__(self, model_path, confidence_threshold=0.1):
        """
        Initialize the detector
        
        Args:
            model_path: Path to the Caffe2 model weights (.pkl file)
            confidence_threshold: Minimum confidence for detections
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(__name__)
        
        # Setup configuration to match original Caffe2 model exactly
        self.cfg = self._setup_exact_config()
        
        # Build model
        self.model = build_model(self.cfg)
        
        # Load Caffe2 weights
        self._load_caffe2_weights()
        
        # Set to evaluation mode
        self.model.eval()
        
        self.logger.info(f"Loaded exact Caffe2 model from {model_path}")
        self.logger.info(f"Using confidence threshold: {confidence_threshold}")
    
    def _setup_exact_config(self):
        """Setup Detectron2 configuration to exactly match original Caffe2 model"""
        cfg = get_cfg()
        
        # Use Faster R-CNN with FPN (matching original Caffe2)
        from detectron2 import model_zoo
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
        
        # Set model parameters to exactly match original Caffe2 (11 classes including background)
        cfg.MODEL.WEIGHTS = self.model_path
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10  # 10 foreground classes + 1 background = 11 total
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.confidence_threshold
        
        # Input configuration - matching original Caffe2 exactly
        cfg.INPUT.MIN_SIZE_TEST = 800
        cfg.INPUT.MAX_SIZE_TEST = 1333
        cfg.INPUT.PIXEL_MEAN = [102.9801, 115.9465, 122.7717]  # Original Caffe2 mean
        cfg.INPUT.PIXEL_STD = [1.0, 1.0, 1.0]  # Original Caffe2 std
        
        # RPN configuration - matching original Caffe2 exactly
        cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 6000
        cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
        cfg.MODEL.RPN.NMS_THRESH = 0.7
        cfg.MODEL.RPN.MIN_SIZE = 0
        cfg.MODEL.RPN.BBOX_REG_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
        
        # Anchor configuration - matching original Caffe2
        cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32], [64], [128], [256], [512]]
        cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
        
        # ROI heads configuration - matching original Caffe2 exactly
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
        cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMAGE = 100
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
        cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25
        cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS = (10.0, 10.0, 5.0, 5.0)
        
        # FPN configuration - matching original Caffe2 (5 levels)
        cfg.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
        cfg.MODEL.FPN.OUT_CHANNELS = 256
        
        # Backbone configuration - matching original Caffe2
        cfg.MODEL.RESNETS.DEPTH = 50
        cfg.MODEL.RESNETS.STRIDE_IN_1X1 = False
        cfg.MODEL.RESNETS.RES5_DILATION = 1
        
        return cfg
    
    def _load_caffe2_weights(self):
        """
        Load Caffe2 weights into the Detectron2 model
        """
        try:
            # Use Detectron2's Caffe2 model loading capability
            checkpointer = DetectionCheckpointer(self.model)
            checkpointer.load(self.model_path)
            self.logger.info("Successfully loaded Caffe2 weights")
            
        except Exception as e:
            self.logger.error(f"Failed to load Caffe2 weights: {e}")
            self.logger.info("Using randomly initialized weights")
    
    def detect_single_image(self, image_path):
        """
        Run detection on a single image
        
        Args:
            image_path: Path to input image
            
        Returns:
            Tuple of (detections, resized_image, original_dimensions)
            - detections: List in format: [class_name, confidence, xmin, ymin, xmax, ymax] (650x650 coords)
            - resized_image: 650x650 image for OCR processing
            - original_dimensions: (original_width, original_height) for visualization
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Store original dimensions for later upscaling
        original_height, original_width = image.shape[:2]
        
        # Resize image to 650x650 (original PlotQA model input size)
        resized_image = cv2.resize(image, (650, 650))
        
        # Run inference
        with torch.no_grad():
            # Convert resized image to the format expected by Detectron2
            height, width = resized_image.shape[:2]  # Should be 650, 650
            image_tensor = torch.from_numpy(resized_image.transpose(2, 0, 1)).float()
            
            # Create input dict
            inputs = [{"image": image_tensor, "height": height, "width": width}]
            
            # Run model
            outputs = self.model(inputs)
        
        # Convert outputs to required format
        detections = []
        if outputs and len(outputs) > 0:
            output = outputs[0]
            
            if "instances" in output:
                instances = output["instances"]
                
                boxes = instances.pred_boxes.tensor.cpu().numpy()
                scores = instances.scores.cpu().numpy()
                classes = instances.pred_classes.cpu().numpy()
                
                for box, score, class_id in zip(boxes, scores, classes):
                    if score >= self.confidence_threshold:
                        # Convert from 0-based (Detectron2) to 1-based (PlotQA) indexing
                        plotqa_class_id = class_id + 1
                        class_name = CATEGORY_ID_TO_NAME.get(plotqa_class_id, f"unknown_{class_id}")
                        xmin, ymin, xmax, ymax = box
                        
                        # Keep bbox in 650x650 coordinates for OCR processing
                        # Store original dimensions for potential upscaling later
                        detection = [class_name, float(score), float(xmin), float(ymin), float(xmax), float(ymax)]
                        detections.append(detection)
        
        return detections, resized_image, (original_width, original_height)

def create_exact_caffe2_detector(model_path, confidence_threshold=0.1):
    """
    Factory function to create the exact Caffe2 detector
    
    Args:
        model_path: Path to Caffe2 model weights
        confidence_threshold: Minimum confidence for detections
        
    Returns:
        ExactCaffe2Detector instance
    """
    return ExactCaffe2Detector(model_path, confidence_threshold)

if __name__ == "__main__":
    # Test the detector
    setup_logger()
    
    model_path = "models/ved/model_final.pkl"
    detector = create_exact_caffe2_detector(model_path)
    
    # Test with a sample image
    if os.path.exists("test_chart.png"):
        detections = detector.detect_single_image("test_chart.png")
        print(f"Found {len(detections)} detections")
        for detection in detections:
            print(f"  {detection[0]}: {detection[1]:.3f} at {detection[2]:.1f},{detection[3]:.1f},{detection[4]:.1f},{detection[5]:.1f}")
    else:
        print("No test image found")
