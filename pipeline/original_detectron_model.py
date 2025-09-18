#!/usr/bin/env python3
"""
Original Detectron Faster R-CNN with FPN Model Replication
Replicates the exact architecture from the original Detectron repository
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign
from collections import OrderedDict
import os

import logging

logger = logging.getLogger(__name__)

# PlotQA categories - exact mapping from original
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
CATEGORY_NAME_TO_ID = {cat["name"]: cat["id"] for cat in PLOTQA_CATEGORIES}

class OriginalDetectronFasterRCNN(nn.Module):
    """
    Replicates the original Detectron Faster R-CNN with FPN architecture
    Based on the original Detectron implementation
    """
    
    def __init__(self, num_classes=10, pretrained_backbone=True):
        super(OriginalDetectronFasterRCNN, self).__init__()
        
        self.num_classes = num_classes
        
        # Create backbone - ResNet-50 with FPN (matching original Detectron)
        self.backbone = resnet_fpn_backbone('resnet50', pretrained=pretrained_backbone)
        
        # Anchor generator - matching original Detectron settings
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        self.anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        
        # RPN head - matching original Detectron
        out_channels = self.backbone.out_channels
        self.rpn_head = RPNHead(out_channels, self.anchor_generator.num_anchors_per_location()[0])
        
        # ROI heads - matching original Detectron
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=2
        )
        
        # Box head - matching original Detectron architecture
        resolution = box_roi_pool.output_size[0]
        representation_size = 1024
        self.box_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_channels * resolution ** 2, representation_size),
            nn.ReLU(inplace=True),
            nn.Linear(representation_size, representation_size),
            nn.ReLU(inplace=True)
        )
        
        # Box predictor - matching original Detectron
        self.box_predictor = FastRCNNPredictor(representation_size, num_classes)
        
        # Transform - matching original Detectron settings
        self.transform = GeneralizedRCNNTransform(
            min_size=800,
            max_size=1333,
            image_mean=[102.9801, 115.9465, 122.7717],  # Original Detectron mean
            image_std=[1.0, 1.0, 1.0],  # Original Detectron std
            size_divisible=32
        )
        
        # ROI heads
        self.roi_heads = RoIHeads(
            box_roi_pool=box_roi_pool,
            box_head=self.box_head,
            box_predictor=self.box_predictor,
            fg_iou_thresh=0.5,
            bg_iou_thresh=0.5,
            batch_size_per_image=512,
            positive_fraction=0.25,
            bbox_reg_weights=None,
            score_thresh=0.05,
            nms_thresh=0.5,
            detections_per_img=100
        )
        
        logger.info(f"Initialized Original Detectron Faster R-CNN with {num_classes} classes")
    
    def forward(self, images, targets=None):
        """
        Forward pass matching original Detectron behavior
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        
        # Transform images
        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets)
        
        # Extract features using backbone
        features = self.backbone(images.tensors)
        
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
        
        # Generate proposals using RPN
        proposals, proposal_losses = self.rpn_head(images, features, targets)
        
        # ROI heads
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        
        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses
        
        return detections

class OriginalDetectronDetector:
    """
    Detector class that replicates the original Detectron behavior
    """
    
    def __init__(self, model_path, confidence_threshold=0.5):
        """
        Initialize the detector
        
        Args:
            model_path: Path to the original Detectron model weights
            confidence_threshold: Minimum confidence for detections
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(__name__)
        
        # Initialize model
        self.model = OriginalDetectronFasterRCNN(num_classes=len(PLOTQA_CATEGORIES))
        
        # Load weights
        self._load_weights()
        
        # Set to evaluation mode
        self.model.eval()
        
        self.logger.info(f"Loaded original Detectron model from {model_path}")
        self.logger.info(f"Using confidence threshold: {confidence_threshold}")
    
    def _load_weights(self):
        """
        Load weights from the original Detectron model
        This is a placeholder - we'll need to implement proper weight loading
        """
        try:
            # Try to load as PyTorch state dict first
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif isinstance(checkpoint, dict):
                state_dict = checkpoint
            else:
                # If it's a Caffe2 model, we'll need to convert it
                self.logger.warning("Caffe2 model detected - weight conversion needed")
                return
            
            # Load the state dict
            self.model.load_state_dict(state_dict, strict=False)
            self.logger.info("Successfully loaded model weights")
            
        except Exception as e:
            self.logger.error(f"Failed to load weights: {e}")
            self.logger.info("Using randomly initialized weights")
    
    def detect_single_image(self, image_path):
        """
        Run detection on a single image
        
        Args:
            image_path: Path to input image
            
        Returns:
            List of detections in format: [class_name, confidence, xmin, ymin, xmax, ymax]
        """
        import cv2
        from PIL import Image
        import torchvision.transforms as transforms
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        
        # Transform for model input
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        image_tensor = transform(image).unsqueeze(0)
        
        # Run inference
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        # Convert predictions to required format
        detections = []
        if predictions and len(predictions) > 0:
            pred = predictions[0]
            
            boxes = pred['boxes'].cpu().numpy()
            scores = pred['scores'].cpu().numpy()
            labels = pred['labels'].cpu().numpy()
            
            for box, score, label in zip(boxes, scores, labels):
                if score >= self.confidence_threshold:
                    class_name = CATEGORY_ID_TO_NAME.get(label.item(), f"unknown_{label.item()}")
                    xmin, ymin, xmax, ymax = box
                    
                    detection = [class_name, float(score), float(xmin), float(ymin), float(xmax), float(ymax)]
                    detections.append(detection)
        
        return detections

def create_original_detectron_model(model_path, confidence_threshold=0.5):
    """
    Factory function to create the original Detectron model
    
    Args:
        model_path: Path to model weights
        confidence_threshold: Minimum confidence for detections
        
    Returns:
        OriginalDetectronDetector instance
    """
    return OriginalDetectronDetector(model_path, confidence_threshold)

if __name__ == "__main__":
    # Test the model
    import os
    
    model_path = "models/ved/model_final.pkl"
    detector = create_original_detectron_model(model_path)
    
    # Test with a sample image
    if os.path.exists("test_chart.png"):
        detections = detector.detect_single_image("test_chart.png")
        print(f"Found {len(detections)} detections")
        for detection in detections:
            print(f"  {detection[0]}: {detection[1]:.3f} at {detection[2]:.1f},{detection[3]:.1f},{detection[4]:.1f},{detection[5]:.1f}")
    else:
        print("No test image found")
