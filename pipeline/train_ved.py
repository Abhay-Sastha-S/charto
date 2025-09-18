#!/usr/bin/env python3
"""
PlotQA VED (Visual Element Detection) Training Script
Trains a Detectron2 Faster R-CNN with FPN model on PlotQA dataset for detecting chart elements.
"""

import argparse
import os
import json
import logging
from pathlib import Path

import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.logger import setup_logger

# PlotQA dataset categories - EXACT mapping from original generate_detections_for_fasterrcnn.py
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

class PlotQATrainer(DefaultTrainer):
    """Custom trainer for PlotQA dataset with COCO evaluator"""
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)

def register_plotqa_datasets(data_dir):
    """Register PlotQA datasets with Detectron2 - matches original dataset_catalog.py structure"""
    
    # Convert data_dir to Path object for easier handling
    data_path = Path(data_dir)
    
    # Register multiple training datasets as in original dataset_catalog.py
    train_configs = [
        ("plotqa_train1", "train_50k_annotations.json"),
        ("plotqa_train2", "train_50k_1l_annotations.json"), 
        ("plotqa_train3", "train_1l_end_annotations.json")
    ]
    
    train_images = str(data_path / "TRAIN")
    
    for dataset_name, annotation_file in train_configs:
        train_annotations = str(data_path / "annotations" / annotation_file)
        if os.path.exists(train_annotations):
            register_coco_instances(dataset_name, {}, train_annotations, train_images)
            print(f"Registered {dataset_name}: {train_annotations}")
        else:
            print(f"Warning: {dataset_name} annotations not found at {train_annotations}")
    
    # Register validation dataset (matches coco_val in original catalog)
    val_images = str(data_path / "VAL")
    val_annotations = str(data_path / "annotations" / "val_annotations.json")
    
    if os.path.exists(val_annotations):
        register_coco_instances("coco_val", {}, val_annotations, val_images)
        print(f"Registered validation dataset: {val_annotations}")
    else:
        print(f"Warning: Validation annotations not found at {val_annotations}")
    
    # Register test dataset (matches coco_val1 in original catalog) 
    test_images = str(data_path / "TEST")
    test_annotations = str(data_path / "annotations" / "test_annotations.json")
    
    if os.path.exists(test_annotations):
        register_coco_instances("coco_val1", {}, test_annotations, test_images)
        print(f"Registered test dataset: {test_annotations}")
    else:
        print(f"Warning: Test annotations not found at {test_annotations}")

def setup_config(args):
    """Setup Detectron2 configuration for PlotQA training"""
    
    cfg = get_cfg()
    
    # Load base Faster R-CNN config
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
    
    # Dataset configuration - use selected training set and validation as in original
    cfg.DATASETS.TRAIN = (args.train_dataset,)  # Use specified training set
    cfg.DATASETS.TEST = ("coco_val",)  # Use validation set for testing during training
    
    # Model configuration
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(PLOTQA_CATEGORIES)  # Number of PlotQA categories
    
    # Training configuration
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = args.learning_rate
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.SOLVER.STEPS = (int(args.max_iter * 0.7), int(args.max_iter * 0.9))  # LR decay steps
    cfg.SOLVER.CHECKPOINT_PERIOD = args.checkpoint_period
    
    # Evaluation configuration
    cfg.TEST.EVAL_PERIOD = args.eval_period
    
    # Output directory
    cfg.OUTPUT_DIR = args.output_dir
    
    # Data loader configuration
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    
    # Input configuration
    cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)  # Multi-scale training
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333
    
    return cfg

def main(args):
    """Main training function"""
    
    # Setup logging
    setup_logger()
    logger = logging.getLogger(__name__)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Register PlotQA datasets
    register_plotqa_datasets(args.data_dir)
    
    # Setup configuration
    cfg = setup_config(args)
    default_setup(cfg, args)
    
    # Save configuration
    with open(os.path.join(args.output_dir, "config.yaml"), "w") as f:
        f.write(cfg.dump())
    
    logger.info(f"Training configuration saved to {args.output_dir}/config.yaml")
    
    # Create trainer and start training
    trainer = PlotQATrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    
    if args.eval_only:
        # Evaluation only mode
        logger.info("Running evaluation only...")
        trainer.test(cfg, trainer.model)
    else:
        # Training mode
        logger.info("Starting training...")
        trainer.train()
        
        # Run final evaluation
        logger.info("Running final evaluation...")
        trainer.test(cfg, trainer.model)
    
    logger.info("Training completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PlotQA VED model")
    
    # Required arguments
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to PlotQA dataset directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save model and outputs")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training (default: 4)")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate (default: 0.001)")
    parser.add_argument("--max_iter", type=int, default=40000,
                        help="Maximum training iterations (default: 40000)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loader workers (default: 4)")
    
    # Dataset selection (matches original PlotQA structure)
    parser.add_argument("--train_dataset", type=str, default="plotqa_train1",
                        choices=["plotqa_train1", "plotqa_train2", "plotqa_train3"],
                        help="Training dataset to use (default: plotqa_train1)")
    
    # Checkpointing and evaluation
    parser.add_argument("--checkpoint_period", type=int, default=5000,
                        help="Checkpoint saving period (default: 5000)")
    parser.add_argument("--eval_period", type=int, default=5000,
                        help="Evaluation period during training (default: 5000)")
    
    # Execution modes
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from checkpoint")
    parser.add_argument("--eval_only", action="store_true",
                        help="Run evaluation only (no training)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.data_dir):
        raise ValueError(f"Data directory does not exist: {args.data_dir}")
    
    # Run training
    main(args)
