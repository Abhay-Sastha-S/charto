#!/usr/bin/env python3
"""
Bbox Upscaling functionality from original PlotQA pipeline
Replicates the upscale_boxes.py from misc/codes/
"""

import numpy as np
import matplotlib.pyplot as plt
from bbox_conversion import getScale, ResizeBox

def upscale_boxes(target_size, box, image, visualise_scaled_box=False):
    """
    Upscale boxes from original PlotQA pipeline
    
    Args:
        target_size: Target size [width, height]
        box: Bounding box [xmin, ymin, xmax, ymax]
        image: Image array
        visualise_scaled_box: Whether to visualize the scaled box
        
    Returns:
        Scaled bounding box [xmin, ymin, xmax, ymax]
    """
    x_scale, y_scale = getScale(650, 650, target_size)
    scaled_box = ResizeBox(box, x_scale, y_scale)
    
    if visualise_scaled_box:
        setup_plot(image)
        add_bboxes_to_plot(scaled_box, 'cyan')
        plt.show()
    
    return scaled_box

def setup_plot(image):
    """Setup matplotlib plot for visualization"""
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.axis('off')

def add_bboxes_to_plot(bbox, color='red'):
    """Add bounding boxes to plot"""
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    
    rect = plt.Rectangle((x1, y1), width, height, 
                        linewidth=2, edgecolor=color, facecolor='none')
    plt.gca().add_patch(rect)
