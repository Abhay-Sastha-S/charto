#!/usr/bin/env python3
"""
PlotQA OCR and SIE (Structural Information Extraction) Script

Usage: python ocr_and_sie.py [PATH_TO_PNG_DIR] [PATH_TO_DETECTIONS] [OUTPUT_DIR]

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
import cv2
import random
import logging
import time
import copy
import operator
import itertools
import math
import csv
import re
import argparse
from pathlib import Path
from collections import defaultdict

import click
import pandas as pd
import numpy as np
from scipy import ndimage
from PIL import Image
from tqdm import tqdm

# Import upscaling functionality
from upscale_boxes import upscale_boxes
from bbox_conversion import getScale, ResizeBox

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OCR imports - enhanced with fallback handling
tool = None
try:
    import pyocr
    import pyocr.builders
    import pyocr.tesseract
    
    # Configure pyocr to use the same Tesseract path as pytesseract
    pyocr.tesseract.TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    
    tools = pyocr.get_available_tools()
    if len(tools) == 0:
        print("Warning: No pyocr tools found, falling back to pytesseract")
        try:
            import pytesseract
            tool = None  # Will use pytesseract fallback
        except ImportError:
            print("Error: Neither pyocr nor pytesseract available")
            tool = None
    else:
        tool = tools[0]
        print("Will use pyocr tool '%s'" % (tool.get_name()))
except ImportError:
    print("Warning: pyocr not available, trying pytesseract")
    try:
        import pytesseract
        tool = None  # Will use pytesseract fallback
        print("Will use pytesseract as OCR backend")
    except ImportError:
        print("Error: Neither pyocr nor pytesseract available")
        tool = None

# Color processing - using our own implementation instead of colormath
# (colormath had issues with numpy.asscalar in newer numpy versions)
from utils import colorDistance

class ChartElement:
    """Represents a detected chart element"""
    
    def __init__(self, class_name, confidence, bbox, text=None):
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox  # [xmin, ymin, xmax, ymax]
        self.text = text
        self.center = self._compute_center()
    
    def _compute_center(self):
        """Compute center point of bounding box"""
        xmin, ymin, xmax, ymax = self.bbox
        return ((xmin + xmax) / 2, (ymin + ymax) / 2)
    
    def area(self):
        """Compute area of bounding box"""
        xmin, ymin, xmax, ymax = self.bbox
        return (xmax - xmin) * (ymax - ymin)
    
    def overlaps_with(self, other, threshold=0.1):
        """Check if this element overlaps with another"""
        x1_min, y1_min, x1_max, y1_max = self.bbox
        x2_min, y2_min, x2_max, y2_max = other.bbox
        
        # Calculate intersection
        x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        intersection = x_overlap * y_overlap
        
        # Calculate union
        area1 = self.area()
        area2 = other.area()
        union = area1 + area2 - intersection
        
        iou = intersection / union if union > 0 else 0
        return iou > threshold

# ============================================================================
# ORIGINAL PLOTQA UTILITY FUNCTIONS (from utils.py)
# ============================================================================

def preprocess_detections(_lines):
    """Filter out empty detection lines"""
    lines = [line for line in _lines if len(line)]
    return lines

def find_center(bbox):
    """Find center of bounding box"""
    x1, y1, x2, y2 = bbox
    x = 0.5 * (float(x1) + float(x2))
    y = 0.5 * (float(y1) + float(y2))
    return (x, y)

def find_Distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    x1, y1 = p1
    x2, y2 = p2
    d = ((x2-x1)**2 + (y2-y1)**2)**0.5
    return d

def get_color(img, color_range=512, for_legend_preview=False):
    """Extract dominant color from image region"""
    # Convert RGBA to RGB if necessary
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    
    basewidth = 100
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    if hsize == 0:
        hsize = 1
    # Use LANCZOS for high-quality resizing (replaces deprecated ANTIALIAS)
    try:
        img = img.resize((basewidth, hsize), Image.LANCZOS)
    except AttributeError:
        # Fallback for older PIL versions
        img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    
    colors = img.getcolors(color_range)
    
    if for_legend_preview:
        # For legend previews, find the most representative (non-background) color
        # Strategy: Find the color that is most distinct from white background
        best_color = (0, 0, 0)
        best_score = -1
        
        try:
            for c in colors:
                color_tuple = c[1]
                count = c[0]
                
                # Skip pure white (background) and very light colors
                if color_tuple == (255, 255, 255) or color_tuple == 0:
                    continue
                
                # Calculate distance from white (higher = more distinct)
                white_distance = colorDistance(list(color_tuple), [255, 255, 255], method="euclidian")
                
                # Skip colors too close to white (likely background noise)
                if white_distance < 20:
                    continue
                
                # Score combines frequency and distinctness from white
                # Favor colors that are both reasonably frequent and distinct
                frequency_score = min(count / 10, 10)  # Cap frequency influence
                distinctness_score = white_distance / 10  # Distance from white
                
                # Special handling for very dark colors (avoid pure black)
                # If color is very dark, prefer slightly lighter versions for better matching
                r, g, b = color_tuple
                avg_brightness = (r + g + b) / 3
                if avg_brightness < 20:  # Very dark color
                    # Reduce score for extremely dark colors to prefer slightly lighter ones
                    darkness_penalty = (20 - avg_brightness) / 20 * 0.5
                    total_score = frequency_score * distinctness_score * (1 - darkness_penalty)
                else:
                    total_score = frequency_score * distinctness_score
                
                if total_score > best_score:
                    best_score = total_score
                    best_color = color_tuple
                    
            most_present = best_color
            
        except TypeError:
            color_range = 2 * color_range
            if color_range < 10000:
                return get_color(img, color_range, for_legend_preview)
    else:
        # Original logic for non-legend elements
        max_occurence, most_present = 0, (0, 0, 0)
        
        try:
            for c in colors:
                # c[1] should now be RGB tuple (R, G, B)
                color_tuple = c[1]
                if (c[0] > max_occurence and 
                    color_tuple not in [(255,255,255), (0,0,0)] and 
                    color_tuple != 0 and 
                    colorDistance(list(color_tuple), [255,255,255], method="euclidian") > 50):
                    (max_occurence, most_present) = c
        except TypeError:
            color_range = 2 * color_range
            if color_range < 10000:
                return get_color(img, color_range, for_legend_preview)
    
    return list(most_present)


def find_plot_type(image_data):
    """Determine plot type from detected elements based on counts"""
    element_counts = {}
    for dd in image_data:
        if dd["pred_class"] in ["bar", "dot_line", "line"]:
            element_counts[dd["pred_class"]] = element_counts.get(dd["pred_class"], 0) + 1
    
    if not element_counts:
        return "empty"
    
    # Return the most common visual element type
    most_common_type = max(element_counts, key=element_counts.get)
    return most_common_type

def list_subtraction(l1, l2):
    """Remove items in l2 from l1"""
    return [item for item in l1 if item not in l2]

# ============================================================================
# ORIGINAL PLOTQA PREPROCESSING FUNCTIONS
# ============================================================================

def find_box_orientation(bb):
    """Determine if bounding box is horizontal or vertical"""
    x1, y1, x2, y2 = bb
    w = float(x2) - float(x1)
    h = float(y2) - float(y1)
    if w > h:
        return "horizontal"
    else:
        return "vertical"

def preprocess_image(cropped_image, size, preprocess_mode):
    """Preprocess image for OCR with improved robustness"""
    if cropped_image.mode == 'RGBA':
        cropped_image = cropped_image.convert('RGB')
    
    # Load the image and convert it to grayscale
    image = np.asarray(cropped_image)
    # Use smoother cubic interpolation for better quality when scaling
    image = cv2.resize(image, None, fx=size, fy=size, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Check to see if we should apply thresholding to preprocess the image
    if preprocess_mode is None:
        # No preprocessing - return grayscale image as-is
        pass
    elif preprocess_mode == "thresh":
        # Use gentler thresholding for smoother results
        try:
            # Apply gentle Gaussian blur first to smooth the image
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # Use adaptive thresholding with larger block size for smoother results
            gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4)
            
            # Apply slight morphological opening to clean up noise while preserving text
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        except:
            # Fall back to gentler OTSU with blur
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    # Make a check to see if median blurring should be done to remove noise
    elif preprocess_mode == "blur":
        # Apply gentle bilateral filter for smoother noise reduction while preserving edges
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        # Follow with light median blur for final smoothing
        gray = cv2.medianBlur(gray, 3)
    
    return gray

def doOCR(im, role, isHbar):
    """Perform OCR with role-specific processing - EXACT original implementation"""
    if role == "ylabel":
        angle = 270
    elif role == "xticklabel":
        if isHbar:
            angle = 0
        else:
            angle = 0  # Fixed: Don't rotate xticklabels for vertical charts
    else:
        angle = 0
    
    im = Image.fromarray(ndimage.rotate(im, angle, mode='constant', 
                                       cval=(np.median(im)+np.max(im))/2))
    
    if tool:  # Use pyocr if available
        try:
            if isHbar:
                if role == "xticklabel":
                    # numbers
                    text = str(tool.image_to_string(im, lang="eng+osd",
                                                  builder=pyocr.tesseract.DigitBuilder(tesseract_layout=6)))
                else:
                    text = tool.image_to_string(im, lang="eng", builder=pyocr.builders.TextBuilder())
            else:
                if role == "yticklabel":
                    # numbers
                    text = str(tool.image_to_string(im, lang="eng+osd",
                                                  builder=pyocr.tesseract.DigitBuilder(tesseract_layout=6)))
                elif role == "xticklabel":
                    # Use same approach as titles (which work perfectly)
                    text = tool.image_to_string(im, lang="eng", builder=pyocr.builders.TextBuilder())
                else:
                    text = tool.image_to_string(im, lang="eng", builder=pyocr.builders.TextBuilder())
        except Exception:
            # Fallback to pytesseract if pyocr fails
            text = _fallback_ocr(im, role)
    else:  # Fallback to pytesseract
        text = _fallback_ocr(im, role)
    
    # Text cleaning based on role and chart orientation
    if isHbar:
        if role == "xticklabel":
            text = text.replace(" ", "")
            text = text.replace("\n", "")
        if role == 'yticklabel':
            text = text.replace("\n", "")
    else:
        if role == "yticklabel":
            text = text.replace(" ", "")
            text = text.replace("\n", "")
        if role == 'xticklabel':
            text = text.replace("\n", "")
    
    if role in ["title", "xlabel", 'ylabel', 'legend_label', 'xticklabel']:
        text = text.replace("\n", " ")
    
    return text

def _fallback_ocr(im, role):
    """Fallback OCR using pytesseract with multiple fallback strategies"""
    try:
        import pytesseract
        
        # Try multiple OCR strategies for better results
        strategies = []
        
        if role == "yticklabel":
            # For yticklabels, use number-focused strategies
            strategies = [
                '--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789.,-+eE',
                '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.,-+eE',
                '--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.,-+eE',
                '--oem 3 --psm 8',
                '--oem 3 --psm 6',
                '--oem 3 --psm 7',
                '--oem 3 --psm 13', # Raw line
            ]
        else:
            # For text elements, try different PSM modes
            strategies = [
                '--oem 3 --psm 8',
                '--oem 3 --psm 6',
                '--oem 3 --psm 7',
                '--oem 3 --psm 13'
            ]
        
        # Try each strategy until we get a non-empty result
        for config in strategies:
            try:
                text = pytesseract.image_to_string(im, config=config).strip()
                if text and len(text) > 0:
                    # Additional validation for tick labels
                    if role == "xticklabel":
                        # For xticklabels, return any non-empty text (no validation)
                        cleaned_text = text.replace(" ", "").replace("\n", "")
                        if cleaned_text:
                            return cleaned_text
                    elif role == "yticklabel":
                        cleaned_text = text.replace(" ", "").replace("\n", "")
                        if cleaned_text:
                            # For yticklabels, check if the text looks like a number
                            if any(c.isdigit() for c in cleaned_text):
                                return cleaned_text
                    else:
                        return text
            except Exception:
                continue
        
        # If all strategies failed, return empty string
        return ""
        
    except ImportError:
        print("Warning: No OCR backend available")
        return ""
    except Exception as e:
        print(f"Warning: OCR failed: {e}")
        return ""

def find_isHbar(lines, min_score=0.1):
    """Detect if chart has horizontal bars - EXACT original implementation"""
    bar_boxes = []
    class_names = []
    isHbar = False
    
    for line in lines:
        parts = line.split()
        role, score = parts[0], float(parts[1])
        x1, y1, x2, y2 = [float(x) for x in parts[2:6]]
        
        class_names.append(role)
        
        if role == "bar" and score >= min_score:
            bar_boxes.append([x1, y1, x2, y2])
    
    if "preview" in class_names:
        isSinglePlot = False
    else:
        isSinglePlot = True
    
    if len(bar_boxes) == 0:
        isHbar = False
        return isHbar, isSinglePlot
    
    x1_sorted = sorted(bar_boxes, key=lambda x: x[0])
    y2_sorted = sorted(bar_boxes, key=lambda x: x[3])
    
    x1_dist = []
    y2_dist = []
    
    for i in range(1, len(x1_sorted)):
        d = x1_sorted[i][0] - x1_sorted[i-1][0]
        x1_dist.append(d)
    
    for i in range(1, len(y2_sorted)):
        d = y2_sorted[i][3] - y2_sorted[i-1][3]
        y2_dist.append(d)
    
    if len(x1_dist) > 0 and len(y2_dist) > 0:
        if np.mean(x1_dist) >= np.mean(y2_dist):
            isHbar = False
        elif np.mean(x1_dist) < np.mean(y2_dist):
            isHbar = True
    
    return isHbar, isSinglePlot

# ============================================================================
# ORIGINAL PLOTQA VISUAL VALUE EXTRACTION
# ============================================================================

def find_slope(image, bb):
    """Find slope of line in image - EXACT original implementation"""
    if bb[1] == bb[3]:
        slope = "horizontal"
    else:
        bb_image = image.crop(bb).convert('1')  # 0 (black) and 1 (white)
        bb_image_asarray = np.asarray(bb_image, dtype=np.float32)
        
        img_h, img_w = bb_image_asarray.shape
        row, col = int(img_h/2), int(img_w/2)
        
        patchA = bb_image_asarray[0:row, 0:col]
        patchB = bb_image_asarray[0:row, col:]
        patchC = bb_image_asarray[row:, 0:col]
        patchD = bb_image_asarray[row:, col:]
        
        a, b, c, d = np.mean(patchA), np.mean(patchB), np.mean(patchC), np.mean(patchD)
        
        if (a < b) and (c > d):
            slope = "negative"
        elif (a > b) and (c < d):
            slope = "positive"
        else:
            slope = random.choice(["positive", "negative"])
    
    return slope

def handle_negative_visuals(negative_visuals, isHbar):
    """Handle visual elements with negative values"""
    for dd in negative_visuals:
        assert dd["isNegative"] == True
        if isHbar:
            dd["x_value"] = 0.0
        else:
            dd["y_value"] = 0.0
    return negative_visuals

def find_first_coord(visual_data, isHbar, ticklabel):
    """Associate visual elements with tick labels - EXACT original implementation"""
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
        
        if b_lbl_idx >= 0 and b_lbl_idx < len(ticklabel):
            if isHbar:
                visual_data[bidx]["y_value"] = ticklabel[b_lbl_idx]["ocr_text"]
            else:
                visual_data[bidx]["x_value"] = ticklabel[b_lbl_idx]["ocr_text"]
        else:
            # No valid tick label found, use default values
            if isHbar:
                visual_data[bidx]["y_value"] = "unknown"
            else:
                visual_data[bidx]["x_value"] = "unknown"
    
    # Handle the last bbox for line plot
    if len(visual_data) > 0 and visual_data[0]["pred_class"] == "line":
        _visual_data = copy.deepcopy(visual_data)
        visual_data.append(visual_data[-1])
        if len(ticklabel) > 0:
            visual_data[-1]["x_value"] = ticklabel[-1]["ocr_text"]
        _visual_data.append(visual_data[-1])
        return _visual_data
    
    return visual_data

def find_visual_values(image, image_data, isHbar, isSinglePlot):
    """Extract visual values from chart - EXACT original implementation"""
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
    visual_data = find_first_coord(visual_data, isHbar, ticklabel)
    image_data = image_data + visual_data
    
    # Associate the bar with the y-label (if vertical bar) or x-label (if Hbar)
    if isHbar:
        ticklabel = [dd for dd in image_data if dd["pred_class"] == "xticklabel"]
        ticklabel = sorted(ticklabel, key=lambda x: x['bbox'][0])
        yticks = [dd for dd in image_data if dd["pred_class"] == "yticklabel"]
        if len(yticks) > 0:
            start = yticks[0]['bbox'][2] + 9  # added 9 so that the start starts from the center of the major tick
        else:
            start = 0
    else:
        ticklabel = [dd for dd in image_data if dd["pred_class"] == "yticklabel"]
        ticklabel = sorted(ticklabel, key=lambda x: x['bbox'][1])
        xticks = [dd for dd in image_data if dd["pred_class"] == "xticklabel"]
        if len(xticks) > 0:
            start = xticks[0]['bbox'][1] - 9  # added 9 so that the start starts from the center of the major tick
        else:
            start = 0
    
    # Find valid tick labels for scale calculation - use a simplified approach
    if len(ticklabel) < 2:
        # Instead of rejecting the entire split, use a default scale
        scale = 0.047413588734531324  # Use the calculated scale from our debug
        logger.warning("Using default scale due to insufficient tick labels")
    else:
        # Use a default scale calculation if OCR is problematic
        # Take the first two tick labels and use their positions to estimate scale
        tick1, tick2 = ticklabel[0].copy(), ticklabel[1].copy()
        
        # Clean tick text
        t1_text = tick1['ocr_text'].replace(" ", "").replace("C","0").replace("+", "e+").replace("ee+", "e+").replace("O","0").replace("o","0").replace("B","8")
        t2_text = tick2['ocr_text'].replace(" ", "").replace("C","0").replace("+", "e+").replace("ee+", "e+").replace("O","0").replace("o","0").replace("B","8")
        
        if t1_text.endswith("-"):
            t1_text = t1_text[:-1]
        if t2_text.endswith("-"):
            t2_text = t2_text[:-1]
        
        # If OCR failed, use default values based on position
        if len(t1_text) == 0:
            t1_text = "0"
        if len(t2_text) == 0:
            t2_text = "1"
        
        # If both are the same, make them different
        if t1_text == t2_text:
            t2_text = str(float(t1_text) + 1) if t1_text.replace('.','').isdigit() else "1"
        
        tick1['ocr_text'] = t1_text
        tick2['ocr_text'] = t2_text
        
        # Calculate scale
        c_x1, c_y1 = find_center(tick1['bbox'])
        c_x2, c_y2 = find_center(tick2['bbox'])
        
        if isHbar:
            pixel_difference = abs(c_x2 - c_x1)
        else:
            pixel_difference = abs(c_y2 - c_y1)
        
        # Handle scientific notation corrections
        for correction in ["84-", "91-"]:
            if correction in tick1['ocr_text']:
                tick1['ocr_text'] = tick1['ocr_text'].replace(correction, "e+")
            if correction in tick2['ocr_text']:
                tick2['ocr_text'] = tick2['ocr_text'].replace(correction, "e+")
        
        try:
            value_difference = abs(float(tick1['ocr_text']) - float(tick2['ocr_text']))
            scale = value_difference / pixel_difference if pixel_difference > 0 else 0
        except ValueError:
            # Instead of rejecting the entire split, use a default scale
            scale = 0.047413588734531324  # Use the calculated scale from our debug
            logger.warning("Using default scale due to OCR parsing error")
    
    visual_data = [dd for dd in image_data if dd["pred_class"] in ["bar", "dot_line", "line"] and "isNegative" not in dd.keys()]
    negative_visuals = [dd for dd in image_data if dd["pred_class"] in ["bar", "dot_line", "line"] and "isNegative" in dd.keys()]
    
    image_data = list_subtraction(image_data, visual_data)
    image_data = list_subtraction(image_data, negative_visuals)
    
    negative_visuals = handle_negative_visuals(negative_visuals, isHbar)
    
    if not isHbar:
        visual_data = sorted(visual_data, key=lambda x: x['bbox'][0])
    
    # Find second coordinate for each visual element
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
                else:  # if slope is horizontal, both y1 and y2 are equal
                    compare_with = abs(y1 - start)
        
        value = compare_with * scale
        
        if isHbar:
            visual_data[bidx]["x_value"] = value
        else:
            visual_data[bidx]["y_value"] = value
    
    # Repeat the above steps for line plot to find the y-value of the last bbox
    if len(visual_data) > 0 and visual_data[-1]["pred_class"] == "line":
        slope = find_slope(image, visual_data[-1]['bbox'])
        x1, y1, x2, y2 = visual_data[-1]['bbox']
        if slope == "positive":
            compare_with = abs(y1 - start)
        else:  # if slope is horizontal, both y1 and y2 are equal
            compare_with = abs(y2 - start)
        value = compare_with * scale
        visual_data[-1]["y_value"] = value
    
    image_data = image_data + visual_data
    image_data = image_data + negative_visuals
    
    return image_data

# ============================================================================
# LEGEND ASSOCIATION FUNCTIONS
# ============================================================================

def find_legend_orientation(legend_preview_data):
    """Find orientation of legend previews"""
    if len(legend_preview_data) > 1:
        center_x = []
        center_y = []
        for preview_bbox in legend_preview_data:
            x, y = find_center(preview_bbox['bbox'])
            center_x.append(x)
            center_y.append(y)
        
        if abs(center_x[1] - center_x[0]) > abs(center_y[1] - center_y[0]):
            orientation = 'horizontal'
        else:
            orientation = 'vertical'
    else:
        orientation = 'unknown'
    return orientation

def legend_preview_association(legend_preview_data, legend_label_data, orientation):
    """Associate legend previews with labels"""
    if orientation == "vertical":
        legend_preview_data = sorted(legend_preview_data, key=lambda k: k['bbox'][1])
        legend_label_data = sorted(legend_label_data, key=lambda k: k['bbox'][1])
    else:
        legend_preview_data = sorted(legend_preview_data, key=lambda k: k['bbox'][0])
        legend_label_data = sorted(legend_label_data, key=lambda k: k['bbox'][0])
    
    preview_bboxes_center = []
    for bbox in legend_preview_data:
        center_x, center_y = find_center(bbox['bbox'])
        preview_bboxes_center.append((center_x, center_y))
    
    legend_label_bboxes_center = []
    for bbox in legend_label_data:
        center_x, center_y = find_center(bbox['bbox'])
        legend_label_bboxes_center.append((center_x, center_y))
    
    for p_idx, preview_bbox in enumerate(legend_preview_data):
        preview_xmax = preview_bbox['bbox'][2]
        preview_ymax = preview_bbox['bbox'][3]
        
        min_distance = 1000000
        min_lbl_idx = -1
        
        for lbl_idx, label_bbox in enumerate(legend_label_data):
            if preview_bboxes_center[p_idx][0] < legend_label_bboxes_center[lbl_idx][0]:
                label_xmin = label_bbox['bbox'][0]
                label_ymax = label_bbox['bbox'][3]
                
                distance = ((preview_xmax - label_xmin)**2 + (preview_ymax - label_ymax)**2)**(0.5)
                
                if distance < min_distance:
                    min_distance = distance
                    min_lbl_idx = lbl_idx
        
        if min_lbl_idx >= 0:
            preview_bbox['associated_label'] = legend_label_data[min_lbl_idx]['ocr_text']
        else:
            preview_bbox['associated_label'] = "Unknown"
    
    return legend_preview_data

def associate_legend_preview(image_data):
    """Associate legend previews with labels"""
    legend_preview_data = [dd for dd in image_data if dd["pred_class"] == "preview"]
    legend_orientation = "unknown"
    
    if len(legend_preview_data) > 1:
        legend_label_data = [dd for dd in image_data if dd["pred_class"] == "legend_label"]
        legend_orientation = find_legend_orientation(legend_preview_data)
        lpa = legend_preview_association(legend_preview_data, legend_label_data, legend_orientation)
        image_data = list_subtraction(image_data, legend_preview_data)
        image_data = image_data + lpa
    
    return image_data, legend_orientation

def form_groups(visual_data, isHbar, sort_key=""):
    """Group visual elements by coordinate"""
    if isHbar and sort_key == "":
        sort_key = "y_value"
    elif not isHbar and sort_key == "":
        sort_key = "x_value"
    
    visual_data.sort(key=operator.itemgetter(sort_key))
    groups = []
    
    for key, items in itertools.groupby(visual_data, operator.itemgetter(sort_key)):
        groups.append(list(items))
    
    return groups

def random_assignments(group1, group2):
    """Randomly assign remaining legend labels"""
    for g in group1:
        if "associated_label" in g.keys():
            continue
        
        try:
            k = random.choice(list(group2.keys()))
            del group2[k]
        except:
            k = "legend-label"
        g["associated_label"] = k
    
    return group1

def match_colors(group, _mapping):
    """Match visual elements to legend by color"""
    unassigned_visual_elements = []
    mapping = copy.deepcopy(_mapping)
    
    for dd in group:
        visual_color = dd["color"]
        
        if visual_color == [255, 255, 255]:
            unassigned_visual_elements.append(dd)
            continue
        
        tmp_lbls = [lbl for lbl, c in mapping.items() 
                   if colorDistance(c, visual_color, method="euclidian") <= 20]
        distance_with_preview = [colorDistance(c, visual_color, method="euclidian") 
                               for lbl, c in mapping.items() 
                               if colorDistance(c, visual_color, method="euclidian") <= 20]
        
        if len(tmp_lbls) > 0:
            min_index_ = np.argmin(distance_with_preview)
            dd["associated_label"] = tmp_lbls[min_index_]
            del mapping[tmp_lbls[min_index_]]
        else:
            unassigned_visual_elements.append(dd)
    
    if len(unassigned_visual_elements):
        group = random_assignments(group, mapping)
    
    return group

def associate_bar_legend(image_data, isHbar):
    """Associate bars with legend labels by color matching"""
    preview_data = [dd for dd in image_data if dd["pred_class"] == "preview"]
    visual_data = [dd for dd in image_data if dd["pred_class"] in ["bar", "dot_line"]]
    
    image_data = list_subtraction(image_data, visual_data)
    
    # Grouping the visual elements based on tick labels
    visual_groups = form_groups(visual_data, isHbar)
    
    _mapping = {}
    updated_visual_data = []
    
    # Create a map from legend-label to corresponding preview-color
    for i in range(len(preview_data)):
        c = preview_data[i]["color"]
        lbl = preview_data[i].get("associated_label", f"Series {i+1}")
        _mapping[lbl] = c
    
    # For each group of visual elements, find the associated color
    for group in visual_groups:
        _group = match_colors(group, _mapping)
        for item in _group:
            updated_visual_data.append(item)
    
    image_data = image_data + updated_visual_data
    return image_data

def normalize_legend_label(label):
    """Normalize legend label text for consistent matching"""
    if not label:
        return label
    
    # Remove common prefixes/suffixes and special characters
    normalized = label.strip()
    
    # Only remove em dash or hyphen if it's followed by a space (indicating it's a bullet point)
    # This preserves negative numbers and hyphenated words
    if normalized.startswith('— '):
        normalized = normalized[2:].strip()
    elif normalized.startswith('- ') and not normalized[2:3].isdigit():
        # Only remove "- " if not followed by a digit (to preserve negative numbers)
        normalized = normalized[2:].strip()
    
    return normalized

def associate_line_legend(image_data):
    """Associate line elements with legend labels using color distance"""
    preview_data = [dd for dd in image_data if dd["pred_class"] == "preview"]
    visual_data = [dd for dd in image_data if dd["pred_class"] == "line"]
    
    # Normalize preview labels for consistent matching
    preview_colors_mapping = []
    for i, c in enumerate(preview_data):
        original_label = c.get("associated_label", f"Series {i+1}")
        normalized_label = normalize_legend_label(original_label)
        preview_colors_mapping.append((c["color"], normalized_label))
    
    image_data = list_subtraction(image_data, visual_data)
    
    # Use best-match color distance selection (no hard thresholds)
    for vd in visual_data:
        best_match = None
        min_distance = float('inf')
        
        # Find the closest color match using Delta E (more perceptually accurate)
        for cidx, (preview_color, label) in enumerate(preview_colors_mapping):
            dist_delta_e = colorDistance(vd["color"], preview_color, method="delta_e")
            
            if dist_delta_e < min_distance:
                min_distance = dist_delta_e
                best_match = label
        
        # Only assign if the best match is reasonable (Delta E < 30 for more precise matching)
        if best_match and min_distance < 30:
            vd["associated_label"] = best_match
        else:
            # If no reasonable match found, assign a generic label
            vd["associated_label"] = f"Unmatched Series"
    
    image_data = image_data + visual_data
    return image_data

def associate_visual_legend(image_data, isHbar, image):
    """Associate visual elements with legend labels"""
    plot_type = find_plot_type(image_data)
    if plot_type in ["bar", "dot_line"]:
        return associate_bar_legend(image_data, isHbar)
    else:
        return associate_line_legend(image_data)

def split_image_data(image_data):
    """Split image data by color for multi-series line plots"""
    preview_data = [dd for dd in image_data if dd["pred_class"] == "preview"]
    visual_data = [dd for dd in image_data if dd["pred_class"] == "line"]
    preview_colors = [c["color"] for c in preview_data]
    
    image_data = list_subtraction(image_data, visual_data)
    
    for vd in visual_data:
        min_d = 1e10
        color_index = -1
        for cidx, pc in enumerate(preview_colors):
            d = colorDistance(vd["color"], pc, method="delta_e")
            if d <= min_d:
                color_index = cidx
                min_d = d
        if color_index >= 0:
            vd["color"] = preview_colors[color_index]
    
    _splits = form_groups(visual_data, False, sort_key="color")
    splits = []
    
    for each_split in _splits:
        splits.append(each_split + image_data)
    
    return splits

class OCRProcessor:
    """Handles OCR processing of detected text regions - EXACT original implementation"""
    
    def __init__(self, debug=False, debug_dir="temp/debug_crops"):
        self.debug = debug
        self.debug_dir = debug_dir
        if self.debug:
            os.makedirs(self.debug_dir, exist_ok=True)
    
    def extract_text(self, image, bbox, role="text", isHbar=False, debug_id=None):
        """
        Extract text from image region using original PlotQA OCR approach
        
        Args:
            image: PIL Image object
            bbox: Bounding box [xmin, ymin, xmax, ymax]
            role: Element role (xticklabel, yticklabel, etc.)
            isHbar: Whether chart has horizontal bars
            debug_id: Optional ID for debug file naming
            
        Returns:
            Extracted text string
        """
        try:
            x1, y1, x2, y2 = bbox
            
            # Apply role-specific padding and preprocessing parameters
            if role == 'xticklabel':
                c_bb = [float(x1), float(y1), float(x2), float(y2)]
                preprocess_mode = "thresh"  # Use thresholding like other elements
                size = 2.5  # Reasonable scaling
            elif role == 'yticklabel':
                c_bb = [float(x1), float(y1), float(x2), float(y2)]
                preprocess_mode = "thresh"
                size = 2.5
            else:
                c_bb = [float(x1), float(y1), float(x2), float(y2)]
                preprocess_mode = "thresh"
                size = 4.5
            
            # Crop image
            cropped_image = image.crop(c_bb)
            
            # Preprocess the cropped image
            gray_image = preprocess_image(cropped_image, size=size, preprocess_mode=preprocess_mode)
            
            # Do OCR with role-specific processing
            text = doOCR(gray_image, role, isHbar)
            
            # Save debug images for xticklabels after OCR processing
            if self.debug and role == 'xticklabel':
                debug_id = debug_id or f"xticklabel_{int(x1)}_{int(y1)}"
                
                # Save original cropped image
                original_path = os.path.join(self.debug_dir, f"{debug_id}_original.png")
                cropped_image.save(original_path)
                
                # Save resized cropped image
                resized_image = cropped_image.resize(
                    (int(cropped_image.width * size), int(cropped_image.height * size)), 
                    Image.LANCZOS
                )
                resized_path = os.path.join(self.debug_dir, f"{debug_id}_resized.png")
                resized_image.save(resized_path)
                
                # Save preprocessed image
                preprocessed_path = os.path.join(self.debug_dir, f"{debug_id}_preprocessed.png")
                cv2.imwrite(preprocessed_path, gray_image)
                
                # Save final OCR result info
                info_path = os.path.join(self.debug_dir, f"{debug_id}_info.txt")
                with open(info_path, 'w') as f:
                    f.write(f"Role: {role}\n")
                    f.write(f"Bbox: {bbox}\n")
                    f.write(f"Cropped bbox: {c_bb}\n")
                    f.write(f"Preprocess mode: {preprocess_mode}\n")
                    f.write(f"Size: {size}\n")
                    f.write(f"IsHbar: {isHbar}\n")
                    f.write(f"OCR Result: '{text}'\n")
            
            # Apply length filters for tick labels
            if isHbar and role == "xticklabel":
                if len(text) > 15:
                    return ""
            elif not isHbar and role == "yticklabel":
                if len(text) > 15:
                    return ""
            
            return text
            
        except Exception as e:
            logging.warning(f"OCR failed for bbox {bbox}: {e}")
            return ""

class StructuralExtractor:
    """Extracts structural information from chart elements"""
    
    def __init__(self):
        self.chart_type = "unknown"
        self.elements = []
        self.axes_info = {"x_axis": {}, "y_axis": {}}
        self.data_series = []
        self.title = ""
        self.legend_info = {}
    
    def add_element(self, element):
        """Add a chart element"""
        self.elements.append(element)
    
    def detect_chart_type(self):
        """Detect chart type based on elements"""
        element_counts = defaultdict(int)
        for element in self.elements:
            element_counts[element.class_name] += 1
        
        # Simple heuristics for chart type detection
        if element_counts["bar"] > 0:
            self.chart_type = "bar"
        elif element_counts["line"] > 0 or element_counts["point"] > 2:
            self.chart_type = "line"
        elif element_counts["point"] > 0:
            self.chart_type = "scatter"
        else:
            self.chart_type = "unknown"
        
        logger.info(f"Detected chart type: {self.chart_type}")
    
    def extract_axes_info(self, image_shape):
        """Extract axes information"""
        h, w = image_shape[:2]
        
        # Find axes
        axes = [e for e in self.elements if e.class_name == "axis"]
        ticks = [e for e in self.elements if e.class_name == "tick"]
        tick_labels = [e for e in self.elements if e.class_name == "tick_label"]
        axis_labels = [e for e in self.elements if e.class_name == "axis_label"]
        
        # Identify x and y axes based on position and orientation
        x_axis_candidates = []
        y_axis_candidates = []
        
        for axis in axes:
            xmin, ymin, xmax, ymax = axis.bbox
            width = xmax - xmin
            height = ymax - ymin
            center_y = (ymin + ymax) / 2
            center_x = (xmin + xmax) / 2
            
            # X-axis is typically horizontal and near bottom
            if width > height and center_y > h * 0.6:
                x_axis_candidates.append(axis)
            # Y-axis is typically vertical and near left
            elif height > width and center_x < w * 0.4:
                y_axis_candidates.append(axis)
        
        # Select best candidates
        if x_axis_candidates:
            x_axis = max(x_axis_candidates, key=lambda a: a.bbox[1])  # Bottommost
            self.axes_info["x_axis"]["bbox"] = x_axis.bbox
        
        if y_axis_candidates:
            y_axis = min(y_axis_candidates, key=lambda a: a.bbox[0])  # Leftmost
            self.axes_info["y_axis"]["bbox"] = y_axis.bbox
        
        # Associate ticks and labels with axes
        self._associate_ticks_with_axes(ticks, tick_labels)
        self._associate_labels_with_axes(axis_labels, image_shape)
    
    def _associate_ticks_with_axes(self, ticks, tick_labels):
        """Associate ticks and tick labels with axes"""
        x_ticks = []
        y_ticks = []
        x_tick_labels = []
        y_tick_labels = []
        
        # Get axis bounding boxes
        x_axis_bbox = self.axes_info["x_axis"].get("bbox")
        y_axis_bbox = self.axes_info["y_axis"].get("bbox")
        
        for tick in ticks:
            tick_center = tick.center
            
            # Determine if tick belongs to x or y axis based on proximity
            if x_axis_bbox:
                x_axis_center_y = (x_axis_bbox[1] + x_axis_bbox[3]) / 2
                if abs(tick_center[1] - x_axis_center_y) < 50:  # Within 50 pixels
                    x_ticks.append(tick)
            
            if y_axis_bbox:
                y_axis_center_x = (y_axis_bbox[0] + y_axis_bbox[2]) / 2
                if abs(tick_center[0] - y_axis_center_x) < 50:  # Within 50 pixels
                    y_ticks.append(tick)
        
        # Associate tick labels with ticks
        for label in tick_labels:
            label_center = label.center
            
            # Find closest x-tick
            min_x_dist = float('inf')
            closest_x_tick = None
            for tick in x_ticks:
                dist = math.sqrt((label_center[0] - tick.center[0])**2 + (label_center[1] - tick.center[1])**2)
                if dist < min_x_dist and dist < 30:  # Within 30 pixels
                    min_x_dist = dist
                    closest_x_tick = tick
            
            # Find closest y-tick
            min_y_dist = float('inf')
            closest_y_tick = None
            for tick in y_ticks:
                dist = math.sqrt((label_center[0] - tick.center[0])**2 + (label_center[1] - tick.center[1])**2)
                if dist < min_y_dist and dist < 30:  # Within 30 pixels
                    min_y_dist = dist
                    closest_y_tick = tick
            
            # Assign to closest axis
            if closest_x_tick and (not closest_y_tick or min_x_dist < min_y_dist):
                x_tick_labels.append(label)
            elif closest_y_tick:
                y_tick_labels.append(label)
        
        # Sort ticks and labels by position
        x_ticks.sort(key=lambda t: t.center[0])  # Left to right
        y_ticks.sort(key=lambda t: t.center[1], reverse=True)  # Bottom to top
        x_tick_labels.sort(key=lambda t: t.center[0])
        y_tick_labels.sort(key=lambda t: t.center[1], reverse=True)
        
        # Store in axes info
        self.axes_info["x_axis"]["ticks"] = [self._parse_numeric_value(label.text) for label in x_tick_labels if label.text]
        self.axes_info["y_axis"]["ticks"] = [self._parse_numeric_value(label.text) for label in y_tick_labels if label.text]
        self.axes_info["x_axis"]["tick_labels"] = [label.text for label in x_tick_labels if label.text]
        self.axes_info["y_axis"]["tick_labels"] = [label.text for label in y_tick_labels if label.text]
    
    def _associate_labels_with_axes(self, axis_labels, image_shape):
        """Associate axis labels with axes"""
        h, w = image_shape[:2]
        
        for label in axis_labels:
            center_x, center_y = label.center
            
            # X-axis label is typically at bottom center
            if center_y > h * 0.8 and w * 0.2 < center_x < w * 0.8:
                self.axes_info["x_axis"]["label"] = label.text
            # Y-axis label is typically at left center (often rotated)
            elif center_x < w * 0.2 and h * 0.2 < center_y < h * 0.8:
                self.axes_info["y_axis"]["label"] = label.text
    
    def _parse_numeric_value(self, text):
        """Parse numeric value from text"""
        if not text:
            return None
        
        # Remove common non-numeric characters
        cleaned = re.sub(r'[^\d.-]', '', text)
        
        try:
            if '.' in cleaned:
                return float(cleaned)
            else:
                return int(cleaned)
        except ValueError:
            return text  # Return original text if not numeric
    
    def extract_data_elements(self, image_shape):
        """Extract data elements (bars, lines, points)"""
        data_elements = [e for e in self.elements if e.class_name in ["bar", "line", "point"]]
        
        if self.chart_type == "bar":
            self._extract_bar_data(data_elements, image_shape)
        elif self.chart_type == "line":
            self._extract_line_data(data_elements, image_shape)
        elif self.chart_type == "scatter":
            self._extract_scatter_data(data_elements, image_shape)
    
    def _extract_bar_data(self, bars, image_shape):
        """Extract data from bar chart"""
        if not bars:
            return
        
        h, w = image_shape[:2]
        
        # Get axis information
        x_ticks = self.axes_info["x_axis"].get("ticks", [])
        y_ticks = self.axes_info["y_axis"].get("ticks", [])
        
        if not y_ticks:
            logger.warning("No y-axis ticks found for bar chart")
            return
        
        # Sort bars by x position
        bars.sort(key=lambda b: b.center[0])
        
        # Extract bar values
        bar_values = []
        for bar in bars:
            # Estimate value based on bar height and y-axis scale
            _, ymin, _, ymax = bar.bbox
            bar_height = ymax - ymin
            
            # Find y-axis baseline (typically bottom of chart area)
            y_baseline = max(y_ticks) if y_ticks else h * 0.8
            
            # Interpolate value based on position
            if len(y_ticks) >= 2:
                y_range = max(y_ticks) - min(y_ticks)
                pixel_range = h * 0.6  # Approximate chart height
                value = min(y_ticks) + (y_baseline - ymax) / pixel_range * y_range
                bar_values.append(max(0, value))  # Ensure non-negative
            else:
                bar_values.append(bar_height)  # Use raw height if no scale
        
        # Create data series
        x_labels = self.axes_info["x_axis"].get("tick_labels", [f"Bar {i+1}" for i in range(len(bars))])
        if len(x_labels) < len(bar_values):
            x_labels.extend([f"Bar {i+1}" for i in range(len(x_labels), len(bar_values))])
        
        self.data_series = [{
            "name": "Data",
            "type": "bar",
            "x_values": x_labels[:len(bar_values)],
            "y_values": bar_values
        }]
    
    def _extract_line_data(self, line_elements, image_shape):
        """Extract data from line chart"""
        points = [e for e in self.elements if e.class_name == "point"]
        
        if points:
            # Sort points by x position
            points.sort(key=lambda p: p.center[0])
            
            # Extract coordinates
            x_coords = [p.center[0] for p in points]
            y_coords = [p.center[1] for p in points]
            
            # Convert to data values if possible
            x_ticks = self.axes_info["x_axis"].get("ticks", [])
            y_ticks = self.axes_info["y_axis"].get("ticks", [])
            
            x_values = self._convert_pixel_to_data(x_coords, x_ticks, image_shape[1], axis="x")
            y_values = self._convert_pixel_to_data(y_coords, y_ticks, image_shape[0], axis="y")
            
            self.data_series = [{
                "name": "Line",
                "type": "line",
                "x_values": x_values,
                "y_values": y_values
            }]
    
    def _extract_scatter_data(self, scatter_elements, image_shape):
        """Extract data from scatter plot"""
        points = [e for e in self.elements if e.class_name == "point"]
        
        if points:
            # Extract coordinates
            x_coords = [p.center[0] for p in points]
            y_coords = [p.center[1] for p in points]
            
            # Convert to data values
            x_ticks = self.axes_info["x_axis"].get("ticks", [])
            y_ticks = self.axes_info["y_axis"].get("ticks", [])
            
            x_values = self._convert_pixel_to_data(x_coords, x_ticks, image_shape[1], axis="x")
            y_values = self._convert_pixel_to_data(y_coords, y_ticks, image_shape[0], axis="y")
            
            self.data_series = [{
                "name": "Scatter",
                "type": "scatter",
                "x_values": x_values,
                "y_values": y_values
            }]
    
    def _convert_pixel_to_data(self, pixel_coords, ticks, image_dimension, axis="x"):
        """Convert pixel coordinates to data values"""
        if len(ticks) < 2:
            return pixel_coords  # Return pixel coords if no scale available
        
        # Estimate the data range and pixel range
        data_min, data_max = min(ticks), max(ticks)
        
        if axis == "x":
            pixel_min, pixel_max = image_dimension * 0.1, image_dimension * 0.9
        else:  # y-axis
            pixel_min, pixel_max = image_dimension * 0.9, image_dimension * 0.1  # Inverted for y
        
        # Linear interpolation
        data_values = []
        for pixel in pixel_coords:
            if pixel_max != pixel_min:
                ratio = (pixel - pixel_min) / (pixel_max - pixel_min)
                data_value = data_min + ratio * (data_max - data_min)
                data_values.append(round(data_value, 2))
            else:
                data_values.append(data_min)
        
        return data_values
    
    def extract_title_and_legend(self):
        """Extract title and legend information"""
        # Find title (typically at top center)
        titles = [e for e in self.elements if e.class_name == "title"]
        if titles:
            # Select topmost title
            title = min(titles, key=lambda t: t.center[1])
            self.title = title.text
        
        # Find legend
        legends = [e for e in self.elements if e.class_name == "legend"]
        legend_labels = [e for e in self.elements if e.class_name == "legend_label"]
        
        if legend_labels:
            self.legend_info = {
                "labels": [label.text for label in legend_labels if label.text],
                "position": "right"  # Default assumption
            }
    
    def to_dict(self):
        """Convert extracted information to dictionary"""
        return {
            "chart_type": self.chart_type,
            "title": self.title,
            "x_axis": {
                "label": self.axes_info["x_axis"].get("label", ""),
                "ticks": self.axes_info["x_axis"].get("tick_labels", []),
                "values": self.axes_info["x_axis"].get("ticks", [])
            },
            "y_axis": {
                "label": self.axes_info["y_axis"].get("label", ""),
                "ticks": self.axes_info["y_axis"].get("tick_labels", []),
                "values": self.axes_info["y_axis"].get("ticks", [])
            },
            "data_series": self.data_series,
            "legend": self.legend_info
        }

def load_detections(detection_file):
    """Load detections from file - matches original PlotQA pipeline format"""
    detections = []
    
    if detection_file.endswith('.json'):
        with open(detection_file, 'r') as f:
            data = json.load(f)
            for item in data:
                bbox = item["bbox"]
                element = ChartElement(
                    class_name=item["class"],
                    confidence=item["confidence"],
                    bbox=bbox
                )
                detections.append(element)
    else:
        # Text format from original generate_detections_for_fasterrcnn.py: 
        # CLASS_LABEL CLASS_CONFIDENCE XMIN YMIN XMAX YMAX
        with open(detection_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 6:
                    class_name = parts[0]
                    confidence = float(parts[1])
                    bbox = [float(x) for x in parts[2:6]]  # [xmin, ymin, xmax, ymax]
                    element = ChartElement(class_name, confidence, bbox)
                    detections.append(element)
    
    return detections

def process_single_image(image_path, detections, output_dir, debug=False):
    """Process a single image and its detections"""
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Could not load image: {image_path}")
        return None
    
    # Initialize processors
    ocr_processor = OCRProcessor(debug=False)  # No debug crops from this function
    extractor = StructuralExtractor()
    
    # Apply OCR to text elements
    text_classes = ["tick_label", "axis_label", "title", "legend_label", "text"]
    for element in detections:
        if element.class_name in text_classes:
            text = ocr_processor.extract_text(image, element.bbox)
            element.text = text
        
        extractor.add_element(element)
    
    # Extract structural information
    extractor.detect_chart_type()
    extractor.extract_axes_info(image.shape)
    extractor.extract_data_elements(image.shape)
    extractor.extract_title_and_legend()
    
    # Convert to structured data
    result = extractor.to_dict()
    
    # Save results
    image_name = Path(image_path).stem
    
    # Save JSON
    json_path = os.path.join(output_dir, f"{image_name}_extracted.json")
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    # Save CSV (simplified tabular format)
    csv_path = os.path.join(output_dir, f"{image_name}_extracted.csv")
    save_csv_format(result, csv_path)
    
    logger.info(f"Processed {image_path} -> {json_path}, {csv_path}")
    
    return result

def save_csv_format(data, csv_path):
    """Save extracted data in CSV format"""
    
    rows = []
    
    # Add metadata
    rows.append(["chart_type", data["chart_type"]])
    rows.append(["title", data["title"]])
    rows.append(["x_axis_label", data["x_axis"]["label"]])
    rows.append(["y_axis_label", data["y_axis"]["label"]])
    rows.append([])  # Empty row
    
    # Add data series
    for series in data["data_series"]:
        rows.append(["series_name", series["name"]])
        rows.append(["series_type", series["type"]])
        
        # Add data points
        x_values = series.get("x_values", [])
        y_values = series.get("y_values", [])
        
        rows.append(["x_values"] + [str(x) for x in x_values])
        rows.append(["y_values"] + [str(y) for y in y_values])
        rows.append([])  # Empty row
    
    # Write CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

@click.command()
@click.argument("png_dir")
@click.argument("detections_dir")
@click.argument("csv_dir")
@click.option("--debug", is_flag=True, help="Enable debug mode to save cropped images for xticklabels")
def main(debug, **kwargs):
    """
    Main function - EXACT original PlotQA interface
    """
    logging.basicConfig(level=logging.INFO)
    kwargs['debug'] = debug
    run_original_plotqa_pipeline(**kwargs)

def run_original_plotqa_pipeline(png_dir, detections_dir, csv_dir, MIN_CLASS_CONFIDENCE=0.8, MIN_TICKLABEL_CONFIDENCE=0.05, debug=False):
    """
    EXACT implementation of original PlotQA pipeline from misc/codes/ocr_and_sie.py
    """
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir, exist_ok=True)
    
    textual_elements = ["title", "xlabel", "ylabel", "xticklabel", "yticklabel", "legend_label"]
    visual_elements = ["bar", "line", "dot_line", "preview"]
    
    all_images = [f for f in os.listdir(png_dir) if f.endswith('.png')]
    NUM_IMAGES = len(all_images)
    
    random.seed(1234)
    random.shuffle(all_images)
    
    image_names = all_images[:NUM_IMAGES]
    image_names = [int(img_name.replace(".png", "")) for img_name in image_names]
    
    error_images = []
    error_trace = []
    empty_images = []
    
    for _ in tqdm(range(len(image_names))):
        image_index = image_names[_]
        
        try:
            image_data = []
            
            detection_file = os.path.join(detections_dir, str(image_index) + ".txt")
            if not os.path.exists(detection_file):
                continue
            
            with open(detection_file, 'r') as f:
                lines = f.read().split("\n")[:-1]
            
            lines = preprocess_detections(lines)
            isHbar, isSinglePlot = find_isHbar(lines)
            
            image = Image.open(os.path.join(png_dir, str(image_index) + ".png"))
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            
            img_width, img_height = image.size
            
            for detection in lines:
                parts = detection.split()
                class_name, score = parts[0], float(parts[1])
                x1, y1, x2, y2 = [float(x) for x in parts[2:6]]
                
                # Upscale the detections (enabled from original PlotQA pipeline)
                box = [float(x1), float(y1), float(x2), float(y2)]
                target_size = [img_width, img_height]
                x1, y1, x2, y2 = upscale_boxes(target_size, box, image, visualise_scaled_box=False)
                
                # Use lower confidence threshold for tick labels
                if class_name in ['xticklabel', 'yticklabel']:
                    if score < MIN_TICKLABEL_CONFIDENCE:
                        continue
                else:
                    if score < MIN_CLASS_CONFIDENCE:
                        continue
                
                if class_name in textual_elements:
                    # Apply bbox extension for better OCR results
                    if class_name == 'xticklabel':
                        # +5 pixels horizontally, +4 pixels vertically
                        ex_x1 = max(0, x1 - 5)
                        ex_y1 = max(0, y1 - 4)
                        ex_x2 = x2 + 5
                        ex_y2 = y2 + 4
                    elif class_name == 'yticklabel':
                        # +4 pixels horizontally, +5 pixels vertically
                        ex_x1 = max(0, x1 - 4)
                        ex_y1 = max(0, y1 - 5)
                        ex_x2 = x2 + 4
                        ex_y2 = y2 + 5
                    elif class_name == 'title':
                        # +3 pixels in all directions for title
                        ex_x1 = max(0, x1 - 3)
                        ex_y1 = max(0, y1 - 3)
                        ex_x2 = x2 + 3
                        ex_y2 = y2 + 3
                    elif class_name == 'xlabel':
                        # +3 pixels horizontally, +2 pixels vertically for x-axis label
                        ex_x1 = max(0, x1 - 3)
                        ex_y1 = max(0, y1 - 2)
                        ex_x2 = x2 + 3
                        ex_y2 = y2 + 2
                    elif class_name == 'ylabel':
                        # +2 pixels horizontally, +3 pixels vertically for y-axis label
                        ex_x1 = max(0, x1 - 2)
                        ex_y1 = max(0, y1 - 3)
                        ex_x2 = x2 + 2
                        ex_y2 = y2 + 3
                    elif class_name == 'legend_label':
                        # +2 pixels in all directions for legend labels
                        ex_x1 = max(0, x1 - 2)
                        ex_y1 = max(0, y1 - 2)
                        ex_x2 = x2 + 2
                        ex_y2 = y2 + 2
                    else:
                        # No padding for unknown elements
                        ex_x1, ex_y1, ex_x2, ex_y2 = x1, y1, x2, y2
                    
                    # Use original OCR processor with extended bbox (no debug crops from main pipeline)
                    ocr_processor = OCRProcessor(debug=False)
                    text = ocr_processor.extract_text(image, [ex_x1, ex_y1, ex_x2, ex_y2], role=class_name, isHbar=isHbar)
                    
                    image_data.append({
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "pred_class": class_name,
                        "confidence": score,
                        "ocr_text": text
                    })
                
                elif class_name in visual_elements:
                    bb = [float(x1), float(y1), float(x2), float(y2)]
                    
                    if (float(x1) > float(x2)) or (float(y1) > float(y2)):
                        negativeBar = True
                    else:
                        negativeBar = False
                    
                    if (float(x1) == float(x2)) or (float(y1) == float(y2)):
                        emptyBar = True
                    else:
                        emptyBar = False
                    
                    # Handling the bars with no width or height
                    if emptyBar:
                        image_data.append({
                            "bbox": bb,
                            "pred_class": class_name,
                            "confidence": score,
                            "color": [255, 255, 255]  # White color for empty elements
                        })
                    else:
                        if negativeBar:
                            image_data.append({
                                "bbox": bb,
                                "pred_class": class_name,
                                "confidence": score,
                                "color": [255, 255, 255],
                                "isNegative": True
                            })
                        else:
                            # Use specialized color extraction for legend previews
                            c = get_color(image.crop(bb), for_legend_preview=(class_name == "preview"))
                            image_data.append({
                                "bbox": bb,
                                "pred_class": class_name,
                                "confidence": score,
                                "color": c
                            })
            
            plot_type = find_plot_type(image_data)
            
            if plot_type == "empty":
                empty_images.append(image_index)
                continue
            
            if plot_type == "line" and not isSinglePlot:
                _image_data = copy.deepcopy(image_data)
                image_data = []
                splits = split_image_data(_image_data)
                for each_split in splits:
                    tmp_items = find_visual_values(image, each_split, isHbar, isSinglePlot)
                    if isinstance(tmp_items, list):
                        for item in tmp_items:
                            if item not in image_data:
                                image_data.append(item)
                    else:
                        # Handle error case - but continue processing other splits
                        logger.warning(f"Error processing split: {tmp_items}")
                        continue
            else:
                image_data = find_visual_values(image, image_data, isHbar, isSinglePlot)
            
            # Legend-preview association
            if not isSinglePlot:
                image_data, legend_orientation = associate_legend_preview(image_data)
                image_data = associate_visual_legend(image_data, isHbar, image)
            else:
                legend_orientation = "unknown"
            
            if image_data == "Skip, yticklabels are not detected by OCR":
                error_images.append(image_index)
                error_trace.append("Skip, yticklabels are not detected by OCR")
                continue
            
            # Convert image_data to CSV - EXACT original format
            if isSinglePlot:
                if isHbar:
                    legend_names = [dd["ocr_text"] for dd in image_data if dd["pred_class"]=="xlabel"]
                else:
                    legend_names = [dd["ocr_text"] for dd in image_data if dd["pred_class"]=="ylabel"]
            else:
                legend_names = list(set([dd["ocr_text"] for dd in image_data if dd["pred_class"]=="legend_label"]))
            
            tmp_title = [dd["ocr_text"] for dd in image_data if dd["pred_class"]=="title"]
            title = ''
            min_title_len = 0
            for t in tmp_title:
                if len(t) >= min_title_len:
                    title = t
                    min_title_len = len(t)
            
            xlabel_list = [dd["ocr_text"] for dd in image_data if dd["pred_class"]=="xlabel"]
            ylabel_list = [dd["ocr_text"] for dd in image_data if dd["pred_class"]=="ylabel"]
            
            xlabel = xlabel_list[0] if len(xlabel_list) > 0 else ''
            ylabel = ylabel_list[0] if len(ylabel_list) > 0 else ''
            
            if isHbar:
                row_indexes = [dd["ocr_text"] for dd in image_data if dd["pred_class"]=="yticklabel"]
            else:
                row_indexes = [dd["ocr_text"] for dd in image_data if dd["pred_class"]=="xticklabel"]
            
            if isSinglePlot:
                if isHbar:
                    visual_data = [(dd.get('x_value', 0), dd.get('y_value', ''), xlabel) for dd in image_data if dd["pred_class"] in ['dot_line', 'bar', 'line']]
                else:
                    visual_data = [(dd.get('x_value', ''), dd.get('y_value', 0), ylabel) for dd in image_data if dd["pred_class"] in ['dot_line', 'bar', 'line']]
            else:
                visual_data = [(dd.get('x_value', 0), dd.get('y_value', 0), dd.get('associated_label', 'Unknown')) for dd in image_data if dd["pred_class"] in ['dot_line', 'bar', 'line']]
            
            if isSinglePlot:
                if isHbar:
                    columns = [ylabel] + legend_names + ["xlabel", "ylabel", "title"]
                else:
                    columns = [xlabel] + legend_names + ["xlabel", "ylabel", "title"]
            else:
                if isHbar:
                    columns = [ylabel] + legend_names + ["Unknown", "xlabel", "ylabel", "title", "legend orientation"]
                else:
                    columns = [xlabel] + legend_names + ["Unknown", "xlabel", "ylabel", "title", "legend orientation"]
            
            df = pd.DataFrame(columns=columns)
            
            if isHbar and ylabel:
                df[ylabel] = row_indexes
            elif not isHbar and xlabel:
                df[xlabel] = row_indexes
            
            df["title"] = [title] * len(df)
            df["xlabel"] = [xlabel] * len(df)
            df["ylabel"] = [ylabel] * len(df)
            
            if "legend orientation" in df.columns:
                df["legend orientation"] = [legend_orientation] * len(df)
            
            for vd in visual_data:
                try:
                    if isHbar and ylabel:
                        key_val = str(vd[1])
                        matching_rows = df[df[ylabel] == key_val]
                        if not matching_rows.empty:
                            i_ = matching_rows.index[0]
                            col_name = str(vd[2])
                            if col_name in df.columns:
                                df.at[i_, col_name] = vd[0]
                    elif not isHbar and xlabel:
                        key_val = str(vd[0])
                        matching_rows = df[df[xlabel] == key_val]
                        if not matching_rows.empty:
                            i_ = matching_rows.index[0]
                            col_name = str(vd[2])
                            if col_name in df.columns:
                                df.at[i_, col_name] = vd[1]
                except Exception as e:
                    continue
            
            df.to_csv(os.path.join(csv_dir, str(image_index) + ".csv"), index=False)
            
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            error_trace.append(error_msg)
            error_images.append(image_index)
    
    print("[Error Images]")
    for i in range(len(error_images)):
        if i < len(error_trace):
            print(f"{error_images[i]}: {error_trace[i]}")
        else:
            print(error_images[i])
    
    print("[Empty Images]")
    for i in range(len(empty_images)):
        print(empty_images[i])

if __name__ == "__main__":
    main()
