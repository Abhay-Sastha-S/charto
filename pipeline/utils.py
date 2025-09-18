"""
Utility functions for the PlotQA pipeline

This module contains utility functions needed by the pipeline components,
including:
- Basic geometric utilities (find_center, find_Distance, list_subtraction)
- Color distance calculation functions (colorDistance, Delta E CIE 2000)
- Fixed slope detection that handles numpy slice indices properly

This replaces dependencies on misc/codes/utils.py and the problematic 
colormath library which has issues with numpy.asscalar in newer numpy versions.
"""

import math
import numpy as np
import random


def find_center(bbox):
    """
    Helper method, used to find the center of a bbox.
    
    Args:
        bbox: [x1, y1, x2, y2] bounding box coordinates
        
    Returns:
        (x, y) tuple: X and Y coordinates of the bbox center
    """
    x1, y1, x2, y2 = bbox

    x = 0.5 * (float(x1) + float(x2))
    y = 0.5 * (float(y1) + float(y2))

    return (x, y)


def find_Distance(p1, p2):
    """
    Calculate Euclidean distance between two points
    
    Args:
        p1: (x1, y1) first point
        p2: (x2, y2) second point
        
    Returns:
        float: Euclidean distance between the points
    """
    x1, y1 = p1
    x2, y2 = p2

    d = ((x2-x1)**2 + (y2-y1)**2)**0.5

    return d


def list_subtraction(l1, l2):
    """
    Remove items in l2 from l1
    
    Args:
        l1: List to subtract from
        l2: List of items to remove
        
    Returns:
        List with items from l2 removed from l1
        
    Example:
        l1 = [1,2,3]
        l2 = [1,3]
        return [2]
    """
    return [item for item in l1 if item not in l2]


def safe_argmin(arr):
    """Convert numpy argmin result to Python int to avoid slice index issues"""
    return int(np.argmin(arr))


def find_slope(image, bb):
    """
    Find slope of the line in an image with proper numpy slice index handling
    
    Args:
        image: PIL Image object
        bb: Bounding box [x1, y1, x2, y2]
        
    Returns:
        str: "horizontal", "positive", or "negative"
    """
    if bb[1] == bb[3]:
        slope = "horizontal"
    else:
        bb_image = image.crop(bb).convert('1')  # 0 (black) and 1 (white)
        bb_image_asarray = np.asarray(bb_image, dtype=np.float32)
        
        img_h, img_w = bb_image_asarray.shape
        
        # Convert to integers for slicing to avoid numpy slice index errors
        row, col = int(img_h/2), int(img_w/2)
        
        patchA = bb_image_asarray[0:row, 0:col]
        patchB = bb_image_asarray[0:row, col:]
        patchC = bb_image_asarray[row:, 0:col]
        patchD = bb_image_asarray[row:, col:]
        
        a, b, c, d = np.mean(patchA), np.mean(patchB), np.mean(patchC), np.mean(patchD)
        
        if (a < b) and (c > d):
            slope = "negative"  # take points x1, y1, x2, y2
        elif (a > b) and (c < d):
            slope = "positive"  # take points x1, y2, x2, y1
        else:
            slope = random.choice(["positive", "negative"])
    
    return slope


def colorDistance(c1, c2, method="euclidian"):
    """
    Calculate color distance between two RGB colors
    
    Args:
        c1: First color as [R, G, B] list/tuple
        c2: Second color as [R, G, B] list/tuple  
        method: "euclidian" for RGB distance, "delta_e" for CIE 2000
        
    Returns:
        float: Color distance value
    """
    if method == "euclidian":
        return _colorDistance(c1, c2)
    elif method == "delta_e":
        return _delta_e_cie2000(c1, c2)
    else:
        return _colorDistance(c1, c2)


def _colorDistance(c1, c2):
    """Euclidean color distance in RGB space"""
    x1, y1, z1 = c1
    x2, y2, z2 = c2
    d = ((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)**0.5
    return d


def _srgb_to_xyz(rgb):
    """Convert sRGB to XYZ color space"""
    r, g, b = [x / 255.0 for x in rgb]
    
    # Gamma correction
    def gamma_correct(c):
        if c > 0.04045:
            return pow((c + 0.055) / 1.055, 2.4)
        else:
            return c / 12.92
    
    r = gamma_correct(r)
    g = gamma_correct(g)
    b = gamma_correct(b)
    
    # Convert to XYZ using sRGB matrix (D65 illuminant)
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
    
    return x, y, z


def _xyz_to_lab(xyz):
    """Convert XYZ to LAB color space"""
    x, y, z = xyz
    
    # D65 illuminant reference white
    xn, yn, zn = 0.95047, 1.00000, 1.08883
    
    x = x / xn
    y = y / yn
    z = z / zn
    
    def f(t):
        if t > 0.008856:
            return pow(t, 1/3)
        else:
            return (7.787 * t) + (16/116)
    
    fx = f(x)
    fy = f(y)
    fz = f(z)
    
    l = (116 * fy) - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    
    return l, a, b


def _delta_e_cie2000(rgb1, rgb2):
    """
    Calculate Delta E CIE 2000 color difference
    
    Implementation based on the CIE 2000 formula for perceptual color difference.
    This is more accurate than simple Euclidean distance in RGB space.
    
    Args:
        rgb1: First color as [R, G, B] list/tuple (0-255)
        rgb2: Second color as [R, G, B] list/tuple (0-255)
        
    Returns:
        float: Delta E CIE 2000 color difference
    """
    # Convert RGB to LAB
    xyz1 = _srgb_to_xyz(rgb1)
    xyz2 = _srgb_to_xyz(rgb2)
    lab1 = _xyz_to_lab(xyz1)
    lab2 = _xyz_to_lab(xyz2)
    
    l1, a1, b1 = lab1
    l2, a2, b2 = lab2
    
    # Calculate chroma
    c1 = (a1**2 + b1**2)**0.5
    c2 = (a2**2 + b2**2)**0.5
    
    # Handle edge case: both colors are achromatic
    if c1 == 0 and c2 == 0:
        return abs(l2 - l1)
    
    # Calculate differences
    delta_l = l2 - l1
    delta_c = c2 - c1
    
    # Calculate hue angles
    def calc_hue(a, b):
        if a == 0 and b == 0:
            return 0
        h = math.atan2(b, a) * 180 / math.pi
        return h if h >= 0 else h + 360
    
    h1 = calc_hue(a1, b1)
    h2 = calc_hue(a2, b2)
    
    # Calculate hue difference
    if c1 == 0 or c2 == 0:
        delta_h = 0
    elif abs(h2 - h1) <= 180:
        delta_h = h2 - h1
    elif h2 - h1 > 180:
        delta_h = h2 - h1 - 360
    else:
        delta_h = h2 - h1 + 360
    
    # Calculate delta H (big H)
    delta_big_h = 2 * (c1 * c2)**0.5 * math.sin(math.radians(delta_h / 2))
    
    # Calculate averages
    l_prime = (l1 + l2) / 2
    c_prime = (c1 + c2) / 2
    
    # Calculate average hue
    if c1 == 0 or c2 == 0:
        h_prime = h1 + h2
    elif abs(h1 - h2) <= 180:
        h_prime = (h1 + h2) / 2
    elif abs(h1 - h2) > 180 and (h1 + h2) < 360:
        h_prime = (h1 + h2 + 360) / 2
    else:
        h_prime = (h1 + h2 - 360) / 2
    
    # Calculate T factor
    t = (1 - 0.17 * math.cos(math.radians(h_prime - 30)) +
         0.24 * math.cos(math.radians(2 * h_prime)) +
         0.32 * math.cos(math.radians(3 * h_prime + 6)) -
         0.20 * math.cos(math.radians(4 * h_prime - 63)))
    
    # Calculate rotation term
    delta_ro = 30 * math.exp(-((h_prime - 275) / 25)**2)
    
    # Calculate weighting functions
    sl = 1 + (0.015 * (l_prime - 50)**2) / (20 + (l_prime - 50)**2)**0.5
    sc = 1 + 0.045 * c_prime
    sh = 1 + 0.015 * c_prime * t
    rt = -2 * (c_prime**7 / (c_prime**7 + 25**7))**0.5 * math.sin(2 * math.radians(delta_ro))
    
    # Calculate final Delta E with weighting factors (kL=kC=kH=1 for standard conditions)
    kl = kc = kh = 1
    
    delta_e = ((delta_l / (kl * sl))**2 + 
               (delta_c / (kc * sc))**2 + 
               (delta_big_h / (kh * sh))**2 + 
               rt * (delta_c / (kc * sc)) * (delta_big_h / (kh * sh)))**0.5
    
    return float(delta_e)
