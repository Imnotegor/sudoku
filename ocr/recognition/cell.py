"""Cell-level digit extraction and recognition."""

import cv2
import numpy as np

from ..config import get_settings
from ..preprocessing.enhance import threshold_cell
from .cnn import get_classifier
from .tesseract import ocr_digit_image


def extract_digit(cell_gray: np.ndarray, cell_mask: np.ndarray) -> tuple[int, int]:
    """
    Extracts and recognizes a digit from a cell.

    Args:
        cell_gray: Grayscale cell image.
        cell_mask: Binary mask of potential digit pixels.

    Returns:
        Tuple of (digit value, confidence). Returns (0, 0) if no digit found.
    """
    size = cell_gray.shape[0]
    margin = int(size * 0.12)

    if size - margin <= margin:
        return 0, 0

    # Crop margins
    gray = cell_gray[margin : size - margin, margin : size - margin]
    mask = cell_mask[margin : size - margin, margin : size - margin]

    if gray.size == 0 or mask.size == 0:
        return 0, 0

    # Clean up mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    mask_ratio = cv2.countNonZero(mask) / float(mask.size)

    # Try mask-based extraction if there's reasonable ink content
    if 0.002 <= mask_ratio <= 0.3:
        value, conf = _ocr_from_mask(gray, mask)
        if value:
            return value, conf

    return _ocr_from_gray(gray)


def _ocr_from_mask(gray: np.ndarray, mask: np.ndarray) -> tuple[int, int]:
    """Recognizes digit using the provided mask."""
    settings = get_settings()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0, 0

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    min_area = max(settings.min_component_area, int(mask.size * 0.006))

    if area < min_area:
        return 0, 0

    x, y, w, h = cv2.boundingRect(contour)
    digit_gray = gray[y : y + h, x : x + w]

    if digit_gray.size == 0:
        return 0, 0

    return _ocr_from_gray(digit_gray)


def _ocr_from_gray(gray: np.ndarray) -> tuple[int, int]:
    """Recognizes digit from grayscale image."""
    settings = get_settings()

    # Normalize and threshold
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = threshold_cell(blur)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    thresh = cv2.dilate(thresh, np.ones((2, 2), np.uint8), iterations=1)

    if cv2.countNonZero(thresh) < max(15, int(thresh.size * 0.008)):
        return 0, 0

    # Find largest contour
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    min_area = max(settings.min_component_area, int(thresh.size * 0.006))

    if area < min_area:
        return 0, 0

    x, y, w, h = cv2.boundingRect(contour)
    digit = thresh[y : y + h, x : x + w]

    if digit.size == 0:
        return 0, 0

    # Try CNN first
    classifier = get_classifier()
    if classifier:
        cnn_value, cnn_conf = classifier.predict(digit)
        if cnn_value:
            return cnn_value, cnn_conf

    # Fallback to Tesseract
    pad = 6
    digit = cv2.copyMakeBorder(digit, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    digit = cv2.resize(digit, (48, 48), interpolation=cv2.INTER_AREA)
    digit = cv2.bitwise_not(digit)

    configs = [
        "--oem 3 --psm 10 -c tessedit_char_whitelist=123456789 -c classify_bln_numeric_mode=1",
        "--oem 3 --psm 13 -c tessedit_char_whitelist=123456789 -c classify_bln_numeric_mode=1",
        "--oem 3 --psm 8 -c tessedit_char_whitelist=123456789 -c classify_bln_numeric_mode=1",
    ]
    return ocr_digit_image(digit, configs)
