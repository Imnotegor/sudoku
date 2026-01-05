"""Image enhancement functions for improving OCR accuracy."""

import cv2
import numpy as np

from ..config import get_settings
from ..utils import maybe_save_debug


def preprocess_warped(warped: np.ndarray) -> np.ndarray:
    """
    Enhances a warped grayscale image for better OCR.

    Args:
        warped: Grayscale warped image of the Sudoku board.

    Returns:
        Enhanced grayscale image.
    """
    denoise = cv2.fastNlMeansDenoising(warped, None, 12, 7, 21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoise)
    blur = cv2.GaussianBlur(enhanced, (0, 0), 1.2)
    enhanced = cv2.addWeighted(enhanced, 1.5, blur, -0.5, 0)
    maybe_save_debug("warp_pre", enhanced)
    return enhanced


def prepare_digit_mask(warped: np.ndarray) -> np.ndarray:
    """
    Prepares a mask highlighting digit pixels while suppressing grid lines.

    Args:
        warped: Grayscale warped image of the Sudoku board.

    Returns:
        Binary mask with digits highlighted.
    """
    settings = get_settings()
    blur = cv2.GaussianBlur(warped, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 3
    )
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    size = thresh.shape[0]
    cell = max(10, size // settings.grid_size)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, cell))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cell, 1))

    vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    grid = cv2.bitwise_or(vertical, horizontal)
    digits = cv2.subtract(thresh, grid)
    digits = cv2.morphologyEx(digits, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    digits = cv2.dilate(digits, np.ones((2, 2), np.uint8), iterations=1)
    return digits


def base_threshold(warped: np.ndarray) -> np.ndarray:
    """
    Creates a basic threshold for digit detection.

    Args:
        warped: Grayscale warped image.

    Returns:
        Binary threshold image.
    """
    blur = cv2.GaussianBlur(warped, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 4
    )
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return thresh


def threshold_cell(gray: np.ndarray) -> np.ndarray:
    """
    Thresholds a cell image for digit extraction.

    Args:
        gray: Grayscale cell image.

    Returns:
        Binary threshold image.
    """
    settings = get_settings()
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    otsu_ratio = cv2.countNonZero(otsu) / float(otsu.size)
    adaptive_ratio = cv2.countNonZero(adaptive) / float(adaptive.size)

    if 0.003 <= adaptive_ratio <= 0.18 and (
        abs(adaptive_ratio - settings.target_fg_ratio)
        <= abs(otsu_ratio - settings.target_fg_ratio)
    ):
        return adaptive

    return otsu
