"""Perspective warp functions for extracting the Sudoku board."""

import cv2
import numpy as np

from ..config import get_settings
from ..utils import maybe_save_debug
from .enhance import preprocess_warped


def order_points(pts: np.ndarray) -> np.ndarray:
    """Orders corner points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def warp_board(image: np.ndarray) -> np.ndarray:
    """
    Extracts and warps the Sudoku board from an image.

    Args:
        image: BGR color image containing a Sudoku puzzle.

    Returns:
        Grayscale image of the warped board, preprocessed for OCR.
    """
    settings = get_settings()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contour = find_board_contour(gray)

    if contour is None:
        resized = cv2.resize(gray, (settings.target_size, settings.target_size))
        return preprocess_warped(resized)

    rect = order_points(contour.reshape(4, 2))
    dst = np.array(
        [
            [0, 0],
            [settings.target_size - 1, 0],
            [settings.target_size - 1, settings.target_size - 1],
            [0, settings.target_size - 1],
        ],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(gray, matrix, (settings.target_size, settings.target_size))
    maybe_save_debug("warp_raw", warped)
    return preprocess_warped(warped)


def find_board_contour(gray: np.ndarray) -> np.ndarray | None:
    """Finds the Sudoku board contour in a grayscale image."""
    contour = find_board_by_edges(gray)
    if contour is not None:
        return contour
    return find_board_by_threshold(gray)


def find_board_by_edges(gray: np.ndarray) -> np.ndarray | None:
    """Finds the board using edge detection."""
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    maybe_save_debug("edges", edges)
    return select_board_contour(gray, edges)


def find_board_by_threshold(gray: np.ndarray) -> np.ndarray | None:
    """Finds the board using adaptive thresholding."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 4
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    maybe_save_debug("threshold_board", thresh)
    return select_board_contour(gray, thresh)


def select_board_contour(gray: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """Selects the best quadrilateral contour from a mask."""
    settings = get_settings()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    image_area = gray.shape[0] * gray.shape[1]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours[:15]:
        area = cv2.contourArea(contour)
        if area < image_area * settings.min_board_area_ratio:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            return approx

    # Fallback to bounding box of largest contour
    contour = contours[0]
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return np.intp(box)
