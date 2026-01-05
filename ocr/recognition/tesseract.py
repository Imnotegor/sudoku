"""Tesseract OCR functions."""

from typing import List

import cv2
import numpy as np
import pytesseract

from ..config import get_settings
from ..preprocessing import prepare_digit_mask, base_threshold
from ..utils import maybe_save_debug


def ocr_by_boxes(warped: np.ndarray) -> tuple[List[List[int]], List[List[int]]]:
    """
    Performs OCR using Tesseract's bounding box detection.

    Args:
        warped: Preprocessed grayscale image of the Sudoku board.

    Returns:
        Tuple of (grid values, confidence grid).
    """
    settings = get_settings()
    digit_mask = prepare_digit_mask(warped)

    if cv2.countNonZero(digit_mask) < 1500:
        digit_mask = base_threshold(warped)

    thresh = cv2.bitwise_not(digit_mask)
    maybe_save_debug("ocr_thresh", thresh)

    config = "--oem 3 --psm 6 -c tessedit_char_whitelist=123456789"
    data = pytesseract.image_to_data(thresh, config=config, output_type=pytesseract.Output.DICT)

    cell_size = settings.target_size / settings.grid_size
    grid = [[0 for _ in range(settings.grid_size)] for _ in range(settings.grid_size)]
    conf_map = [[0 for _ in range(settings.grid_size)] for _ in range(settings.grid_size)]

    for i, text in enumerate(data.get("text", [])):
        text = text.strip()
        if not text.isdigit():
            continue

        value = int(text)
        if value < 1 or value > 9:
            continue

        try:
            conf = int(float(data.get("conf", [0])[i]))
        except (ValueError, TypeError):
            conf = 0

        if conf < settings.min_ocr_confidence:
            continue

        x = data.get("left", [0])[i]
        y = data.get("top", [0])[i]
        w = data.get("width", [0])[i]
        h = data.get("height", [0])[i]
        cx = x + w / 2
        cy = y + h / 2
        row = int(cy / cell_size)
        col = int(cx / cell_size)

        if 0 <= row < settings.grid_size and 0 <= col < settings.grid_size:
            if conf > conf_map[row][col]:
                conf_map[row][col] = conf
                grid[row][col] = value

    return grid, conf_map


def ocr_digit_image(digit: np.ndarray, configs: list[str]) -> tuple[int, int]:
    """
    Recognizes a digit using multiple Tesseract configurations.

    Args:
        digit: Preprocessed digit image (white digit on black background).
        configs: List of Tesseract configuration strings to try.

    Returns:
        Tuple of (digit value, confidence). Returns (0, 0) if no digit found.
    """
    settings = get_settings()
    best_value = 0
    best_conf = -1

    for config in configs:
        data = pytesseract.image_to_data(
            digit, config=config, output_type=pytesseract.Output.DICT
        )

        for text, conf in zip(data.get("text", []), data.get("conf", [])):
            text = text.strip()
            if not text.isdigit():
                continue

            try:
                conf_value = int(float(conf))
            except (ValueError, TypeError):
                continue

            if conf_value > best_conf:
                value = int(text)
                if 1 <= value <= 9:
                    best_conf = conf_value
                    best_value = value

        if best_conf >= settings.min_cell_confidence:
            return best_value, best_conf

    return 0, 0
