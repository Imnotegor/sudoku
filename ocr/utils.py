"""Utility functions for the OCR service."""

import os

import cv2
import numpy as np

from .config import get_settings


def maybe_save_debug(stage: str, image: np.ndarray) -> None:
    """
    Saves an image to the debug directory if debug mode is enabled.

    Args:
        stage: Name of the processing stage (used in filename).
        image: Image to save.
    """
    settings = get_settings()

    if not settings.debug_ocr:
        return

    os.makedirs(settings.debug_ocr_dir, exist_ok=True)
    cv2.imwrite(os.path.join(settings.debug_ocr_dir, f"{stage}.png"), image)
