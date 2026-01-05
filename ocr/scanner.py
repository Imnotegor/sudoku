"""Main scanner class for Sudoku OCR."""

from typing import List

import cv2
import numpy as np

from .grid import scan_grid


class SudokuScanner:
    """
    Sudoku scanner that extracts digits from images.

    Example usage:
        scanner = SudokuScanner()
        grid = scanner.scan(image)
    """

    def scan(self, image: np.ndarray) -> List[List[int]]:
        """
        Scans a Sudoku puzzle from an image.

        Args:
            image: BGR color image containing a Sudoku puzzle.

        Returns:
            9x9 list of digit values (0 for empty cells).
        """
        return scan_grid(image)

    def scan_from_bytes(self, image_bytes: bytes) -> List[List[int]]:
        """
        Scans a Sudoku puzzle from image bytes.

        Args:
            image_bytes: Raw image bytes (JPEG, PNG, etc.).

        Returns:
            9x9 list of digit values (0 for empty cells).

        Raises:
            ValueError: If the image cannot be decoded.
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Could not decode image")

        return self.scan(image)
