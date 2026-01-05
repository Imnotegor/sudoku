"""Grid scanning and digit detection."""

from typing import List

import cv2

from ..config import get_settings
from ..preprocessing import warp_board, prepare_digit_mask, base_threshold
from ..recognition import extract_digit, ocr_by_boxes
from ..utils import maybe_save_debug
from .conflicts import suppress_conflicts, prune_unsolvable


def scan_grid(image) -> List[List[int]]:
    """
    Scans a Sudoku grid from an image.

    Args:
        image: BGR color image containing a Sudoku puzzle.

    Returns:
        9x9 list of digit values (0 for empty cells).
    """
    settings = get_settings()
    warped = warp_board(image)
    warped = cv2.resize(warped, (settings.target_size, settings.target_size))

    digit_mask = prepare_digit_mask(warped)
    if cv2.countNonZero(digit_mask) < 1500:
        digit_mask = base_threshold(warped)

    maybe_save_debug("warp", warped)
    maybe_save_debug("digit_mask", digit_mask)

    cell_size = settings.target_size // settings.grid_size

    grid = []
    conf_grid = []

    for row in range(settings.grid_size):
        row_values = []
        row_conf = []

        for col in range(settings.grid_size):
            y1 = row * cell_size
            y2 = (row + 1) * cell_size
            x1 = col * cell_size
            x2 = (col + 1) * cell_size

            cell_gray = warped[y1:y2, x1:x2]
            cell_mask = digit_mask[y1:y2, x1:x2]
            value, conf = extract_digit(cell_gray, cell_mask)

            row_values.append(value)
            row_conf.append(conf)

        grid.append(row_values)
        conf_grid.append(row_conf)

    # If cell-by-cell detection found few digits, try bounding box approach
    if count_digits(grid) < 20:
        box_grid, box_conf = ocr_by_boxes(warped)
        for row in range(settings.grid_size):
            for col in range(settings.grid_size):
                if grid[row][col] == 0 and box_grid[row][col] != 0:
                    grid[row][col] = box_grid[row][col]
                    conf_grid[row][col] = box_conf[row][col]

    suppress_conflicts(grid, conf_grid)
    prune_unsolvable(grid, conf_grid)

    return grid


def count_digits(grid: List[List[int]]) -> int:
    """Counts non-zero digits in the grid."""
    return sum(1 for row in grid for value in row if value > 0)
