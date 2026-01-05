"""Conflict resolution for OCR results."""

from typing import List

from ..config import get_settings
from ..solver import is_solvable


def suppress_conflicts(grid: List[List[int]], conf_grid: List[List[int]]) -> None:
    """
    Removes conflicting digits, keeping the one with highest confidence.

    Args:
        grid: 9x9 grid of digit values (modified in place).
        conf_grid: 9x9 grid of confidence values.
    """
    settings = get_settings()

    # Check rows
    for row in range(settings.grid_size):
        cells = [(row, col) for col in range(settings.grid_size)]
        _remove_conflicts_in_line(cells, grid, conf_grid)

    # Check columns
    for col in range(settings.grid_size):
        cells = [(row, col) for row in range(settings.grid_size)]
        _remove_conflicts_in_line(cells, grid, conf_grid)

    # Check 3x3 boxes
    for box_row in range(0, settings.grid_size, 3):
        for box_col in range(0, settings.grid_size, 3):
            cells = [
                (box_row + r, box_col + c) for r in range(3) for c in range(3)
            ]
            _remove_conflicts_in_line(cells, grid, conf_grid)


def _remove_conflicts_in_line(
    cells: List[tuple[int, int]],
    grid: List[List[int]],
    conf_grid: List[List[int]],
) -> None:
    """Removes duplicate values in a line, keeping the highest confidence one."""
    positions: dict[int, list[tuple[int, int]]] = {}

    for row, col in cells:
        value = grid[row][col]
        if value == 0:
            continue
        positions.setdefault(value, []).append((row, col))

    for spots in positions.values():
        if len(spots) <= 1:
            continue

        # Keep the spot with highest confidence
        best = max(spots, key=lambda rc: conf_grid[rc[0]][rc[1]])
        for row, col in spots:
            if (row, col) != best:
                grid[row][col] = 0


def prune_unsolvable(grid: List[List[int]], conf_grid: List[List[int]]) -> None:
    """
    Removes low-confidence digits until the grid becomes solvable.

    Args:
        grid: 9x9 grid of digit values (modified in place).
        conf_grid: 9x9 grid of confidence values.
    """
    if is_solvable(grid):
        return

    settings = get_settings()

    # Collect all filled cells with their confidence
    candidates = []
    for row in range(settings.grid_size):
        for col in range(settings.grid_size):
            if grid[row][col] != 0:
                candidates.append((conf_grid[row][col], row, col))

    # Sort by confidence (lowest first)
    candidates.sort()

    # Remove cells one by one until solvable
    for _, row, col in candidates:
        grid[row][col] = 0
        conf_grid[row][col] = 0
        if is_solvable(grid):
            return
