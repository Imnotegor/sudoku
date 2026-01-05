"""Sudoku solver for validation."""

from typing import List, Optional

from .config import get_settings


def is_solvable(grid: List[List[int]]) -> bool:
    """
    Checks if a Sudoku grid is solvable.

    Args:
        grid: 9x9 grid of digit values.

    Returns:
        True if the grid has a valid solution.
    """
    grid_copy = [row[:] for row in grid]
    return solve_grid(grid_copy)


def solve_grid(grid: List[List[int]]) -> bool:
    """
    Solves a Sudoku grid in place using backtracking.

    Args:
        grid: 9x9 grid of digit values (modified in place).

    Returns:
        True if the grid was solved successfully.
    """
    cell = _find_best_cell(grid)
    if cell is None:
        return True

    row, col, options = cell
    if not options:
        return False

    for value in options:
        grid[row][col] = value
        if solve_grid(grid):
            return True

    grid[row][col] = 0
    return False


def _find_best_cell(grid: List[List[int]]) -> Optional[tuple[int, int, List[int]]]:
    """Finds the empty cell with the fewest valid candidates."""
    settings = get_settings()
    best = None
    best_options: List[int] = []

    for row in range(settings.grid_size):
        for col in range(settings.grid_size):
            if grid[row][col] != 0:
                continue

            options = _get_candidates(grid, row, col)

            if best is None or len(options) < len(best_options):
                best = (row, col, options)
                best_options = options

            if best is not None and len(best_options) <= 1:
                return best

    return best


def _get_candidates(grid: List[List[int]], row: int, col: int) -> List[int]:
    """Returns valid candidates for a cell."""
    settings = get_settings()
    used = set()

    # Row values
    used.update(grid[row])

    # Column values
    used.update(grid[r][col] for r in range(settings.grid_size))

    # Box values
    box_row = (row // 3) * 3
    box_col = (col // 3) * 3
    for r in range(box_row, box_row + 3):
        for c in range(box_col, box_col + 3):
            used.add(grid[r][c])

    return [n for n in range(1, 10) if n not in used]
