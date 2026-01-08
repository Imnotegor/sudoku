import { GRID_SIZE } from "../constants";
import { gameState, findConflicts, findIncorrect, isRelatedCell } from "../game";

let boardElement: HTMLElement | null = null;
let cells: HTMLButtonElement[] = [];
let lastRender: CellRenderState[] = [];

type CellRenderState = {
  value: number;
  notes: number;
  fixed: boolean;
  selected: boolean;
  related: boolean;
  conflict: boolean;
  incorrect: boolean;
  same: boolean;
};

/** Initialize the board module with the board element */
export function initBoard(element: HTMLElement): void {
  boardElement = element;
  buildBoard();
}

/** Build the board grid */
function buildBoard(): void {
  if (!boardElement) return;

  boardElement.innerHTML = "";
  cells = [];
  lastRender = [];

  for (let row = 0; row < GRID_SIZE; row++) {
    for (let col = 0; col < GRID_SIZE; col++) {
      const cell = document.createElement("button");
      cell.className = "cell";
      cell.type = "button";
      cell.dataset.row = String(row);
      cell.dataset.col = String(col);
      cell.setAttribute("role", "gridcell");
      cell.setAttribute("aria-selected", "false");

      // Add borders for 3x3 boxes
      if (col === 2 || col === 5) cell.classList.add("br");
      if (row === 2 || row === 5) cell.classList.add("bb");

      boardElement.appendChild(cell);
      cells.push(cell);
      lastRender.push({
        value: -1,
        notes: -1,
        fixed: false,
        selected: false,
        related: false,
        conflict: false,
        incorrect: false,
        same: false,
      });
    }
  }
}

/** Render notes as a 3x3 grid */
function renderNotes(mask: number): string {
  const items: string[] = [];
  for (let i = 1; i <= 9; i++) {
    const bit = 1 << (i - 1);
    items.push(`<span>${mask & bit ? i : ""}</span>`);
  }
  return `<div class="notes-grid">${items.join("")}</div>`;
}

/** Update the board display */
export function updateBoard(): void {
  if (!boardElement) return;

  const displayGrid = gameState.getDisplayGrid();
  const conflicts = findConflicts(displayGrid);
  const incorrectCells = findIncorrect(gameState.current, gameState.solution, gameState.puzzle);
  const { row: selectedRow, col: selectedCol } = gameState.selected;
  const selectedValue = displayGrid[selectedRow][selectedCol];
  const puzzle = gameState.puzzle;
  const notes = gameState.notes;

  cells.forEach((cell) => {
    const row = Number(cell.dataset.row);
    const col = Number(cell.dataset.col);
    const index = row * GRID_SIZE + col;
    const value = displayGrid[row][col];
    const noteMask = notes[row][col];
    const fixed = puzzle[row][col] !== 0;
    const selected = row === selectedRow && col === selectedCol;
    const related = isRelatedCell(row, col, selectedRow, selectedCol);
    const conflict = conflicts.has(index);
    const incorrect = incorrectCells.has(index);
    const same = value !== 0 && value === selectedValue && !selected;

    const previous = lastRender[index];
    const next: CellRenderState = {
      value,
      notes: noteMask,
      fixed,
      selected,
      related,
      conflict,
      incorrect,
      same,
    };

    if (
      previous &&
      previous.value === next.value &&
      previous.notes === next.notes &&
      previous.fixed === next.fixed &&
      previous.selected === next.selected &&
      previous.related === next.related &&
      previous.conflict === next.conflict &&
      previous.incorrect === next.incorrect &&
      previous.same === next.same
    ) {
      return;
    }

    if (value === 0) {
      if (noteMask) {
        cell.innerHTML = renderNotes(noteMask);
        cell.classList.add("notes");
      } else {
        cell.textContent = "";
        cell.classList.remove("notes");
      }
    } else {
      cell.textContent = String(value);
      cell.classList.remove("notes");
    }

    cell.classList.toggle("fixed", fixed);
    cell.classList.toggle("selected", selected);
    cell.classList.toggle("related", related);
    cell.classList.toggle("conflict", conflict);
    cell.classList.toggle("incorrect", incorrect);
    cell.classList.toggle("same", same);
    cell.setAttribute("aria-selected", selected ? "true" : "false");

    lastRender[index] = next;
  });
}

/** Handle board cell click */
export function handleBoardClick(callback: (row: number, col: number) => void): void {
  if (!boardElement) return;

  boardElement.addEventListener("click", (event) => {
    const cell = (event.target as HTMLElement).closest(".cell");
    if (!cell || !(cell instanceof HTMLElement)) return;

    const row = Number(cell.dataset.row);
    const col = Number(cell.dataset.col);
    callback(row, col);
  });
}

/** Set loading state on the board */
export function setBoardLoading(loading: boolean): void {
  boardElement?.classList.toggle("loading", loading);
}
