import type { CellPosition, Difficulty, Grid, HistoryEntry, NotesGrid } from "../types";
import { MAX_HISTORY_SIZE, GRID_SIZE } from "../constants";
import { copyGrid, createEmptyGrid, createEmptyNotes } from "./grid";
import { findConflicts } from "./validation";

/** Callback type for state change notifications */
export type StateChangeListener = () => void;

/** Central game state manager */
export class GameState {
  private _puzzle: Grid = createEmptyGrid();
  private _solution: Grid | null = null;
  private _current: Grid = createEmptyGrid();
  private _notes: NotesGrid = createEmptyNotes();
  private _difficulty: Difficulty = "easy";
  private _selected: CellPosition = { row: 0, col: 0 };
  private _startTime: number = Date.now();
  private _elapsedMs: number = 0;
  private _showingSolution: boolean = false;
  private _noteMode: boolean = false;
  private _history: HistoryEntry[] = [];
  private _listeners: Set<StateChangeListener> = new Set();

  get puzzle(): Grid {
    return this._puzzle;
  }

  get solution(): Grid | null {
    return this._solution;
  }

  get current(): Grid {
    return this._current;
  }

  get notes(): NotesGrid {
    return this._notes;
  }

  get difficulty(): Difficulty {
    return this._difficulty;
  }

  get selected(): CellPosition {
    return this._selected;
  }

  get startTime(): number {
    return this._startTime;
  }

  get elapsedMs(): number {
    return this._elapsedMs;
  }

  get showingSolution(): boolean {
    return this._showingSolution;
  }

  get noteMode(): boolean {
    return this._noteMode;
  }

  get canUndo(): boolean {
    return this._history.length > 0;
  }

  /** Subscribe to state changes */
  subscribe(listener: StateChangeListener): () => void {
    this._listeners.add(listener);
    return () => this._listeners.delete(listener);
  }

  private notify(): void {
    this._listeners.forEach((listener) => listener());
  }

  /** Initialize a new game */
  newGame(puzzle: Grid, solution: Grid, difficulty: Difficulty): void {
    this._puzzle = puzzle;
    this._solution = solution;
    this._current = copyGrid(puzzle);
    this._notes = createEmptyNotes();
    this._difficulty = difficulty;
    this._startTime = Date.now();
    this._elapsedMs = 0;
    this._showingSolution = false;
    this._noteMode = false;
    this._history = [];
    this.selectFirstEmpty();
    this.notify();
  }

  /** Start manual entry mode */
  startManualEntry(grid?: Grid): void {
    this._puzzle = createEmptyGrid();
    this._solution = null;
    this._current = grid ? copyGrid(grid) : createEmptyGrid();
    this._notes = createEmptyNotes();
    this._difficulty = "manual";
    this._startTime = Date.now();
    this._elapsedMs = 0;
    this._showingSolution = false;
    this._noteMode = false;
    this._history = [];
    this.selectFirstEmpty();
    this.notify();
  }

  /** Restore game from saved state */
  restore(
    puzzle: Grid,
    solution: Grid | null,
    current: Grid,
    notes: NotesGrid,
    difficulty: Difficulty,
    elapsedMs: number
  ): void {
    this._puzzle = puzzle;
    this._solution = solution;
    this._current = current;
    this._notes = notes;
    this._difficulty = difficulty;
    this._startTime = Date.now() - elapsedMs;
    this._elapsedMs = elapsedMs;
    this._showingSolution = false;
    this._noteMode = false;
    this._history = [];
    this.selectFirstEmpty();
    this.notify();
  }

  /** Apply a scanned grid */
  applyScannedGrid(puzzleGrid: Grid, solutionGrid: Grid): void {
    this._puzzle = copyGrid(puzzleGrid);
    this._solution = solutionGrid;
    this._current = copyGrid(puzzleGrid);
    this._notes = createEmptyNotes();
    this._difficulty = "manual";
    this._startTime = Date.now();
    this._elapsedMs = 0;
    this._showingSolution = false;
    this._noteMode = false;
    this._history = [];
    this.selectFirstEmpty();
    this.notify();
  }

  /** Apply a manual solution */
  applyManualSolution(solutionGrid: Grid): void {
    const manualPuzzle = copyGrid(this._current);
    this._puzzle = manualPuzzle;
    this._solution = solutionGrid;
    this._current = copyGrid(manualPuzzle);
    this._notes = createEmptyNotes();
    this._showingSolution = true;
    this._noteMode = false;
    this._history = [];
    this.notify();
  }

  /** Select a cell */
  selectCell(row: number, col: number): void {
    this._selected = { row, col };
    this.notify();
  }

  /** Select the first empty cell */
  selectFirstEmpty(): void {
    for (let row = 0; row < GRID_SIZE; row++) {
      for (let col = 0; col < GRID_SIZE; col++) {
        if (this._puzzle[row][col] === 0) {
          this._selected = { row, col };
          return;
        }
      }
    }
    this._selected = { row: 0, col: 0 };
  }

  /** Set a cell value */
  setCellValue(row: number, col: number, value: number): boolean {
    if (this._showingSolution) return false;
    if (this._puzzle[row][col] !== 0) return false;
    if (this._current[row][col] === value) return false;

    this.pushHistory(row, col);
    this._current[row][col] = value;
    this._notes[row][col] = 0;
    this.notify();
    return true;
  }

  /** Clear a cell */
  clearCell(row: number, col: number): boolean {
    if (this._showingSolution) return false;
    if (this._puzzle[row][col] !== 0) return false;
    if (this._current[row][col] === 0 && this._notes[row][col] === 0) return false;

    this.pushHistory(row, col);
    this._current[row][col] = 0;
    this._notes[row][col] = 0;
    this.notify();
    return true;
  }

  /** Toggle a note value */
  toggleNote(row: number, col: number, value: number): boolean {
    if (this._showingSolution) return false;
    if (this._puzzle[row][col] !== 0) return false;
    if (this._current[row][col] !== 0) return false;

    this.pushHistory(row, col);
    const bit = 1 << (value - 1);
    this._notes[row][col] ^= bit;
    this.notify();
    return true;
  }

  /** Clear notes for a cell */
  clearNotes(row: number, col: number): boolean {
    if (this._showingSolution) return false;
    if (this._puzzle[row][col] !== 0) return false;
    if (this._notes[row][col] === 0) return false;

    this.pushHistory(row, col);
    this._notes[row][col] = 0;
    this.notify();
    return true;
  }

  /** Toggle note mode */
  toggleNoteMode(): void {
    this._noteMode = !this._noteMode;
    this.notify();
  }

  /** Toggle solution visibility */
  toggleSolution(): void {
    if (!this._solution) return;
    this._showingSolution = !this._showingSolution;
    this.notify();
  }

  /** Undo the last action */
  undo(): boolean {
    const entry = this._history.pop();
    if (!entry) return false;

    this._current[entry.row][entry.col] = entry.current;
    this._notes[entry.row][entry.col] = entry.notes;
    this.notify();
    return true;
  }

  private pushHistory(row: number, col: number): void {
    this._history.push({
      row,
      col,
      current: this._current[row][col],
      notes: this._notes[row][col],
    });
    if (this._history.length > MAX_HISTORY_SIZE) {
      this._history.shift();
    }
  }

  /** Get the currently displayed grid (current or solution) */
  getDisplayGrid(): Grid {
    return this._showingSolution && this._solution ? this._solution : this._current;
  }

  /** Check if the puzzle is solved */
  isSolved(): boolean {
    if (this._showingSolution) return false;
    if (!this._solution) return false;

    const conflicts = findConflicts(this._current);
    if (conflicts.size > 0) return false;

    for (let row = 0; row < GRID_SIZE; row++) {
      for (let col = 0; col < GRID_SIZE; col++) {
        if (this._current[row][col] !== this._solution[row][col]) {
          return false;
        }
      }
    }
    return true;
  }

  /** Get elapsed time in milliseconds */
  getElapsedMs(): number {
    return this._showingSolution
      ? this._elapsedMs
      : Date.now() - this._startTime;
  }

  /** Get state for saving */
  getSaveState(): {
    puzzle: Grid;
    solution: Grid | null;
    current: Grid;
    notes: NotesGrid;
    difficulty: Difficulty;
    elapsedMs: number;
  } {
    return {
      puzzle: this._puzzle,
      solution: this._solution,
      current: this._current,
      notes: this._notes,
      difficulty: this._difficulty,
      elapsedMs: Date.now() - this._startTime,
    };
  }
}

/** Singleton game state instance */
export const gameState = new GameState();
