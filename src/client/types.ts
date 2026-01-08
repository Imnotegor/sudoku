/** A single row of the Sudoku grid (0 = empty, 1-9 = digit) */
export type GridRow = [number, number, number, number, number, number, number, number, number];

/** 9x9 Sudoku grid */
export type Grid = [GridRow, GridRow, GridRow, GridRow, GridRow, GridRow, GridRow, GridRow, GridRow];

/** Notes grid - each cell stores a bitmask of possible values (1-9) */
export type NotesGrid = number[][];

/** Cell position */
export interface CellPosition {
  readonly row: number;
  readonly col: number;
}

/** Difficulty level */
export type Difficulty = "easy" | "medium" | "hard" | "expert" | "manual";

/** Supported locales */
export type Locale = "ru" | "en";

/** Translation keys */
export interface Translations {
  readonly title: string;
  readonly eyebrow: string;
  readonly hero_title: string;
  readonly hero_sub: string;
  readonly menu: string;
  readonly settings: string;
  readonly board_label: string;
  readonly difficulty_heading: string;
  readonly actions_heading: string;
  readonly numbers_heading: string;
  readonly scan_heading: string;
  readonly manual_heading: string;
  readonly manual_placeholder: string;
  readonly hint: string;
  readonly hint_action: string;
  readonly new_game: string;
  readonly manual_entry: string;
  readonly solve_manual: string;
  readonly manual_import: string;
  readonly manual_export: string;
  readonly clear_cell: string;
  readonly undo: string;
  readonly notes: string;
  readonly show_solution: string;
  readonly hide_solution: string;
  readonly open_camera: string;
  readonly upload_photo: string;
  readonly camera_close: string;
  readonly camera_capture: string;
  readonly difficulty_label: string;
  readonly difficulty_easy: string;
  readonly difficulty_medium: string;
  readonly difficulty_hard: string;
  readonly difficulty_expert: string;
  readonly difficulty_manual: string;
  readonly online: string;
  readonly offline: string;
  readonly toast_new_game: string;
  readonly toast_solved: string;
  readonly toast_manual_ready: string;
  readonly toast_manual_imported: string;
  readonly toast_manual_exported: string;
  readonly toast_manual_copied: string;
  readonly toast_manual_invalid: string;
  readonly toast_manual_empty: string;
  readonly toast_manual_solved: string;
  readonly toast_solve_error: string;
  readonly toast_hint_used: string;
  readonly toast_hint_unavailable: string;
  readonly toast_scan_start: string;
  readonly toast_scan_done: string;
  readonly toast_scan_error: string;
  readonly toast_camera_denied: string;
  readonly toast_camera_unsupported: string;
}

/** History entry for undo functionality */
export interface HistoryEntry {
  readonly row: number;
  readonly col: number;
  readonly current: number;
  readonly notes: number;
}

/** Game state stored in localStorage */
export interface SavedGameState {
  readonly puzzle: Grid;
  readonly solution: Grid | null;
  readonly current: Grid;
  readonly notes: NotesGrid;
  readonly difficulty: Difficulty;
  readonly elapsedMs: number;
}

/** Scan API response */
export interface ScanResponse {
  readonly puzzle: number[][];
  readonly solution: number[][];
}
