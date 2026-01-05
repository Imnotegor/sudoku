import type { Difficulty, Grid } from "./types";
import { GRID_SIZE, UI_UPDATE_DELAY_MS } from "./constants";
import { detectLocale, setLocale, t, applyTranslations } from "./i18n";
import {
  gameState,
  generatePuzzle,
  copyGrid,
  hasAnyDigits,
  findConflicts,
  solveGrid,
  parseManualText,
  sanitizeGrid,
  parsePuzzleGrid,
  parseSolvedGrid,
  formatGrid,
} from "./game";
import {
  initToast,
  showToast,
  initTimer,
  updateTimer,
  startTimer,
  initNumpad,
  updateNumpadAvailability,
  handleNumpadClick,
  initBoard,
  updateBoard,
  handleBoardClick,
  setBoardLoading,
  initCamera,
  openCamera,
  bindCameraEvents,
  initDifficulty,
  updateDifficultyButtons,
  handleDifficultyClick,
  initOnlineStatus,
} from "./ui";
import { saveGame, loadGame } from "./storage";
import { ApiError, scanImage } from "./api";

/** DOM element references */
function getElements() {
  return {
    board: document.getElementById("board")!,
    difficulty: document.getElementById("difficulty")!,
    difficultyLabel: document.getElementById("difficulty-label")!,
    timer: document.getElementById("timer")!,
    onlineStatus: document.getElementById("online-status")!,
    newGameBtn: document.getElementById("new-game") as HTMLButtonElement,
    manualEntryBtn: document.getElementById("manual-entry") as HTMLButtonElement | null,
    solveManualBtn: document.getElementById("solve-manual") as HTMLButtonElement | null,
    manualInput: document.getElementById("manual-input") as HTMLTextAreaElement | null,
    manualImportBtn: document.getElementById("manual-import") as HTMLButtonElement | null,
    manualExportBtn: document.getElementById("manual-export") as HTMLButtonElement | null,
    clearBtn: document.getElementById("clear-cell") as HTMLButtonElement,
    undoBtn: document.getElementById("undo-move") as HTMLButtonElement | null,
    toggleNotesBtn: document.getElementById("toggle-notes") as HTMLButtonElement | null,
    showSolutionBtn: document.getElementById("show-solution") as HTMLButtonElement,
    numpad: document.getElementById("numpad")!,
    toast: document.getElementById("toast")!,
    openCameraBtn: document.getElementById("open-camera") as HTMLButtonElement | null,
    uploadPhotoBtn: document.getElementById("upload-photo") as HTMLButtonElement | null,
    photoInput: document.getElementById("photo-input") as HTMLInputElement | null,
    camera: document.getElementById("camera") as HTMLElement | null,
    cameraVideo: document.getElementById("camera-video") as HTMLVideoElement | null,
    cameraCaptureBtn: document.getElementById("camera-capture") as HTMLButtonElement | null,
    cameraCloseBtn: document.getElementById("camera-close") as HTMLButtonElement | null,
    cameraCanvas: document.getElementById("camera-canvas") as HTMLCanvasElement | null,
    panel: document.querySelector(".panel") as HTMLElement | null,
    numbersSection: document.querySelector(".numbers-section") as HTMLElement | null,
    mobileNumpadHost: document.getElementById("mobile-numpad-host") as HTMLElement | null,
    mobileMenuOverlay: document.getElementById("mobile-menu-overlay") as HTMLElement | null,
    mobileMenuBody: document.getElementById("mobile-menu-body") as HTMLElement | null,
    mobileMenuBtn: document.getElementById("mobile-menu") as HTMLButtonElement | null,
    mobileMenuCloseBtn: document.getElementById("mobile-menu-close") as HTMLButtonElement | null,
    mobileTimer: document.getElementById("mobile-timer") as HTMLElement | null,
    mobileUndoBtn: document.getElementById("mobile-undo") as HTMLButtonElement | null,
    mobileNotesBtn: document.getElementById("mobile-notes") as HTMLButtonElement | null,
    mobileSettingsBtn: document.getElementById("mobile-settings") as HTMLButtonElement | null,
    mobileQuickActions: document.querySelector(".mobile-quick-actions") as HTMLElement | null,
    mobileQuickUndoBtn: document.querySelector(
      ".mobile-quick-actions [data-action='undo']"
    ) as HTMLButtonElement | null,
    mobileQuickNotesBtn: document.querySelector(
      ".mobile-quick-actions [data-action='notes']"
    ) as HTMLButtonElement | null,
    mobileQuickSolutionBtn: document.querySelector(
      ".mobile-quick-actions [data-action='solution']"
    ) as HTMLButtonElement | null,
  };
}

let elements: ReturnType<typeof getElements>;
const MOBILE_BREAKPOINT_PX = 720;
let panelSections: HTMLElement[] = [];
let menuSections: HTMLElement[] = [];
let numbersSection: HTMLElement | null = null;
let mobileLayoutQuery: MediaQueryList | null = null;

type PuzzleWorkerMessage = {
  id: number;
  difficulty: Difficulty;
};

type PuzzleWorkerResponse = {
  id: number;
  difficulty: Difficulty;
  puzzle: Grid;
  solution: Grid;
};

const PUZZLE_WORKER_PATH = "/dist/worker.js";

let puzzleWorker: Worker | null = null;
let generationRequestId = 0;
let pendingDifficulty: Difficulty | null = null;

function getPuzzleWorker(): Worker | null {
  if (puzzleWorker) return puzzleWorker;
  if (typeof Worker === "undefined") return null;
  if (window.location.protocol !== "http:" && window.location.protocol !== "https:") return null;

  try {
    puzzleWorker = new Worker(PUZZLE_WORKER_PATH);
    puzzleWorker.addEventListener("message", handleWorkerMessage);
    puzzleWorker.addEventListener("error", handleWorkerError);
    return puzzleWorker;
  } catch {
    puzzleWorker = null;
    return null;
  }
}

function handleWorkerMessage(event: MessageEvent<PuzzleWorkerResponse>): void {
  const { id, puzzle, solution, difficulty } = event.data;
  if (id !== generationRequestId) return;
  finalizeNewGame(puzzle, solution, difficulty);
}

function handleWorkerError(): void {
  if (puzzleWorker) {
    puzzleWorker.terminate();
    puzzleWorker = null;
  }

  if (!pendingDifficulty) return;
  const requestId = generationRequestId;
  const difficulty = pendingDifficulty;
  fallbackGenerate(difficulty, requestId);
}

function finalizeNewGame(puzzle: Grid, solution: Grid, difficulty: Difficulty): void {
  pendingDifficulty = null;
  gameState.newGame(puzzle, solution, difficulty);
  saveGame(gameState.getSaveState());
  setBoardLoading(false);
  showToast(t("toast_new_game"));
}

function fallbackGenerate(difficulty: Difficulty, requestId: number): void {
  setTimeout(() => {
    if (requestId !== generationRequestId) return;
    const { puzzle, solution } = generatePuzzle(difficulty);
    finalizeNewGame(puzzle, solution, difficulty);
  }, UI_UPDATE_DELAY_MS);
}

function cancelPendingGeneration(): void {
  generationRequestId += 1;
  pendingDifficulty = null;
}

/** Initialize the application */
function init(): void {
  setLocale(detectLocale());
  applyTranslations();

  elements = getElements();

  // Initialize UI modules
  initToast(elements.toast);
  initTimer(elements.timer, elements.mobileTimer);
  initBoard(elements.board);
  initNumpad(elements.numpad);
  initDifficulty(elements.difficulty, elements.difficultyLabel);
  initOnlineStatus(elements.onlineStatus);

  initMobileLayout();

  if (
    elements.camera &&
    elements.cameraVideo &&
    elements.cameraCaptureBtn &&
    elements.cameraCloseBtn &&
    elements.cameraCanvas
  ) {
    initCamera(
      elements.camera,
      elements.cameraVideo,
      elements.cameraCaptureBtn,
      elements.cameraCloseBtn,
      elements.cameraCanvas
    );
  }

  // Bind event handlers
  bindEvents();

  // Subscribe to state changes
  gameState.subscribe(updateUI);

  // Restore or start new game
  restoreGame();

  // Start timer
  startTimer();

  // Register service worker
  registerServiceWorker();

  document.documentElement.dataset.appReady = "true";
}

function initMobileLayout(): void {
  if (!elements.panel) return;

  panelSections = Array.from(elements.panel.querySelectorAll<HTMLElement>(".panel-section"));
  numbersSection =
    panelSections.find((section) => section.classList.contains("numbers-section")) ?? null;
  menuSections = panelSections.filter((section) => section !== numbersSection);

  mobileLayoutQuery = window.matchMedia(`(max-width: ${MOBILE_BREAKPOINT_PX}px)`);
  mobileLayoutQuery.addEventListener("change", syncMobileLayout);
  syncMobileLayout();
}

function syncMobileLayout(): void {
  const isMobile = mobileLayoutQuery
    ? mobileLayoutQuery.matches
    : window.matchMedia(`(max-width: ${MOBILE_BREAKPOINT_PX}px)`).matches;

  if (!elements.panel) return;

  if (isMobile) {
    if (elements.mobileMenuBody) {
      menuSections.forEach((section) => elements.mobileMenuBody?.appendChild(section));
    }
    if (elements.mobileNumpadHost && numbersSection) {
      elements.mobileNumpadHost.appendChild(numbersSection);
    }
    return;
  }

  closeMobileMenu();
  panelSections.forEach((section) => elements.panel?.appendChild(section));
}

function openMobileMenu(): void {
  if (!elements.mobileMenuOverlay) return;
  elements.mobileMenuOverlay.hidden = false;
  document.body.classList.add("mobile-menu-open");
}

function closeMobileMenu(): void {
  if (!elements.mobileMenuOverlay) return;
  elements.mobileMenuOverlay.hidden = true;
  document.body.classList.remove("mobile-menu-open");
}

function toggleMobileMenu(): void {
  if (!elements.mobileMenuOverlay) return;
  if (elements.mobileMenuOverlay.hidden) {
    openMobileMenu();
  } else {
    closeMobileMenu();
  }
}

/** Bind all event handlers */
function bindEvents(): void {
  // Board click
  handleBoardClick((row, col) => {
    gameState.selectCell(row, col);
  });

  // Difficulty selection
  handleDifficultyClick((level) => {
    startNewGame(level);
  });

  // Numpad click
  handleNumpadClick((value) => {
    const { row, col } = gameState.selected;
    if (gameState.noteMode) {
      gameState.toggleNote(row, col, value);
    } else {
      gameState.setCellValue(row, col, value);
    }
    checkSolved();
  });

  // New game button
  elements.newGameBtn.addEventListener("click", () => {
    startNewGame(gameState.difficulty === "manual" ? "easy" : gameState.difficulty);
  });

  // Manual entry button
  elements.manualEntryBtn?.addEventListener("click", () => {
    gameState.startManualEntry();
    showToast(t("toast_manual_ready"));
  });

  // Solve manual button
  elements.solveManualBtn?.addEventListener("click", solveManualGrid);

  // Manual import button
  elements.manualImportBtn?.addEventListener("click", handleManualImport);

  // Manual export button
  elements.manualExportBtn?.addEventListener("click", handleManualExport);

  // Clear cell button
  elements.clearBtn.addEventListener("click", () => {
    const { row, col } = gameState.selected;
    if (gameState.noteMode) {
      gameState.clearNotes(row, col);
    } else {
      gameState.clearCell(row, col);
    }
  });

  // Undo button
  elements.undoBtn?.addEventListener("click", () => {
    gameState.undo();
  });

  // Toggle notes button
  elements.toggleNotesBtn?.addEventListener("click", () => {
    gameState.toggleNoteMode();
  });

  // Show solution button
  elements.showSolutionBtn.addEventListener("click", () => {
    gameState.toggleSolution();
  });

  // Camera button
  elements.openCameraBtn?.addEventListener("click", openCamera);

  // Upload photo button
  elements.uploadPhotoBtn?.addEventListener("click", () => {
    elements.photoInput?.click();
  });

  // Photo input change
  elements.photoInput?.addEventListener("change", handlePhotoInput);

  // Camera events
  if (
    elements.camera &&
    elements.cameraVideo &&
    elements.cameraCaptureBtn &&
    elements.cameraCloseBtn
  ) {
    bindCameraEvents(sendImageToServer);
  }

  elements.mobileMenuBtn?.addEventListener("click", toggleMobileMenu);
  elements.mobileMenuCloseBtn?.addEventListener("click", closeMobileMenu);
  elements.mobileMenuOverlay?.addEventListener("click", (event) => {
    if (event.target === elements.mobileMenuOverlay) {
      closeMobileMenu();
    }
  });

  elements.mobileUndoBtn?.addEventListener("click", () => {
    gameState.undo();
  });

  elements.mobileNotesBtn?.addEventListener("click", () => {
    gameState.toggleNoteMode();
  });

  elements.mobileSettingsBtn?.addEventListener("click", toggleMobileMenu);

  elements.mobileQuickActions?.addEventListener("click", (event) => {
    const target = (event.target as HTMLElement).closest<HTMLButtonElement>("[data-action]");
    if (!target) return;
    handleQuickAction(target.dataset.action);
  });

  // Keyboard events
  document.addEventListener("keydown", handleKeydown);

  // Save game before unload
  window.addEventListener("beforeunload", () => {
    saveGame(gameState.getSaveState());
  });
}

function handleQuickAction(action?: string): void {
  if (!action) return;

  switch (action) {
    case "undo":
      gameState.undo();
      return;
    case "clear": {
      const { row, col } = gameState.selected;
      if (gameState.noteMode) {
        gameState.clearNotes(row, col);
      } else {
        gameState.clearCell(row, col);
      }
      return;
    }
    case "notes":
      gameState.toggleNoteMode();
      return;
    case "hint":
      applyHint();
      return;
    case "solution":
      gameState.toggleSolution();
      return;
    default:
      return;
  }
}

function applyHint(): void {
  if (gameState.showingSolution) {
    showToast(t("toast_hint_unavailable"));
    return;
  }

  const solution = gameState.solution;
  if (!solution) {
    showToast(t("toast_hint_unavailable"));
    return;
  }

  const puzzle = gameState.puzzle;
  const current = gameState.current;
  let { row, col } = gameState.selected;

  if (puzzle[row][col] !== 0 || current[row][col] !== 0) {
    let found = false;
    for (let r = 0; r < GRID_SIZE; r++) {
      for (let c = 0; c < GRID_SIZE; c++) {
        if (puzzle[r][c] === 0 && current[r][c] === 0) {
          row = r;
          col = c;
          found = true;
          break;
        }
      }
      if (found) break;
    }

    if (!found) {
      showToast(t("toast_hint_unavailable"));
      return;
    }
  }

  const value = solution[row][col];
  if (!value) {
    showToast(t("toast_hint_unavailable"));
    return;
  }

  if (gameState.setCellValue(row, col, value)) {
    checkSolved();
    showToast(t("toast_hint_used"));
  } else {
    showToast(t("toast_hint_unavailable"));
  }
}

/** Handle keyboard input */
function handleKeydown(event: KeyboardEvent): void {
  if (event.defaultPrevented) return;

  const key = event.key;

  // Undo with Ctrl/Cmd + Z
  if ((event.ctrlKey || event.metaKey) && key.toLowerCase() === "z") {
    gameState.undo();
    event.preventDefault();
    return;
  }

  // Number input
  if (key >= "1" && key <= "9") {
    const { row, col } = gameState.selected;
    if (gameState.noteMode) {
      gameState.toggleNote(row, col, Number(key));
    } else {
      gameState.setCellValue(row, col, Number(key));
      checkSolved();
    }
    event.preventDefault();
    return;
  }

  // Clear cell
  if (key === "Backspace" || key === "Delete" || key === "0") {
    const { row, col } = gameState.selected;
    if (gameState.noteMode) {
      gameState.clearNotes(row, col);
    } else {
      gameState.clearCell(row, col);
    }
    event.preventDefault();
    return;
  }

  // Arrow key navigation
  const moves: Record<string, [number, number]> = {
    ArrowUp: [-1, 0],
    ArrowDown: [1, 0],
    ArrowLeft: [0, -1],
    ArrowRight: [0, 1],
  };

  if (moves[key]) {
    const [dr, dc] = moves[key];
    const { row, col } = gameState.selected;
    const nextRow = (row + dr + GRID_SIZE) % GRID_SIZE;
    const nextCol = (col + dc + GRID_SIZE) % GRID_SIZE;
    gameState.selectCell(nextRow, nextCol);
    event.preventDefault();
  }
}

/** Start a new game with the given difficulty */
function startNewGame(difficulty: Difficulty): void {
  setBoardLoading(true);
  pendingDifficulty = difficulty;
  const requestId = ++generationRequestId;
  const worker = getPuzzleWorker();

  if (worker) {
    const message: PuzzleWorkerMessage = { id: requestId, difficulty };
    worker.postMessage(message);
    return;
  }

  fallbackGenerate(difficulty, requestId);
}

/** Restore game from localStorage */
function restoreGame(): void {
  const saved = loadGame();
  if (saved) {
    gameState.restore(
      saved.puzzle,
      saved.solution,
      saved.current,
      saved.notes,
      saved.difficulty,
      saved.elapsedMs
    );
  } else {
    startNewGame("easy");
  }
}

/** Update all UI components */
function updateUI(): void {
  updateBoard();
  updateDifficultyButtons();
  updateNumpadAvailability();
  updateTimer();
  updateNoteModeUI();
  updateSolutionButtonText();
  updateUndoButtonState();
}

/** Update note mode button UI */
function updateNoteModeUI(): void {
  const isActive = gameState.noteMode;

  elements.toggleNotesBtn?.classList.toggle("active", isActive);
  elements.toggleNotesBtn?.setAttribute("aria-pressed", isActive ? "true" : "false");
  elements.mobileNotesBtn?.classList.toggle("active", isActive);
  elements.mobileNotesBtn?.setAttribute("aria-pressed", isActive ? "true" : "false");
  elements.mobileQuickNotesBtn?.classList.toggle("active", isActive);
  elements.mobileQuickNotesBtn?.setAttribute("aria-pressed", isActive ? "true" : "false");
}

/** Update solution button text */
function updateSolutionButtonText(): void {
  const label = gameState.showingSolution ? t("hide_solution") : t("show_solution");
  elements.showSolutionBtn.textContent = label;
  elements.mobileQuickSolutionBtn?.classList.toggle("active", gameState.showingSolution);
  elements.mobileQuickSolutionBtn?.setAttribute("aria-pressed", gameState.showingSolution ? "true" : "false");
  elements.mobileQuickSolutionBtn?.setAttribute("aria-label", label);
}

/** Update undo button state */
function updateUndoButtonState(): void {
  const disabled = !gameState.canUndo;
  if (elements.undoBtn) {
    elements.undoBtn.disabled = disabled;
  }
  if (elements.mobileUndoBtn) {
    elements.mobileUndoBtn.disabled = disabled;
  }
  if (elements.mobileQuickUndoBtn) {
    elements.mobileQuickUndoBtn.disabled = disabled;
  }
}

/** Check if the puzzle is solved */
function checkSolved(): void {
  if (gameState.isSolved()) {
    showToast(t("toast_solved"));
  }
}

/** Handle photo input */
function handlePhotoInput(): void {
  const file = elements.photoInput?.files?.[0];
  if (!file) return;
  elements.photoInput!.value = "";
  sendImageToServer(file);
}

/** Send image to server for scanning */
async function sendImageToServer(file: File): Promise<void> {
  cancelPendingGeneration();
  setBoardLoading(true);
  showToast(t("toast_scan_start"));

  try {
    const data = await scanImage(file);
    const puzzleGrid = parsePuzzleGrid(data.puzzle);
    const solutionGrid = parseSolvedGrid(data.solution);
    gameState.applyScannedGrid(puzzleGrid, solutionGrid);
    saveGame(gameState.getSaveState());
    showToast(t("toast_scan_done"));
  } catch (error) {
    const message =
      error instanceof ApiError && error.message ? error.message : t("toast_scan_error");
    showToast(message);
    console.error(error);
  } finally {
    setBoardLoading(false);
  }
}

/** Handle manual import */
function handleManualImport(): void {
  const text = elements.manualInput?.value ?? "";
  try {
    const grid = parseManualText(text);
    gameState.startManualEntry(grid);
    saveGame(gameState.getSaveState());
    showToast(t("toast_manual_imported"));
  } catch {
    showToast(t("toast_manual_invalid"));
  }
}

/** Handle manual export */
async function handleManualExport(): Promise<void> {
  const text = formatGrid(gameState.current);
  if (elements.manualInput) {
    elements.manualInput.value = text;
  }

  try {
    await navigator.clipboard.writeText(text);
    showToast(t("toast_manual_copied"));
  } catch {
    if (elements.manualInput) {
      elements.manualInput.focus();
      elements.manualInput.select();
    }
    showToast(t("toast_manual_exported"));
  }
}

/** Solve the manually entered grid */
function solveManualGrid(): void {
  if (!hasAnyDigits(gameState.current)) {
    showToast(t("toast_manual_empty"));
    return;
  }

  if (findConflicts(gameState.current).size > 0) {
    showToast(t("toast_solve_error"));
    return;
  }

  const attempt = copyGrid(gameState.current);
  if (!solveGrid(attempt)) {
    showToast(t("toast_solve_error"));
    return;
  }

  const sanitized = sanitizeGrid(attempt);
  gameState.applyManualSolution(sanitized);
  saveGame(gameState.getSaveState());
  showToast(t("toast_manual_solved"));
}

/** Register the service worker */
function registerServiceWorker(): void {
  if ("serviceWorker" in navigator) {
    navigator.serviceWorker.register("/sw.js").catch(() => {
      // Service worker registration failed
    });
  }
}

// Initialize the application when the DOM is ready
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
