const DIFFICULTY_BLANKS = {
  easy: 36,
  medium: 44,
  hard: 52,
  expert: 58,
};

const STORAGE_KEY = "sudoku:game";
const translations = {
  ru: {
    title: "Судоку офлайн",
    eyebrow: "Судоку",
    hero_title: "Играйте без интернета на любом устройстве",
    hero_sub: "Автосохранение прогресса, адаптивная сетка и уровни сложности.",
    board_label: "Поле судоку",
    difficulty_heading: "Сложность",
    actions_heading: "Действия",
    numbers_heading: "Числа",
    scan_heading: "Сканирование",
    manual_heading: "Ручной ввод",
    manual_placeholder: "Вставьте 9 строк по 9 символов (1–9, 0 или . для пустых).",
    hint: "Можно вводить цифры 1–9 с клавиатуры. Стрелки перемещают выделение.",
    new_game: "Новая игра",
    manual_entry: "Ручной ввод",
    solve_manual: "Решить ввод",
    manual_import: "Импорт",
    manual_export: "Экспорт",
    clear_cell: "Очистить",
    undo: "Отменить",
    notes: "Заметки",
    show_solution: "Показать решение",
    hide_solution: "Скрыть решение",
    open_camera: "Камера",
    upload_photo: "Загрузить фото",
    camera_close: "Закрыть",
    camera_capture: "Снять",
    difficulty_label: "Сложность",
    difficulty_easy: "Легко",
    difficulty_medium: "Нормально",
    difficulty_hard: "Сложно",
    difficulty_expert: "Эксперт",
    difficulty_manual: "Ручной ввод",
    online: "Онлайн",
    offline: "Офлайн",
    toast_new_game: "Новая игра готова",
    toast_solved: "Отлично! Судоку решено.",
    toast_manual_ready: "Можно вводить судоку вручную.",
    toast_manual_imported: "Поле загружено.",
    toast_manual_exported: "Строка готова.",
    toast_manual_copied: "Скопировано в буфер.",
    toast_manual_invalid: "Неверный формат. Нужно 81 символ (1–9, 0 или .).",
    toast_manual_empty: "Введите хотя бы одну цифру.",
    toast_manual_solved: "Решение готово.",
    toast_solve_error: "Не удалось решить судоку.",
    toast_scan_start: "Обрабатываю фото…",
    toast_scan_done: "Готово! Поле заполнено.",
    toast_scan_error: "Не удалось распознать фото.",
    toast_camera_denied: "Нет доступа к камере.",
    toast_camera_unsupported: "Камера не поддерживается.",
  },
  en: {
    title: "Offline Sudoku",
    eyebrow: "Sudoku",
    hero_title: "Play offline on any device",
    hero_sub: "Autosave progress, responsive grid, and difficulty levels.",
    board_label: "Sudoku board",
    difficulty_heading: "Difficulty",
    actions_heading: "Actions",
    numbers_heading: "Numbers",
    scan_heading: "Scan",
    manual_heading: "Manual entry",
    manual_placeholder: "Paste 9 lines of 9 symbols (1–9, 0 or . for empty).",
    hint: "Use keys 1–9 to enter numbers. Arrow keys move selection.",
    new_game: "New game",
    manual_entry: "Manual entry",
    solve_manual: "Solve input",
    manual_import: "Import",
    manual_export: "Export",
    clear_cell: "Clear",
    undo: "Undo",
    notes: "Notes",
    show_solution: "Show solution",
    hide_solution: "Hide solution",
    open_camera: "Camera",
    upload_photo: "Upload photo",
    camera_close: "Close",
    camera_capture: "Capture",
    difficulty_label: "Difficulty",
    difficulty_easy: "Easy",
    difficulty_medium: "Medium",
    difficulty_hard: "Hard",
    difficulty_expert: "Expert",
    difficulty_manual: "Manual entry",
    online: "Online",
    offline: "Offline",
    toast_new_game: "New game ready",
    toast_solved: "Great! Sudoku solved.",
    toast_manual_ready: "Manual entry ready.",
    toast_manual_imported: "Grid loaded.",
    toast_manual_exported: "String ready.",
    toast_manual_copied: "Copied to clipboard.",
    toast_manual_invalid: "Invalid format. Need 81 symbols (1–9, 0 or .).",
    toast_manual_empty: "Enter at least one digit.",
    toast_manual_solved: "Solution ready.",
    toast_solve_error: "Could not solve the puzzle.",
    toast_scan_start: "Processing photo…",
    toast_scan_done: "Done! Grid filled.",
    toast_scan_error: "Could not read the photo.",
    toast_camera_denied: "Camera access denied.",
    toast_camera_unsupported: "Camera not supported.",
  },
};

const boardEl = document.getElementById("board");
const difficultyEl = document.getElementById("difficulty");
const difficultyLabel = document.getElementById("difficulty-label");
const timerEl = document.getElementById("timer");
const onlineStatusEl = document.getElementById("online-status");
const newGameBtn = document.getElementById("new-game");
const manualEntryBtn = document.getElementById("manual-entry");
const solveManualBtn = document.getElementById("solve-manual");
const manualInput = document.getElementById("manual-input");
const manualImportBtn = document.getElementById("manual-import");
const manualExportBtn = document.getElementById("manual-export");
const clearBtn = document.getElementById("clear-cell");
const undoBtn = document.getElementById("undo-move");
const toggleNotesBtn = document.getElementById("toggle-notes");
const showSolutionBtn = document.getElementById("show-solution");
const numpadEl = document.getElementById("numpad");
const toastEl = document.getElementById("toast");
const openCameraBtn = document.getElementById("open-camera");
const uploadPhotoBtn = document.getElementById("upload-photo");
const photoInput = document.getElementById("photo-input");
const cameraEl = document.getElementById("camera");
const cameraVideo = document.getElementById("camera-video");
const cameraCaptureBtn = document.getElementById("camera-capture");
const cameraCloseBtn = document.getElementById("camera-close");
const cameraCanvas = document.getElementById("camera-canvas");

let puzzle = [];
let solution = [];
let current = [];
let difficulty = "easy";
let selected = { row: 0, col: 0 };
let cells = [];
let startTime = Date.now();
let elapsedMs = 0;
let timerId = null;
let showingSolution = false;
let currentLocale = "ru";
let cameraStream = null;
let noteMode = false;
let notes = createEmptyNotes();
let history = [];

function init() {
  setLocale(detectLocale());
  buildBoard();
  buildNumpad();
  bindEvents();
  restoreGame();
  startTimer();
  registerServiceWorker();
}

function buildBoard() {
  boardEl.innerHTML = "";
  cells = [];
  for (let row = 0; row < 9; row += 1) {
    for (let col = 0; col < 9; col += 1) {
      const cell = document.createElement("button");
      cell.className = "cell";
      cell.type = "button";
      cell.dataset.row = String(row);
      cell.dataset.col = String(col);
      cell.setAttribute("role", "gridcell");
      cell.setAttribute("aria-selected", "false");
      if (col === 2 || col === 5) cell.classList.add("br");
      if (row === 2 || row === 5) cell.classList.add("bb");
      boardEl.appendChild(cell);
      cells.push(cell);
    }
  }
}

function buildNumpad() {
  numpadEl.innerHTML = "";
  for (let i = 1; i <= 9; i += 1) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = String(i);
    btn.dataset.value = String(i);
    numpadEl.appendChild(btn);
  }
}

function bindEvents() {
  boardEl.addEventListener("click", (event) => {
    const cell = event.target.closest(".cell");
    if (!cell) return;
    selectCell(Number(cell.dataset.row), Number(cell.dataset.col));
  });

  difficultyEl.addEventListener("click", (event) => {
    const button = event.target.closest("[data-level]");
    if (!button) return;
    difficulty = button.dataset.level;
    startNewGame();
  });

  newGameBtn.addEventListener("click", () => startNewGame());
  manualEntryBtn?.addEventListener("click", startManualEntry);
  solveManualBtn?.addEventListener("click", solveManualGrid);
  manualImportBtn?.addEventListener("click", handleManualImport);
  manualExportBtn?.addEventListener("click", handleManualExport);
  clearBtn.addEventListener("click", () => clearCell(selected.row, selected.col));
  undoBtn?.addEventListener("click", undoLastAction);
  toggleNotesBtn?.addEventListener("click", toggleNotesMode);
  showSolutionBtn.addEventListener("click", toggleSolution);
  if (openCameraBtn) openCameraBtn.addEventListener("click", openCamera);
  if (uploadPhotoBtn) uploadPhotoBtn.addEventListener("click", () => photoInput?.click());
  if (photoInput) photoInput.addEventListener("change", handlePhotoInput);
  if (cameraCloseBtn) cameraCloseBtn.addEventListener("click", closeCamera);
  if (cameraCaptureBtn) cameraCaptureBtn.addEventListener("click", captureFromCamera);

  numpadEl.addEventListener("click", (event) => {
    const button = event.target.closest("button");
    if (!button) return;
    if (button.disabled) return;
    const value = Number(button.dataset.value);
    if (noteMode) {
      toggleNoteValue(selected.row, selected.col, value);
    } else {
      setCellValue(selected.row, selected.col, value);
    }
  });

  document.addEventListener("keydown", handleKeydown);
  window.addEventListener("beforeunload", saveGame);
  window.addEventListener("online", updateOnlineStatus);
  window.addEventListener("offline", updateOnlineStatus);
}

function handleKeydown(event) {
  if (event.defaultPrevented) return;
  const key = event.key;
  if ((event.ctrlKey || event.metaKey) && key.toLowerCase() === "z") {
    undoLastAction();
    event.preventDefault();
    return;
  }
  if (key >= "1" && key <= "9") {
    if (noteMode) {
      toggleNoteValue(selected.row, selected.col, Number(key));
    } else {
      setCellValue(selected.row, selected.col, Number(key));
    }
    event.preventDefault();
    return;
  }

  if (key === "Backspace" || key === "Delete" || key === "0") {
    if (noteMode) {
      clearNotes(selected.row, selected.col);
    } else {
      clearCell(selected.row, selected.col);
    }
    event.preventDefault();
    return;
  }

  const moves = {
    ArrowUp: [-1, 0],
    ArrowDown: [1, 0],
    ArrowLeft: [0, -1],
    ArrowRight: [0, 1],
  };
  if (moves[key]) {
    const [dr, dc] = moves[key];
    const nextRow = (selected.row + dr + 9) % 9;
    const nextCol = (selected.col + dc + 9) % 9;
    selectCell(nextRow, nextCol);
    event.preventDefault();
  }
}

function startNewGame() {
  showingSolution = false;
  showSolutionBtn.textContent = t("show_solution");
  boardEl.classList.add("loading");
  setTimeout(() => {
    const generated = generatePuzzle(difficulty);
    puzzle = generated.puzzle;
    solution = generated.solution;
    current = copyGrid(puzzle);
    notes = createEmptyNotes();
    history = [];
    noteMode = false;
    startTime = Date.now();
    elapsedMs = 0;
    selectFirstEmpty();
    updateUI();
    saveGame();
    boardEl.classList.remove("loading");
    showToast(t("toast_new_game"));
  }, 30);
}

function startManualEntry(grid) {
  showingSolution = false;
  showSolutionBtn.textContent = t("show_solution");
  puzzle = createEmptyGrid();
  solution = [];
  const hasGrid = Array.isArray(grid);
  current = hasGrid ? copyGrid(grid) : createEmptyGrid();
  notes = createEmptyNotes();
  history = [];
  noteMode = false;
  difficulty = "manual";
  startTime = Date.now();
  elapsedMs = 0;
  selectFirstEmpty();
  updateUI();
  saveGame();
  if (!hasGrid) {
    showToast(t("toast_manual_ready"));
  }
}

function restoreGame() {
  const savedRaw = localStorage.getItem(STORAGE_KEY);
  if (!savedRaw) {
    startNewGame();
    return;
  }
  try {
    const saved = JSON.parse(savedRaw);
    if (!saved || !Array.isArray(saved.puzzle) || saved.puzzle.length !== 9) {
      startNewGame();
      return;
    }
    puzzle = saved.puzzle;
    solution = saved.solution;
    current = saved.current;
    notes = restoreNotes(saved.notes);
    difficulty = saved.difficulty || "easy";
    startTime = Date.now() - (saved.elapsedMs || 0);
    elapsedMs = saved.elapsedMs || 0;
    history = [];
    selectFirstEmpty();
    updateUI();
  } catch (error) {
    startNewGame();
  }
}

function saveGame() {
  if (!puzzle.length) return;
  const payload = {
    puzzle,
    solution,
    current,
    notes,
    difficulty,
    elapsedMs: Date.now() - startTime,
  };
  localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
}

function selectCell(row, col) {
  selected = { row, col };
  updateUI();
}

function selectFirstEmpty() {
  for (let row = 0; row < 9; row += 1) {
    for (let col = 0; col < 9; col += 1) {
      if (puzzle[row][col] === 0) {
        selected = { row, col };
        return;
      }
    }
  }
  selected = { row: 0, col: 0 };
}

function setCellValue(row, col, value) {
  if (showingSolution) return;
  if (puzzle[row][col] !== 0) return;
  if (current[row][col] === value) return;
  pushHistory();
  current[row][col] = value;
  notes[row][col] = 0;
  updateUI();
  saveGame();
  checkSolved();
}

function clearCell(row, col) {
  if (showingSolution) return;
  if (puzzle[row][col] !== 0) return;
  if (current[row][col] === 0 && notes[row][col] === 0) return;
  pushHistory();
  current[row][col] = 0;
  notes[row][col] = 0;
  updateUI();
  saveGame();
}

function toggleSolution() {
  if (!solution.length) return;
  showingSolution = !showingSolution;
  showSolutionBtn.textContent = showingSolution ? t("hide_solution") : t("show_solution");
  updateUI();
}

function toggleNotesMode() {
  noteMode = !noteMode;
  updateNoteModeUI();
}

function updateNoteModeUI() {
  if (!toggleNotesBtn) return;
  toggleNotesBtn.classList.toggle("active", noteMode);
  toggleNotesBtn.setAttribute("aria-pressed", noteMode ? "true" : "false");
  if (undoBtn) {
    undoBtn.disabled = history.length === 0;
  }
}

function toggleNoteValue(row, col, value) {
  if (showingSolution) return;
  if (puzzle[row][col] !== 0) return;
  if (current[row][col] !== 0) return;
  pushHistory();
  const bit = 1 << (value - 1);
  notes[row][col] ^= bit;
  updateUI();
  saveGame();
}

function clearNotes(row, col) {
  if (showingSolution) return;
  if (puzzle[row][col] !== 0) return;
  if (notes[row][col] === 0) return;
  pushHistory();
  notes[row][col] = 0;
  updateUI();
  saveGame();
}

function pushHistory() {
  history.push({ current: copyGrid(current), notes: copyNotes(notes) });
  if (history.length > 200) history.shift();
}

function undoLastAction() {
  const entry = history.pop();
  if (!entry) return;
  current = copyGrid(entry.current);
  notes = copyNotes(entry.notes);
  updateUI();
  saveGame();
}

function updateNumpadAvailability() {
  const buttons = numpadEl.querySelectorAll("button");
  const value = current[selected.row]?.[selected.col] ?? 0;
  if (puzzle[selected.row]?.[selected.col] !== 0 || showingSolution) {
    buttons.forEach((btn) => {
      btn.disabled = true;
      btn.classList.remove("is-hidden");
    });
    return;
  }
  const boxNumbers = new Set();
  const boxRow = Math.floor(selected.row / 3) * 3;
  const boxCol = Math.floor(selected.col / 3) * 3;
  for (let r = boxRow; r < boxRow + 3; r += 1) {
    for (let c = boxCol; c < boxCol + 3; c += 1) {
      const cellValue = current[r][c];
      if (cellValue > 0) boxNumbers.add(cellValue);
    }
  }
  if (value > 0) boxNumbers.delete(value);
  buttons.forEach((btn) => {
    const num = Number(btn.dataset.value);
    const hidden = boxNumbers.has(num);
    btn.disabled = hidden;
    btn.classList.toggle("is-hidden", hidden);
  });
}

function renderNotes(mask) {
  const items = [];
  for (let i = 1; i <= 9; i += 1) {
    const bit = 1 << (i - 1);
    items.push(`<span>${mask & bit ? i : ""}</span>`);
  }
  return `<div class="notes-grid">${items.join("")}</div>`;
}

function handlePhotoInput() {
  const file = photoInput?.files?.[0];
  if (!file) return;
  photoInput.value = "";
  sendImageToServer(file);
}

async function openCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    showToast(t("toast_camera_unsupported"));
    return;
  }
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" } },
      audio: false,
    });
    cameraVideo.srcObject = cameraStream;
    cameraEl.hidden = false;
  } catch (error) {
    showToast(t("toast_camera_denied"));
  }
}

function closeCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
  }
  cameraStream = null;
  if (cameraVideo) cameraVideo.srcObject = null;
  if (cameraEl) cameraEl.hidden = true;
}

function captureFromCamera() {
  if (!cameraStream || !cameraVideo) return;
  if (!cameraVideo.videoWidth) {
    showToast(t("toast_camera_denied"));
    return;
  }
  const size = Math.min(cameraVideo.videoWidth, cameraVideo.videoHeight);
  const sx = (cameraVideo.videoWidth - size) / 2;
  const sy = (cameraVideo.videoHeight - size) / 2;
  cameraCanvas.width = size;
  cameraCanvas.height = size;
  const ctx = cameraCanvas.getContext("2d");
  if (!ctx) return;
  ctx.drawImage(cameraVideo, sx, sy, size, size, 0, 0, size, size);
  cameraCanvas.toBlob(
    (blob) => {
      if (!blob) return;
      const file = new File([blob], "capture.jpg", { type: "image/jpeg" });
      sendImageToServer(file);
    },
    "image/jpeg",
    0.9
  );
  closeCamera();
}

async function sendImageToServer(file) {
  boardEl.classList.add("loading");
  showToast(t("toast_scan_start"));
  try {
    const formData = new FormData();
    formData.append("image", file);
    const response = await fetch("/api/scan", { method: "POST", body: formData });
    if (!response.ok) {
      const errorText = await response.text().catch(() => "");
      throw new Error(errorText || "Scan failed");
    }
    const data = await response.json();
    if (!data?.puzzle || !data?.solution) {
      throw new Error("Invalid response");
    }
    applyScannedGrid(data.puzzle, data.solution);
    showToast(t("toast_scan_done"));
  } catch (error) {
    showToast(t("toast_scan_error"));
    console.error(error);
  } finally {
    boardEl.classList.remove("loading");
  }
}

function handleManualImport() {
  const text = manualInput?.value ?? "";
  try {
    const grid = parseManualText(text);
    startManualEntry(grid);
    showToast(t("toast_manual_imported"));
  } catch (error) {
    showToast(t("toast_manual_invalid"));
  }
}

async function handleManualExport() {
  const text = formatGrid(current);
  if (manualInput) manualInput.value = text;
  try {
    await navigator.clipboard.writeText(text);
    showToast(t("toast_manual_copied"));
  } catch {
    if (manualInput) {
      manualInput.focus();
      manualInput.select();
    }
    showToast(t("toast_manual_exported"));
  }
}

function solveManualGrid() {
  if (!hasAnyDigits(current)) {
    showToast(t("toast_manual_empty"));
    return;
  }
  if (findConflicts(current).size > 0) {
    showToast(t("toast_solve_error"));
    return;
  }
  const attempt = copyGrid(current);
  if (!solveGrid(attempt)) {
    showToast(t("toast_solve_error"));
    return;
  }
  applyManualSolution(attempt);
  showToast(t("toast_manual_solved"));
}

function applyScannedGrid(puzzleGrid, solutionGrid) {
  const parsedPuzzle = parsePuzzleGrid(puzzleGrid);
  const parsedSolution = parseSolvedGrid(solutionGrid);
  puzzle = createEmptyGrid();
  solution = copyGrid(parsedSolution);
  current = copyGrid(parsedPuzzle);
  notes = createEmptyNotes();
  history = [];
  noteMode = false;
  showingSolution = false;
  showSolutionBtn.textContent = t("show_solution");
  difficulty = "manual";
  startTime = Date.now();
  elapsedMs = 0;
  selectFirstEmpty();
  updateUI();
  saveGame();
}

function applyManualSolution(solutionGrid) {
  const solved = parseSolvedGrid(solutionGrid);
  const manualPuzzle = sanitizeGrid(current);
  puzzle = copyGrid(manualPuzzle);
  solution = copyGrid(solved);
  current = copyGrid(manualPuzzle);
  notes = createEmptyNotes();
  history = [];
  noteMode = false;
  showingSolution = true;
  showSolutionBtn.textContent = t("hide_solution");
  startTime = Date.now();
  elapsedMs = 0;
  selectFirstEmpty();
  updateUI();
  saveGame();
}

function sanitizeGrid(grid) {
  return grid.map((row) =>
    row.map((value) => {
      const num = Number(value);
      if (Number.isInteger(num) && num >= 0 && num <= 9) return num;
      return 0;
    })
  );
}

function parseManualText(text) {
  const tokens = String(text).match(/[0-9.]/g) || [];
  if (tokens.length !== 81) {
    throw new Error("Invalid manual input");
  }
  const grid = createEmptyGrid();
  tokens.forEach((char, index) => {
    const value = char === "." ? 0 : Number(char);
    if (!Number.isInteger(value) || value < 0 || value > 9) {
      throw new Error("Invalid manual input");
    }
    const row = Math.floor(index / 9);
    const col = index % 9;
    grid[row][col] = value;
  });
  return grid;
}

function formatGrid(grid) {
  return grid
    .map((row) =>
      row
        .map((value) => {
          if (Number.isInteger(value) && value > 0) return String(value);
          return ".";
        })
        .join("")
    )
    .join("\n");
}

function parseSolvedGrid(grid) {
  if (!Array.isArray(grid) || grid.length !== 9) {
    throw new Error("Invalid grid");
  }
  const parsed = grid.map((row) => {
    if (!Array.isArray(row) || row.length !== 9) {
      throw new Error("Invalid grid");
    }
    return row.map((value) => {
      const num = Number(value);
      if (!Number.isInteger(num) || num < 1 || num > 9) {
        throw new Error("Invalid value");
      }
      return num;
    });
  });
  return parsed;
}

function parsePuzzleGrid(grid) {
  if (!Array.isArray(grid) || grid.length !== 9) {
    throw new Error("Invalid grid");
  }
  const parsed = grid.map((row) => {
    if (!Array.isArray(row) || row.length !== 9) {
      throw new Error("Invalid grid");
    }
    return row.map((value) => {
      const num = Number(value);
      if (!Number.isInteger(num) || num < 0 || num > 9) {
        throw new Error("Invalid value");
      }
      return num;
    });
  });
  return parsed;
}

function updateUI() {
  updateDifficultyButtons();
  updateBoard();
  updateNumpadAvailability();
  updateNoteModeUI();
  updateTimer();
}

function updateDifficultyButtons() {
  const buttons = difficultyEl.querySelectorAll("[data-level]");
  buttons.forEach((button) => {
    const level = button.dataset.level;
    button.classList.toggle("active", level === difficulty);
    button.textContent = getDifficultyLabel(level);
  });
  const label = getDifficultyLabel(difficulty);
  difficultyLabel.textContent = `${t("difficulty_label")}: ${label}`;
}

function updateBoard() {
  const conflicts = findConflicts(showingSolution ? solution : current);
  const selectedValue = showingSolution ? solution[selected.row][selected.col] : current[selected.row][selected.col];
  cells.forEach((cell) => {
    const row = Number(cell.dataset.row);
    const col = Number(cell.dataset.col);
    const index = row * 9 + col;
    const value = showingSolution ? solution[row][col] : current[row][col];
    if (value === 0) {
      const noteMask = notes[row][col];
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
    cell.classList.toggle("fixed", puzzle[row][col] !== 0);
    cell.classList.toggle("selected", row === selected.row && col === selected.col);
    cell.classList.toggle("related", isRelatedCell(row, col, selected.row, selected.col));
    cell.classList.toggle("conflict", conflicts.has(index));
    cell.classList.toggle("same", value !== 0 && value === selectedValue && !cell.classList.contains("selected"));
    cell.setAttribute("aria-selected", row === selected.row && col === selected.col ? "true" : "false");
  });
}

function isRelatedCell(row, col, focusRow, focusCol) {
  const sameBox = Math.floor(row / 3) === Math.floor(focusRow / 3) && Math.floor(col / 3) === Math.floor(focusCol / 3);
  return row === focusRow || col === focusCol || sameBox;
}

function updateTimer() {
  const now = Date.now();
  const totalMs = showingSolution ? elapsedMs : now - startTime;
  const totalSec = Math.floor(totalMs / 1000);
  const minutes = String(Math.floor(totalSec / 60)).padStart(2, "0");
  const seconds = String(totalSec % 60).padStart(2, "0");
  timerEl.textContent = `${minutes}:${seconds}`;
}

function startTimer() {
  if (timerId) clearInterval(timerId);
  timerId = setInterval(updateTimer, 1000);
}

function updateOnlineStatus() {
  const online = navigator.onLine;
  onlineStatusEl.textContent = online ? t("online") : t("offline");
  onlineStatusEl.style.background = online ? "#fff" : "#000";
  onlineStatusEl.style.color = online ? "#000" : "#fff";
}

function checkSolved() {
  if (showingSolution) return;
  if (!Array.isArray(solution) || solution.length !== 9) return;
  const conflicts = findConflicts(current);
  if (conflicts.size > 0) return;
  for (let row = 0; row < 9; row += 1) {
    const solutionRow = solution[row];
    if (!Array.isArray(solutionRow) || solutionRow.length !== 9) return;
    for (let col = 0; col < 9; col += 1) {
      if (current[row][col] !== solutionRow[col]) return;
    }
  }
  showToast(t("toast_solved"));
}

function showToast(message) {
  toastEl.textContent = message;
  toastEl.classList.add("show");
  clearTimeout(showToast._timeoutId);
  showToast._timeoutId = setTimeout(() => {
    toastEl.classList.remove("show");
  }, 2000);
}

function copyGrid(grid) {
  return grid.map((row) => row.slice());
}

function createEmptyNotes() {
  return Array.from({ length: 9 }, () => Array(9).fill(0));
}

function copyNotes(noteGrid) {
  return noteGrid.map((row) => row.slice());
}

function restoreNotes(saved) {
  if (!Array.isArray(saved) || saved.length !== 9) {
    return createEmptyNotes();
  }
  return saved.map((row) => {
    if (!Array.isArray(row) || row.length !== 9) {
      return Array(9).fill(0);
    }
    return row.map((value) => Number(value) || 0);
  });
}

function generatePuzzle(level) {
  const blanks = DIFFICULTY_BLANKS[level] ?? DIFFICULTY_BLANKS.easy;
  const solutionGrid = generateSolvedGrid();
  let puzzleGrid = copyGrid(solutionGrid);
  const positions = shuffle([...Array(81).keys()]);
  let removed = 0;

  for (const pos of positions) {
    if (removed >= blanks) break;
    const row = Math.floor(pos / 9);
    const col = pos % 9;
    if (puzzleGrid[row][col] === 0) continue;
    const backup = puzzleGrid[row][col];
    puzzleGrid[row][col] = 0;
    if (countSolutions(puzzleGrid, 2) !== 1) {
      puzzleGrid[row][col] = backup;
    } else {
      removed += 1;
    }
  }

  let attempts = 0;
  while (removed < blanks && attempts < 1200) {
    const pos = Math.floor(Math.random() * 81);
    const row = Math.floor(pos / 9);
    const col = pos % 9;
    if (puzzleGrid[row][col] === 0) {
      attempts += 1;
      continue;
    }
    const backup = puzzleGrid[row][col];
    puzzleGrid[row][col] = 0;
    if (countSolutions(puzzleGrid, 2) !== 1) {
      puzzleGrid[row][col] = backup;
    } else {
      removed += 1;
    }
    attempts += 1;
  }

  return { puzzle: puzzleGrid, solution: solutionGrid };
}

function generateSolvedGrid() {
  const grid = createEmptyGrid();
  solveGrid(grid);
  return grid;
}

function hasAnyDigits(grid) {
  return grid.some((row) => row.some((value) => value !== 0));
}

function createEmptyGrid() {
  return Array.from({ length: 9 }, () => Array(9).fill(0));
}

function solveGrid(grid) {
  const next = findBestEmpty(grid);
  if (!next) return true;
  const { row, col, candidates } = next;
  const order = shuffle(candidates);
  for (const num of order) {
    if (isSafe(grid, row, col, num)) {
      grid[row][col] = num;
      if (solveGrid(grid)) return true;
      grid[row][col] = 0;
    }
  }
  return false;
}

function countSolutions(grid, limit) {
  let count = 0;

  function backtrack() {
    if (count >= limit) return;
    const next = findBestEmpty(grid);
    if (!next) {
      count += 1;
      return;
    }
    const { row, col, candidates } = next;
    for (const num of candidates) {
      if (isSafe(grid, row, col, num)) {
        grid[row][col] = num;
        backtrack();
        grid[row][col] = 0;
        if (count >= limit) return;
      }
    }
  }

  backtrack();
  return count;
}

function findBestEmpty(grid) {
  let best = null;
  let bestCount = 10;
  for (let row = 0; row < 9; row += 1) {
    for (let col = 0; col < 9; col += 1) {
      if (grid[row][col] !== 0) continue;
      const candidates = [];
      for (let num = 1; num <= 9; num += 1) {
        if (isSafe(grid, row, col, num)) candidates.push(num);
      }
      if (candidates.length === 0) return { row, col, candidates };
      if (candidates.length < bestCount) {
        best = { row, col, candidates };
        bestCount = candidates.length;
        if (bestCount === 1) return best;
      }
    }
  }
  return best;
}

function isSafe(grid, row, col, num) {
  for (let i = 0; i < 9; i += 1) {
    if (grid[row][i] === num) return false;
    if (grid[i][col] === num) return false;
  }
  const boxRow = Math.floor(row / 3) * 3;
  const boxCol = Math.floor(col / 3) * 3;
  for (let r = boxRow; r < boxRow + 3; r += 1) {
    for (let c = boxCol; c < boxCol + 3; c += 1) {
      if (grid[r][c] === num) return false;
    }
  }
  return true;
}

function findConflicts(grid) {
  const conflicts = new Set();

  for (let row = 0; row < 9; row += 1) {
    const buckets = new Map();
    for (let col = 0; col < 9; col += 1) {
      const value = grid[row][col];
      if (value === 0) continue;
      if (!buckets.has(value)) buckets.set(value, []);
      buckets.get(value).push(row * 9 + col);
    }
    for (const indexes of buckets.values()) {
      if (indexes.length > 1) indexes.forEach((idx) => conflicts.add(idx));
    }
  }

  for (let col = 0; col < 9; col += 1) {
    const buckets = new Map();
    for (let row = 0; row < 9; row += 1) {
      const value = grid[row][col];
      if (value === 0) continue;
      if (!buckets.has(value)) buckets.set(value, []);
      buckets.get(value).push(row * 9 + col);
    }
    for (const indexes of buckets.values()) {
      if (indexes.length > 1) indexes.forEach((idx) => conflicts.add(idx));
    }
  }

  for (let boxRow = 0; boxRow < 3; boxRow += 1) {
    for (let boxCol = 0; boxCol < 3; boxCol += 1) {
      const buckets = new Map();
      for (let row = boxRow * 3; row < boxRow * 3 + 3; row += 1) {
        for (let col = boxCol * 3; col < boxCol * 3 + 3; col += 1) {
          const value = grid[row][col];
          if (value === 0) continue;
          if (!buckets.has(value)) buckets.set(value, []);
          buckets.get(value).push(row * 9 + col);
        }
      }
      for (const indexes of buckets.values()) {
        if (indexes.length > 1) indexes.forEach((idx) => conflicts.add(idx));
      }
    }
  }

  return conflicts;
}

function shuffle(array) {
  const arr = array.slice();
  for (let i = arr.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

function registerServiceWorker() {
  if (!("serviceWorker" in navigator)) return;
  navigator.serviceWorker.register("./sw.js").catch(() => {
    // Service worker is optional; ignore failures on file:// or unsupported environments.
  });
}

function detectLocale() {
  const languages = navigator.languages && navigator.languages.length ? navigator.languages : [navigator.language || "en"];
  const normalized = languages.map((lang) => String(lang).toLowerCase());
  if (normalized.some((lang) => lang.startsWith("ru"))) return "ru";
  if (normalized.some((lang) => lang.startsWith("en"))) return "en";
  return "en";
}

function setLocale(locale) {
  currentLocale = translations[locale] ? locale : "en";
  applyTranslations();
}

function t(key) {
  return translations[currentLocale]?.[key] ?? translations.en[key] ?? key;
}

function applyTranslations() {
  document.documentElement.lang = currentLocale;
  document.title = t("title");
  document.querySelectorAll("[data-i18n]").forEach((element) => {
    const key = element.dataset.i18n;
    const attribute = element.dataset.i18nAttr;
    const value = t(key);
    if (attribute) {
      element.setAttribute(attribute, value);
    } else {
      element.textContent = value;
    }
  });
  boardEl.setAttribute("aria-label", t("board_label"));
  showSolutionBtn.textContent = showingSolution ? t("hide_solution") : t("show_solution");
  updateDifficultyButtons();
  updateOnlineStatus();
  updateNoteModeUI();
}

function getDifficultyLabel(level) {
  const key = `difficulty_${level}`;
  return translations[currentLocale]?.[key] ?? translations.en[key] ?? level;
}

init();
