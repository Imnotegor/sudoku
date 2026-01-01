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
    hint: "Можно вводить цифры 1–9 с клавиатуры. Стрелки перемещают выделение.",
    new_game: "Новая игра",
    clear_cell: "Очистить",
    show_solution: "Показать решение",
    hide_solution: "Скрыть решение",
    difficulty_label: "Сложность",
    difficulty_easy: "Легко",
    difficulty_medium: "Нормально",
    difficulty_hard: "Сложно",
    difficulty_expert: "Эксперт",
    online: "Онлайн",
    offline: "Офлайн",
    toast_new_game: "Новая игра готова",
    toast_solved: "Отлично! Судоку решено.",
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
    hint: "Use keys 1–9 to enter numbers. Arrow keys move selection.",
    new_game: "New game",
    clear_cell: "Clear",
    show_solution: "Show solution",
    hide_solution: "Hide solution",
    difficulty_label: "Difficulty",
    difficulty_easy: "Easy",
    difficulty_medium: "Medium",
    difficulty_hard: "Hard",
    difficulty_expert: "Expert",
    online: "Online",
    offline: "Offline",
    toast_new_game: "New game ready",
    toast_solved: "Great! Sudoku solved.",
  },
};

const boardEl = document.getElementById("board");
const difficultyEl = document.getElementById("difficulty");
const difficultyLabel = document.getElementById("difficulty-label");
const timerEl = document.getElementById("timer");
const onlineStatusEl = document.getElementById("online-status");
const newGameBtn = document.getElementById("new-game");
const clearBtn = document.getElementById("clear-cell");
const showSolutionBtn = document.getElementById("show-solution");
const numpadEl = document.getElementById("numpad");
const toastEl = document.getElementById("toast");

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
  clearBtn.addEventListener("click", () => setCellValue(selected.row, selected.col, 0));
  showSolutionBtn.addEventListener("click", toggleSolution);

  numpadEl.addEventListener("click", (event) => {
    const button = event.target.closest("button");
    if (!button) return;
    const value = Number(button.dataset.value);
    setCellValue(selected.row, selected.col, value);
  });

  document.addEventListener("keydown", handleKeydown);
  window.addEventListener("beforeunload", saveGame);
  window.addEventListener("online", updateOnlineStatus);
  window.addEventListener("offline", updateOnlineStatus);
}

function handleKeydown(event) {
  if (event.defaultPrevented) return;
  const key = event.key;
  if (key >= "1" && key <= "9") {
    setCellValue(selected.row, selected.col, Number(key));
    event.preventDefault();
    return;
  }

  if (key === "Backspace" || key === "Delete" || key === "0") {
    setCellValue(selected.row, selected.col, 0);
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
    startTime = Date.now();
    elapsedMs = 0;
    selectFirstEmpty();
    updateUI();
    saveGame();
    boardEl.classList.remove("loading");
    showToast(t("toast_new_game"));
  }, 30);
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
    difficulty = saved.difficulty || "easy";
    startTime = Date.now() - (saved.elapsedMs || 0);
    elapsedMs = saved.elapsedMs || 0;
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
  current[row][col] = value;
  updateUI();
  saveGame();
  checkSolved();
}

function toggleSolution() {
  if (!solution.length) return;
  showingSolution = !showingSolution;
  showSolutionBtn.textContent = showingSolution ? t("hide_solution") : t("show_solution");
  updateUI();
}

function updateUI() {
  updateDifficultyButtons();
  updateBoard();
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
    cell.textContent = value === 0 ? "" : String(value);
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
  const conflicts = findConflicts(current);
  if (conflicts.size > 0) return;
  for (let row = 0; row < 9; row += 1) {
    for (let col = 0; col < 9; col += 1) {
      if (current[row][col] !== solution[row][col]) return;
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
}

function getDifficultyLabel(level) {
  const key = `difficulty_${level}`;
  return translations[currentLocale]?.[key] ?? translations.en[key] ?? level;
}

init();
