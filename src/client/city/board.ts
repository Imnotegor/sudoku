import {
  gameState,
  parseManualText,
  createEmptyNotes,
  restoreNotes,
} from "../game";
import {
  initBoard,
  updateBoard,
  handleBoardClick,
  initNumpad,
  handleNumpadClick,
  updateNumpadAvailability,
} from "../ui";
import { Difficulty } from "../types";
import {
  ApiError,
  button,
  esc,
  level,
  main,
  note,
  problem,
  read,
  store,
} from "./common";

export interface BoardOptions {
  name: string;
  givens: string;
  difficulty: string;
  key: string;
  back: string;
  board?: string;
  revision?: number;
  finish: (grid: string) => Promise<void>;
  save?: (grid: string, revision: number) => Promise<number>;
  report?: () => Promise<void>;
}
interface Saved {
  board: string;
  notes: number[][];
  elapsed: number;
  revision: number;
}
export function openBoard(o: BoardOptions) {
  const saved = read<Saved | null>(o.key, null);
  let revision = o.revision ?? 0;
  const usable =
    saved && saved.revision === revision && /^[0-9]{81}$/.test(saved.board);
  const initial = usable ? saved.board : (o.board ?? o.givens);
  gameState.restore(
    parseManualText(o.givens),
    null,
    parseManualText(initial),
    usable ? restoreNotes(saved.notes) : createEmptyNotes(),
    o.difficulty as Difficulty,
    usable ? saved.elapsed : 0,
  );
  main.innerHTML = `<a class="back" href="${esc(o.back)}">← Карта города</a><div class="board-view">
 <div><div id="city-board" class="board" role="grid" aria-label="Судоку"></div></div>
 <aside class="board-tools"><div class="over">${esc(level(o.difficulty))} партия</div><h1>${esc(o.name)}</h1>
 <p id="board-status" aria-live="polite"></p><div id="keys" class="numpad"></div><div class="actions">
 <button id="notes">Заметки</button><button id="undo">Отмена</button><button id="erase">Стереть</button></div>
 <button class="primary" id="finish">Проверить решение</button>
 ${o.report ? '<button class="plain" id="report">Сообщить о задаче</button>' : ""}</aside></div>`;
  const board = document.getElementById("city-board")!;
  initBoard(board);
  initNumpad(document.getElementById("keys")!);
  const grid = () => gameState.current.flat().join("");
  let sent = o.board ?? o.givens;
  let timer: number | undefined;
  let saving = false;
  let finished = false;
  let blocked = false;
  const local = () => {
    if (!finished)
      store(o.key, {
        board: grid(),
        notes: gameState.notes,
        elapsed: gameState.getElapsedMs(),
        revision,
      });
  };
  async function sync() {
    if (!o.save || saving || finished || blocked || sent === grid()) return;
    saving = true;
    const value = grid();
    let retry = false;
    try {
      revision = await o.save(value, revision);
      sent = value;
      local();
    } catch (e) {
      problem(e);
      blocked = e instanceof ApiError && e.status < 500;
      retry = !blocked;
    } finally {
      saving = false;
      if (!blocked && sent !== grid() && !finished) {
        clearTimeout(timer);
        timer = window.setTimeout(sync, retry ? 4000 : 0);
      }
    }
  }
  function refresh() {
    updateBoard();
    updateNumpadAvailability();
    document
      .getElementById("notes")!
      .setAttribute("aria-pressed", String(gameState.noteMode));
    local();
    clearTimeout(timer);
    if (o.save && sent !== grid()) timer = window.setTimeout(sync, 500);
  }
  handleBoardClick((r, c) => gameState.selectCell(r, c));
  function enter(n: number) {
    const { row, col } = gameState.selected;
    if (n === 0) gameState.clearCell(row, col);
    else if (gameState.noteMode) gameState.toggleNote(row, col, n);
    else gameState.setCellValue(row, col, n);
  }
  handleNumpadClick(enter);
  gameState.subscribe(refresh);
  refresh();
  button("notes", () => gameState.toggleNoteMode());
  button("undo", () => {
    gameState.undo();
  });
  button("erase", () => enter(0));
  if (o.report) button("report", o.report);
  document.addEventListener("keydown", (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (/^[0-9]$/.test(e.key)) {
      e.preventDefault();
      enter(Number(e.key));
    }
    if (e.key === "Backspace" || e.key === "Delete") {
      e.preventDefault();
      enter(0);
    }
    const moves: Record<string, [number, number]> = {
      ArrowUp: [-1, 0],
      ArrowDown: [1, 0],
      ArrowLeft: [0, -1],
      ArrowRight: [0, 1],
    };
    const move = moves[e.key];
    if (move) {
      e.preventDefault();
      const { row, col } = gameState.selected;
      gameState.selectCell((row + move[0] + 9) % 9, (col + move[1] + 9) % 9);
    }
  });
  window.addEventListener("pagehide", local);
  button("finish", async () => {
    if (grid().includes("0")) {
      note("Заполните пустые клетки");
      return;
    }
    await o.finish(grid());
    finished = true;
    clearTimeout(timer);
    localStorage.removeItem(o.key);
    location.assign(o.back);
  });
}
