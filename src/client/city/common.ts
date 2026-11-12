export interface Player {
  id: number;
  name: string;
  admin: boolean;
  district?: string;
}
export interface District {
  id: number;
  slot: number;
  name: string;
  x: number;
  y: number;
  start?: boolean;
  finish?: boolean;
  puzzle?: number;
  givens?: string;
  difficulty?: string;
  scores?: [number, number];
}
export interface CityMap {
  name: string;
  campaign_id: number;
  version_id: number;
  districts: District[];
  links: number[][];
  done: number[];
}
export interface Team {
  name: string;
  side: number;
  invite: string;
  players: Player[];
}
export interface Event {
  id: number;
  player: string;
  district: string;
  district_id: number;
  side: number;
  enabled: boolean;
  puzzle_id: number;
}
export interface Match {
  code: string;
  map: string;
  host: number;
  me: number;
  status: "lobby" | "active" | "finished";
  minutes: number;
  now: number;
  ends: number | null;
  teams: Team[];
  districts: District[];
  links: number[][];
  events: Event[];
}
export interface MatchItem {
  code: string;
  name: string;
  status: Match["status"];
}
export const main = document.getElementById("city")!;
export const params = new URLSearchParams(location.search);
export const esc = (v: unknown): string =>
  String(v ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ]!,
  );
export const level = (v?: string) =>
  ({ easy: "Лёгкая", medium: "Средняя", hard: "Сложная", expert: "Эксперт" })[
    v ?? ""
  ] ?? "";
export const statusText = (s: string) =>
  ({ lobby: "Сбор команд", active: "Идёт матч", finished: "Матч завершён" })[
    s
  ] ?? s;
export const roomURL = (code: string) =>
  "/?mode=battle&room=" + encodeURIComponent(code);
export function note(text: string) {
  document.getElementById("notice")!.textContent = text;
}
export function problem(e: unknown) {
  note(e instanceof Error ? e.message : "Не удалось выполнить действие");
}
export function read<T>(key: string, defaultValue: T): T {
  try {
    return JSON.parse(localStorage.getItem(key) ?? "null") ?? defaultValue;
  } catch {
    return defaultValue;
  }
}
export function store(key: string, value: unknown) {
  localStorage.setItem(key, JSON.stringify(value));
}
export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}
export async function api<T>(path: string, data?: unknown): Promise<T> {
  const resp = await fetch("/api/city" + path, {
    method: data === undefined ? "GET" : "POST",
    credentials: "same-origin",
    cache: "no-store",
    headers: { "Content-Type": "application/json", "X-City-Request": "1" },
    ...(data === undefined ? {} : { body: JSON.stringify(data) }),
  });
  if (!resp.ok) {
    const error = await resp.json().catch(() => ({}));
    throw new ApiError(
      resp.status,
      error.error ?? "Сервис временно недоступен",
    );
  }
  return resp.json() as Promise<T>;
}
let session: Promise<Player> | undefined;
export function identity() {
  return (session ??= api<Player>("/session", {}).catch((e) => {
    session = undefined;
    throw e;
  }));
}
export function button(id: string, fn: () => Promise<unknown> | void) {
  const el = document.getElementById(id) as HTMLButtonElement | null;
  if (!el) return;
  el.onclick = async () => {
    el.disabled = true;
    note("");
    try {
      await fn();
    } catch (e) {
      problem(e);
    } finally {
      el.disabled = false;
    }
  };
}
export function form(id: string, fn: (data: FormData) => Promise<unknown>) {
  const el = document.getElementById(id) as HTMLFormElement | null;
  if (!el) return;
  el.onsubmit = async (ev) => {
    ev.preventDefault();
    if (el.dataset.busy) return;
    el.dataset.busy = "1";
    note("");
    try {
      await fn(new FormData(el));
    } catch (e) {
      problem(e);
    } finally {
      delete el.dataset.busy;
    }
  };
}
export function heading(parent: string, title: string, sub = "") {
  return `<div class="intro"><div class="over">${esc(parent)}</div><h1>${esc(title)}</h1>${sub ? "<p>" + esc(sub) + "</p>" : ""}</div>`;
}
export function rooms(items: MatchItem[]) {
  return items
    .map(
      (m) =>
        `<a class="room-row" href="${roomURL(m.code)}"><strong>${esc(m.name)}</strong><code>${esc(m.code)}</code><span>${esc(statusText(m.status))} →</span></a>`,
    )
    .join("");
}
