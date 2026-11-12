import {
  api,
  identity,
  CityMap,
  MatchItem,
  Player,
  rooms,
  roomURL,
  heading,
  main,
  esc,
  form,
  button,
  level,
  note,
} from "./common";
import { drawMap } from "./map";
interface Dashboard {
  map: CityMap;
  versions: { version_id: number; name: string }[];
  puzzles: { id: number; difficulty: string }[];
  matches: MatchItem[];
  reports: {
    id: number;
    code: string;
    district: string;
    player: string;
    puzzle_id: number;
    reason: string;
    resolved: boolean;
  }[];
  actions: {
    action: string;
    detail: string;
    actor: string;
    created_at: string;
  }[];
}
function edit(data: Dashboard) {
  const m = data.map;
  let selected = 0,
    preview = false;
  const visited = new Set<number>();
  const host = document.getElementById("editor")!;
  function update() {
    const d = m.districts[selected];
    const linked = (a: number, b: number) =>
      m.links.some(([x, y]) => (x === a && y === b) || (x === b && y === a));
    host.innerHTML = `<div class="editor-head"><h2>Редактор карты</h2><button id="preview">${preview ? "Вернуться в редактор" : "Пройти маршрут"}</button></div>
   <div class="atlas-layout"><div id="edit-map"></div><div class="info" id="fields"></div></div>`;
    const map = document.getElementById("edit-map")!;
    if (preview) {
      const reachable = (id: number) =>
        m.districts.some((x) => x.id === id && x.start) ||
        Array.from(visited).some((a) => linked(a, id));
      drawMap(
        map,
        m.districts,
        m.links,
        (x) => ({
          label: visited.has(x.id)
            ? "Пройден"
            : x.start
              ? "Старт"
              : x.finish
                ? "Финиш"
                : "",
          disabled: !reachable(x.id),
          state: visited.has(x.id)
            ? "complete"
            : reachable(x.id)
              ? "available"
              : "locked",
        }),
        (x) => {
          visited.add(x.id);
          update();
        },
      );
      document.getElementById("fields")!.innerHTML =
        `<h3>Предпросмотр маршрута</h3><p>Нажмите доступный район, чтобы отметить его пройденным и проверить следующую развилку.</p><p>Пройдено ${visited.size} из 5.</p>`;
    } else {
      drawMap(
        map,
        m.districts,
        m.links,
        (x) => ({
          label: x.start ? "Старт" : x.finish ? "Финиш" : "",
          state: x.id === d.id ? "complete" : "available",
        }),
        (x) => {
          selected = m.districts.indexOf(x);
          update();
        },
      );
      map.querySelector("canvas")!.onclick = (e) => {
        const r = map.getBoundingClientRect();
        d.x = Math.max(
          8,
          Math.min(92, Math.round(((e.clientX - r.left) / r.width) * 100)),
        );
        d.y = Math.max(
          8,
          Math.min(92, Math.round(((e.clientY - r.top) / r.height) * 100)),
        );
        update();
      };
      const options = (id: number) =>
        m.districts
          .map(
            (x) =>
              `<option value="${x.id}" ${x.id === id ? "selected" : ""}>${esc(x.name)}</option>`,
          )
          .join("");
      document.getElementById("fields")!.innerHTML =
        `<label>Название карты<input id="map-title" maxlength="60" value="${esc(m.name)}"></label>
    <label>Выбранный район<input id="district-name" maxlength="32" value="${esc(d.name)}"></label>
    <p class="hint">Выберите район и нажмите на карту, чтобы переместить его.</p>
    <label>Задача района<select id="puzzle">${data.puzzles.map((p) => `<option value="${p.id}" ${p.id === d.puzzle ? "selected" : ""}>№ ${p.id} · ${level(p.difficulty)}</option>`).join("")}</select></label>
    <div class="actions"><label>Старт<select id="start">${options(m.districts.find((x) => x.start)!.id)}</select></label><label>Финиш<select id="end">${options(m.districts.find((x) => x.finish)!.id)}</select></label></div>
    <fieldset><legend>Дороги из района</legend>${m.districts
      .filter((x) => x.id !== d.id)
      .map(
        (x) =>
          `<label class="check"><input type="checkbox" value="${x.id}" ${linked(d.id, x.id) ? "checked" : ""}>${esc(x.name)}</label>`,
      )
      .join("")}</fieldset>
    <button class="primary" id="publish">Опубликовать карту</button>`;
      const input = (id: string, fn: (v: string) => void) => {
        const el = document.getElementById(id) as HTMLInputElement;
        el.oninput = () => fn(el.value);
      };
      input("map-title", (v) => (m.name = v));
      input("district-name", (v) => {
        d.name = v;
        map.querySelectorAll("b")[selected].textContent = v;
      });
      input("puzzle", (v) => (d.puzzle = Number(v)));
      input("start", (v) => {
        m.districts.forEach((x) => (x.start = x.id === Number(v)));
        update();
      });
      input("end", (v) => {
        m.districts.forEach((x) => (x.finish = x.id === Number(v)));
        update();
      });
      host.querySelectorAll<HTMLInputElement>("input[type=checkbox]").forEach(
        (box) =>
          (box.onchange = () => {
            const id = Number(box.value);
            m.links = m.links.filter(
              (link) => !(link.includes(d.id) && link.includes(id)),
            );
            if (box.checked) m.links.push([d.id, id]);
            update();
          }),
      );
      button("publish", async () => {
        const idx = (id: number) => m.districts.findIndex((x) => x.id === id);
        await api("/admin/maps", {
          base: m.version_id,
          name: m.name,
          districts: m.districts,
          start: m.districts.findIndex((x) => x.start),
          finish: m.districts.findIndex((x) => x.finish),
          links: m.links.map(([a, b]) => [idx(a), idx(b)]),
        });
        await dashboard();
        note("Карта опубликована");
      });
    }
    button("preview", () => {
      preview = !preview;
      visited.clear();
      update();
    });
  }
  update();
}
async function dashboard() {
  const d = await api<Dashboard>("/admin");
  main.innerHTML =
    heading("Организатор", "Управление городом") +
    `<section id="editor"></section><section class="portal">
 <div class="info"><h2>Провести матч</h2><form id="host"><label>Карта<select name="version">${d.versions.map((v) => `<option value="${v.version_id}">${esc(v.name)} · v${v.version_id}</option>`).join("")}</select></label>
 <label>Длительность, минут<input type="number" min="5" max="60" name="minutes" value="25" required></label><button class="primary">Открыть комнату</button></form></div>
 <div class="info"><h2>Матчи</h2>${rooms(d.matches) || "<p>Матчей пока нет.</p>"}</div></section>
 <section><h2>Спорные задачи</h2>${
   d.reports
     .map(
       (
         r,
       ) => `<article class="case"><a href="${roomURL(r.code)}">${esc(r.district)} · № ${r.puzzle_id}</a><p>${esc(r.player)}: ${esc(r.reason)}</p>
 ${r.resolved ? "Рассмотрено" : `<form id="case-${r.id}"><label>Причина решения<input required maxlength="500" name="reason"></label><select name="decision"><option value="close">Оставить результат</option><option value="withdraw">Снять задачу и пересчитать очки</option></select><button>Применить</button></form>`}</article>`,
     )
     .join("") || "<p>Жалоб пока нет.</p>"
 }</section>
 <section><h2>Журнал действий</h2>${d.actions.map((a) => `<div class="room-row"><strong>${esc(a.action)}</strong><span>${esc(a.detail)}</span><small>${esc(a.actor)} · ${esc(new Date(a.created_at).toLocaleString("ru"))}</small></div>`).join("")}</section>`;
  edit(d);
  form("host", async (f) => {
    const m = await api<{ code: string }>("/matches", {
      version: Number(f.get("version")),
      minutes: Number(f.get("minutes")),
      spectate: true,
    });
    location.assign(roomURL(m.code));
  });
  for (const r of d.reports)
    form("case-" + r.id, async (f) => {
      await api("/admin/reports/" + r.id, {
        withdraw: f.get("decision") === "withdraw",
        reason: f.get("reason"),
      });
      await dashboard();
      note("Решение записано");
    });
}
export async function admin() {
  const p: Player = await identity();
  if (p.admin) {
    await dashboard();
    return;
  }
  main.innerHTML =
    heading("Организатор", "Вход в админку") +
    `<form class="info login" id="login"><label>Ключ администратора<input name="key" type="password" autocomplete="off" required></label><button class="primary">Войти</button></form>`;
  form("login", async (f) => {
    await api("/admin/login", { key: f.get("key") });
    await dashboard();
  });
}
