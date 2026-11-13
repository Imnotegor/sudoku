import {
  api,
  identity,
  Player,
  Match,
  MatchItem,
  District,
  heading,
  esc,
  roomURL,
  main,
  note,
  params,
  form,
  button,
  rooms,
  problem,
} from "./common";
import { openBoard } from "./board";
import { drawMap } from "./map";

const state = (d: District) => {
  const [a, b] = d.scores!;
  return a > b ? "blue" : b > a ? "amber" : a ? "contested" : "neutral";
};
function remaining(m: Match) {
  const sec = Math.max(0, (m.ends ?? m.now) - m.now);
  return Math.floor(sec / 60) + ":" + String(sec % 60).padStart(2, "0");
}
function result(m: Match) {
  const a = m.districts.filter((d) => state(d) === "blue").length,
    b = m.districts.filter((d) => state(d) === "amber").length;
  return {
    a,
    b,
    winner: a === b ? "Ничья" : a > b ? "Победили синие" : "Победили янтарные",
  };
}
async function report(id: number) {
  const reason = window.prompt("Что случилось с задачей?");
  if (!reason?.trim()) return;
  await api("/games/" + id + "/report", { reason });
  note("Организатор получил сообщение");
}
async function copyLink(link: string) {
  try {
    await navigator.clipboard.writeText(location.origin + link);
    note("Ссылка скопирована");
  } catch {
    window.prompt("Скопируйте ссылку", location.origin + link);
  }
}
async function portal(p: Player) {
  const recent = await api<MatchItem[]>("/matches");
  main.innerHTML =
    heading(
      "Командная игра",
      "Битва за город",
      "Две команды по два игрока. За 25 минут захватите больше районов, чем соперник.",
    ) +
    `<div class="portal"><section class="info"><h2>Комната матча</h2><p>Создайте комнату и отправьте ссылку трём игрокам.</p><button class="primary" id="create">Создать матч 2×2</button>
 <form id="join-code"><label>Код или ссылка на комнату<input name="code" required autocomplete="off"></label><button>Войти в комнату</button></form></section>
 <section class="info"><h2>Ваш игрок</h2><form id="profile"><label>Имя в команде<input name="name" value="${esc(p.name)}" maxlength="32" required></label><button>Сохранить имя</button></form></section></div>
 ${recent.length ? "<section><h2>Мои матчи</h2>" + rooms(recent) + "</section>" : ""}`;
  button("create", async () => {
    const m = await api<{ code: string }>("/matches", {});
    location.assign(roomURL(m.code));
  });
  form("profile", async (f) => {
    await api("/profile", { name: f.get("name") });
    note("Имя сохранено");
  });
  form("join-code", async (f) => {
    const raw = String(f.get("code")).trim();
    let code = raw;
    let invite = "";
    if (raw.includes("://")) {
      const u = new URL(raw);
      code = u.searchParams.get("room") ?? "";
      invite = u.searchParams.get("invite") ?? "";
    }
    if (!/^[a-f0-9]{10}$/i.test(code))
      throw new Error("Введите код из десяти символов");
    await api("/matches/" + code.toLowerCase());
    location.assign(
      roomURL(code.toLowerCase()) +
        (invite ? "&invite=" + encodeURIComponent(invite) : ""),
    );
  });
}
export async function battle() {
  const p = await identity(),
    code = params.get("room");
  if (!code) {
    await portal(p);
    return;
  }
  const path = "/matches/" + encodeURIComponent(code);
  let m = await api<Match>(path);
  let fetched = Date.now();
  const me = () => m.teams.find((t) => t.players.some((x) => x.id === p.id));
  const requested = Number(params.get("district"));
  const chosen = m.districts.find((d) => d.id === requested);
  const board = !!(chosen && me() && m.status === "active");
  const join = async (side: number) => {
    await api(path + "/join", {
      side,
      ...(params.get("invite") ? { invite: params.get("invite") } : {}),
    });
    location.replace(roomURL(m.code));
  };
  let historyDistrict = 0;
  function draw() {
    const events = m.events.filter(
      (e) => !historyDistrict || e.district_id === historyDistrict,
    );
    const r = result(m),
      mine = me(),
      invite = params.get("invite");
    main.innerHTML =
      heading(
        "Матч · " + m.code,
        m.map,
        m.status === "lobby"
          ? "Матч начнётся, когда обе команды займут по два места."
          : m.status === "finished"
            ? r.winner
            : "Каждое верное решение даёт району одно очко влияния.",
      ) +
      `<div class="scorebar"><b class="blue-text">Синие · ${r.a}</b><strong id="clock">${m.status === "lobby" ? m.minutes + " мин" : m.status === "finished" ? "Финал" : remaining(m)}</strong><b class="amber-text">Янтарные · ${r.b}</b></div>
 <div class="atlas-layout"><div id="map"></div><aside class="info"><div class="over">Команды</div>
 ${m.teams
   .map(
     (
       t,
     ) => `<section class="roster"><h3>${esc(t.name)} <small>${t.players.length}/2</small></h3>
 <p>${t.players.map((u) => esc(u.name) + (u.id === p.id ? " (вы)" : "") + (u.district ? " · " + esc(u.district) : "")).join("<br>") || "Свободные места"}</p>
 ${m.status === "lobby" && !mine && t.players.length < 2 && (!invite || invite === t.invite) ? '<button id="join-' + t.side + '">Занять место</button>' : ""}
 ${m.status === "lobby" && mine?.side === t.side ? '<button class="plain" id="team-link">Пригласить в команду</button>' : ""}</section>`,
   )
   .join("")}
 <button id="share">Ссылка на комнату</button></aside></div><section id="history"><h2>История захватов</h2><label>Район<select id="history-filter"><option value="0">Все районы</option>${m.districts.map((d) => '<option value="' + d.id + '" ' + (historyDistrict === d.id ? "selected" : "") + ">" + esc(d.name) + "</option>").join("")}</select></label>
 ${events.length ? events.map((e) => `<div class="room-row"><strong>${esc(e.district)}</strong><span>${esc(e.player)} · ${esc(m.teams[e.side].name)}</span><small>${e.enabled ? "+1" : "Очко снято"}</small>${mine || p.admin ? '<button class="plain event-report" data-id="' + e.id + '">Оспорить</button>' : ""}</div>`).join("") : "<p>Здесь появятся подтверждённые решения.</p>"}</section>`;
    drawMap(
      document.getElementById("map")!,
      m.districts,
      m.links,
      (d) => ({
        label: d.scores!.join(" : "),
        state: state(d),
        disabled: m.status === "lobby",
      }),
      (d) => {
        if (mine && m.status === "active") {
          location.assign(roomURL(code!) + "&district=" + d.id);
          return;
        }
        historyDistrict = d.id;
        draw();
        document.getElementById("history")!.scrollIntoView();
      },
      m.code,
    );
    for (const t of m.teams) button("join-" + t.side, () => join(t.side));
    const filter = document.getElementById(
      "history-filter",
    ) as HTMLSelectElement;
    filter.onchange = () => {
      historyDistrict = Number(filter.value);
      draw();
      document.getElementById("history")!.scrollIntoView();
    };
    button("share", () => copyLink(roomURL(m.code)));
    if (mine)
      button("team-link", () =>
        copyLink(roomURL(m.code) + "&invite=" + mine.invite),
      );
    main.querySelectorAll<HTMLButtonElement>(".event-report").forEach(
      (b) =>
        (b.onclick = () => {
          void report(Number(b.dataset.id)).catch(problem);
        }),
    );
  }
  if (board && chosen) {
    const game = await api<{
      id: number;
      board: string;
      revision: number;
      givens: string;
      difficulty: string;
    }>(path + "/play", { district: chosen.id });
    openBoard({
      ...game,
      name: chosen.name,
      key: "sudoku:match:" + p.id + ":" + game.id,
      back: roomURL(code),
      save: async (board, revision) =>
        (
          await api<{ revision: number }>("/games/" + game.id + "/move", {
            board,
            revision,
          })
        ).revision,
      finish: async (board) => {
        await api("/games/" + game.id + "/finish", { board });
      },
      report: () => report(game.id),
    });
  } else draw();
  function tick() {
    const clock = document.getElementById(board ? "board-status" : "clock");
    if (clock && (board || m.status === "active")) {
      const d = m.districts.find((d) => d.id === requested);
      const time = remaining({
        ...m,
        now: m.now + Math.floor((Date.now() - fetched) / 1000),
      });
      clock.textContent = board
        ? d?.name +
          " · " +
          d?.scores?.join(" : ") +
          " · " +
          (m.status === "finished" ? "Матч завершён" : time)
        : time;
    }
    if (board && m.status === "finished") {
      const submit = document.getElementById("finish") as HTMLButtonElement;
      submit.disabled = true;
    }
  }
  tick();
  setInterval(tick, 1000);
  let polling = false;
  setInterval(async () => {
    if (polling || document.hidden || m.status === "finished") return;
    polling = true;
    try {
      const next = await api<Match>(path);
      const changed =
        JSON.stringify({ ...next, now: 0 }) !==
        JSON.stringify({ ...m, now: 0 });
      m = next;
      fetched = Date.now();
      if (!board && changed) draw();
      tick();
    } catch (e) {
      problem(e);
    } finally {
      polling = false;
    }
  }, 3500);
}
