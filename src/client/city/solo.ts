import {
  identity,
  api,
  CityMap,
  District,
  esc,
  heading,
  main,
  note,
  params,
  read,
  store,
  level,
  ApiError,
} from "./common";
import { drawMap } from "./map";
import { openBoard } from "./board";
import { parseManualText, solveGrid } from "../game";
interface Pending {
  campaign: number;
  district: number;
  board: string;
}
const cache = "sudoku:city:campaign";
const queue = "sudoku:city:completions";
async function flush() {
  for (const item of read<Pending[]>(queue, [])) {
    await api("/campaign/" + item.campaign + "/finish", item);
    store(
      queue,
      read<Pending[]>(queue, []).filter(
        (x) => x.campaign !== item.campaign || x.district !== item.district,
      ),
    );
  }
}
function available(m: CityMap, d: District) {
  return (
    d.start ||
    m.done.includes(d.id) ||
    m.links.some(
      ([a, b]) =>
        (a === d.id && m.done.includes(b)) ||
        (b === d.id && m.done.includes(a)),
    )
  );
}
export async function solo() {
  let map = read<CityMap | null>(cache, null);
  try {
    await identity();
    await flush();
    const campaign =
      map && params.get("latest") !== "1" ? "?campaign=" + map.campaign_id : "";
    map = await api<CityMap>("/map" + campaign);
    store(cache, map);
  } catch (e) {
    if (!map || (e instanceof ApiError && e.status < 500)) throw e;
    note("Вы играете офлайн. Прогресс сохранится на этом устройстве.");
  }
  const m = map!;
  const d = m.districts.find((x) => x.id === Number(params.get("district")));
  if (d && available(m, d)) {
    openBoard({
      name: d.name,
      givens: d.givens!,
      difficulty: d.difficulty!,
      key: "sudoku:solo:" + m.campaign_id + ":" + d.id,
      back: "/?mode=solo",
      finish: async (board) => {
        const solved = parseManualText(d.givens!);
        if (!solveGrid(solved) || solved.flat().join("") !== board)
          throw new Error("Проверьте решение");
        if (!m.done.includes(d.id)) {
          const item = { campaign: m.campaign_id, district: d.id, board };
          store(queue, [...read<Pending[]>(queue, []), item]);
          m.done.push(d.id);
          store(cache, m);
        }
        try {
          await flush();
        } catch (e) {
          if (e instanceof ApiError && e.status < 500) throw e;
        }
      },
    });
    document.getElementById("board-status")!.textContent =
      "Личный маршрут · прогресс сохраняется автоматически";
    return;
  }
  const finish = m.districts.find((d) => d.finish)!;
  main.innerHTML =
    heading("Одиночная игра", "Мой город", m.name) +
    `<div class="atlas-layout"><div id="map"></div><aside class="info"><div class="over">Личный маршрут</div>
 <h2>${m.done.includes(finish.id) ? "Маршрут пройден" : "До района «" + esc(finish.name) + "»"}</h2><p>Решайте судоку, чтобы открыть соседние районы.</p>
 <strong class="metric">${m.done.length} <small>/ ${m.districts.length}</small></strong><p>районов пройдено</p>
 <ul class="district-list">${m.districts.map((d) => `<li>${available(m, d) ? '<a href="/?mode=solo&district=' + d.id + '">' + esc(d.name) + "</a>" : esc(d.name)}<small>${m.done.includes(d.id) ? "Пройден" : available(m, d) ? esc(level(d.difficulty)) : "Закрыт"}</small></li>`).join("")}</ul>
 <a class="plain" href="/?mode=solo&latest=1">Открыть актуальную карту</a></aside></div>`;
  drawMap(
    document.getElementById("map")!,
    m.districts,
    m.links,
    (d) => ({
      label: m.done.includes(d.id)
        ? "Пройден"
        : d.start
          ? "Старт"
          : d.finish
            ? "Финиш"
            : level(d.difficulty),
      state: m.done.includes(d.id)
        ? "complete"
        : available(m, d)
          ? "available"
          : "locked",
      disabled: !available(m, d),
    }),
    (d) => location.assign("/?mode=solo&district=" + d.id),
  );
}
