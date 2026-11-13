import { main, params, problem, read, esc } from "./common";
import { landmarks, roads, drawMap } from "./map";
import { landscapes } from "./scenery";
import { solo } from "./solo";
import { battle } from "./battle";
import { admin } from "./admin";

function home() {
  const saved = read<unknown>("sudoku:game", null);
  main.innerHTML = `<div class="home-hero"><div class="hero-copy"><div class="over">Командное судоку</div><h1>Битва <br>за город</h1><p>Решайте судоку и захватывайте районы. Через 25 минут победит команда, которая удержит больше районов.</p><a class="primary" href="/?mode=battle">Войти в битву <span>→</span></a></div>
 <div class="map-card"><div class="map-cap"><b>Речной город</b><button type="button" class="map-cycle" id="map-cycle" aria-label="Следующий пейзаж карты"></button></div><div id="home-map"></div><div class="map-foot"><span>● Синие</span><span>● Янтарные</span><span>5 районов · 2 × 2</span></div></div></div>
 <section class="solo-choices"><h2>Играть одному</h2><div class="choice-grid">
 <a class="choice" href="/quick.html"><div class="tile-symbol grid-symbol" aria-hidden="true"></div><div><h3>${saved ? "Продолжить партию" : "Быстрая партия"}</h3><p>Классическое судоку, заметки и сканирование. Можно играть офлайн.</p></div><b>↗</b></a>
 <a class="choice" href="/?mode=solo"><div class="tile-symbol route-symbol" aria-hidden="true">⌖</div><div><h3>Мой город</h3><p>Откройте путь от Вокзала до Маяка в своём темпе.</p></div><b>↗</b></a></div></section>`;
  let selected = 0;
  const cycle = document.getElementById("map-cycle")!;
  function preview() {
    const { title, kind } = landscapes[selected];
    cycle.textContent = title + "  ›";
    drawMap(
      document.getElementById("home-map")!,
      landmarks,
      roads,
      () => ({ label: "" }),
      undefined,
      "preview-" + kind,
      selected,
    );
  }
  cycle.onclick = () => {
    selected = (selected + 1) % landscapes.length;
    preview();
  };
  preview();
}
async function start() {
  switch (params.get("mode")) {
    case "solo":
      await solo();
      break;
    case "battle":
      await battle();
      break;
    case "admin":
      await admin();
      break;
    default:
      home();
  }
}
void start().catch((e) => {
  main.innerHTML = `<h1>Не удалось открыть режим</h1><p>${esc(e.message)}</p><a class="primary" href="/">К выбору режима</a>`;
  problem(e);
});
if ("serviceWorker" in navigator)
  void navigator.serviceWorker.register("/sw.js").catch(() => {});
