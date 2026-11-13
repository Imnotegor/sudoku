import { District, esc } from "./common";
import { districtIcon } from "./district-icons";
import { createScene, WIDTH, HEIGHT, landscapes, mapSeed } from "./scenery";
export const landmarks: District[] = [
  { id: 1, slot: 1, name: "Вокзал", x: 17, y: 55 },
  { id: 2, slot: 2, name: "Парк", x: 40, y: 22 },
  { id: 3, slot: 3, name: "Университет", x: 48, y: 76 },
  { id: 4, slot: 4, name: "Порт", x: 68, y: 42 },
  { id: 5, slot: 5, name: "Маяк", x: 86, y: 70 },
];
export const roads = [
  [1, 2],
  [1, 3],
  [2, 4],
  [3, 4],
  [3, 5],
  [4, 5],
];
export type Pin = { label: string; state?: string; disabled?: boolean };
type Scene = ReturnType<typeof createScene>;
let pausedAt = performance.now();
let delay = 0;
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
let paused = false;
try {
  paused = localStorage.getItem("sudoku:map-motion") === "paused";
} catch {
  /* optional preference */
}

// The room view and the editor replace their markup on update. Let the DOM
// own the lifetime of each observer and animation loop.
class AtlasScenery extends HTMLElement {
  scene?: Scene;
  private terrain = document.createElement("canvas");
  private actors = document.createElement("canvas");
  private frame = 0;
  private onScreen = false;
  private lastPaint = -100;
  private resize = new ResizeObserver(() => this.paintTerrain());
  private visibility = new IntersectionObserver(([entry]) => {
    this.onScreen = entry.isIntersecting;
    this.updateMotion();
  });

  connectedCallback() {
    this.setAttribute("aria-hidden", "true");
    this.replaceChildren(this.terrain, this.actors);
    this.resize.observe(this);
    this.visibility.observe(this);
    document.addEventListener("visibilitychange", this.updateMotion);
    reducedMotion.addEventListener("change", this.updateMotion);
  }

  disconnectedCallback() {
    this.resize.disconnect();
    this.visibility.disconnect();
    document.removeEventListener("visibilitychange", this.updateMotion);
    reducedMotion.removeEventListener("change", this.updateMotion);
    cancelAnimationFrame(this.frame);
    this.frame = 0;
  }

  private paintTerrain() {
    const scale = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.round(this.clientWidth * scale),
      height = Math.round(this.clientHeight * scale);
    if (!width || !height) return;
    for (const layer of [this.terrain, this.actors]) {
      layer.width = width;
      layer.height = height;
      layer
        .getContext("2d")
        ?.setTransform(width / WIDTH, 0, 0, height / HEIGHT, 0, 0);
    }
    const ctx = this.terrain.getContext("2d");
    if (ctx) this.scene?.background(ctx);
    this.paintActors(performance.now());
  }

  private paintActors(now: number) {
    const ctx = this.actors.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, WIDTH, HEIGHT);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    this.scene?.motion(ctx, reducedMotion.matches ? 0 : ((paused ? pausedAt : now) - delay) / 1000);
  }

  private tick = (now: number) => {
    if (now - this.lastPaint >= 1000 / 24) {
      this.paintActors(now);
      this.lastPaint = now;
    }
    this.frame = requestAnimationFrame(this.tick);
  };

  updateMotion = () => {
    cancelAnimationFrame(this.frame);
    this.frame = 0;
    if (
      this.isConnected &&
      !document.hidden &&
      this.onScreen &&
      !paused &&
      !reducedMotion.matches
    ) {
      this.frame = requestAnimationFrame(this.tick);
    }
    const btn =
      this.parentElement?.querySelector<HTMLButtonElement>(".map-motion");
    if (btn) {
      btn.hidden = reducedMotion.matches;
      btn.textContent = paused ? "▶" : "Ⅱ";
      btn.title = paused
        ? "Включить анимацию карты"
        : "Приостановить анимацию карты";
      btn.setAttribute("aria-label", btn.title);
    }
  };
}
customElements.define("atlas-scenery", AtlasScenery);

export function drawMap(
  host: HTMLElement,
  ds: District[],
  links: number[][],
  pin: (d: District) => Pin,
  click?: (d: District) => void,
  key: string = "city-" + ds.map((d) => d.id).join("-"),
  preview?: number,
) {
  const seed = mapSeed(key);
  const variant = landscapes[preview ?? seed % landscapes.length];
  const scene = new AtlasScenery();
  scene.scene = createScene(
    variant.kind,
    seed,
    ds.map((d) => ({
      ...d,
      x: (d.x * WIDTH) / 100,
      y: (d.y * HEIGHT) / 100,
    })),
    links,
  );
  host.classList.add("city-map");
  host.dataset.landscape = variant.kind;
  host.replaceChildren(scene);
  for (const d of ds) {
    const p = pin(d),
      b = document.createElement(click ? "button" : "span");
    b.className = "map-pin " + (p.state ?? "");
    b.style.left = d.x + "%";
    b.style.top = d.y + "%";
    b.innerHTML = `<b>${esc(d.name)}</b>${p.label ? "<small>" + esc(p.label) + "</small>" : ""}`;
    b.prepend(districtIcon(d.slot));
    if (b instanceof HTMLButtonElement) {
      b.type = "button";
      b.disabled = !!p.disabled;
      b.onclick = () => click?.(d);
    }
    host.append(b);
  }
  const control = document.createElement("button");
  control.type = "button";
  control.className = "map-motion";
  control.onclick = () => {
    const now = performance.now();
    if (paused) delay += now - pausedAt;
    else pausedAt = now;
    paused = !paused;
    try {
      localStorage.setItem("sudoku:map-motion", paused ? "paused" : "playing");
    } catch {
      /* optional preference */
    }
    scene.updateMotion();
  };
  host.append(control);
  scene.updateMotion();
}
