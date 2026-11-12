import { District, esc } from "./common";
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
export function drawMap(
  host: HTMLElement,
  ds: District[],
  links: number[][],
  pin: (d: District) => Pin,
  click?: (d: District) => void,
) {
  host.classList.add("city-map");
  host.innerHTML = '<canvas aria-hidden="true"></canvas>';
  const canvas = host.querySelector("canvas")!;
  const ctx = canvas.getContext("2d");
  if (ctx) {
    canvas.width = 1000;
    canvas.height = 540;
    ctx.scale(10, 5.4);
    ctx.fillStyle = "#edeadc";
    ctx.fillRect(0, 0, 100, 100);
    ctx.save();
    ctx.translate(50, 50);
    ctx.rotate(-0.28);
    for (let x = -75; x < 75; x += 4.3)
      for (let y = -85; y < 85; y += 6.2) {
        ctx.fillStyle = Math.round(x + y) % 5 === 0 ? "#d2dcc2" : "#dfdecd";
        ctx.fillRect(x, y, 3.5, 5.1);
      }
    ctx.restore();
    ctx.strokeStyle = "#9bc9ca";
    ctx.lineWidth = 6;
    ctx.beginPath();
    ctx.moveTo(-5, 92);
    ctx.bezierCurveTo(27, 63, 33, 100, 57, 72);
    ctx.bezierCurveTo(76, 52, 73, 12, 107, 5);
    ctx.stroke();
    ctx.strokeStyle = "#fbf7e7";
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(0, 17);
    ctx.lineTo(100, 82);
    ctx.stroke();
    const byId = new Map(ds.map((d) => [d.id, d]));
    for (const [a, b] of links) {
      const from = byId.get(a),
        to = byId.get(b);
      if (!from || !to) continue;
      ctx.beginPath();
      ctx.moveTo(from.x, from.y);
      ctx.lineTo(to.x, to.y);
      ctx.strokeStyle = "#fffdf2";
      ctx.lineWidth = 1.7;
      ctx.stroke();
      ctx.strokeStyle = "#687d70";
      ctx.lineWidth = 0.55;
      ctx.stroke();
    }
  }
  for (const d of ds) {
    const p = pin(d),
      b = document.createElement(click ? "button" : "span");
    b.className = "map-pin " + (p.state ?? "");
    b.style.left = d.x + "%";
    b.style.top = d.y + "%";
    b.innerHTML = `<i></i><b>${esc(d.name)}</b>${p.label ? "<small>" + esc(p.label) + "</small>" : ""}`;
    if (b instanceof HTMLButtonElement) {
      b.type = "button";
      b.disabled = !!p.disabled;
      b.onclick = () => click?.(d);
    }
    host.append(b);
  }
}
