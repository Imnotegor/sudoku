// Illustrated in a fixed atlas coordinate system; district coordinates stay in percentages.
type Pen = CanvasRenderingContext2D;
type Point = { x: number; y: number };
export const WIDTH = 900,
  HEIGHT = 520;
const tau = Math.PI * 2;
const ink = "#514c3f";
export type Landscape = "river" | "islands" | "valley";
export const landscapes: { kind: Landscape; title: string }[] = [
  { kind: "river", title: "Речные кварталы" },
  { kind: "islands", title: "Архипелаг" },
  { kind: "valley", title: "Горная долина" },
];

export function mapSeed(key: string) {
  let n = 2166136261;
  for (const c of key) n = Math.imul(n ^ c.charCodeAt(0), 16777619);
  return n >>> 0;
}
function generator(seed: number) {
  let state = seed;
  return () =>
    (state = (Math.imul(state, 1664525) + 1013904223) >>> 0) / 4294967296;
}
function shape(c: Pen, data: string, fill: string, stroke = ink, width = 1.3) {
  const p = new Path2D(data);
  if (fill) {
    c.fillStyle = fill;
    c.fill(p);
  }
  if (stroke) {
    c.strokeStyle = stroke;
    c.lineWidth = width;
    c.stroke(p);
  }
}
function oval(
  c: Pen,
  x: number,
  y: number,
  rx: number,
  ry: number,
  color: string,
) {
  c.beginPath();
  c.ellipse(x, y, rx, ry, 0, 0, tau);
  c.fillStyle = color;
  c.fill();
}
function stroke(c: Pen, p: Path2D, color: string, width: number) {
  c.strokeStyle = color;
  c.lineWidth = width;
  c.stroke(p);
}
function at(c: Pen, x: number, y: number, size: number, draw: () => void) {
  c.save();
  c.translate(x, y);
  c.scale(size, size);
  draw();
  c.restore();
}

function tree(
  c: Pen,
  x: number,
  y: number,
  size: number,
  pine: boolean,
  tint: number,
) {
  at(c, x, y, size, () => {
    oval(c, 3, 1, 8, 2.6, "#71764c44");
    shape(c, "M0 0v-13", "", "#706148", 2);
    if (pine) {
      shape(
        c,
        "M0 -26l-6 11h3l-5 8 4 -1 -5 6 15 -1 -5 -8 3 1 -5 -8h3Z",
        tint > 0.5 ? "#657e60" : "#78906a",
        "#526649",
        0.8,
      );
      shape(c, "M0 -23l-4 10 3 -1 -4 9", "", "#a7b58c", 0.9);
    } else {
      shape(
        c,
        "M-2 -4C-12 -4 -10 -13 -7 -16C-8 -24 2 -27 5 -19C13 -18 12 -5 3 -5Z",
        tint > 0.5 ? "#92aa76" : "#aec18a",
        "#768961",
        0.85,
      );
      shape(c, "M-5 -14q0 -6 5 -7", "", "#c7d29e", 1.6);
    }
  });
}

function house(c: Pen, x: number, y: number, size: number, roof: string) {
  at(c, x, y, size, () => {
    shape(c, "M-9 0l9 4 13 -7 -9 -4Z", "#918c753d", "");
    shape(c, "M-10 -10l13 -5v12l-13 5Z", "#f1ddae");
    shape(c, "M3 -15l8 5v11L3 -3Z", "#c39b73");
    shape(c, "M-12 -10l4 -7 13 -5 -3 8Z", roof);
    shape(c, "M2 -14l3 -8 8 12 -2 1Z", "#b97854");
    shape(c, "M-7 -7v4m5 -6v4m9 -8v4", "", "#65655a", 1.5);
  });
}

function landmark(c: Pen, slot: number) {
  oval(c, 5, 2, 33, 7, "#4d554333");
  switch (slot) {
    case 1: // A station hall with a clock tower and striped platforms.
      shape(c, "M-39 -1l53 -12 28 9 -54 13Z", "#c8bbaa");
      shape(c, "M-30 -16l38 -9v19l-38 9Z", "#e7c99e");
      shape(c, "M8 -25l22 9V0L8 -6Z", "#bd946e");
      shape(c, "M-34 -16l9 -13 36 -9 -6 14Z", "#7b9991");
      shape(c, "M5 -24l6 -14 22 21 -3 2Z", "#597a77");
      for (let x = -25; x < 2; x += 7)
        shape(c, `M${x} ${-10 - (x + 25) * 0.23}v7`, "", "#576a65", 2.4);
      shape(c, "M-11 -3v-31l10 -3v31Z", "#f5dfae");
      shape(c, "M-1 -37l8 5v28l-8 -2Z", "#b4936b");
      shape(c, "M-14 -34l10 -15 12 16 -9 -3Z", "#506d65");
      oval(c, -6, -25, 3.4, 4, "#ffefc8");
      shape(c, "M-6 -28v3l2 1", "", ink, 1.1);
      break;
    case 2: // Glasshouse in the botanical garden.
      tree(c, -25, 0, 1.1, false, 0);
      tree(c, 26, -3, 1.4, false, 1);
      shape(c, "M-21 -2v-9q0 -13 18 -17 17 -4 22 9v11Z", "#b5d3ba");
      shape(
        c,
        "M-21 -11l40 -9m-39 13l39 -9M-14 -4v-11q2 -10 10 -12M-5 -6v-14q3 -9 6 -9M5 -9v-12q0 -7 -2 -8M14 -11v-12",
        "",
        "#657d69",
        1.3,
      );
      shape(c, "M-23 -2l43 -10v4L-23 3Z", "#e1c89c");
      tree(c, -31, 4, 0.7, false, 1);
      break;
    case 3: // Copper dome and colonnade.
      shape(c, "M-34 0l36 -9 34 8 -39 11Z", "#d6bea0");
      shape(c, "M-28 -3v-20l30 -8 29 10v17L1 3Z", "#e7d2af");
      shape(c, "M1 -31l30 10v17L1 3Z", "#c6a383");
      shape(c, "M-32 -23l32 -15 34 16 -32 -4Z", "#ae7661");
      shape(c, "M-15 -26v-7l16 -5 14 6v6l-14 -4Z", "#f2d8a9");
      shape(c, "M-15 -33q1 -16 16 -21 14 4 14 22l-14 -5Z", "#709a90");
      shape(c, "M0 -54v-6m0 8q-6 9 -6 16", "", ink, 1.25);
      for (let i = 0; i < 5; i++)
        shape(c, `M${-23 + i * 5} ${-18 - i}v13`, "", "#fff1d6", 2.6);
      shape(c, "M7 -19v10m8 -8v9m7 -6v8", "", "#746d60", 2);
      break;
    case 4: // Working docks and a lattice jib crane.
      shape(c, "M-33 0l46 -13 27 9 -47 14Z", "#caa174");
      for (let i = 0; i < 5; i++)
        shape(c, `M${-27 + i * 9} ${1 - i * 2.5}l17 5`, "", "#9c7855", 1);
      house(c, -12, -4, 0.85, "#a9654f");
      shape(c, "M5 -5l3 -36 5 -1 4 34ZM7 -9l7 -9 -6 -8 6 -9", "", "#6e6550", 2);
      shape(c, "M-4 -37l10 -7 36 -15 1 4 -34 20Z", "#d8a654");
      shape(c, "M9 -43l5 5 6 -9 5 3 7 -8", "", "#836841", 1.1);
      shape(c, "M37 -53v23q0 4 -4 2", "", ink, 1.25);
      break;
    default: // A striped lighthouse with a glazed lantern.
      shape(c, "M-22 2l7 -9 18 -5 16 8 7 6 -24 7Z", "#b6ad97");
      shape(c, "M-9 1l4 -47 9 -1 6 45 -8 4Z", "#f6e6c5");
      shape(
        c,
        "M-7 -16l14 -3 1 9 -16 3ZM-5 -36l10 -2 1 8 -12 1Z",
        "#bb6955",
        "",
        0,
      );
      shape(c, "M-8 -46l18 -3v4l-18 3Z", "#466662");
      shape(c, "M-5 -46v-11l9 -2 4 2v10Z", "#efd58a");
      shape(c, "M-9 -56l10 -10 10 8 -7 -1Z", "#587772");
      shape(c, "M1 -58v10M0 -66v-4M-1 0v-6", "", ink, 1.4);
      break;
  }
}

const terrain = {
  river: {
    land: "#e9dcb8",
    water: "#85b5b3",
    grove: "#b6c493",
    roof: "#bc7960",
  },
  islands: {
    land: "#e6d3a8",
    water: "#7db7bc",
    grove: "#aec394",
    roof: "#be7661",
  },
  valley: {
    land: "#d8d7b1",
    water: "#9cbfba",
    grove: "#a5b48a",
    roof: "#9c7763",
  },
};
type Node = Point & { id: number; slot: number };
export function createScene(
  kind: Landscape,
  seed: number,
  nodes: Node[],
  links: number[][],
) {
  const rand = generator(seed),
    phase = rand() * tau;
  const p = terrain[kind];
  const routeY = (x: number) => {
    if (kind === "river")
      return (
        304 + 74 * Math.sin(x / 205 + phase) + 35 * Math.sin(x / 106 + 0.8)
      );
    if (kind === "valley")
      return 370 - x * 0.19 + 43 * Math.sin(x / 112 + phase);
    return 22 + 11 * Math.sin(x / 112 + phase);
  };
  const river = new Path2D();
  for (let x = -10; x <= WIDTH + 10; x += 5) {
    if (x === -10) river.moveTo(x, routeY(x));
    else river.lineTo(x, routeY(x));
  }
  const riverWidth = kind === "valley" ? 23 : 62;
  const islands = nodes.map((n) => ({
    x: n.x,
    y: n.y,
    rx: 75 + rand() * 37,
    ry: 49 + rand() * 18,
    k: rand() * tau,
  }));
  const coastRadius = (angle: number, k: number) =>
    1 + 0.1 * Math.sin(3 * angle + k) + 0.07 * Math.cos(5 * angle - k);
  function islandPath(i: (typeof islands)[number]) {
    const path = new Path2D();
    for (let n = 0; n < 96; n++) {
      const a = (tau * n) / 96,
        r = coastRadius(a, i.k);
      const x = i.x + r * i.rx * Math.cos(a),
        y = i.y + r * i.ry * Math.sin(a);
      if (n === 0) path.moveTo(x, y);
      else path.lineTo(x, y);
    }
    path.closePath();
    return path;
  }
  function onLand(x: number, y: number) {
    if (kind !== "islands") return Math.abs(y - routeY(x)) > riverWidth / 2 + 9;
    return islands.some((i) => {
      const dx = (x - i.x) / i.rx,
        dy = (y - i.y) / i.ry;
      return Math.hypot(dx, dy) < coastRadius(Math.atan2(dy, dx), i.k) - 0.12;
    });
  }
  const points = new Map(nodes.map((n) => [n.id, n]));
  const connections = links.flatMap(([a, b]) => {
    const from = points.get(a),
      to = points.get(b);
    return from && to ? [{ from, to }] : [];
  });
  const nearRoad = (x: number, y: number) =>
    connections.some(({ from, to }) => {
      const dx = to.x - from.x,
        dy = to.y - from.y;
      const t = Math.max(
        0,
        Math.min(
          1,
          ((x - from.x) * dx + (y - from.y) * dy) / (dx * dx + dy * dy || 1),
        ),
      );
      return Math.hypot(x - from.x - t * dx, y - from.y - t * dy) < 12;
    });
  const objects: {
    x: number;
    y: number;
    size: number;
    type: "house" | "tree";
    tint: number;
  }[] = [];
  for (let group = 0; group < 110; group++) {
    const originX = rand() * 960 - 30,
      originY = rand() * 560 - 20;
    const trees = rand() < (kind === "valley" ? 0.8 : 0.45);
    for (let row = 0; row < 2; row++)
      for (let col = 0; col < 3; col++) {
        const x = originX + col * 19 + row * 7 + rand() * 5,
          y = originY - col * 5 + row * 12 + rand() * 6;
        if (
          !onLand(x, y) ||
          nearRoad(x, y) ||
          nodes.some((n) => Math.hypot(n.x - x, (n.y - y) * 1.2) < 48)
        )
          continue;
        objects.push({
          x,
          y,
          type: trees ? "tree" : "house",
          size: 0.42 + rand() * 0.35,
          tint: rand(),
        });
      }
  }
  objects.sort((a, b) => a.y - b.y);
  function background(c: Pen) {
    const random = generator(seed);
    c.lineJoin = "round";
    c.lineCap = "round";
    c.fillStyle = kind === "islands" ? p.water : p.land;
    c.fillRect(0, 0, WIDTH, HEIGHT);
    if (kind === "islands") {
      for (const i of islands) {
        const path = islandPath(i);
        stroke(c, path, "#96c5c4", 23);
        stroke(c, path, "#c3d7c7", 12);
        stroke(c, path, "#eadfbe", 5);
        c.fillStyle = p.land;
        c.fill(path);
      }
    } else {
      // Nested contour lines in the hills; terraced fields on the floodplain.
      for (let i = 0; i < (kind === "valley" ? 25 : 53); i++) {
        const x = random() * 980 - 40,
          y = random() * 580 - 30,
          rx = 32 + random() * 67,
          ry = 16 + random() * 35;
        if (kind === "valley") {
          for (let j = 0; j < 5; j++) {
            c.beginPath();
            c.ellipse(
              x,
              y,
              rx * (1 - j * 0.16),
              ry * (1 - j * 0.16),
              -0.38,
              0,
              tau,
            );
            if (j === 0) {
              c.fillStyle = "#bcc59b80";
              c.fill();
            }
            c.strokeStyle = "#9eaf855c";
            c.lineWidth = 1;
            c.stroke();
          }
        } else {
          c.save();
          c.translate(x, y);
          c.rotate(-0.3);
          c.fillStyle = i % 3 === 0 ? "#c5cda3" : "#ddcda5";
          c.fillRect(0, 0, rx, ry);
          c.strokeStyle = "#eee0bf";
          c.lineWidth = 1;
          for (let k = 4; k < rx; k += 5) {
            c.beginPath();
            c.moveTo(k, 1);
            c.lineTo(k, ry - 1);
            c.stroke();
          }
          c.restore();
        }
      }
      stroke(c, river, "#b9c4a0", riverWidth + 15);
      stroke(c, river, "#f3e2bd", riverWidth + 7);
      stroke(c, river, p.water, riverWidth);
    }
    if (kind === "valley") {
      // A distant ridge gives the cableway a skyline of its own.
      for (let i = 0; i < 10; i++) {
        const x = i * 109 - 40 + random() * 30, base = 38 + random() * 40;
        at(c, x, base, .7 + random() * .52, () => {
          shape(c, "M-72 6l28 -22 13 -23 10 6 21 -43 20 34 8 -4 17 26 24 24Z", "#a1ad96", "#7c8b79", 1);
          shape(c, "M0 -76l-5 46 -14 18 19 -9 20 14 32 9 -24 -26 -9 -19Z", "#82998c", "", 0);
          shape(c, "M-14 -48l14 -28 17 30 -10 -5 -7 9 -5 -13Z", "#e8e8d2", "", 0);
          shape(c, "M-37 -15l6 -15m9 18l11 -13m34 13l-6 -10", "", "#798b75", 1.2);
        });
      }
    }
    for (let i = 0; i < 1900; i++) {
      const x = random() * WIDTH,
        y = random() * HEIGHT;
      if (!onLand(x, y)) {
        c.fillStyle = "#e4f0dc66";
        c.fillRect(x, y, 2 + random() * 6, 0.95);
      } else {
        c.fillStyle = "#887e6040";
        c.fillRect(x, y, 1.1, 0.8);
      }
    }
    for (const o of objects) {
      if (o.type === "house")
        house(c, o.x, o.y, o.size, o.tint > 0.55 ? p.roof : "#8e9990");
      else
        tree(c, o.x, o.y, o.size, kind === "valley" || o.tint > 0.84, o.tint);
    }
    for (const { from, to } of connections) {
      const road = new Path2D();
      road.moveTo(from.x, from.y);
      road.lineTo(to.x, to.y);
      stroke(c, road, "#8d876c", 7);
      stroke(c, road, "#f6e7c4", 5);
      c.setLineDash([2, 6]);
      stroke(c, road, "#9b8868", 1.1);
      c.setLineDash([]);
    }
    for (const n of [...nodes].sort((a, b) => a.y - b.y))
      at(c, n.x, n.y, 1.15, () => landmark(c, n.slot));
  }
  function boat(c: Pen, x: number, y: number, angle: number, size: number) {
    c.save();
    c.translate(x, y);
    c.rotate(angle);
    c.scale(size, size);
    shape(c, "M-9 -2q-7 0 -16 -4m16 8q-8 0 -18 3", "", "#e3eee0", 1.4);
    shape(c, "M-11 -4Q3 -6 13 0Q2 6 -11 4Z", "#614c43", "#524b43", 1);
    shape(c, "M-9 -2l12 -1 4 3 -4 2 -12 0Z", "#f2dfa5", "", 0);
    shape(c, "M-4 -3h6v5h-6Z", "#b76b4f", "", 0);
    c.restore();
  }
  function motion(c: Pen, time: number) {
    const u = (time / 54 + phase / tau) % 1;
    if (kind === "valley" && connections.length) {
      const { from, to } = connections[seed % connections.length];
      const t = 0.5 - 0.45 * Math.cos(tau * u),
        dx = to.x - from.x,
        dy = to.y - from.y;
      const x = from.x + t * dx,
        y = from.y + t * dy;
      shape(c, `M${x} ${y - 2}v8`, "", ink, 1.2);
      at(c, x, y + 9, 0.9, () => {
        shape(c, "M-8 -4l14 -2 3 3v8l-14 2 -3 -3Z", "#b65c43");
        shape(c, "M-6 -3l11 -1v4l-11 1Z", "#d9e0c6", "", 0);
      });
    } else {
      const x = -24 + u * (WIDTH + 48);
      boat(c, x, routeY(x), Math.atan2(routeY(x + 2) - routeY(x - 2), 4), 1);
    }
    const bx = ((time * 5.3 + (seed % WIDTH)) % (WIDTH + 80)) - 40,
      by = 80 + 18 * Math.sin(bx / 90 + phase);
    for (let i = 0; i < 3; i++) {
      const flap = 2 + 1.4 * Math.sin(time * 2 + i);
      at(c, bx - i * 15, by + (i === 1 ? -7 : i * 2), 0.9, () =>
        shape(
          c,
          `M-5 0q3 -${flap} 5 1q2 -${flap + 1} 5 -1`,
          "",
          "#535f54",
          1.45,
        ),
      );
    }
  }
  return { background, motion };
}
