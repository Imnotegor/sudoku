// Miniatures share a ground line at y=0. Light comes from the upper left.
type Pen = CanvasRenderingContext2D;
const ink = "#394841";
const stone = "#f3deb1";
const shade = "#c49970";
const glass = "#486f70";
const copper = "#45837c";
const darkCopper = "#305e5c";
const brick = "#b85d44";
const ochre = "#deb15e";

function path(
  c: Pen,
  d: string,
  fill: string | null,
  edge = ink,
  weight = 1.25,
) {
  const p = new Path2D(d);
  if (fill) {
    c.fillStyle = fill;
    c.fill(p);
  }
  if (edge) {
    c.strokeStyle = edge;
    c.lineWidth = weight;
    c.stroke(p);
  }
}
function line(c: Pen, d: string, color = ink, width = 1) {
  path(c, d, null, color, width);
}
function ellipse(
  c: Pen,
  x: number,
  y: number,
  rx: number,
  ry: number,
  fill: string,
  edge?: string,
) {
  c.beginPath();
  c.ellipse(x, y, rx, ry, 0, 0, Math.PI * 2);
  c.fillStyle = fill;
  c.fill();
  if (edge) {
    c.strokeStyle = edge;
    c.lineWidth = 1.15;
    c.stroke();
  }
}
function arch(
  c: Pen,
  x: number,
  base: number,
  w: number,
  h: number,
  color = glass,
) {
  c.save();
  c.translate(x, base);
  path(
    c,
    `M0 0V${-h + w / 2}a${w / 2} ${w / 2} 0 0 1 ${w} 0V0Z`,
    color,
    ink,
    0.85,
  );
  c.restore();
}

function station(c: Pen) {
  // The train hall sits behind the red-brick clock facade.
  path(c, "M-49 -2l72 -7 26 8 -70 6Z", "#caba99");
  path(c, "M-42 -30h57l24 7v16l-22 5h-59Z", "#d7ad81");
  path(c, "M-43 -29q8 -18 17 -15h46q11 2 20 21l-24 -6Z", "#85aca1");
  path(c, "M16 -29q-4 -10 -12 -15h16q11 2 20 21Z", darkCopper);
  line(
    c,
    "M-31 -31q3 -7 8 -11m8 11q3 -7 8 -11m8 11q3 -7 8 -11",
    "#d7e1c6",
    1.3,
  );
  path(c, "M-42 -24h60v19h-60Z", stone);
  for (const x of [-38, -29, -20, 4, 13]) arch(c, x, -8, 5, 11);
  path(c, "M17 -24l21 -5v20l-21 5Z", shade);
  line(c, "M24 -21v10m7 -12v10", "#565d52", 2.5);
  path(c, "M-47 -26h68l-4 4h-63Z", brick);
  path(c, "M-20 -3v-31l4 -7v-9l8 -9 9 9v10l5 7v30Z", brick);
  path(
    c,
    "M-21 -34l6 -9v-9l7 -9 9 10v8l6 9 -3 2 -6 -9v-8l-6 -6 -5 6v8l-6 10Z",
    "#e7c48a",
  );
  ellipse(c, -8, -44, 6.5, 6.5, "#fff1ce", ink);
  line(c, "M-8 -48v4l3 1", ink, 1.4);
  arch(c, -15, -4, 14, 23, darkCopper);
  line(c, "M-17 -5v-16a9 9 0 0 1 18 0v16", "#f3d598", 2.2);
  line(c, "M-14 -18h12m-6 -8v21", "#ccb88c", 1.35);
  path(c, "M-20 -4H5l3 5h-31Z", "#e7cda1");
  line(c, "M-21 0H6", "#917b60");
}

function park(c: Pen) {
  // A Victorian palm house: a central octagonal conservatory and two wings.
  path(c, "M-49 -2q7 -11 26 -11l45 1 24 7 -5 8 -70 2Z", "#9dab76", "#717f5b");
  path(c, "M-37 -7v-24m-1 9 -6 -7m7 0 5 -6", null, "#6e674e", 2.7);
  path(
    c,
    "M-43 -23q-10 -5 -5 -13 -3 -8 5 -11 5 -9 12 -2 10 -1 9 10 6 8 -1 13 -8 8 -20 3Z",
    "#79936a",
    "#526e54",
  );
  path(c, "M-47 -38q2 -7 7 -6 3 -9 10 -2 -8 0 -6 6Z", "#aac08b", "");
  path(c, "M33 -6v-22m0 9 5 -5", null, "#6e674e", 2.4);
  path(
    c,
    "M25 -22q-4 -5 0 -9 -3 -7 3 -9 5 -7 10 0 8 3 4 10 3 7 -4 10Z",
    "#8fa678",
    "#637957",
  );
  path(c, "M-34 -17q2 -15 12 -16h44q11 1 14 14v12h-70Z", "#c4ded0");
  path(c, "M-33 -17h68v11h-68Z", "#95b7a6");
  for (const x of [-28, -21, 19, 27])
    line(c, `M${x} -7v-11q0 -7 3 -12`, "#547c70", 1.15);
  line(c, "M-34 -17h70M-32 -10h65", "#567669", 1.1);
  path(c, "M-18 -8v-19q1 -13 13 -20l5 -5 4 5q12 6 15 19v21l-17 5Z", "#b7d9c8");
  path(c, "M0 -50q13 11 16 22v19l-7 3v-22q-1 -14 -9 -22Z", "#7eac9f", "");
  line(
    c,
    "M0 -50q-10 9 -11 23v17m10 -38q-4 10 -4 22v17m6 -39q5 11 6 22v17",
    "#4d796e",
    1.15,
  );
  path(c, "M-20 -27l19 3 21 -4v3l-21 4 -19 -3Z", "#4e796b");
  line(c, "M-17 -36l16 2 16 -3", "#7c9c87", 1.2);
  path(c, "M-3 -50v-4h5v4Z", ochre);
  line(c, "M0 -54v-4", ink, 1.25);
  path(c, "M-10 -5v-10l8 -6 9 6v10Z", "#ead9b6");
  arch(c, -5, -5, 8, 11, darkCopper);
  path(c, "M-35 -7l18 1 15 3 20 -4 18 -2v4l-37 8 -34 -6Z", "#d6c5a0");
  path(c, "M-8 -4H5l6 7H-13Z", "#ebd9b4", "#8c9175", 0.9);
}

function university(c: Pen) {
  path(c, "M-49 -2l39 -8 57 5 3 5 -48 6Z", "#cfc1a3");
  path(c, "M-42 -23h67l16 5v15l-15 3h-68Z", stone);
  path(c, "M25 -23l16 5v15l-16 3Z", shade);
  path(c, "M-45 -24l8 -9h56l25 12 -19 2Z", "#b47756");
  for (const x of [-37, -29, 19, 27]) arch(c, x, -6, 4.5, 10);
  path(c, "M-18 -29v-11q16 -9 34 0v12Z", "#dbbb8c");
  for (const x of [-13, -5, 4, 11]) arch(c, x, -31, 3.6, 6);
  path(c, "M-21 -39q2 -16 17 -22l4 -5 4 5q14 6 17 22 -19 6 -42 0Z", copper);
  path(c, "M1 -62q10 10 10 24l9 -2q-2 -13 -16 -22Z", darkCopper, "");
  line(c, "M-4 -59q-10 11 -10 19m13 -20q-5 9 -6 21", "#a0c4a9", 1.5);
  path(c, "M-23 -40q21 5 46 0l-1 4q-20 5 -44 0Z", "#c0b389");
  path(c, "M-3 -61v-5l3 -3 3 3v5Z", "#c9b77f");
  line(c, "M0 -69v-4", ink, 1);
  path(c, "M-24 -22H21v20h-45Z", "#705f51");
  for (const x of [-20, -10, 1, 12]) {
    path(c, `M${x} -20h5l-1 16h-3Z`, "#f2ddb3", "#927b61", 0.85);
    line(c, `M${x - 1} -20h7m-7 16h7`, "#f5e2bd", 2);
  }
  path(c, "M-30 -22l27 -15 29 15 -2 3h-52Z", "#f6e1b7");
  path(c, "M-22 -23l19 -10 20 10Z", "#bda17e", "#8e785f", 0.9);
  ellipse(c, -3, -26, 2.2, 2.2, "#7b8f7f");
  path(c, "M-24 -3h46l3 3h-52Zm-4 4h54l3 3h-60Z", "#dcc8a2", "#887d67", 0.85);
}

function port(c: Pen) {
  // The crane, suspended load and tug read as one quayside silhouette.
  path(c, "M-47 -6l54 -11 40 9 -57 14Z", "#c7a77a");
  path(c, "M-47 -6l37 12 57 -14v5l-57 15 -37 -12Z", "#8c775c");
  path(c, "M-25 -11l10 -3 9 3 -10 3Z", "#ddaa6f");
  path(c, "M-25 -11l9 3v8l-9 -3Zm9 3 10 -3v8l-10 3Z", "#ad8457");
  path(c, "M-8 -10l4 -33 7 -3 7 33 -5 1 -5 -22 -4 25Z", "#e2ae58");
  line(c, "M-5 -16l9 -9 -7 -6 6 -6", ink, 1.45);
  path(c, "M-9 -44l7 -8 9 2 31 -18 4 2 -34 25 -8 7Z", "#e4ac55");
  line(c, "M4 -47l7 -1 2 -6 7 -1 2 -5 7 -1 3 -5", ink, 1.2);
  path(c, "M-14 -47l9 -5 8 4v10l-9 4 -8 -4Z", "#bd863e");
  path(c, "M-12 -46l6 2v5l-6 -3Zm8 2 5 -2v5l-5 2Z", "#a4c6bc", ink, 0.85);
  line(c, "M36 -63v25m0 0 -4 5m4 -5 5 4", ink, 1.4);
  path(c, "M29 -33l8 -2 8 3 -8 3Z", "#dfb372");
  path(c, "M29 -33l8 4v8l-8 -4Z", "#bf864a");
  path(c, "M37 -29l8 -3v8l-8 3Z", "#9a693e");
  path(c, "M-40 -7q14 5 31 0l13 -4 -6 9q-14 8 -33 4Z", "#8d4b39");
  path(c, "M-41 -9q16 5 32 0l13 -4 -3 5q-20 10 -40 4Z", "#426d70");
  path(c, "M-30 -10v-8l13 -2 5 5v7Z", "#ecd5a7");
  path(c, "M-32 -18l14 -3 7 5 -4 2 -13 -2Z", "#446b68");
  line(c, "M-25 -15v3m5 -3v3", "#537b7c", 2.2);
  path(c, "M-10 -11l-2 -9 4 -1 2 9Z", "#b96a4e");
  line(c, "M-13 -19l5 -1", ink, 2.4);
}

function lighthouse(c: Pen) {
  path(
    c,
    "M-38 -3l8 -9 12 1 9 -11 12 3 12 -2 10 9 9 2 4 7 -14 4 -16 -2 -16 4Z",
    "#9a9e86",
  );
  path(
    c,
    "M-39 -2l10 -8 9 1 -3 6Zm32 -5 11 -12 10 -1 -4 10Zm15 9 15 -6 13 1 -10 6Z",
    "#d8cdb0",
    "",
  );
  path(c, "M-25 -4v-15l10 -5 12 6v15Z", "#efd7a6");
  path(c, "M-29 -17l9 -12 9 -1 11 12 -5 2 -12 -8 -8 8Z", brick);
  arch(c, -23, -4, 5, 8, darkCopper);
  path(c, "M-9 -5l5 -47 15 -2 9 48q-12 8 -29 1Z", "#f8e5bc");
  path(c, "M5 -52l6 -2 9 48q-4 3 -9 4Z", "#cab898", "");
  path(c, "M-6 -33q9 3 21 -1l2 10q-11 5 -24 1Z", brick, "");
  path(c, "M-8 -15q12 4 26 -2l2 11q-12 6 -29 1Z", brick, "");
  path(c, "M-9 -52l10 3 15 -4 -2 4 -13 4 -9 -3Z", "#415e5c");
  path(c, "M-6 -54v-12l9 -4 9 3v13l-10 3Z", "#e9bc68");
  path(c, "M-3 -64l6 -3v11l-6 1Z", "#fff0b5", "");
  line(c, "M2 -68v15m7 -13v11", "#596555", 1.6);
  path(c, "M-10 -65l11 -11 15 9 -4 2 -10 -3 -9 4Z", darkCopper);
  line(c, "M1 -76v-3", ink, 1.35);
  line(
    c,
    "M-11 -57v6l12 4 16 -5v-6m-28 2 12 4 16 -5m-23 3v5m15 -7v5",
    ink,
    1.05,
  );
  arch(c, 2, -5, 6, 9, glass);
  line(c, "M4 -37v4", ink, 2);
}

const symbols = [station, park, university, port, lighthouse];

// A separate label-sized canvas keeps the architecture in proportion to the
// type and provides a hit target that matches the visible building.
export function districtIcon(slot: number) {
  const canvas = document.createElement("canvas");
  canvas.className = "district-icon";
  canvas.setAttribute("aria-hidden", "true");
  const scale = 4;
  canvas.width = 112 * scale;
  canvas.height = 88 * scale;
  const c = canvas.getContext("2d");
  if (c) {
    c.setTransform(scale, 0, 0, scale, 56 * scale, 80 * scale);
    c.lineJoin = "round";
    c.lineCap = "round";
    ellipse(c, 3, 1, 37, 5, "rgba(67,62,44,.22)");
    symbols[slot - 1]?.(c);
  }
  return canvas;
}
