// MK-SH-V decision gate, an anatomy figure. Build:
//   node demo/figure/build.mjs [--light]
// Reads demo_cases.json (three REAL calibrated cases from E42 per-case files;
// every number the figure shows comes from it) and writes decision-gate.html.
// Depends on the anatomy kit (C:\Users\ahmia\.claude\skills\anatomy); only the
// built HTML ships in the repo and the Space.

import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { pathToFileURL } from "node:url";
const SKILL = process.env.ANATOMY_SKILL || "C:/Users/ahmia/.claude/skills/anatomy";
const k = await import(pathToFileURL(SKILL + "/kit/iso-kit.mjs").href);
const { kitScript } = await import(pathToFileURL(SKILL + "/scripts/inline-kit.mjs").href);

const HERE = dirname(fileURLToPath(import.meta.url));
const theme = process.argv.includes("--light") ? "light" : "dark";
const CASES = JSON.parse(readFileSync(join(HERE, "..", "demo_cases.json"), "utf-8"));
// ordered left to right on the carriage: abstain, edge, high (the story ascends)
const CASE_ORDER = ["abstain", "edge", "high"];
const CASE_LIST = CASE_ORDER.map((key) => ({ key, ...CASES[key] }));

const WIDTH = 660;
const HEIGHT = 420;
const TAU = 0.55;

const PLATE = { x: -150, y: -70, w: 300, d: 140, r: 8 };
const PLATE_H = 12;
const FOOT_H = 5;

// page path (left half): rails, carriage with three case pages, head above
const RAIL_FROM = -140;
const RAIL_TO = -4;
const HEAD_X = -70;            // the reading head sits here; the case under it is live
const CASE_W = 26;
const CASE_STEP = 40;
const CARRIAGE = { x: -120, y: -26, w: 106, d: 52, r: 3 };
const CASE_XS = CASE_LIST.map((_, i) => CARRIAGE.x + 14 + CASE_STEP * i + CASE_W / 2); // local case centres
const STOP_CARRIAGE_X = CASE_XS.map((cx) => HEAD_X - cx);   // carriage x putting case i under the head

// single tower behind the rails
const TOWER = [-70, -50];
const TOWER_TOP = 96;

// probability dial (right)
const DIAL = [56, -26];
const DIAL_R = 21;
const DIAL_Z = 52;
const DIAL_H = 10;
const DIAL_TOP = DIAL_Z + DIAL_H;

// output slot with the gate (front right)
const SLOT = { x: 66, y: 34, w: 58, d: 30 };
const HINGE = [68, 32];

const P = k.fitProjection(
  [...k.boxCorners(PLATE, -PLATE_H - FOOT_H, 0), [TOWER[0], TOWER[1], TOWER_TOP], [DIAL[0] + DIAL_R + 10, DIAL[1], DIAL_TOP], [SLOT.x + SLOT.w, SLOT.y + SLOT.d, 6]],
  WIDTH,
  HEIGHT,
  { pad: 26, azimuth: 40 },
);

const at = (point) => k.iso(point, P);
const S = (paths, style) => k.solidSvg(paths, style);
const L = (d, style) => k.lineSvg(d, style);
const D = (points, style) => k.dotsSvg(points, style);
const byDepth = (points) => [...points].sort((a, b) => k.depthOf([a[0], a[1], 0], P) - k.depthOf([b[0], b[1], 0], P));

function plate() {
  const out = [`<path class="iso-halo" d="${k.haloOf(PLATE, -PLATE_H - FOOT_H, PLATE_H, P)}"/>`];
  for (const [x, y] of byDepth(k.corners(PLATE, 24))) {
    out.push(S(k.cylinder(x, y, 7, -PLATE_H - FOOT_H, 1.6, P, 28), { tone: "lo" }));
    out.push(S(k.cylinder(x, y, 5.4, -PLATE_H - FOOT_H + 1.6, FOOT_H - 1.6, P, 28), { tone: "mid" }));
  }
  out.push(S(k.slabOf(PLATE, -PLATE_H, PLATE_H, P, 8, 1.6), { tone: "mid" }));
  out.push(L(k.planOutline(k.insetPlan(PLATE, 5), 0, P), { tone: "lo" }));
  const screws = k.corners(PLATE, 10);
  out.push(L(screws.map(([x, y]) => k.ring(x, y, 2.2, 0, P, 16)).join(""), { tone: "lo" }));
  out.push(D(screws.map(([x, y]) => at([x, y, 0])), { size: 0.5 }));
  // label plate, back left: name + two rule lines (engraved, not floating)
  const label = { x: 92, y: -68, w: 54, d: 16, r: 2 };
  out.push(S(k.slabOf(label, 0, 0.8, P, 2), { tone: "lo" }));
  out.push(k.faceTextSvg(k.topMatrix([label.x + 6, label.y + 9.5], 0.8, P, "x"), "MK-SH-V", { size: 6.2, tone: "mid", anchor: "start" }));
  out.push(k.faceTextSvg(k.topMatrix([label.x + 6, label.y + 4], 0.8, P, "x"), "decision gate", { size: 2.4, tone: "lo", anchor: "start" }));
  out.push(L(k.lineOnTop([label.x + 2, label.y + 13.5], [label.x + label.w - 2, label.y + 13.5], 0.8, P), { tone: "lo" }));
  // ruler strip along the front edge: the plate is a measured surface
  const strip = { x: -146, y: 47, w: 178, d: 7, r: 1 };
  out.push(S(k.slabOf(strip, 0, 0.6, P, 3), { tone: "lo", crease: "none" }));
  const ruler = k.topTicks(strip.x + 4, strip.x + strip.w - 4, 5, 5, 50.5, 0.6, [2, 3.6], P);
  out.push(L(ruler.minor, { tone: "lo" }), L(ruler.major, { tone: "mid" }));
  out.push(D([at([strip.x + 2, strip.y + 3.5, 0.6]), at([strip.x + strip.w - 2, strip.y + 3.5, 0.6])], { size: 0.45 }));
  return out.join("");
}

function rails() {
  const out = [];
  for (const y of [-22, 16]) {
    out.push(S(k.slabOf({ x: RAIL_FROM, y, w: RAIL_TO - RAIL_FROM, d: 5, r: 1 }, 0, 3, P, 3, 0.4), { tone: "mid" }));
    out.push(D([RAIL_FROM + 6, RAIL_TO - 6].map((x) => at([x, y + 2.5, 3])), { size: 0.4 }));
  }
  for (const x of [RAIL_FROM - 5, RAIL_TO]) {
    out.push(S(k.slabOf({ x, y: -25, w: 5, d: 50, r: 1.2 }, 0, 7, P, 3, 0.5), { tone: "mid" }));
    out.push(D([at([x + 2.5, -20, 7]), at([x + 2.5, 20, 7])], { size: 0.45 }));
  }
  return out.join("");
}

function tower() {
  const out = [];
  const [tx, ty] = TOWER;
  const foot = { x: tx - 16, y: ty - 11, w: 32, d: 22, r: 3 };
  out.push(S(k.slabOf(foot, 0, 9, P, 4, 0.8), { tone: "mid" }));
  out.push(L(k.planOutline(k.insetPlan(foot, 2.4), 9, P), { tone: "faint" }));
  out.push(D([at([tx - 12, ty + 7, 9]), at([tx + 12, ty + 7, 9]), at([tx - 12, ty - 7, 9])], { size: 0.45 }));
  out.push(S(k.cylinder(tx, ty, 5, 9, TOWER_TOP - 26 - 9, P, 28), { tone: "mid", crease: "none" }));
  for (const z of [26, 40, 54, 68, 82]) {  // cooling fins: the tower is one machined column
    out.push(S(k.slabOf({ x: tx - 8, y: ty - 7, w: 16, d: 14, r: 2 }, z, 1.4, P, 2, 0.4), { tone: "mid" }));
  }
  out.push(S(k.cylinder(tx, ty, 5, TOWER_TOP - 4, 4, P, 28), { tone: "mid" }));
  out.push(S(k.cylinder(tx, ty, 5.6, TOWER_TOP, 2, P, 28), { tone: "mid" }));
  out.push(L(k.ring(tx, ty, 2.4, TOWER_TOP + 2, P, 16), { tone: "lo" }));
  out.push(D([at([tx, ty, TOWER_TOP + 2])], { size: 0.5 }));
  // hold lamp on the cap front edge: lit (accent) while the gate is closed
  const [lx, ly] = at([tx + 3, ty + 6, TOWER_TOP]);
  out.push(`<circle data-part="lamp" cx="${lx}" cy="${ly}" r="1.6" class="iso-dot" data-tone="lo"/>`);
  // arm forward over the page path, then the reading head hangs from it
  const arm = { x: tx - 6, y: ty + 10, w: 12, d: 42, r: 2 };
  out.push(S(k.slabOf(arm, TOWER_TOP - 6, 4, P, 3, 0.5), { tone: "mid" }));
  out.push(D([at([tx, ty + 16, TOWER_TOP - 2]), at([tx, arm.y + arm.d - 4, TOWER_TOP - 2])], { size: 0.4 }));
  out.push(S(k.cylinder(tx, 2, 2.4, 38, TOWER_TOP - 6 - 38, P, 20), { tone: "mid", crease: "none" }));
  out.push(L([40, 50].map((z) => k.sideArc(tx, 2, 2.4, z, P, 12)).join(""), { tone: "faint" }));
  const head = { x: tx - 12, y: -10, w: 24, d: 22, r: 2.5 };
  out.push(S(k.slabOf(head, 26, 12, P, 4, 0.6), { tone: "mid" }));
  out.push(L(k.planOutline(k.insetPlan(head, 2.2), 38, P), { tone: "faint" }));
  out.push(D([at([tx - 8, 6, 38]), at([tx + 8, 6, 38]), at([tx, -6, 38])], { size: 0.4 }));
  // three marker pins, one per option slot on the page below; tips ~1.5 above the paper
  for (const px of [tx - 7, tx, tx + 7]) {
    out.push(S(k.cylinder(px, -1, 1.5, 9, 17, P, 14), { tone: "hi", crease: "none", lit: true }));
    out.push(S(k.cylinder(px, -1, 1.9, 22, 4, P, 14), { tone: "hi" }));
  }
  return out.join("");
}

function carriage() {
  const out = [`<path class="iso-halo" d="${k.haloOf(k.insetPlan(CARRIAGE, 3), 3, 0.1, P)}" style="filter:blur(4px)"/>`];
  out.push(S(k.slabOf(CARRIAGE, 0, 5, P, 4, 0.6), { tone: "mid" }));
  out.push(D([-104, -84, -44, -24].map((x) => at([x, CARRIAGE.y + CARRIAGE.d - 2.4, 5])), { size: 0.4, tone: "lo" }));
  // three case pages: bright paper, each with its own layout of faint content lines
  CASE_LIST.forEach((_, i) => {
    const plan = { x: CARRIAGE.x + 14 + CASE_STEP * i, y: CARRIAGE.y + 8, w: CASE_W, d: 36, r: 1 };
    out.push(S(k.slabOf(plan, 5, 2.6, P, 1.2, 0.3), { tone: "hi", crease: "lo" }));
    const rows = i === 2 ? 5 : i === 1 ? 4 : 3;   // denser page = more confident read? no: just variety
    for (let rI = 0; rI < rows; rI++) {
      const y = plan.y + 6 + rI * 5.4;
      const wLine = (plan.w - 8) * (0.55 + 0.4 * ((rI * 7 + i * 3) % 3) / 2);
      out.push(L(k.lineOnTop([plan.x + 4, y], [plan.x + 4 + wLine, y], 7.7, P), { tone: "faint" }));
    }
    out.push(L(k.lineOnTop([plan.x + 4, plan.y + 3.2], [plan.x + plan.w - 4, plan.y + 3.2], 7.7, P), { tone: "lo" }));
  });
  // handle knob at the front
  const handle = [CARRIAGE.x + CARRIAGE.w - 8, CARRIAGE.y + CARRIAGE.d + 4];
  out.push(S(k.slabOf({ x: handle[0] - 2, y: CARRIAGE.y + CARRIAGE.d - 1, w: 4, d: 4, r: 1 }, 1, 3, P, 2), { tone: "mid", crease: "none" }));
  out.push(S(k.cylinder(handle[0], handle[1] + 1.4, 3, 0, 5, P, 20), { tone: "mid" }));
  out.push(L(k.knurl(handle[0], handle[1] + 1.4, 3, 0, 5, 18, P), { tone: "lo" }));
  return out.join("");
}

function dial() {
  const out = [];
  const [x, y] = DIAL;
  out.push(S(k.cylinder(x, y, 4.6, 0, 1.6, P, 24), { tone: "mid" }));
  out.push(L(k.ring(x, y, 4.6, 0, P, 24), { tone: "lo" }));
  out.push(S(k.cylinder(x, y, 2.9, 1.6, DIAL_Z - 1.6, P, 20), { tone: "mid" }));
  out.push(S(k.cylinder(x, y, DIAL_R, DIAL_Z, DIAL_H, P, 64, 0.8), { tone: "hi", crease: "lo" }));
  out.push(L(k.knurl(x, y, DIAL_R, DIAL_TOP - 3.4, DIAL_TOP, 84, P, 0.3), { tone: "lo" }));
  out.push(L(k.ring(x, y, DIAL_R - 2.2, DIAL_TOP, P, 64), { tone: "lo" }));
  const ticks = k.radialTicks(x, y, DIAL_R - 3, 56, 5, DIAL_TOP, [1.3, 2.8], P);
  out.push(L(ticks.minor, { tone: "lo" }), L(ticks.major, { tone: "mid" }));
  out.push(L(k.ring(x, y, 1.6, DIAL_TOP, P, 16), { tone: "mid" }));
  const ang = (p) => (-225 + p * 270) * Math.PI / 180;
  const notch = (p, r0, r1) => k.lineOnTop([x + Math.sin(ang(p)) * r0, y - Math.cos(ang(p)) * r0], [x + Math.sin(ang(p)) * r1, y - Math.cos(ang(p)) * r1], DIAL_TOP + 0.2, P);
  out.push(`<path class="iso-line" d="${notch(TAU, DIAL_R - 6.5, DIAL_R - 2.5)}" style="stroke:var(--iso-acc,#e8a13c);stroke-width:1.4"/>`);
  const dialText = (pv, r, s, txt, tone) => k.faceTextSvg(k.topMatrix([x + Math.sin(ang(pv)) * r, y - Math.cos(ang(pv)) * r], DIAL_TOP, P, "x"), txt, { size: s, tone });
  out.push(dialText(0, DIAL_R - 9.5, 5.4, "0", "mid"));
  out.push(dialText(1, DIAL_R - 9.5, 5.4, "1", "mid"));
  out.push(dialText(TAU, DIAL_R - 9, 5.4, "T", "hi"));
  return out.join("");
}

function outputSlot() {
  const out = [];
  // frame: two side walls and a back wall, opening toward the viewer
  for (const y of [SLOT.y, SLOT.y + SLOT.d - 5]) {
    out.push(S(k.slabOf({ x: SLOT.x, y, w: SLOT.w, d: 5, r: 1 }, 0, 15, P, 2, 0.5), { tone: "mid" }));
    out.push(L(k.lineOnTop([SLOT.x + 2, y + 2.5], [SLOT.x + SLOT.w - 2, y + 2.5], 15, P), { tone: "faint" }));
  }
  out.push(S(k.slabOf({ x: SLOT.x + SLOT.w - 5, y: SLOT.y, w: 5, d: SLOT.d, r: 1 }, 0, 15, P, 2, 0.5), { tone: "mid" }));
  // the choice card drops inside the frame when the gate opens (live part)
  const card = { x: SLOT.x + 8, y: SLOT.y + 11, w: 30, d: 14, r: 1 };
  out.push(`<g data-part="card">${S(k.slabOf(card, 2, 3.5, P, 1.2, 0.3), { tone: "hi", crease: "lo" })}
    ${L(k.lineOnTop([card.x + 4, card.y + 11], [card.x + card.w - 4, card.y + 11], 5.7, P), { tone: "faint" })}
    ${L(k.lineOnTop([card.x + 4, card.y + 6], [card.x + card.w - 12, card.y + 6], 5.7, P), { tone: "faint" })}</g>`);
  // tall hinge post + gate lever lying across the opening when closed (live part)
  out.push(S(k.cylinder(HINGE[0], HINGE[1], 2.2, 0, 19, P, 16), { tone: "mid" }));
  out.push(S(k.cylinder(HINGE[0], HINGE[1], 3, 19, 2.4, P, 16), { tone: "mid" }));
  const armLen = SLOT.w - 4;
  const accLine = `<path class="iso-line" d="${k.lineOnTop([HINGE[0] + 5, HINGE[1]], [HINGE[0] + armLen - 3, HINGE[1]], 18.2, P)}" style="stroke:var(--iso-acc,#e8a13c);stroke-width:1.2"/>`;
  out.push(`<g data-part="gate">${S(k.slabOf({ x: HINGE[0] + 2, y: HINGE[1] - 2.5, w: armLen, d: 5, r: 1.5 }, 14, 4, P, 1.5, 0.4), { tone: "mid" })}${accLine}</g>`);
  out.push(D([at([HINGE[0], HINGE[1], 21.5])], { size: 0.5 }));
  return out.join("");
}

const needleAt = (p) => {
  const rad = (-225 + p * 270) * Math.PI / 180;
  const [x, y] = DIAL;
  return k.lineOnTop([x - Math.sin(rad) * 3.2, y + Math.cos(rad) * 3.2], [x + Math.sin(rad) * (DIAL_R - 4.5), y - Math.cos(rad) * (DIAL_R - 4.5)], DIAL_TOP + 0.2, P);
};

const startCase = 2; // open on the confident case
const svg = k.figureSvg({
  width: WIDTH,
  height: HEIGHT,
  label:
    "MK-SH-V as a bench instrument. A single tower reads the page under its marker pins; the dial shows the calibrated probability of the chosen option; the gate arm holds the output slot shut unless the needle clears the tau mark at 0.55. The three pages are three real calibration cases with their measured probabilities.",
  body: [
    plate(),
    tower(),
    rails(),
    `<g data-part="carriage" transform="${k.translateAlong(P, [STOP_CARRIAGE_X[startCase], 0, 0])}">${carriage()}</g>`,
    dial(),
    outputSlot(),
    `<path data-part="needle" class="iso-line" data-tone="lit" data-free="start" d="${needleAt(CASE_LIST[startCase].pmax)}" style="stroke-width:1.6"/>`,
  ],
});

const live = `
const P = ${JSON.stringify(P)};
const CASES = ${JSON.stringify(CASE_LIST.map((c) => ({ id: c.id, key: c.key, p: c.pmax, choice: c.choice, correct: c.correct })))};
const STOPS = ${JSON.stringify(STOP_CARRIAGE_X)};
const HEAD = ${HEAD_X}, TAU = ${TAU};
const DIAL = ${JSON.stringify(DIAL)}, DIAL_R = ${DIAL_R}, DIAL_TOP = ${DIAL_TOP};
const stage = document.querySelector(".iso-stage");
const svg = stage.querySelector("svg");
const carriage = svg.querySelector('[data-part="carriage"]');
const needle = svg.querySelector('[data-part="needle"]');
const gate = svg.querySelector('[data-part="gate"]');
const card = svg.querySelector('[data-part="card"]');
const lamp = svg.querySelector('[data-part="lamp"]');
const readout = document.querySelector("[data-readout]");
const still = matchMedia("(prefers-reduced-motion: reduce)");
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const needleAt = ${needleAt.toString().replace("k.lineOnTop", "lineOnTop")};
// nearest case index by carriage position (case i is under the head when cx = STOPS[i])
const nearest = (cx) => STOPS.reduce((best, s, i) => Math.abs(cx - s) < Math.abs(cx - STOPS[best]) ? i : best, 0);
let cx = STOPS[${startCase}], target = cx;
let needleP = { at: CASES[${startCase}].p, rate: 0 };
let cardZ = 0, cardTarget = 0, gateOpen = null;
let frame = 0, last = 0, clock = 0, touring = true, visible = false, idleTimer = 0;
const tourAt = (t) => STOPS[Math.floor(t / 2.4) % STOPS.length];
function draw() {
  carriage.setAttribute("transform", translateAlong(P, [cx, 0, 0]));
  needle.setAttribute("d", needleAt(needleP.at));
  const p = needleP.at;
  const open = p >= TAU;
  if (open !== gateOpen) {
    gateOpen = open;
    const [hx, hy] = ${JSON.stringify(at([HINGE[0], HINGE[1], 10]))};
    gate.setAttribute("transform", open ? \`rotate(-64, \${hx}, \${hy})\` : "");
    lamp.setAttribute("data-tone", open ? "lo" : "acc");
    cardTarget = open ? 1 : 0;
  }
  card.setAttribute("transform", translateAlong(P, [0, 0, 2 + cardZ * 5]) + (cardZ < 0.05 ? \` translate(0, \${-cardZ * 900})\` : ""));
  card.style.opacity = String(clamp(cardZ * 3, 0, 1));
  const c = CASES[nearest(target)] || CASES[0];
  const tag = c.correct ? "" : " (pick would err)";
  const text = open
    ? "choice " + c.choice + " · p " + p.toFixed(2) + " >= tau" + tag
    : "abstain · p " + p.toFixed(2) + " < tau 0.55" + tag;
  if (readout.textContent !== text) readout.textContent = text;
  stage.setAttribute("aria-valuenow", String(Math.round(p * 100)));
  stage.setAttribute("aria-valuetext", text);
}
function tick(now) {
  const dt = last ? Math.min((now - last) / 1000, 1 / 30) : 1 / 60;
  last = now;
  const calm = still.matches;
  if (touring) { clock += dt; target = tourAt(clock); }
  cx = calm ? target : cx + (target - cx) * (1 - Math.exp(-dt / 0.3));
  if (Math.abs(target - cx) < 0.01) cx = target;
  const want = CASES[nearest(cx)].p;
  const stiff = 300, drag = calm ? 2 * Math.sqrt(300) : 22;
  for (let i = 0; i < 4; i++) { needleP.rate += ((want - needleP.at) * stiff - needleP.rate * drag) * dt / 4; needleP.at += needleP.rate * dt / 4; }
  if (Math.abs(want - needleP.at) < 0.004 && Math.abs(needleP.rate) < 0.004) { needleP.at = want; needleP.rate = 0; }
  cardZ = calm ? cardTarget : cardZ + (cardTarget - cardZ) * (1 - Math.exp(-dt / 0.2));
  draw();
  const settled = !touring && cx === target && needleP.rate === 0 && needleP.at === want && Math.abs(cardZ - cardTarget) < 0.01;
  frame = visible && !settled ? requestAnimationFrame(tick) : 0;
}
const run = () => { if (!frame && visible) { last = 0; frame = requestAnimationFrame(tick); } };
const steer = (next) => { touring = false; target = clamp(next, STOPS[0] - 14, STOPS[STOPS.length - 1] + 14); clearTimeout(idleTimer); idleTimer = setTimeout(() => { touring = !still.matches; clock = 0; run(); }, 3600); run(); };
new IntersectionObserver((entries) => { visible = entries[entries.length - 1].isIntersecting; if (visible) run(); }, { rootMargin: "120px 0px" }).observe(stage);
if (still.matches) touring = false;
stage.addEventListener("pointermove", (event) => {
  const box = svg.getBoundingClientRect();
  const share = clamp(((event.clientX - box.left) / box.width - 0.08) / 0.5, 0, 1);
  steer(STOPS[0] + share * (STOPS[STOPS.length - 1] - STOPS[0]));
});
stage.addEventListener("pointerleave", () => steer(target));
stage.addEventListener("keydown", (event) => {
  const here = nearest(target);
  let to = null;
  if (event.key === "ArrowRight" || event.key === "ArrowUp") to = clamp(here + 1, 0, STOPS.length - 1);
  if (event.key === "ArrowLeft" || event.key === "ArrowDown") to = clamp(here - 1, 0, STOPS.length - 1);
  if (event.key === "Home") to = 0;
  if (event.key === "End") to = STOPS.length - 1;
  if (to === null) return;
  event.preventDefault();
  steer(STOPS[to]);
});
draw();
`;

const body = k.plateHtml({
  fig: "Fig 1",
  title: "The decision gate",
  hint: "Hover or use arrow keys to slide the next case under the head",
  readout: "choice opt1 · p 0.92 >= tau",
  keys: [
    { mark: "lit", label: "Needle: calibrated probability of the chosen option" },
    { mark: "raised", label: "The case page under the marker pins (three real calibration cases)" },
  ],
  caption:
    "One tower, one pass: the marker pins score the options of the page beneath them, the dial shows the calibrated probability at the shipped temperature, and the gate arm keeps the output slot shut unless the needle clears tau = 0.55. The left page (urgency, p 0.32) is refused; its pick would have been wrong.",
  body: `<div class="iso-stage" tabindex="0" role="slider" aria-label="MK-SH-V decision gate: arrow keys move case to case" aria-valuemin="0" aria-valuemax="100" aria-valuenow="92">${svg}</div>`,
});

writeFileSync(join(HERE, theme === "light" ? "decision-gate-light.html" : "decision-gate.html"), k.pageHtml({ title: "MK-SH-V decision gate", theme, body, script: kitScript() + live }));
console.log("wrote", theme === "light" ? "decision-gate-light.html" : "decision-gate.html");
