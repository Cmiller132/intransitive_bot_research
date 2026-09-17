/* Small SVG charts, no library: a line chart with a hover readout, the evaluation ruler, the lineage staircase and
   the sequential test's track. */
(function () {
  const NS = "http://www.w3.org/2000/svg";

  function el(name, attrs, parent) {
    const node = document.createElementNS(NS, name);
    for (const [k, v] of Object.entries(attrs || {})) if (v !== undefined && v !== null) node.setAttribute(k, v);
    if (parent) parent.appendChild(node);
    return node;
  }
  function text(parent, x, y, str, attrs) {
    const t = el("text", { x, y, ...attrs }, parent);
    t.textContent = str;
    return t;
  }
  const minus = (s) => String(s).replace(/^-/, "\u2212");
  const signedInt = (v) => (Math.round(v) > 0 ? "+" : "") + minus(Math.round(v));

  function niceTicks(min, max, count) {
    if (!(isFinite(min) && isFinite(max))) return [];
    if (min === max) { const pad = Math.abs(min) * 0.05 || 1; min -= pad; max += pad; }
    const raw = (max - min) / Math.max(1, count);
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) || raw;
    const ticks = [];
    for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) ticks.push(+v.toPrecision(12));
    return ticks;
  }

  const tip = () => document.getElementById("tip");
  function showTip(evt, html) {
    const t = tip();
    t.innerHTML = html;
    t.hidden = false;
    const pad = 14;
    const r = t.getBoundingClientRect();
    let x = evt.clientX + pad, y = evt.clientY + pad;
    if (x + r.width > window.innerWidth - 8) x = evt.clientX - r.width - pad;
    if (y + r.height > window.innerHeight - 8) y = evt.clientY - r.height - pad;
    t.style.left = x + "px";
    t.style.top = y + "px";
  }
  function hideTip() { tip().hidden = true; }

  /**
   * lineChart(container, {series: [{name, color, values, dash, width, dots}], x, height, format, xTitle, markers,
   *   zeroLine, band: {lo, hi, label}})
   */
  function lineChart(container, opts) {
    container.innerHTML = "";
    const width = Math.max(260, container.clientWidth);
    const height = opts.height || 280;
    const m = { top: 14, right: 16, bottom: opts.xTitle ? 38 : 24, left: opts.left || 58 };
    const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": opts.label || "chart" }, container);
    const xs = opts.x;
    const all = opts.series.flatMap((s) => s.values.filter((v) => typeof v === "number" && isFinite(v)));
    if (!all.length || !xs.length) {
      text(svg, width / 2, height / 2, "No values for this metric", { "text-anchor": "middle" });
      return;
    }
    let lo = Math.min(...all), hi = Math.max(...all);
    if (opts.band) { lo = Math.min(lo, opts.band.lo); hi = Math.max(hi, opts.band.hi); }
    if (opts.zeroLine) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
    const span = hi - lo || Math.abs(hi) * 0.1 || 1;
    lo -= span * 0.08; hi += span * 0.08;
    const iw = width - m.left - m.right, ih = height - m.top - m.bottom;
    const X = (i) => m.left + (xs.length === 1 ? iw / 2 : (i / (xs.length - 1)) * iw);
    const Y = (v) => m.top + ih - ((v - lo) / (hi - lo)) * ih;
    const fmt = opts.format || ((v) => v.toPrecision(3));

    if (opts.band) {
      el("rect", { x: m.left, width: iw, y: Y(opts.band.hi), height: Math.max(0, Y(opts.band.lo) - Y(opts.band.hi)),
        fill: "var(--teal-soft)", opacity: 0.7 }, svg);
      if (opts.band.label) text(svg, m.left + iw - 4, Y(opts.band.hi) + 13, opts.band.label, { "text-anchor": "end" });
    }
    for (const t of niceTicks(lo, hi, Math.max(3, Math.round(ih / 55)))) {
      el("line", { x1: m.left, x2: m.left + iw, y1: Y(t), y2: Y(t), class: "grid" }, svg);
      text(svg, m.left - 8, Y(t) + 4, minus(fmt(t)), { "text-anchor": "end" });
    }
    if (opts.zeroLine && lo < 0 && hi > 0) el("line", { x1: m.left, x2: m.left + iw, y1: Y(0), y2: Y(0), class: "axis" }, svg);
    const every = Math.ceil(xs.length / Math.max(2, Math.floor(iw / 44)));
    xs.forEach((label, i) => {
      if (i % every === 0 || i === xs.length - 1) text(svg, X(i), m.top + ih + 16, label, { "text-anchor": "middle" });
    });
    if (opts.xTitle) text(svg, m.left + iw / 2, height - 4, opts.xTitle, { "text-anchor": "middle" });

    for (const mk of opts.markers || []) {
      el("line", { x1: X(mk.i), x2: X(mk.i), y1: m.top, y2: m.top + ih, stroke: "var(--ink-3)", "stroke-dasharray": "2 3" }, svg);
      const right = X(mk.i) > m.left + iw * 0.8;
      text(svg, X(mk.i) + (right ? -5 : 5), m.top + 10, mk.label, { "text-anchor": right ? "end" : "start", class: "strong-label" });
    }

    for (const s of opts.series) {
      let d = "";
      s.values.forEach((v, i) => {
        if (typeof v !== "number" || !isFinite(v)) return;
        d += (d ? "L" : "M") + X(i).toFixed(1) + " " + Y(v).toFixed(1);
      });
      el("path", { d, fill: "none", stroke: s.color, "stroke-width": s.width || 2, "stroke-dasharray": s.dash,
        "stroke-linejoin": "round", "stroke-linecap": "round", opacity: s.opacity }, svg);
      for (const i of s.dots || []) {
        if (typeof s.values[i] === "number") el("circle", { cx: X(i), cy: Y(s.values[i]), r: 4, fill: "var(--surface)", stroke: s.color, "stroke-width": 2 }, svg);
      }
    }

    const cross = el("line", { y1: m.top, y2: m.top + ih, stroke: "var(--ink-3)", opacity: 0 }, svg);
    const hit = el("rect", { x: m.left, y: m.top, width: iw, height: ih, fill: "transparent" }, svg);
    hit.addEventListener("mousemove", (evt) => {
      const box = svg.getBoundingClientRect();
      const px = ((evt.clientX - box.left) / box.width) * width;
      const i = Math.max(0, Math.min(xs.length - 1, Math.round(((px - m.left) / iw) * (xs.length - 1))));
      cross.setAttribute("x1", X(i)); cross.setAttribute("x2", X(i)); cross.setAttribute("opacity", 0.6);
      const rows = opts.series
        .filter((s) => typeof s.values[i] === "number")
        .map((s) => `<div class="row"><span><span class="swatch" style="background:${s.color}"></span> ${s.name}</span><b>${(opts.tipFormat || fmt)(s.values[i])}</b></div>`)
        .join("");
      showTip(evt, `<b>${opts.xName || "epoch"} ${xs[i]}</b>${rows}`);
    });
    hit.addEventListener("mouseleave", () => { cross.setAttribute("opacity", 0); hideTip(); });
  }

  /**
   * ruler(container, {rows: [{label, lo, hi, point, tone, ghost, detail}], clear, clearLabel})
   * Scores on a horizontal scale around .5 with Elo underneath.
   */
  function ruler(container, opts) {
    container.innerHTML = "";
    const width = Math.max(300, container.clientWidth);
    const rowH = opts.rows.length > 3 ? 40 : 52;
    const labelW = Math.min(130, width * 0.26);
    const top = 34, bottom = 46;
    const height = top + rowH * opts.rows.length + bottom;
    const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": "evaluation intervals" }, container);
    let half = 0.05;
    for (const r of opts.rows) half = Math.max(half, Math.abs(r.lo - 0.5) + 0.01, Math.abs(r.hi - 0.5) + 0.01);
    if (opts.clear) half = Math.max(half, Math.abs(opts.clear - 0.5) + 0.01);
    half = Math.ceil(half / 0.025) * 0.025;
    const left = labelW + 10, right = width - 14, iw = right - left;
    const X = (s) => left + ((s - (0.5 - half)) / (2 * half)) * iw;
    const y0 = top, y1 = top + rowH * opts.rows.length;

    el("rect", { x: left, y: y0 - 6, width: X(0.5) - left, height: y1 - y0 + 12, fill: "var(--red-soft)", opacity: 0.45 }, svg);
    el("rect", { x: X(0.5), y: y0 - 6, width: right - X(0.5), height: y1 - y0 + 12, fill: "var(--teal-soft)", opacity: 0.45 }, svg);
    const short = iw < 420;
    const versus = opts.versus || "its start";
    const fits = !short && versus.length < 24;
    text(svg, left + 6, 16, fits ? `weaker than ${versus}` : "weaker", { class: "strong-label" }).style.fill = "var(--red)";
    text(svg, right - 6, 16, fits ? `stronger than ${versus}` : "stronger", { "text-anchor": "end", class: "strong-label" }).style.fill = "var(--teal)";

    for (const t of niceTicks(0.5 - half, 0.5 + half, Math.max(4, Math.round(iw / 80)))) {
      el("line", { x1: X(t), x2: X(t), y1: y1 + 6, y2: y1 + 11, class: "axis" }, svg);
      text(svg, X(t), y1 + 24, t.toFixed(Math.abs(t * 100 - Math.round(t * 100)) > 1e-6 ? 3 : 2).replace(/^0/, ""), { "text-anchor": "middle" });
      const elo = -400 * Math.log10(1 / Math.min(Math.max(t, 1e-6), 1 - 1e-6) - 1);
      text(svg, X(t), y1 + 38, signedInt(elo), { "text-anchor": "middle", opacity: 0.8 });
    }
    text(svg, left - 24, y1 + 24, "score", { "text-anchor": "end" });
    text(svg, left - 24, y1 + 38, "Elo", { "text-anchor": "end" });

    el("line", { x1: X(0.5), x2: X(0.5), y1: y0 - 10, y2: y1 + 8, stroke: "var(--ink)", "stroke-width": 1.5 }, svg);
    if (opts.clear && opts.clear < 0.5 + half) {
      el("line", { x1: X(opts.clear), x2: X(opts.clear), y1: y0 - 6, y2: y1 + 6, stroke: "var(--teal)", "stroke-dasharray": "4 3", "stroke-width": 1.5 }, svg);
      const flip = X(opts.clear) > right - 120;
      text(svg, X(opts.clear) + (flip ? -5 : 5), y0 + 4, opts.clearLabel, { "text-anchor": flip ? "end" : "start" }).style.fill = "var(--teal)";
    }

    opts.rows.forEach((r, k) => {
      const cy = top + rowH * k + rowH / 2 + 4;
      const g = el("g", { opacity: r.ghost ? 0.55 : 1 }, svg);
      text(g, labelW, cy + 4, r.label, { "text-anchor": "end", class: r.ghost ? "" : "strong-label" });
      el("line", { x1: X(r.lo), x2: X(r.hi), y1: cy, y2: cy, stroke: r.tone, "stroke-width": r.ghost ? 5 : 9, "stroke-linecap": "round" }, g);
      el("circle", { cx: X(r.point), cy, r: r.ghost ? 4 : 6.5, fill: "var(--surface)", stroke: r.tone, "stroke-width": 2.5 }, g);
      const hit = el("rect", { x: left, y: cy - rowH / 2, width: iw, height: rowH, fill: "transparent" }, g);
      hit.addEventListener("mousemove", (evt) => showTip(evt, r.detail));
      hit.addEventListener("mouseleave", hideTip);
    });
  }

  /**
   * staircase(container, {start, steps: [{name, elo, lo, hi, total, totalLo, totalHi, tone, attempts, nulls, open, current}]})
   * The lineage as a climb: each adopted network steps up (or down) by its measured gain against its own start; the
   * shaded band is the running total's 95 % range, which widens with every step.
   */
  function staircase(container, opts) {
    container.innerHTML = "";
    const steps = opts.steps;
    const width = Math.max(300, container.clientWidth);
    const height = opts.height || 250;
    const m = { top: 22, right: 22, bottom: 58, left: 52 };
    const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": "Elo along the lineage" }, container);
    const cols = steps.length + 1;
    const iw = width - m.left - m.right, ih = height - m.top - m.bottom;
    const X = (i) => m.left + (i + 0.5) * (iw / cols);
    const known = steps.filter((s) => typeof s.total === "number");
    const narrow = iw / cols < 120;
    const values = [0, ...known.flatMap((s) => [s.totalLo, s.totalHi, s.total, s.total - s.elo + s.lo, s.total - s.elo + s.hi]),
      ...steps.filter((s) => s.direct).flatMap((s) => [s.direct.lo, s.direct.hi])];
    let lo = Math.min(...values), hi = Math.max(...values);
    const span = hi - lo || 40;
    lo -= span * 0.1; hi += span * 0.12;
    const Y = (v) => m.top + ih - ((v - lo) / (hi - lo)) * ih;

    for (const t of niceTicks(lo, hi, Math.max(3, Math.round(ih / 45)))) {
      el("line", { x1: m.left, x2: m.left + iw, y1: Y(t), y2: Y(t), class: t === 0 ? "axis" : "grid" }, svg);
      text(svg, m.left - 8, Y(t) + 4, signedInt(t), { "text-anchor": "end" });
    }
    text(svg, m.left - 8, m.top - 8, "Elo", { "text-anchor": "end" });

    // the running total's band
    let band = "", back = "";
    let prev = { x: X(0), total: 0, lo: 0, hi: 0 };
    const pts = [prev];
    steps.forEach((s, i) => {
      if (typeof s.total !== "number" || !s.counted) return;
      pts.push({ x: X(i + 1), total: s.total, lo: s.totalLo, hi: s.totalHi });
    });
    pts.forEach((p, i) => { band += (i ? "L" : "M") + p.x + " " + Y(p.hi); });
    [...pts].reverse().forEach((p) => { back += "L" + p.x + " " + Y(p.lo); });
    if (pts.length > 1) el("path", { d: band + back + "Z", fill: "var(--blue-soft)", class: "band" }, svg);

    // the climb itself: flat to the next column, then the step
    let d = `M${X(0)} ${Y(0)}`;
    let level = 0;
    let tried = "";
    steps.forEach((s, i) => {
      if (typeof s.total !== "number") return;
      const seg = `L${X(i + 1) - 12} ${Y(level)}L${X(i + 1) - 12} ${Y(s.total)}L${X(i + 1)} ${Y(s.total)}`;
      if (s.counted) { d += seg; level = s.total; }
      else tried = `M${X(i)} ${Y(level)}` + seg;
    });
    el("path", { d, fill: "none", stroke: "var(--blue)", "stroke-width": 2.5, "stroke-linejoin": "round" }, svg);
    if (tried) el("path", { d: tried, fill: "none", stroke: "var(--ink-3)", "stroke-width": 2, "stroke-dasharray": "4 3" }, svg);

    // the start
    el("circle", { cx: X(0), cy: Y(0), r: 5, fill: "var(--ink)" }, svg);
    text(svg, X(0), height - m.bottom + 18, opts.start, { "text-anchor": "middle", class: "strong-label" });
    if (!narrow) text(svg, X(0), height - m.bottom + 33, "start", { "text-anchor": "middle" });

    level = 0;
    steps.forEach((s, i) => {
      const x = X(i + 1);
      const g = el("g", {}, svg);
      if (typeof s.total === "number") {
        // this step's own interval, drawn from the level it started at
        el("line", { x1: x - 12, x2: x - 12, y1: Y(level + s.lo), y2: Y(level + s.hi), stroke: s.tone, "stroke-width": 5, "stroke-linecap": "round", opacity: 0.5 }, g);
        el("circle", { cx: x, cy: Y(s.total), r: s.current ? 7 : 5.5, fill: s.current ? s.tone : "var(--surface)", stroke: s.tone, "stroke-width": 2.5 }, g);
        const label = `${signedInt(s.elo)}`;
        text(g, x + 9, Y(s.total) - 8, label, { class: "strong-label" }).style.fill = s.tone;
        if (s.counted) level = s.total;
      } else {
        el("circle", { cx: x, cy: Y(level), r: 5.5, fill: "none", stroke: "var(--grey)", "stroke-width": 2, "stroke-dasharray": "3 2" }, g);
      }
      const name = narrow && s.name.length > 10 ? s.name.slice(0, 9) + "…" : s.name;
      if (s.direct) {  // measured straight against the line's first network
        const dx = x + 30;
        el("line", { x1: dx, x2: dx, y1: Y(s.direct.lo), y2: Y(s.direct.hi), stroke: "var(--plum)", "stroke-width": 2 }, g);
        el("path", { d: `M${dx} ${Y(s.direct.elo) - 6}l6 6-6 6-6-6z`, fill: "var(--surface)", stroke: "var(--plum)", "stroke-width": 2 }, g);
      }
      text(g, x, height - m.bottom + 18, name, { "text-anchor": "middle", class: s.current ? "strong-label" : "" });
      const tries = s.attempts > 1 ? `${s.attempts} batches from its start` : "first try";
      if (!narrow) text(g, x, height - m.bottom + 33, s.open ? "not decided yet" : tries, { "text-anchor": "middle" });
      const hit = el("rect", { x: x - iw / cols / 2, y: m.top, width: iw / cols, height: ih + 40, fill: "transparent" }, g);
      hit.addEventListener("mousemove", (evt) => showTip(evt, s.detail));
      hit.addEventListener("mouseleave", hideTip);
    });
  }

  /** sprtTrack(container, {llr, bound, pairs, cap, stop}): where the log-likelihood ratio stands between its bounds. */
  function sprtTrack(container, opts) {
    container.innerHTML = "";
    const width = Math.max(240, container.clientWidth);
    const height = 64;
    const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": "sequential test progress" }, container);
    const b = opts.bound;
    const span = Math.max(b * 1.25, Math.abs(opts.llr || 0) * 1.1);
    const X = (v) => 10 + ((v + span) / (2 * span)) * (width - 20);
    el("rect", { x: 10, y: 22, width: width - 20, height: 10, rx: 5, fill: "var(--surface-2)" }, svg);
    el("rect", { x: X(-span), y: 22, width: X(-b) - X(-span), height: 10, rx: 5, fill: "var(--red-soft)" }, svg);
    el("rect", { x: X(b), y: 22, width: X(span) - X(b), height: 10, rx: 5, fill: "var(--teal-soft)" }, svg);
    for (const [v, label] of [[-b, "reject"], [0, "0"], [b, "accept"]]) {
      el("line", { x1: X(v), x2: X(v), y1: 16, y2: 38, stroke: v ? "var(--ink-3)" : "var(--line-strong)" }, svg);
      text(svg, X(v), 52, label, { "text-anchor": "middle" });
    }
    if (typeof opts.llr === "number") {
      const tone = opts.llr >= b ? "var(--teal)" : opts.llr <= -b ? "var(--red)" : "var(--ochre)";
      el("circle", { cx: X(opts.llr), cy: 27, r: 7, fill: "var(--surface)", stroke: tone, "stroke-width": 3 }, svg);
      const t = text(svg, X(opts.llr), 11, `LLR ${opts.llr >= 0 ? "+" : "\u2212"}${Math.abs(opts.llr).toFixed(2)}`, { "text-anchor": "middle", class: "strong-label" });
      t.style.fill = tone;
    }
  }


  /** barChart(container, {values, labels, height, format, tip(i), color}) */
  function barChart(container, opts) {
    container.innerHTML = "";
    const width = Math.max(260, container.clientWidth);
    const height = opts.height || 180;
    const m = { top: 10, right: 8, bottom: 30, left: 44 };
    const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": opts.label || "bar chart" }, container);
    const n = opts.values.length;
    if (!n) return;
    const max = Math.max(...opts.values, 1);
    const iw = width - m.left - m.right, ih = height - m.top - m.bottom;
    const bw = iw / n;
    const Y = (v) => m.top + ih - (v / max) * ih;
    for (const t of niceTicks(0, max, 3)) {
      el("line", { x1: m.left, x2: m.left + iw, y1: Y(t), y2: Y(t), class: "grid" }, svg);
      text(svg, m.left - 6, Y(t) + 4, (opts.format || String)(t), { "text-anchor": "end" });
    }
    const every = Math.ceil(n / Math.max(2, Math.floor(iw / 38)));
    opts.values.forEach((v, i) => {
      const x = m.left + i * bw;
      const bar = el("rect", { x: x + Math.min(1, bw * 0.1), width: Math.max(1, bw - Math.min(2, bw * 0.2)), y: Y(v),
        height: Math.max(0, m.top + ih - Y(v)), fill: (opts.colors && opts.colors[i]) || opts.color || "var(--blue)", rx: 1 }, svg);
      if (i % every === 0) text(svg, x, m.top + ih + 14, opts.labels[i], { "text-anchor": "start" });
      const hit = el("rect", { x, y: m.top, width: bw, height: ih, fill: "transparent" }, svg);
      hit.addEventListener("mousemove", (evt) => { bar.setAttribute("opacity", 0.75); showTip(evt, opts.tip(i)); });
      hit.addEventListener("mouseleave", () => { bar.removeAttribute("opacity"); hideTip(); });
    });
    if (opts.xTitle) text(svg, m.left + iw / 2, height - 2, opts.xTitle, { "text-anchor": "middle" });
  }

  const PIECE = ["", "R", "P", "S"];
  /**
   * board(container, {cells: 81 codes (0 empty, 1-3 Blue, 4-6 Red), move: {from, to}, best: {from, to}, small, labels})
   * Absolute frame: a1 bottom left is Blue's home, i9 top right is Red's.
   */
  function board(container, opts) {
    container.innerHTML = "";
    const size = opts.small ? 120 : 420;
    const pad = opts.small ? 2 : 22;
    const cell = (size - pad * 2) / 9;
    const svg = el("svg", { viewBox: `0 0 ${size} ${size}`, class: "board", role: "img", "aria-label": opts.label || "board" }, container);
    const X = (sq) => pad + (sq % 9) * cell, Y = (sq) => pad + (8 - Math.floor(sq / 9)) * cell;
    for (let sq = 0; sq < 81; sq++) {
      const dark = (Math.floor(sq / 9) + (sq % 9)) % 2 === 1;
      let fill = dark ? "var(--board-dark)" : "var(--board-light)";
      if (sq === 0) fill = "var(--blue-soft)";
      if (sq === 80) fill = "var(--red-soft)";
      if (opts.move && (sq === opts.move.from || sq === opts.move.to)) fill = "var(--ochre-soft)";
      el("rect", { x: X(sq), y: Y(sq), width: cell, height: cell, fill }, svg);
    }
    el("rect", { x: pad, y: pad, width: cell * 9, height: cell * 9, fill: "none", stroke: "var(--line-strong)" }, svg);
    if (!opts.small) {
      for (let i = 0; i < 9; i++) {
        text(svg, pad + i * cell + cell / 2, size - 6, "abcdefghi"[i], { "text-anchor": "middle" });
        text(svg, 9, pad + (8 - i) * cell + cell / 2 + 4, String(i + 1), { "text-anchor": "middle" });
      }
    }
    const arrow = (m, stroke, dash) => {
      if (!m) return;
      const cx = (sq) => X(sq) + cell / 2, cy = (sq) => Y(sq) + cell / 2;
      el("line", { x1: cx(m.from), y1: cy(m.from), x2: cx(m.to), y2: cy(m.to), stroke, "stroke-width": opts.small ? 1.5 : 3,
        "stroke-linecap": "round", "stroke-dasharray": dash, opacity: 0.8 }, svg);
    };
    for (let sq = 0; sq < 81; sq++) {
      const c = opts.cells[sq];
      if (!c) continue;
      const blue = c <= 3, kind = blue ? c : c - 3;
      const fill = blue ? "var(--blue)" : "var(--red)";
      const cx = X(sq) + cell / 2, cy = Y(sq) + cell / 2, r = cell * 0.36;
      if (kind === 1) el("circle", { cx, cy, r, fill }, svg);
      else if (kind === 2) el("rect", { x: cx - r, y: cy - r, width: 2 * r, height: 2 * r, rx: r * 0.25, fill }, svg);
      else el("path", { d: `M${cx} ${cy - r * 1.15}L${cx + r * 1.1} ${cy + r * 0.85}L${cx - r * 1.1} ${cy + r * 0.85}Z`, fill }, svg);
      if (!opts.small) text(svg, cx, cy + (kind === 3 ? 6 : 4), PIECE[kind], { "text-anchor": "middle", class: "piece-letter" });
    }
    arrow(opts.best, "var(--ink-3)", "4 4");
    arrow(opts.move, "var(--ochre)");
  }

  /** evalGraph(container, {values: per ply, Blue's expected result or null, cursor, marks: [ply], onSeek(ply)}) */
  function evalGraph(container, opts) {
    container.innerHTML = "";
    const width = Math.max(260, container.clientWidth);
    const height = opts.height || 130;
    const m = { top: 8, right: 8, bottom: 20, left: 34 };
    const iw = width - m.left - m.right, ih = height - m.top - m.bottom;
    const n = Math.max(1, opts.values.length);
    const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": "evaluation by ply" }, container);
    const X = (i) => m.left + (i / Math.max(1, n - 1)) * iw, Y = (v) => m.top + ih / 2 - (v * ih) / 2;
    el("rect", { x: m.left, y: m.top, width: iw, height: ih / 2, fill: "var(--blue-soft)", opacity: 0.35 }, svg);
    el("rect", { x: m.left, y: m.top + ih / 2, width: iw, height: ih / 2, fill: "var(--red-soft)", opacity: 0.35 }, svg);
    el("line", { x1: m.left, x2: m.left + iw, y1: Y(0), y2: Y(0), class: "axis" }, svg);
    text(svg, m.left - 5, Y(1) + 8, "Blue", { "text-anchor": "end" });
    text(svg, m.left - 5, Y(-1), "Red", { "text-anchor": "end" });
    for (const p of opts.marks || []) el("line", { x1: X(p), x2: X(p), y1: m.top, y2: m.top + ih, stroke: "var(--ochre)", opacity: 0.5 }, svg);
    let d = "";
    opts.values.forEach((v, i) => { if (typeof v === "number") d += (d ? "L" : "M") + X(i).toFixed(1) + " " + Y(v).toFixed(1); });
    el("path", { d, fill: "none", stroke: "var(--ink)", "stroke-width": 1.6, "stroke-linejoin": "round" }, svg);
    for (const t of niceTicks(0, n - 1, Math.max(3, Math.floor(iw / 70)))) text(svg, X(t), height - 5, String(t), { "text-anchor": "middle" });
    const cursor = el("line", { x1: X(opts.cursor), x2: X(opts.cursor), y1: m.top, y2: m.top + ih, stroke: "var(--blue)", "stroke-width": 2 }, svg);
    const hit = el("rect", { x: m.left, y: m.top, width: iw, height: ih, fill: "transparent", style: "cursor:pointer" }, svg);
    const plyAt = (evt) => {
      const box = svg.getBoundingClientRect();
      return Math.max(0, Math.min(n - 1, Math.round((((evt.clientX - box.left) / box.width) * width - m.left) / iw * (n - 1))));
    };
    hit.addEventListener("click", (evt) => opts.onSeek && opts.onSeek(plyAt(evt)));
    hit.addEventListener("mousemove", (evt) => {
      const i = plyAt(evt), v = opts.values[i];
      showTip(evt, `<b>ply ${i}</b><div class="row"><span>Blue's expected result</span><b>${typeof v === "number" ? (v >= 0 ? "+" : "\u2212") + Math.abs(v).toFixed(2) : "no search"}</b></div>`);
    });
    hit.addEventListener("mouseleave", hideTip);
    return { move: (i) => { cursor.setAttribute("x1", X(i)); cursor.setAttribute("x2", X(i)); } };
  }

  window.Charts = { lineChart, ruler, staircase, sprtTrack, barChart, board, evalGraph, hideTip };
})();
