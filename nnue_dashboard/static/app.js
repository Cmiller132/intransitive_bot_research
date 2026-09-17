(function () {
  const state = { data: null, raw: "", control: null, rawControl: "", run: null, metric: "run:objective", sets: null,
    compare: "", evalFile: null, open: new Set(["log-failed"]) };
  const SET_COLORS = ["var(--blue)", "var(--red)", "var(--teal)", "var(--ochre)", "var(--plum)", "var(--grey)", "#6f8f2f", "#3d7fb8"];
  const TONE = { better: "var(--teal)", better_small: "var(--teal)", retest: "var(--ochre)", worse: "var(--red)", null: "var(--ochre)",
    running: "var(--grey)", invalid: "var(--grey)" };
  const RUN_METRICS = ["objective", "loss", "value", "consistency", "best", "lr", "seconds"];
  const SET_METRICS = ["mse", "mae", "bias", "symmetry_range_mean", "symmetry_range_p95"];
  const SLICE_METRICS = ["mse", "mae", "bias", "rows"];
  const STEPS = [["check", "Check"], ["generate", "Games"], ["import", "Import"], ["encode", "Encode"], ["train", "Train"], ["evaluate", "Evaluate"]];
  const LIVE = ["running", "queued", "paused", "generating", "preparing", "training", "evaluating"];
  const $ = (sel, root = document) => root.querySelector(sel);
  const TOKEN = document.querySelector('meta[name="token"]').content;

  // ------------------------------------------------------------------ helpers
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const num = (v, digits = 4) => (typeof v === "number" && isFinite(v) ? (Math.abs(v) >= 1000 ? Math.round(v).toLocaleString() : +v.toPrecision(digits) + "") : "–");
  const pct = (v, d = 1) => (typeof v === "number" ? (v * 100).toFixed(d) + " %" : "–");
  const elo = (v) => (typeof v === "number" ? (Math.round(v) > 0 ? "+" : Math.round(v) < 0 ? "−" : "") + Math.abs(Math.round(v)) : "–");
  const score = (v) => (typeof v === "number" ? v.toFixed(3).replace(/^0/, "") : "–");
  const millions = (v) => (v >= 1e6 ? (v / 1e6).toFixed(v >= 1e7 ? 0 : 1) + " M" : Math.round(v / 1000) + " k");
  const setLabel = (s) => s.replace(/^selfplay_/, "");
  const colorFor = (sets, s) => SET_COLORS[Math.max(0, sets.indexOf(s)) % SET_COLORS.length];
  const duration = (s) => {
    if (typeof s !== "number" || !isFinite(s)) return "";
    if (s < 90) return "under 2 minutes";
    if (s < 5400) return `about ${Math.round(s / 60)} minutes`;
    return `about ${(s / 3600).toFixed(s < 36000 ? 1 : 0)} hours`;
  };
  const ago = (t) => {
    const s = Date.now() / 1000 - t;
    return s < 90 ? "just now" : s < 5400 ? `${Math.round(s / 60)} minutes ago` : s < 129600 ? `${Math.round(s / 3600)} hours ago` : `${Math.round(s / 86400)} days ago`;
  };
  const controlOn = () => !!state.control?.enabled;

  const VERDICT_WORDS = { adopt: "adopt", more_data: "more data", reeval: "test again", abort: "go back",
    evaluate: "not evaluated", running: "running", stopped: "stopped", incomplete: "incomplete" };
  function mark(verdict) {
    const shapes = {
      adopt: '<path d="M6 1.5 11 10.5H1z" fill="currentColor"/>',
      abort: '<path d="M6 10.5 11 1.5H1z" fill="currentColor"/>',
      more_data: '<path d="M6 1 11 6 6 11 1 6z" fill="currentColor"/>',
      reeval: '<path d="M6 1.8 10.2 6 6 10.2 1.8 6z" fill="none" stroke="currentColor" stroke-width="1.6"/>',
      running: '<circle cx="6" cy="6" r="4.2" fill="none" stroke="currentColor" stroke-width="1.6" stroke-dasharray="3 2"/>',
      stopped: '<rect x="2.5" y="2.5" width="7" height="7" fill="currentColor"/>',
    };
    const shape = shapes[verdict] || '<circle cx="6" cy="6" r="4" fill="none" stroke="currentColor" stroke-width="1.6"/>';
    return `<svg class="mark ${verdict}" viewBox="0 0 12 12" aria-hidden="true">${shape}</svg>`;
  }

  function toast(message, bad) {
    const t = $("#toast");
    t.textContent = message;
    t.classList.toggle("bad", !!bad);
    t.hidden = false;
    clearTimeout(toast.timer);
    toast.timer = setTimeout(() => (t.hidden = true), bad ? 7000 : 3200);
  }

  async function api(path, body) {
    const res = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json", "X-Token": TOKEN },
      body: JSON.stringify(body || {}) });
    const data = await res.json().catch(() => ({ error: res.statusText }));
    if (!res.ok) throw new Error(data.error || res.statusText);
    return data;
  }
  async function act(path, body, done) {
    try {
      const data = await api(path, body);
      if (done) toast(done);
      await loadControl(true);
      await loadRuns(true);
      return data;
    } catch (err) {
      toast(err.message, true);
      throw err;
    }
  }

  // ------------------------------------------------------------------ data
  async function loadRuns(force) {
    try {
      const res = await fetch("/api/runs", { cache: "no-store" });
      const body = await res.text();
      if (!res.ok) throw new Error(JSON.parse(body).error || res.statusText);
      $("#refreshed").textContent = "Read " + new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
      if (body === state.raw && !force) return;
      state.raw = body;
      const data = JSON.parse(body);
      // progress numbers change every few seconds while a job runs; only the rail shows them, so a change in them
      // alone redraws the rail and leaves the page (open lists, charts, scroll) as it is
      const signature = JSON.stringify(data, (k, v) => (VOLATILE.has(k) ? undefined : v));
      const changed = force || signature !== state.signature;
      state.signature = signature;
      state.data = data;
      $("#runs-dir").textContent = state.data.runs_dir;
      const names = state.data.runs.map((r) => r.name);
      const wanted = decodeURIComponent(location.hash.slice(1));
      if (!names.includes(state.run) && !pendingJob(state.run)) state.run = names.includes(wanted) ? wanted : pickDefault(state.data.runs);
      if (changed) renderWhenIdle(); else renderTree();
    } catch (err) {
      $("#main").innerHTML = `<div class="empty"><h2>The runs could not be read</h2><p>${esc(err.message)}</p>
        <p>Start the dashboard from the repository root, or pass <code>--repo</code>.</p></div>`;
    }
  }
  async function loadControl(force) {
    try {
      const res = await fetch("/api/control", { cache: "no-store" });
      const body = await res.text();
      if (body === state.rawControl && !force) return;
      const before = state.control?.version;
      state.rawControl = body;
      state.control = JSON.parse(body);
      renderControl();
      renderJobPanel();
      syncNotify();
      // parts of the page depend on which runs have a live job (the retrain and next-step buttons)
      const live = state.control.jobs.filter((j) => ["running", "queued", "paused"].includes(j.state)).map((j) => `${j.run}:${j.state}`).join(",");
      if (state.data && live !== state.liveJobs) { state.liveJobs = live; renderWhenIdle(); }
      if (state.data && before !== undefined && before !== state.control.version) loadRuns();
    } catch (err) { /* the runs view reports a dead server */ }
  }
  function pickDefault(runs) {
    const jobs = state.control?.jobs || [];
    const live = jobs.filter((j) => ["running", "queued", "paused"].includes(j.state) && j.run);
    if (live.length) return live[live.length - 1].run;
    if (!runs.length) return null;
    return [...runs].sort((a, b) => b.mtime - a.mtime)[0].name;
  }
  const byName = (n) => state.data?.runs.find((r) => r.name === n);
  const jobsFor = (name) => (state.control?.jobs || []).filter((j) => j.run === name);
  const jobFor = (name) => {
    const jobs = [...jobsFor(name)].sort((a, b) => a.updated - b.updated);
    return jobs.filter((j) => j.state === "running")[0] || [...jobs].reverse().find((j) => !["done", "failed", "cancelled"].includes(j.state))
      || jobs[jobs.length - 1];
  };
  const pendingJob = (name) => name && !byName(name) && jobFor(name);
  const VOLATILE = new Set(["now", "fraction", "eta_seconds", "detail", "touched", "games_done", "in_flight", "games_per_hour",
    "games_per_core_hour", "mtime", "started"]);

  // A redraw replaces the page's elements; while a list is open or a field has the focus it waits until you leave it.
  function busyInMain() {
    const el = document.activeElement;
    return el && $("#main").contains(el) && ["SELECT", "INPUT", "TEXTAREA"].includes(el.tagName);
  }
  function renderWhenIdle() {
    if (busyInMain()) { state.renderPending = true; renderTree(); return; }
    state.renderPending = false;
    render();
  }
  document.addEventListener("focusout", () => setTimeout(() => { if (state.renderPending && !busyInMain()) renderWhenIdle(); }, 0));
  const current = () => byName(state.run);

  function select(name) {
    Object.assign(state, { run: name, sets: null, evalFile: null, compare: "" });
    history.replaceState(null, "", "#" + encodeURIComponent(name));
    render();
    window.scrollTo(0, 0);
  }

  // ------------------------------------------------------------------ control rail
  const seenStates = new Map();
  function notifyChanges(c) {
    const words = { done: "finished", failed: "failed", paused: "paused" };
    for (const j of c.jobs) {
      const before = seenStates.get(j.id);
      seenStates.set(j.id, j.state);
      if (!before || before === j.state || !words[j.state] || before === "paused") continue;
      const detail = j.state === "failed" ? j.error : j.state === "paused" ? j.pause?.detail : j.steps?.evaluate?.note || "";
      notify(`${j.run || "build"} ${words[j.state]}`, detail || "");
    }
    if (notifyChanges.auto !== undefined && c.auto.message !== notifyChanges.auto && /^Auto stopped/.test(c.auto.message || "")) notify("Auto stopped", c.auto.message);
    notifyChanges.auto = c.auto.message;
  }
  function notify(title, body) {
    let on = false;
    try { on = localStorage.getItem("nnue-notify") === "on"; } catch {}
    if (on && "Notification" in window && Notification.permission === "granted" && document.hidden) new Notification(title, { body, tag: title });
  }

  function renderControl() {
    const c = state.control;
    const box = $("#control");
    if (!c) return;
    if (c.enabled) {
      notifyChanges(c);
      const job = c.jobs.find((j) => j.id === c.current);
      const p = job?.progress;
      const share = p && p.total ? ` ${Math.floor((100 * p.done) / p.total)} %` : "";
      const paused = c.jobs.some((j) => j.state === "paused");
      document.title = job ? `${job.run || "build"}:${share} ${(p?.words || "").toLowerCase()} – NNUE runs` : paused ? "Paused – NNUE runs" : "NNUE runs";
    }

    if (!c.enabled) {
      box.innerHTML = `<p class="control-off">${esc(c.reason)} The page shows the runs and the commands to run by hand.</p>`;
      return;
    }
    const running = c.jobs.find((j) => j.id === c.current);
    const p = running?.progress;
    const frac = p && p.total ? Math.min(1, p.done / p.total) : null;
    const runner = running
      ? `<button class="runner" type="button" data-goto="${esc(running.run || "")}">
          <span class="runner-top"><span class="pulse" aria-hidden="true"></span><b>${esc(running.run || "build")}</b><span>${esc(p?.words || running.kind)}${frac !== null ? ` ${Math.floor(frac * 100)} %` : ""}</span></span>
          <span class="mini-bar${frac === null ? " unknown" : ""}"><span style="width:${frac === null ? 100 : frac * 100}%"></span></span></button>`
      : `<p class="runner idle">Nothing running.</p>`;
    const a = c.auto;
    const auto = a.enabled
      ? `<div class="auto on"><span><b>Auto</b> ${a.adopted} of ${a.max_generations} generations, ${a.batches} of ${a.max_batches} batches</span>
          <button class="small-button" type="button" data-auto-off>Stop auto</button></div>`
      : `<div class="auto"><span><b>Auto</b> off</span><button class="small-button" type="button" data-auto-dialog>Set up…</button></div>`;
    const waiting = c.jobs.filter((j) => (j.state === "queued" || j.state === "paused") && j.id !== c.current);
    const queue = waiting.length
      ? `<ul class="queue">${waiting.map((j) => `<li><button class="link" type="button" data-goto="${esc(j.run || "")}">${esc(j.run || j.kind)}</button>
          <span>${j.state === "paused" ? "paused" : "waiting"}${j.kind === "eval" ? ", evaluation" : ""}</span></li>`).join("")}</ul>` : "";
    const held = a.enabled && !running && c.jobs.find((j) => j.state === "paused");
    const note = held ? `Auto waits: ${held.run} is paused.` : a.message;
    box.innerHTML = `${runner}${auto}${note ? `<p class="auto-message">${esc(note)}</p>` : ""}${queue}
      <div class="control-buttons">
        <button class="small-button" type="button" data-new-line>New line…</button>
        ${c.bot_exists ? "" : `<button class="small-button warn" type="button" data-build>Build the bot</button>`}
      </div>`;
    box.querySelectorAll("[data-goto]").forEach((b) => b.addEventListener("click", () => b.dataset.goto && select(b.dataset.goto)));
    box.querySelector("[data-auto-off]")?.addEventListener("click", () => act("/api/auto", { enabled: false }, "Auto stopped; the running job finishes"));
    box.querySelector("[data-auto-dialog]")?.addEventListener("click", () => autoDialog());
    box.querySelector("[data-new-line]")?.addEventListener("click", () => pipelineDialog(newLineSettings(), "Start a new line", "Start"));
    box.querySelector("[data-build]")?.addEventListener("click", () => act("/api/jobs/build", {}, "Building the bot"));
  }

  function newLineSettings() {
    const d = state.control.defaults;
    const taken = new Set([...(state.data?.runs || []).map((r) => r.name), ...state.control.jobs.map((j) => j.run)]);
    const name = ["first", "line_a", "line_b", "line_c"].find((n) => !taken.has(n)) || `line_${Date.now() % 10000}`;
    return { ...d, name, eval: { ...d.eval } };
  }

  // ------------------------------------------------------------------ job panel
  function renderJobPanel() {
    const slot = $("#job-panel");
    if (!slot) return;
    const name = state.run;
    const job = name && jobFor(name);
    slot.innerHTML = job ? jobPanel(job) : "";
    const q = job?.progress?.sequential;
    if (q && $("#job-sprt")) Charts.sprtTrack($("#job-sprt"), { llr: q.llr, bound: q.bound, pairs: q.pairs, cap: q.cap });
    bindJobPanel(slot, job);
    restoreOpen(slot);
  }

  function jobPanel(job) {
    const steps = job.kind === "pipeline" ? STEPS : job.kind === "eval" ? [["evaluate", "Evaluate"]] : [["build", "Build"]];
    const stepper = `<ol class="stepper">${steps.map(([key, label]) => {
      const st = job.steps[key] || { state: "pending" };
      return `<li class="${st.state}${job.step === key && job.state !== "done" ? " current" : ""}" title="${esc(st.note || st.state)}">
        <span class="dot" aria-hidden="true"></span><span class="label">${label}</span></li>`;
    }).join("")}</ol>`;
    const p = job.progress;
    let bar = "";
    if (p?.sequential && (job.state === "running" || job.state === "paused")) {
      const q = p.sequential;
      const lean = q.llr === null || Math.abs(q.llr) < 0.3 ? "no lean yet" : `leaning ${q.lean}`;
      const bound = q.llr === null ? "" : q.llr >= 0 ? `accepts at +${q.bound.toFixed(2)}` : `rejects at −${q.bound.toFixed(2)}`;
      const crossed = q.llr !== null && Math.abs(q.llr) >= q.bound;
      const f = q.forecast;
      const perPair = q.pairs_per_second ? 1 / q.pairs_per_second : null;
      const pace = crossed ? ` It has crossed the ${q.llr > 0 ? "accept" : "reject"} bound and stops at the next check.`
        : !f ? ""
        : ` Likely stops within about ${f.median_pairs.toLocaleString()} more pairs${perPair ? ` (${duration(f.median_pairs * perPair)})` : ""},
            9 in 10 within ${f.p90_pairs.toLocaleString()}${perPair ? ` (${duration(f.p90_pairs * perPair)})` : ""}; it accepts in ${Math.round(f.accept_chance * 100)} % of the forecasts${f.cap_chance >= 0.05 ? ` and reaches the cap in ${Math.round(f.cap_chance * 100)} %` : ""}.`;
      bar = `<div class="job-progress sequential">
        <div class="progress-top"><b>Sequential test, .50 against ${String(+q.s1.toFixed(3)).replace(/^0/, "")}</b><span>${q.pairs.toLocaleString()} of at most ${q.cap.toLocaleString()} pairs, score ${score(q.score)}</span></div>
        <div class="progress-bar${job.state === "paused" ? " paused" : ""}" title="how close the test is to stopping: a bound or the cap, whichever is nearer"><span style="width:${q.closeness * 100}%"></span></div>
        <div id="job-sprt" class="job-sprt"></div>
        <p class="job-note">${q.pairs < 128 ? `The first decision comes at 128 pairs; the value below is provisional.` : `${lean[0].toUpperCase() + lean.slice(1)}: LLR ${q.llr >= 0 ? "+" : "−"}${Math.abs(q.llr).toFixed(2)}, ${bound}; next check at ${q.next_check.toLocaleString()} pairs.`}${q.pairs >= 32 ? pace : ""}
        ${q.invalid ? `<br><b>Invalid:</b> ${esc(q.invalid)}; the test cannot decide.` : ""}</p>
        ${q.pairs >= q.decide_from ? `<p class="small-note">Taking too long? <b>Stop and decide now</b> reads the ${q.pairs.toLocaleString()} pairs like a test that reached its cap: adopted if the lower bound is above .50, otherwise more data.</p>` : ""}
      </div>`;
    } else if (p && (job.state === "running" || job.state === "paused")) {
      const frac = p.total ? Math.min(1, p.done / p.total) : null;
      const count = p.total ? `${(p.done ?? 0).toLocaleString()} of ${p.total.toLocaleString()} ${p.unit}` : p.done ? `${p.done.toLocaleString()} ${p.unit}` : "";
      const extra = p.in_flight && job.state === "running" ? `, ${p.in_flight} in flight` : "";
      const eta = p.eta_seconds && job.state === "running" ? `, ${duration(p.eta_seconds)} left` : "";
      bar = `<div class="job-progress"><div class="progress-top"><b>${esc(p.words)}</b><span>${count}${extra}${eta}</span></div>
        <div class="progress-bar${frac === null && job.state === "running" ? " unknown" : ""}${job.state === "paused" ? " paused" : ""}"><span style="width:${frac === null ? (job.state === "running" ? 100 : 0) : frac * 100}%"></span></div></div>`;
    }
    const head = {
      running: `Running${job.auto ? " (auto)" : ""}`, queued: "Waiting for the running job", paused: "Paused",
      done: `Finished ${ago(job.updated)}`, failed: "Failed", cancelled: "Cancelled",
    }[job.state];
    const pause = job.state === "paused" && job.pause ? `<p class="job-note">${esc(job.pause.detail)}</p>` : "";
    const error = job.state === "failed" ? `<p class="job-error">${esc(job.error)}</p>` : "";
    const notes = Object.entries(job.steps).filter(([, s]) => s.note && ["done", "skipped"].includes(s.state))
      .map(([k, s]) => `<li><b>${esc(STEPS.find(([key]) => key === k)?.[1] || k)}</b> ${esc(s.note)}</li>`).join("");
    const commands = job.commands?.length ? `<details data-key="cmds-${job.id}"><summary>Commands run</summary>
      <pre class="block small">${job.commands.map((c) => esc(c.command)).join("\n\n")}</pre></details>` : "";
    const log = job.tail?.length ? `<details data-key="${job.state === "failed" ? "log-failed" : `log-${job.id}`}"><summary>Output of ${esc(job.step || "the last step")}</summary>
      <pre class="block small log">${esc(job.tail.join("\n"))}</pre></details>` : "";
    return `<div class="job ${job.state}">
      <div class="job-head"><h3>${head}</h3>${stepper}</div>
      ${bar}${pause}${error}
      <div class="job-actions">${jobButtons(job)}</div>
      ${notes ? `<ul class="job-notes">${notes}</ul>` : ""}
      ${commands}${log}
    </div>`;
  }

  function jobButtons(job) {
    if (!controlOn()) return "";
    const b = (action, label, cls = "") => `<button class="button ${cls}" type="button" data-job-action="${action}">${label}</button>`;
    const training = job.step === "train";
    const out = [];
    const q = job.progress?.sequential;
    const decide = q && q.pairs >= q.decide_from && ["running", "paused"].includes(job.state) ? b("decide_now", "Stop and decide now") : "";
    if (job.state === "running") {
      if (decide) out.push(decide);
      if (training) out.push(job.pause_after_epoch ? `<span class="job-wait">Pauses when this epoch ends.</span>` : b("pause_after_epoch", "Pause after this epoch", "primary"));
      out.push(b("pause", training ? "Pause now" : "Pause"), b("cancel", "Cancel", "quiet-button"));
    } else if (job.state === "queued") {
      out.push(b("pause", "Hold"), b("cancel", "Cancel", "quiet-button"));
    } else if (job.state === "paused") {
      if (training) {
        out.push(b("next_epoch", "Train one more epoch", "primary"), b("run_to_end", "Train to the end"),
          b("evaluate_now", "Evaluate best.nnue now…"), b("end_training", "Stop training and evaluate"),
          b("restart", "Restart training with other settings…"));
      } else {
        out.push(b("continue", "Continue", "primary"));
        if (decide) out.push(decide);
        if (job.step === "evaluate" && byName(job.run)?.config) out.push(b("restart", "Retrain with other settings…"));
      }
      out.push(b("cancel", "Cancel", "quiet-button"));
    } else if (job.state === "failed") {
      out.push(b("continue", "Try again", "primary"));
      if (training || /other settings/.test(job.error || "") || byName(job.run)?.config) out.push(b("restart", training ? "Restart training with other settings…" : "Retrain with other settings…"));
      out.push(b("remove", "Dismiss", "quiet-button"));
    } else if (job.run && byName(job.run)?.config) {
      out.push(b("restart", "Retrain with other settings…"));
    }
    return out.join("");
  }

  function bindJobPanel(slot, job) {
    if (!job) return;
    slot.querySelectorAll("[data-job-action]").forEach((btn) => btn.addEventListener("click", () => {
      const action = btn.dataset.jobAction;
      if (action === "evaluate_now") return evalDialog(job.run, { mode: "fixed", sims: 8, pairs: 100 }, "Evaluate the best network so far");
      if (action === "restart") return restartDialog(job);
      if (action === "cancel" && !confirm("Cancel this job? Its files stay; a batch or training can be continued later by starting it again.")) return;
      if (action === "decide_now" && !confirm("Stop the test and decide from the pairs played so far?")) return;
      const words = { decide_now: "Deciding from the pairs so far", pause_after_epoch: "Pausing when this epoch ends", pause: "Pausing", next_epoch: "Training one more epoch",
        run_to_end: "Training to the end", end_training: "Evaluating the best network so far", continue: "Continuing", cancel: "Cancelled", remove: "Dismissed" };
      btn.disabled = true;
      act(`/api/jobs/${job.id}/${action}`, {}, words[action]).catch(() => (btn.disabled = false));
    }));
    slot.querySelectorAll("[data-goto]").forEach((b) => b.addEventListener("click", () => select(b.dataset.goto)));
  }

  function rememberOpen(root) {
    root.querySelectorAll("details[data-key]").forEach((d) => (d.open ? state.open.add(d.dataset.key) : state.open.delete(d.dataset.key)));
  }
  function restoreOpen(root) {
    root.querySelectorAll("details[data-key]").forEach((d) => {
      if (state.open.has(d.dataset.key)) d.open = true;
      d.addEventListener("toggle", () => (d.open ? state.open.add(d.dataset.key) : state.open.delete(d.dataset.key)));
    });
  }

  // ------------------------------------------------------------------ dialogs
  const dialog = () => $("#dialog");
  function openDialog(html, onSubmit) {
    const dlg = dialog();
    dlg.innerHTML = `<form method="dialog" class="dialog-form">${html}
      <p class="form-error" hidden></p>
      <div class="dialog-buttons"><button class="button quiet-button" type="button" data-close>Cancel</button>
      <button class="button primary" type="submit" data-submit>${esc(dlg.dataset.submit || "Start")}</button></div></form>`;
    const form = dlg.querySelector("form");
    dlg.querySelector("[data-close]").addEventListener("click", () => dlg.close());
    form.addEventListener("submit", async (evt) => {
      evt.preventDefault();
      const err = form.querySelector(".form-error");
      const submit = form.querySelector("[data-submit]");
      submit.disabled = true;
      try {
        await onSubmit(new FormData(form), form);
        dlg.close();
      } catch (e) {
        err.textContent = e.message;
        err.hidden = false;
      } finally {
        submit.disabled = false;
      }
    });
    form.querySelectorAll("[data-toggle]").forEach((el) => {
      // a hidden field is disabled too, so it neither blocks the form's validation nor gets sent
      const sync = () => form.querySelectorAll(`[data-when="${el.name}"]`).forEach((t) => {
        t.hidden = !t.dataset.value.split(",").includes(el.value);
        t.querySelectorAll("input, select").forEach((i) => (i.disabled = t.hidden));
      });
      el.addEventListener("change", sync);
      sync();
    });
    dlg.showModal();
    form.querySelector("input:not([type=hidden]), select")?.focus();
  }

  const field = (label, input, hint = "") => `<label class="form-field"><span>${label}</span>${input}${hint ? `<small>${hint}</small>` : ""}</label>`;
  const numberInput = (name, value, attrs = "") => `<input type="number" name="${name}" value="${esc(value)}" ${attrs}>`;
  const choice = (name, value, options, attrs = "") => `<select name="${name}" ${attrs}>${options.map(([v, l]) =>
    `<option value="${esc(v)}" ${String(v) === String(value) ? "selected" : ""}>${esc(l)}</option>`).join("")}</select>`;
  const budgets = [[8, "20k nodes, fast"], [16, "40k nodes, decides"], [32, "80k nodes"], [96, "240k nodes, long"]];

  function trainingFields(s) {
    const emaMode = s.ema === "auto" || s.ema === "off" ? s.ema : "custom";
    return `<fieldset><legend>Training</legend><div class="grid">
      ${field("Epochs", choice("epochs_mode", s.epochs === "auto" ? "auto" : "custom", [["auto", "Auto (from the data)"], ["custom", "A fixed number"]], "data-toggle"),
        "how often it validates and saves; the amount of training is the passes")}
      <label class="form-field" data-when="epochs_mode" data-value="custom"><span>Number of epochs</span>${numberInput("epochs_value", s.epochs === "auto" ? 20 : s.epochs, 'min="1" max="200"')}</label>
      ${field("Passes over the rows", numberInput("passes", s.passes, 'min="0.5" max="20" step="0.5"'), "3 to 5 does not memorise a batch")}
      ${field("Batch", choice("batch", s.batch, [[1024, "1,024"], [2048, "2,048"], [4096, "4,096"], [8192, "8,192 (the recipe)"], [16384, "16,384"], ["auto", "Auto (experimental)"]]),
        "auto halves it on small data; it played weaker in a direct match here")}
      ${field("Learning rate", numberInput("lr", s.lr, 'min="0.000001" max="0.01" step="any"'), "1e-4 continues a network; 3e-4 from scratch")}
      ${field("Weight averaging", choice("ema_mode", emaMode, [["auto", "About one epoch (auto)"], ["off", "Off"], ["custom", "Custom decay"]], "data-toggle"))}
      <label class="form-field" data-when="ema_mode" data-value="custom"><span>Decay</span>${numberInput("ema_value", typeof s.ema === "number" ? s.ema : 0.999, 'min="0.5" max="0.99999" step="any"')}</label>
      ${field("Device", choice("device", s.device, [["cpu", "CPU"], ["mps", "Apple GPU (mps)"], ["cuda", "NVIDIA GPU (cuda)"]]))}
      ${field("Format", choice("version", s.version, [["auto", "Keep the start's format"], ["9", "Format 9 (unmeasured)"]]))}
      ${field("Stop early", choice("patience", s.patience || "", [["", "Never"], [3, "After 3 epochs without improving"], [5, "After 5 epochs without improving"], [8, "After 8 epochs without improving"]]),
        "saves time on a run that memorises; best.nnue is the best epoch either way")}
    </div>
    <label class="check"><input type="checkbox" name="pause_every_epoch" ${s.pause_every_epoch ? "checked" : ""}> Pause after every epoch</label></fieldset>`;
  }
  function evalFields(e, legend = "Evaluation") {
    return `<fieldset><legend>${legend}</legend><div class="grid">
      ${field("Kind", choice("eval_mode", e.mode, [["sprt", "Sequential test"], ["fixed", "Fixed openings"]], "data-toggle"))}
      ${field("Budget", choice("eval_sims", e.sims, budgets))}
      <label class="form-field" data-when="eval_mode" data-value="fixed"><span>Opening pairs</span>${numberInput("eval_pairs", e.pairs || 100, 'min="2" max="5000"')}<small>100 is a reading, 400 a decision</small></label>
      <label class="form-field" data-when="eval_mode" data-value="sprt"><span>Most pairs</span>${numberInput("eval_cap", e.cap || 3008, 'min="128" step="16"')}<small>usually stops long before; about a third of the batch's search</small></label>
      <label class="form-field" data-when="eval_mode" data-value="sprt"><span>Gain looked for</span>${choice("eval_target", e.target || 0.52, [[0.51, ".51, +7 Elo (slow)"], [0.515, ".515, +10 Elo"], [0.52, ".52, +14 Elo (the base)"], [0.53, ".53, +21 Elo"], [0.54, ".54, +28 Elo"], [0.55, ".55, +35 Elo (generation 0)"], [0.56, ".56, +42 Elo (fast)"]])}<small>higher settles sooner and rejects smaller real gains</small></label>
    </div></fieldset>`;
  }
  const readTraining = (f) => ({
    epochs: f.get("epochs_mode") === "custom" ? +f.get("epochs_value") : "auto", patience: f.get("patience") ? +f.get("patience") : null, passes: +f.get("passes"), batch: f.get("batch") === "auto" ? "auto" : +f.get("batch"), lr: +f.get("lr"), device: f.get("device"),
    version: f.get("version"), pause_every_epoch: f.get("pause_every_epoch") === "on",
    ema: f.get("ema_mode") === "custom" ? +f.get("ema_value") : f.get("ema_mode"),
  });
  const readEval = (f) => ({ mode: f.get("eval_mode"), sims: +f.get("eval_sims"), pairs: +f.get("eval_pairs"),
    cap: +f.get("eval_cap"), target: +f.get("eval_target") });

  function pipelineDialog(s, title, submit) {
    dialog().dataset.submit = submit;
    const sets = (state.data?.sets_on_disk || []).filter((n) => n !== `selfplay_${s.name}`);
    const chosen = new Map((s.extra || []).map((e) => [e.set, e.share]));
    const networks = state.data?.networks || [];
    const html = `<h2>${esc(title)}</h2>
      <fieldset><legend>Start</legend><div class="grid two-col">
        ${field("Starting network", `<input name="start" value="${esc(s.start)}" list="networks" required>
          <datalist id="networks">${networks.map((n) => `<option value="${esc(n)}">`).join("")}</datalist>`)}
        ${field("Name", `<input name="name" value="${esc(s.name)}" required pattern="[A-Za-z0-9_\\-][A-Za-z0-9._\\-]*">`, "a new name; the same name continues that run")}
      </div></fieldset>
      <fieldset><legend>Games</legend><div class="grid">
        ${field("Games", numberInput("games", s.games, 'min="8" step="8"'), "the ladder's size for this generation")}
        ${field("Nodes per move", choice("nodes", s.nodes, [[25000, "25k"], [50000, "50k (the loop's labels)"], [100000, "100k"]]))}
        ${field("Games at once", numberInput("threads", s.threads, 'min="1" max="256"'))}
      </div>
      <label class="check"><input type="checkbox" name="multipv" ${s.multipv === 2 ? "checked" : ""}> Vary the games with near-best alternatives (about 18 % slower)</label>
      <label class="check"><input type="checkbox" name="quiet" ${s.quiet ? "checked" : ""}> Import only quiet positions (unmeasured)</label>
      ${sets.length ? `<div class="sets-pick"><span>Earlier batches to train on</span>
        <div>${sets.map((n) => `<label class="check"><input type="checkbox" name="extra" value="${esc(n)}" ${chosen.has(n) ? "checked" : ""}> ${esc(setLabel(n))}</label>`).join("")}</div></div>` : ""}
      </fieldset>
      ${trainingFields(s)}
      ${evalFields(s.eval)}`;
    openDialog(html, async (f) => {
      const m = /^runs\/([^/]+)\/best\.nnue$/.exec(f.get("start").trim());
      const parent = m && byName(m[1]);
      const evalSettings = readEval(f);
      const capFor = (games) => Math.min(3008, Math.max(128, Math.round((games * (+f.get("nodes") || 50000) / (evalSettings.sims * 2500) / 2 / 3) / 16) * 16));
      if (evalSettings.mode === "sprt" && +f.get("games") !== s.games && evalSettings.cap === (s.eval.cap || 3008)) evalSettings.cap = capFor(+f.get("games"));
      if (parent && s.start !== f.get("start").trim() && evalSettings.mode === "sprt" && Math.abs(evalSettings.target - targetFor(0)) < 1e-9) {
        evalSettings.target = targetFor(parent.depth + 1);  // the start changed to a later generation: its target
      }
      const settings = { ...s, start: f.get("start").trim(), name: f.get("name").trim(), games: +f.get("games"), nodes: +f.get("nodes"),
        threads: +f.get("threads"), multipv: f.get("multipv") === "on" ? 2 : 1, quiet: f.get("quiet") === "on",
        extra: f.getAll("extra").map((n) => ({ set: n, share: chosen.get(n) ?? 1 })), ...readTraining(f), eval: evalSettings };
      await act("/api/jobs/pipeline", settings, `Started ${settings.name}`);
      select(settings.name);
    });
  }

  const targetFor = (depth) => {
    const l = state.data?.ladder || [];
    return l.length ? l[Math.min(Math.max(depth || 0, 0), l.length - 1)].target : 0.52;
  };
  function evalDialog(run, e, title = "Evaluate again") {
    dialog().dataset.submit = "Run the evaluation";
    const r = byName(run);
    const rung = (state.data?.ladder || [])[Math.min(r?.depth || 0, Math.max(0, (state.data?.ladder || []).length - 1))];
    const d = { ...state.control.defaults.eval, target: targetFor(r?.depth), cap: rung?.cap || 3008 };
    const others = [...new Set([...(state.data?.networks || []), byName(r?.chain?.[0]?.name)?.init].filter((n) => n && n !== r?.init && n !== `runs/${run}/best.nnue`))];
    openDialog(`<h2>${esc(title)}</h2><p class="dialog-lede">runs/${esc(run)}/best.nnue. Against its start the result can decide;
      against any other network it measures the distance to that network and is kept apart. It waits for the running job if there is one.</p>
      <div class="grid two-col">${field("Against", choice("reference", e.reference || "", [["", `its start, ${r?.init || "unknown"}`], ...others.map((n) => [n, n])]))}</div>
      ${evalFields({ ...d, ...e }, "Settings")}`, async (f) => {
      await act("/api/jobs/eval", { run, settings: { ...readEval(f), reference: f.get("reference") || null } }, "Evaluation queued");
    });
  }

  function restartDialog(job) {
    // paused inside training: restart in place; anything else retrains from the run's saved configuration
    if (job.kind === "pipeline" && job.step === "train" && ["paused", "failed"].includes(job.state)) {
      dialog().dataset.submit = "Restart training";
      openDialog(`<h2>Restart training with other settings</h2>
        <p class="dialog-lede">A checkpoint only resumes with the settings it began with, so changing them trains again from
        ${esc(job.settings.start)}. The games stay; the current run moves to <code>runs/${esc(job.run)}.replaced_…</code>.</p>
        ${trainingFields(job.settings)}${evalFields(job.settings.eval)}`, async (f) => {
        await act(`/api/jobs/${job.id}/restart_training`, { ...readTraining(f), eval: readEval(f) }, "Training restarts");
      });
      return;
    }
    retrainDialog(job.run);
  }

  function retrainDialog(name) {
    const run = byName(name);
    const cfg = run?.config || {};
    if (!run || !run.config) return toast(`${name} has not been trained yet`, true);
    const waiting = (state.control?.jobs || []).filter((j) => j.run === name && ["queued", "paused"].includes(j.state));
    const rows = Object.values(run.datasets || {}).reduce((a, d) => a + (d?.rows || 0), 0);
    const used = rows && cfg.steps_per_epoch ? (cfg.steps_per_epoch * cfg.epochs * cfg.batch) / rows : null;
    // a tiny batch padded to the 10-step minimum reports hundreds of passes; offer the recipe's 4 then
    const passes = used && used >= 1 && used <= 10 ? Math.round(used * 2) / 2 : 4;
    const settings = { epochs: "auto", passes, batch: cfg.batch || 8192, lr: cfg.lr || 0.0001, ema: cfg.ema ? "auto" : "off",
      device: cfg.device || "cpu", version: cfg.version === 9 ? "9" : "auto", patience: null, pause_every_epoch: false };
    dialog().dataset.submit = "Retrain";
    openDialog(`<h2>Retrain ${esc(name)}</h2>
      <p class="dialog-lede">Trains again from ${esc(cfg.init || "its start")} on exactly the ${run.mixture.length} batch${run.mixture.length === 1 ? "" : "es"}
      it was trained on (${rows ? `${millions(rows)} rows` : "the same rows"}); no games are played. The current run moves to
      <code>runs/${esc(name)}.replaced_…</code> when training starts, with its evaluations.
      ${waiting.length ? `<b>Its ${waiting.length === 1 ? "waiting job is" : `${waiting.length} waiting jobs are`} cancelled.</b>` : ""}
      Last time: ${cfg.epochs} epochs of ${cfg.steps_per_epoch} steps, batch ${cfg.batch}, learning rate ${cfg.lr}${used ? `, about ${used >= 20 ? Math.round(used) : used.toFixed(1)} passes${used > 10 ? " (the batch is too small for the 10-step minimum)" : ""}` : ""}.</p>
      ${trainingFields(settings)}${evalFields({ ...state.control.defaults.eval, target: targetFor(run.depth),
        cap: (state.data.ladder[Math.min(run.depth, state.data.ladder.length - 1)] || {}).cap || 3008 })}`, async (f) => {
      await act("/api/jobs/retrain", { run: name, settings: { ...readTraining(f), eval: readEval(f) } }, `Retraining ${name}`);
    });
  }

  function autoDialog(fromRun) {
    const a = state.control.auto;
    const run = byName(fromRun || state.run);
    const next = run?.decision?.next;
    dialog().dataset.submit = next ? "Start auto" : "Save";
    openDialog(`<h2>Automatic mode</h2>
      <p class="dialog-lede">After each evaluation the dashboard takes the decision itself: the next generation after an
        adoption, another batch from the same start after a null, a sequential test after a reading. It stops at the limits
        below, when a job fails, and when you stop it.</p>
      <div class="grid">
        ${field("Stop after this many new generations", numberInput("max_generations", a.max_generations, 'min="1" max="50"'))}
        ${field("Stop after this many batches", numberInput("max_batches", a.max_batches, 'min="1" max="200"'))}
      </div>
      <label class="check"><input type="checkbox" name="pause_on_problems" ${a.pause_on_problems ? "checked" : ""}> Stop when a training or data check fails</label>
      <label class="check"><input type="checkbox" name="pause_every_epoch" ${a.pause_every_epoch ? "checked" : ""}> Pause its training after every epoch</label>
      ${next ? `<p class="dialog-lede">It starts from <b>${esc(run.name)}</b>: ${esc(run.decision.title.toLowerCase())}.</p>`
        : `<p class="dialog-lede">Select a run with a decision to start from; the settings are saved for then.</p>`}`, async (f) => {
      const cfg = { max_generations: +f.get("max_generations"), max_batches: +f.get("max_batches"),
        pause_on_problems: f.get("pause_on_problems") === "on", pause_every_epoch: f.get("pause_every_epoch") === "on" };
      await api("/api/auto", cfg);
      if (next) await act("/api/auto/start", { run: run.name }, "Auto started");
      else await loadControl(true);
    });
  }

  // ------------------------------------------------------------------ rail
  // A line is a flat list of generations: the batches played from one start, then the batches from the network adopted
  // among them, and so on. Depth never indents, so a line 30 generations deep reads like one 3 deep; older generations
  // fold away, and a second network used as a start (a fork) gets its own line below.
  function lines() {
    const runs = state.data.runs;
    const pending = (state.control?.jobs || []).filter((j) => j.run && !byName(j.run) && ["running", "queued", "paused"].includes(j.state));
    const groups = new Map();
    const add = (key, item) => { if (!groups.has(key)) groups.set(key, []); groups.get(key).push(item); };
    for (const r of runs) add(r.parent || `init:${r.init}`, { run: r, name: r.name, started: r.started });
    for (const j of pending) {
      const m = /^runs\/([^/]+)\/best\.nnue$/.exec(j.settings.start || "");
      add(m && byName(m[1]) ? m[1] : `init:${j.settings.start}`, { job: j, name: j.run, started: j.created });
    }
    // the main line follows the network with the longest line of descendants, then the most recent one
    const reach = new Map();
    const deepest = (name, guard = 0) => {
      if (reach.has(name)) return reach.get(name);
      const kids = (groups.get(name) || []).filter((i) => i.run && groups.has(i.name));
      const value = guard > 2000 ? 0 : 1 + Math.max(0, ...kids.map((k) => deepest(k.name, guard + 1)));
      reach.set(name, value);
      return value;
    };
    const latestChild = (name) => Math.max(0, ...(groups.get(name) || []).map((i) => i.started));
    const out = [];
    const queue = [...groups.keys()].filter((k) => k.startsWith("init:"))
      .sort((a, b) => Math.min(...groups.get(a).map((i) => i.started)) - Math.min(...groups.get(b).map((i) => i.started)))
      .map((k) => ({ key: k, title: `From ${k.slice(5).split("/").pop() || "an unknown network"}`, depth: 0 }));
    const done = new Set();
    while (queue.length) {
      const { key, title, depth: first } = queue.shift();
      const gens = [];
      let current = key, depth = first;
      while (current && groups.has(current) && !done.has(current) && gens.length < 1000) {
        done.add(current);
        const items = [...groups.get(current)].sort((a, b) => a.started - b.started);
        gens.push({ depth, items });
        const starts = items.filter((i) => i.run && groups.has(i.name))
          .sort((a, b) => deepest(b.name) - deepest(a.name) || latestChild(b.name) - latestChild(a.name));
        starts.slice(1).forEach((f) => queue.push({ key: f.name, title: `Branch from ${f.name}`, depth: depth + 1 }));
        current = starts[0]?.name;
        depth += 1;
      }
      if (gens.length) out.push({ key, title, gens });
    }
    return out;
  }

  function renderTree() {
    const nav = $("#tree");
    const all = lines();
    if (!all.length) { nav.innerHTML = '<p class="orphan-note">No runs yet.</p>'; return; }
    const SHOW = 4;
    const node = (item) => {
      if (item.job) {
        const j = item.job;
        return `<button class="node" data-run="${esc(j.run)}" aria-current="${j.run === state.run}">${mark("running")}
          <span class="node-name">${esc(j.run)}</span><span class="node-score">${j.state === "queued" ? "waiting" : j.state}</span></button>`;
      }
      const r = item.run, st = r.status;
      const live = LIVE.includes(st.state);
      const right = live ? (typeof st.fraction === "number" ? `${Math.floor(st.fraction * 100)} %` : st.state === "paused" ? "paused" : "…") : r.eval ? score(r.eval.score) : "";
      return `<button class="node${r.outcome === "adopt" ? " adopted" : ""}" data-run="${esc(r.name)}" aria-current="${r.name === state.run}" title="${esc(VERDICT_WORDS[r.decision.verdict] || "")}">
        ${mark(r.decision.verdict)}<span class="node-name">${esc(r.name)}</span><span class="node-score">${right}</span></button>`;
    };
    nav.innerHTML = all.map((line) => {
      const holdsCurrent = (g) => g.items.some((i) => i.name === state.run);
      const expanded = state.open.has(`line-${line.key}`) || (!state.open.has(`folded-${line.key}`) && line.gens.slice(0, -SHOW).some(holdsCurrent));
      const hidden = expanded ? 0 : Math.max(0, line.gens.length - SHOW);
      const fold = line.gens.length > SHOW
        ? `<button class="fold" type="button" data-fold="${esc(line.key)}">${expanded ? "Fold older generations"
          : `Generations ${line.gens[0].depth} to ${line.gens[hidden - 1].depth} (${line.gens.slice(0, hidden).reduce((a, g) => a + g.items.length, 0)} batches)`}</button>` : "";
      return `<div class="line"><p class="line-title">${esc(line.title)}</p>${fold}
        ${line.gens.slice(hidden).map((g) => `<div class="gen"><span class="gen-label" title="generation ${g.depth}">${g.depth}</span>
          <div class="gen-runs">${g.items.map(node).join("")}</div></div>`).join("")}</div>`;
    }).join("");
    nav.querySelectorAll(".node").forEach((b) => b.addEventListener("click", () => select(b.dataset.run)));
    nav.querySelectorAll("[data-fold]").forEach((b) => b.addEventListener("click", () => {
      const key = b.dataset.fold;
      if (b.textContent.startsWith("Fold")) { state.open.delete(`line-${key}`); state.open.add(`folded-${key}`); }
      else { state.open.add(`line-${key}`); state.open.delete(`folded-${key}`); }
      renderTree();
    }));
    nav.querySelector('[aria-current="true"]')?.scrollIntoView({ block: "nearest" });
  }

  // ------------------------------------------------------------------ main
  function render() {
    renderTree();
    const main = $("#main");
    rememberOpen(main);
    const run = current();
    if (!run) {
      const job = pendingJob(state.run);
      if (job) {
        main.innerHTML = `<header class="run-head"><div><h2>${esc(job.run)}</h2>
          <p class="run-meta">From <code>${esc(job.settings.start)}</code>, ${(job.settings.games || 0).toLocaleString()} games. Its files appear as the first step writes them.</p></div></header>
          <div id="job-panel"></div>`;
        renderJobPanel();
        return;
      }
      const plan = state.data.plan;
      main.innerHTML = `<div class="empty"><h2>No runs in this directory yet</h2>
        ${controlOn() ? `<p>Start the first line from the example network: ${plan.ladder[0].toLocaleString()} games at ${(plan.nodes / 1000).toFixed(0)}k nodes,
          trained, then tested by the sequential test. Every step runs from this page.</p>
          <p><button class="button primary" type="button" id="first-line">Start a new line…</button></p>`
        : `<p>Start the first one from the repository root:</p>`}
        <details data-key="first-cli" ${controlOn() ? "" : "open"}><summary>The command-line equivalent</summary><pre class="block">GAMES=${plan.ladder[0]} NODES=${plan.nodes} SPRT=1 SIMS=${plan.decision_sims} \\
  models/nnue/quickstart.sh models/nnue/examples/example.nnue first</pre></details></div>`;
      $("#first-line")?.addEventListener("click", () => pipelineDialog(newLineSettings(), "Start a new line", "Start"));
      return;
    }
    main.innerHTML = [aims(), head(run), `<div id="job-panel"></div>`, decision(run), lineage(run), evaluation(run), gamesSection(run), training(run), checks(run), data(run)].join("");
    renderJobPanel();
    bind(run);
    restoreOpen(main);
    drawCharts(run);
    if (state.games[run.name]?.summary) drawGames(run);
    refreshGames(run, true);
  }

  function aims() {
    const p = state.data.plan;
    return `<p class="aims">Aiming for labels at <b>${(p.nodes / 1000).toFixed(0)}k nodes</b>, decisions by the sequential test at
      <b>${(p.decision_sims * 2500 / 1000).toFixed(0)}k nodes</b> (a gain of ${score(p.sprt_targets[0])} in generation 0, easing to ${score(p.sprt_targets[p.sprt_targets.length - 1])}, about +14 Elo), and batches growing from
      <b>${p.ladder[0].toLocaleString()}</b> to <b>${p.ladder[p.ladder.length - 1].toLocaleString()}</b> games as the line gets stronger.</p>`;
  }

  function head(run) {
    const cfg = run.config || {};
    const gen = run.generation;
    const from = run.parent
      ? `<button class="link" data-goto="${esc(run.parent)}">${esc(run.parent)}</button>`
      : run.init ? `<code>${esc(run.init)}</code>` : "an unknown network";
    const bits = [
      `From ${from}`,
      `generation ${run.depth}`,
      gen ? `${gen.games_target?.toLocaleString() ?? "?"} games at ${(gen.nodes / 1000).toFixed(0)}k nodes` : null,
      cfg.hidden ? `H${cfg.hidden}, format ${cfg.version}` : null,
      cfg.device && cfg.device !== "cpu" ? `trained on ${cfg.device}` : null,
    ].filter(Boolean);
    const kids = run.children.length
      ? `. Continued by ${run.children.map((c) => `<button class="link" data-goto="${esc(c)}">${esc(c)}</button>`).join(", ")}`
      : "";
    const guessed = run.status.source === "files" && ["generating", "preparing", "training", "evaluating", "stopped"].includes(run.status.state)
      ? `<p class="guess">Started outside the dashboard: ${esc(run.status.detail)}${run.status.state === "stopped" ? ", nothing written for 30 minutes" : ", judged from when its files last changed"}.</p>` : "";
    return `<header class="run-head"><div><h2>${esc(run.name)}</h2>
      <p class="run-meta">${bits.join(", ")}${kids}.</p>${guessed}</div></header>`;
  }

  // ------------------------------------------------------------------ decision
  function ladderPanel(run) {
    const d = run.decision;
    const plan = state.data.plan;
    const rungs = state.data.ladder;
    const batch = d.batch;
    const max = Math.log2(Math.max(rungs[rungs.length - 1].games * plan.retry_growth_max, batch ? batch.games : 0) / rungs[0].games) + 1;
    const width = (g) => (100 * (Math.log2(g / rungs[0].games) + 1)) / max;
    const rows = rungs.map((r, i) => {
      const last = i === rungs.length - 1;
      const on = batch && Math.min(batch.generation, rungs.length - 1) === i;
      const label = last ? `${i} and on` : `${i}`;
      const grown = on && !batch.current && batch.games > batch.base
        ? `<span class="rung-grow" style="width:${width(batch.games) - width(batch.base)}%" title="doubled after ${batch.nulls} null batch${batch.nulls === 1 ? "" : "es"}"></span>` : "";
      return `<div class="rung ${on ? "on" : ""}">
        <span class="rung-gen">${label}</span>
        <span class="rung-track"><span class="rung-bar" style="width:${width(r.games)}%"></span>${grown}</span>
        <span class="rung-games">${(on ? batch.games : r.games).toLocaleString()}</span>
        <span class="rung-target" title="the sequential test's target at this generation">${String(+r.target.toFixed(3)).replace(/^0/, "")}</span>
      </div>`;
    }).join("");
    const note = batch.current
      ? `<p class="ladder-note">This run's batch, ${batch.games.toLocaleString()} games at generation ${batch.generation}${batch.games < batch.base ? `, under the ${batch.base.toLocaleString()} planned` : ""}. The next batch is proposed once it has a result.</p>`
      : `<p class="ladder-note">The proposed batch: about ${millions(batch.rows)} fresh rows${batch.games > batch.base ? `, ${batch.games / batch.base}× the plan after ${batch.nulls} batch${batch.nulls === 1 ? "" : "es"} without a gain` : ""}.</p>`;
    return `<div class="ladder" aria-label="Batch size by generation">
      <h4>Games and test target by generation</h4>${rows}${note}</div>`;
  }

  function decision(run) {
    const d = run.decision;
    if (d.verdict === "running" && run.status.source === "runner") return "";  // the job panel above says it
    const on = controlOn();
    const cli = (label, command) => `<details class="cli" data-key="cli-${esc(run.name)}-${esc(label)}"><summary>${on ? "The command-line equivalent" : esc(label)}</summary>
        <div class="command-top"><span></span><button class="copy" type="button" data-copy="${esc(command.replace(/ \\\n\s*/g, " "))}">Copy</button></div>
        <pre>${esc(command)}</pre></details>`;
    const next = d.next;
    let actions = "";
    if (on && next && d.verdict !== "running") {
      const s = next.settings || {};
      const label = next.kind === "eval" ? "Run the sequential test"
        : d.verdict === "stopped" ? `Continue ${esc(run.name)}`
        : `Start ${esc(next.run)}: ${(s.games || 0).toLocaleString()} games`;
      const auto = state.control.auto.enabled ? "" : `<button class="button" type="button" data-auto-from>Hand over to auto…</button>`;
      if (d.followed_up && d.verdict === "adopt") {
        return decisionShell(d, `<div class="command done"><div class="command-top"><span>${esc(d.followed_up)}</span></div>${d.command ? cli(d.command_label, d.command) : ""}</div>`);
      }
      actions = `<div class="command actions">
        ${d.followed_up ? `<p class="followed">${esc(d.followed_up)}</p>` : ""}
        <div class="action-row">
          <button class="button ${d.followed_up ? "" : "primary"}" type="button" data-next>${label}${d.followed_up ? " anyway" : ""}</button>
          ${next.kind === "pipeline" ? `<button class="button" type="button" data-next-edit>Change settings first…</button>` : ""}
          ${d.followed_up ? "" : auto}
        </div>
        ${d.command ? cli(d.command_label, d.command) : ""}</div>`;
    } else if (d.command) {
      actions = d.followed_up
        ? `<div class="command done"><div class="command-top"><span>${esc(d.followed_up)}</span>
             <button class="copy" type="button" data-copy="${esc(d.command.replace(/ \\\n\s*/g, " "))}">Copy the command anyway</button></div></div>`
        : `<div class="command"><div class="command-top"><span>${esc(d.command_label)}</span>
             <button class="copy" type="button" data-copy="${esc(d.command.replace(/ \\\n\s*/g, " "))}">Copy</button></div>
             <pre>${esc(d.command)}</pre></div>`;
    }
    return decisionShell(d, actions);

    function decisionShell(d, actions) {
      const rule = { adopt: "It cleared the line: the next generation starts from it.",
        more_data: "Neither a gain nor a loss yet: more data from the same start decides it.",
        abort: "Weaker than its start: the network goes, its games stay.",
        reeval: / again at /.test(d.title) ? "" : "A reading is not a decision: the sequential test settles it.",
        evaluate: "The decision needs an evaluation.", running: "", stopped: "", incomplete: "" }[d.verdict];
      return `<div class="decision tone-${d.verdict}">
        <div class="verdict">
          <h3>${esc(d.title)}</h3>
          <p class="summary">${esc(d.summary)}${rule ? ` <span class="rule">${esc(rule)}</span>` : ""}</p>
          <ul>${d.reasons.map((r) => `<li>${esc(r)}</li>`).join("")}</ul>
        </div>
        ${ladderPanel(run)}
        ${actions}
      </div>`;
    }
  }

  // ------------------------------------------------------------------ lineage
  function lineage(run) {
    const steps = run.chain;
    if (!steps.some((s) => typeof s.elo === "number")) return "";
    const root = byName(steps[0].name)?.init;
    const directs = steps.filter((s) => s.direct).length;
    const measure = controlOn() && run.has_net && root && root !== run.init
      ? `<button class="button" type="button" data-measure="${esc(root)}">Measure against ${esc(root.split("/").slice(-2).join("/").replace("/best.nnue", ""))}…</button>` : "";
    const last = steps[steps.length - 1];
    const total = [...steps].reverse().find((s) => s.counted && typeof s.total === "number");
    return `<section>
      <h3>Strength along this line</h3>
      <p class="lede">Each network's gain against the network it started from, stacked from
        <b>${esc(steps[0] && byName(steps[0].name)?.start_label)}</b>.
        ${total ? `The line stands at <b>${elo(total.total)} Elo</b> (${elo(total.total_interval[0])} to ${elo(total.total_interval[1])}).` : ""}
        Steps are measured against different opponents at a shallow budget, so the total is a guide, and its range widens with every step.</p>
      <div class="chart-main"><div id="staircase"></div>
        <div class="legend"><span><span class="swatch" style="background:var(--blue)"></span>stacked from each step's test</span>
        <span><span class="swatch" style="background:var(--plum)"></span>measured directly against the first network${directs ? "" : " (none yet)"}</span></div></div>
      ${measure ? `<div class="lineage-tools"><p class="small-note">A direct match against the first network has no stacking error; it is the better number for how far the line has come.</p>${measure}</div>` : ""}
      ${!last.counted && typeof last.elo === "number"
        ? `<p class="small-note">${esc(run.name)} is not adopted, so its step is dashed and left out of the total.</p>` : ""}
    </section>`;
  }

  function staircaseSteps(run) {
    return run.chain.map((s, i) => {
      const r = byName(s.name);
      const tone = r.eval ? TONE[r.eval.verdict] : "var(--grey)";
      return {
        name: s.name, elo: s.elo, lo: s.elo_interval?.[0], hi: s.elo_interval?.[1], total: s.total,
        direct: s.direct ? { elo: s.direct.elo, lo: s.direct.elo_interval[0], hi: s.direct.elo_interval[1] } : null,
        totalLo: s.total_interval?.[0], totalHi: s.total_interval?.[1], tone, attempts: s.attempts, nulls: s.nulls,
        open: !r.eval, current: i === run.chain.length - 1, counted: s.counted,
        detail: `<b>${esc(s.name)}</b>
          <div class="row"><span>step</span><b>${elo(s.elo)} (${elo(s.elo_interval?.[0])} to ${elo(s.elo_interval?.[1])})</b></div>
          <div class="row"><span>line total</span><b>${elo(s.total)}</b></div>
          <div class="row"><span>measured</span><b>${s.sims ? (s.sims * 2500).toLocaleString() + " nodes" : "–"}, ${esc(s.kind || "")}</b></div>
          <div class="row"><span>batches from its start</span><b>${s.attempts} (${s.nulls} without a gain)</b></div>
          ${s.direct ? `<div class="row"><span>measured directly</span><b>${elo(s.direct.elo)} (${elo(s.direct.elo_interval[0])} to ${elo(s.direct.elo_interval[1])}), ${s.direct.pairs} pairs</b></div>` : ""}`,
      };
    });
  }

  // ------------------------------------------------------------------ evaluation
  const activeEval = (run) => (run.all_evals || run.evals).find((e) => e.file === state.evalFile) || run.eval || (run.all_evals || [])[0];

  function evaluation(run) {
    const all = run.all_evals || run.evals;
    const ev = activeEval(run);
    const again = controlOn() && run.has_net ? `<button class="button" type="button" data-eval-again>Evaluate${all.length ? " again" : ""}…</button>` : "";
    if (!ev) {
      return `<section><div class="section-head"><h3>Evaluation</h3>${again}</div>
        <p class="missing">No evaluation of this run's best.nnue yet.</p></section>`;
    }
    const versus = ev.against_start ? "its start" : ev.opponent;
    const table = all.length > 1 || !ev.against_start ? `<div class="table-scroll"><table class="kv evals-table">
      <thead><tr><th>Against</th><th>Test</th><th>Pairs</th><th>Score</th><th>Elo</th><th>Result</th><th>When</th></tr></thead><tbody>
      ${all.map((e) => {
        const result = e.against_start
          ? ({ better: "stronger", better_small: "stronger (small)", retest: "retest", worse: "weaker", null: "undecided", running: "running", invalid: "invalid" }[e.verdict] || e.verdict)
          : e.score_interval[0] > 0.5 ? "stronger" : e.score_interval[1] < 0.5 ? "weaker"
            : e.los < 0.1 ? `leans weaker (${Math.round(e.los * 100)} % stronger)` : e.los > 0.9 ? `leans stronger (${Math.round(e.los * 100)} %)` : "no clear difference";
        const test = e.kind === "sequential" ? `sequential ${String(+(e.sprt_target || 0.52).toFixed(3)).replace(/^0/, "")}${e.stop ? `, ${e.stop}` : ""}` : "fixed";
        return `<tr data-eval="${esc(e.file)}" tabindex="0" class="${e.file === ev.file ? "on" : ""}${e.stale ? " stale" : ""}">
          <th scope="row">${esc(e.opponent)}${e.against_start ? "" : ' <small class="muted">measurement</small>'}</th>
          <td>${esc(test)}, ${(e.sims * 2500 / 1000).toFixed(0)}k</td><td>${e.pairs.toLocaleString()}</td>
          <td>${score(e.score)} <small class="muted">${score(e.score_interval[0])} to ${score(e.score_interval[1])}</small></td>
          <td>${elo(e.elo)}</td><td>${esc(result)}${e.stale ? ' <small class="muted">older network</small>' : ""}</td>
          <td>${new Date(e.mtime * 1000).toLocaleString([], { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" })}</td></tr>`;
      }).join("")}</tbody></table></div>` : "";
    const sequential = ev.kind === "sequential"
      ? `<div class="sprt"><div id="sprt-track"></div>
          <p class="small-note">${ev.pairs.toLocaleString()} of at most ${ev.sprt_cap.toLocaleString()} pairs, testing .50 against ${ev.sprt_target}.
          ${ev.stop === "inconclusive" ? "Stopped at the cap." : ev.stop === "stopped" ? "Stopped by hand." : ""}</p></div>`
      : "";
    const verdictWord = ev.against_start
      ? { better: "stronger", better_small: "stronger by a small margin", retest: "below the raised target but likely better", worse: "weaker", null: "not decided", running: "still running", invalid: "invalid" }[ev.verdict]
      : ev.score_interval[0] > 0.5 ? `stronger than ${ev.opponent}` : ev.score_interval[1] < 0.5 ? `weaker than ${ev.opponent}`
        : ev.los < 0.1 ? `probably weaker than ${ev.opponent} (${Math.round(ev.los * 100)} % chance it is stronger)`
        : ev.los > 0.9 ? `probably stronger than ${ev.opponent} (${Math.round(ev.los * 100)} %)` : `no clear difference from ${ev.opponent}`;
    const lede = !ev.against_start
      ? `A direct match against <b>${esc(ev.opponent)}</b>, ${ev.pairs} pairs at ${(ev.sims * 2500).toLocaleString()} nodes: <b>${esc(verdictWord)}</b>. A measurement, never the decision.`
      : ev.kind === "sequential"
        ? `A sequential test at ${(ev.sims * 2500).toLocaleString()} nodes: <b>${verdictWord}</b>. Its interval is descriptive; the stop decides.`
        : `${ev.pairs} opening pairs at ${(ev.sims * 2500).toLocaleString()} nodes, ${ev.grade === "decision" ? "a decision" : "a reading"}: <b>${verdictWord}</b>.`;
    const total = ev.games || 1;
    return `<section>
      <div class="section-head"><h3>Evaluation</h3><div class="section-tools">${again}</div></div>
      <p class="lede">${lede} ${ev.stale ? "<b>This result is older than the current best.nnue.</b> " : ""}Bars are 95 % intervals for the score against ${esc(versus)}; faded ones are other results against the same opponent${ev.against_start ? " and other batches from the same start" : ""}.${table ? " Pick a row to show it." : ""}</p>
      ${table}
      <div class="eval-grid">
        <div>
          <div class="ruler" id="ruler"></div>
          ${sequential}
        </div>
        <div>
          <div class="wdl">
            <div class="wdl-bar" aria-hidden="true">
              <span style="width:${(100 * ev.wins) / total}%;background:var(--blue)"></span>
              <span style="width:${(100 * ev.draws) / total}%;background:var(--line-strong)"></span>
              <span style="width:${(100 * ev.losses) / total}%;background:var(--red)"></span>
            </div>
            <div class="wdl-legend"><span>${ev.wins} won</span><span>${ev.draws} drawn</span><span>${ev.losses} lost</span></div>
          </div>
          <dl class="facts">
            <dt>Score</dt><dd>${score(ev.score)} <span class="range">${score(ev.score_interval[0])} to ${score(ev.score_interval[1])}</span></dd>
            <dt>Elo</dt><dd>${elo(ev.elo)} <span class="range">${elo(ev.elo_interval[0])} to ${elo(ev.elo_interval[1])}</span></dd>
            <dt title="Gain divided by the spread of the pair results: unaffected by the draw rate">Normalised Elo</dt><dd>${elo(ev.nelo)}</dd>
            <dt title="Probability that the score is above .50 given this sample">Chance it is stronger</dt><dd>${pct(ev.los, ev.los > 0.99 || ev.los < 0.01 ? 1 : 0)}</dd>
            <dt title="The smallest gain this sample could tell from zero">Resolution</dt><dd>±${Math.round(ev.resolution_elo ?? 0)} <small>Elo</small></dd>
            <dt>Sample</dt><dd>${ev.pairs.toLocaleString()} <small>pairs</small> <span class="range">${ev.games.toLocaleString()} games, ${pct(ev.draw_rate, 0)} drawn</span></dd>
            ${ev.pairs_needed && ev.pairs_needed > ev.pairs ? `<dt>To show this gain</dt><dd>${ev.pairs_needed.toLocaleString()} <small>pairs</small></dd>` : ""}
            <dt>Game length</dt><dd>${num(ev.mean_plies, 3)} <small>plies</small></dd>
          </dl>
          ${ev.against_start ? `<p class="elo-note">${esc(state.data.depth_note)}</p>` : ""}
        </div>
      </div>
    </section>`;
  }

  function rulerRows(run) {
    const rows = [];
    const detail = (r, e) => `<b>${esc(r.name)}</b> <span>against ${esc(e.opponent || "its start")}</span>
        <div class="row"><span>score</span><b>${score(e.score)} (${score(e.score_interval[0])} to ${score(e.score_interval[1])})</b></div>
        <div class="row"><span>Elo</span><b>${elo(e.elo)} (${elo(e.elo_interval[0])} to ${elo(e.elo_interval[1])})</b></div>
        <div class="row"><span>pairs, nodes</span><b>${e.pairs}, ${(e.sims * 2500).toLocaleString()}</b></div>`;
    const add = (r, e, ghost, label) => rows.push({ label, lo: e.score_interval[0], hi: e.score_interval[1], point: e.score,
      tone: e.against_start === false ? (e.score_interval[0] > 0.5 ? "var(--teal)" : e.score_interval[1] < 0.5 ? "var(--red)" : "var(--ochre)") : TONE[e.verdict],
      ghost, detail: detail(r, e) });
    const shown = activeEval(run);
    const tag = (e) => `${e.pairs}p ${(e.sims * 2500 / 1000).toFixed(0)}k`;
    const same = (run.all_evals || run.evals).filter((e) => e.file !== shown.file && (shown.against_start ? e.against_start : e.reference_path === shown.reference_path));
    add(run, shown, false, same.length ? `${run.name}, ${tag(shown)}` : run.name);
    same.forEach((e) => add(run, e, true, tag(e)));
    if (shown.against_start) run.siblings.map(byName).filter((s) => s && s.eval).forEach((s) => add(s, s.eval, true, s.name));
    return rows;
  }

  // ------------------------------------------------------------------ games
  const END_WORDS = { Goal: "reached the goal", Elimination: "eliminated", Stalemate: "stalemate", CaptureClock: "capture clock",
    PlyCap: "ply cap", Forfeit: "forfeit", Interrupted: "interrupted" };
  const END_COLORS = { Goal: "var(--blue)", Elimination: "var(--teal)", Stalemate: "var(--plum)", CaptureClock: "var(--ochre)",
    PlyCap: "var(--grey)", Forfeit: "var(--red)", Interrupted: "var(--grey)" };
  const resultWord = (g) => (g.winner === 0 ? "Blue won" : g.winner === 1 ? "Red won" : g.censored ? "no result" : "draw");
  state.games = {};

  function gamesSection(run) {
    if (!run.generation) return "";
    return `<section id="games">
      <div class="section-head"><h3>Games</h3><span class="muted small-note" id="games-progress"></span></div>
      <p class="lede">What this batch's self-play looks like. Blue moves first from a1 and heads for i9; a game ends on a goal
        square, when a side loses its last piece or has no move, or after too many plies without a capture.</p>
      <div id="games-body"><p class="missing">Reading the games…</p></div>
    </section>`;
  }

  async function refreshGames(run, full) {
    const body = $("#games-body");
    if (!body || !run.generation) return;
    const g = state.games[run.name] || (state.games[run.name] = { end: "", sort: "id", offset: 0, limit: 10 });
    try {
      const [summary, list] = await Promise.all([
        fetch(`/api/games?run=${encodeURIComponent(run.name)}`).then((r) => r.json()),
        fetch(`/api/games/list?run=${encodeURIComponent(run.name)}&end=${g.end}&sort=${g.sort}&offset=${g.offset}&limit=${g.limit}`).then((r) => r.json()),
      ]);
      if (state.run !== run.name || !$("#games-body")) return;
      g.summary = summary;
      g.list = list;
      if ((full || !g.trends) && run.chain.length > 1) {
        const names = run.chain.map((c) => c.name).join(",");
        g.trends = (await fetch(`/api/games/trends?runs=${encodeURIComponent(names)}`).then((r) => r.json())).runs;
      }
      drawGames(run);
    } catch (err) {
      body.innerHTML = `<p class="missing">The games could not be read: ${esc(err.message)}</p>`;
    }
    clearTimeout(refreshGames.timer);
    const st = g.summary?.stats;
    const reading = st && st.shards_ready < st.shards;
    const trendsReading = (g.trends || []).some((t) => t.shards_ready < t.shards);
    if (state.run === run.name && (reading || trendsReading || (g.summary?.live || []).length)) {
      refreshGames.timer = setTimeout(() => current() && refreshGames(current(), reading || trendsReading), reading || trendsReading ? 1500 : 3000);
    }
  }

  function drawGames(run) {
    const g = state.games[run.name];
    const el = document.activeElement;
    if (el && $("#games-body")?.contains(el) && el.tagName === "SELECT") {
      el.addEventListener("blur", () => current()?.name === run.name && drawGames(run), { once: true });
      return;
    }
    const st = g.summary.stats, live = g.summary.live || [], list = g.list;
    const body = $("#games-body");
    if (!body) return;
    $("#games-progress").textContent = st.shards_ready < st.shards ? `reading ${st.shards_ready} of ${st.shards} shards` : "";
    if (!st.games && !live.length) {
      body.innerHTML = `<p class="missing">No finished games published yet${st.shards ? " (still reading)" : ""}.</p>`;
      return;
    }
    const ends = Object.entries(st.ends || {}).sort((a, b) => b[1] - a[1]);
    const endBar = st.games ? `<div class="ends"><div class="ends-bar">${ends.map(([e, n]) =>
      `<span style="width:${(100 * n) / st.games}%;background:${END_COLORS[e] || "var(--grey)"}" title="${esc(e)}: ${n}"></span>`).join("")}</div>
      <ul class="ends-legend">${ends.map(([e, n]) => `<li><span class="swatch" style="background:${END_COLORS[e] || "var(--grey)"}"></span>
        ${esc(END_WORDS[e] || e)} <b>${pct(n / st.games, n / st.games < 0.1 ? 1 : 0)}</b></li>`).join("")}</ul></div>` : "";
    const facts = st.games ? `<dl class="game-facts">
      <div><dt>Games</dt><dd>${st.games.toLocaleString()}</dd></div>
      <div><dt>Length</dt><dd>${Math.round(st.length_median)} <small>plies</small><span class="range">${st.length_p10} to ${st.length_p90}</span></dd></div>
      <div><dt title="Blue moves first; .50 means no first-move edge">First mover's score</dt><dd>${score(st.blue_score)}</dd></div>
      <div><dt>Draws</dt><dd>${pct(st.draw_share, 1)}</dd></div>
      <div><dt title="The ply from which the search's value stayed on the winner's side (above .9)">Decided by</dt><dd>${st.decided_median ? `ply ${Math.round(st.decided_median)}` : "–"}<span class="range">${pct(st.decided_share, 0)} of games</span></dd></div>
      <div><dt>Captures per game</dt><dd>${num(st.captures_mean, 3)}</dd></div>
      <div><dt>Search depth</dt><dd>${num(st.depth_mean, 3)}</dd></div>
      <div><dt>Random moves per game</dt><dd>${num(st.random_mean, 2)}</dd></div>
    </dl>` : "";
    const trends = (g.trends || []).filter((t) => t.games);
    const trendTable = trends.length > 1 ? `<h4 class="sub">Along the line</h4><div class="table-scroll"><table class="kv trend">
      <thead><tr><th>Batch</th><th>Games</th><th>Median length</th><th>Draws</th><th>Clock draws</th><th>First mover</th><th>Decided by</th></tr></thead><tbody>
      ${trends.map((t) => `<tr class="${t.run === run.name ? "on" : ""}"><th scope="row"><button class="link" data-goto="${esc(t.run)}">${esc(t.run)}</button></th>
        <td>${t.games.toLocaleString()}</td><td>${Math.round(t.length_median)}</td><td>${pct(t.draw_share, 1)}</td>
        <td>${pct((t.ends?.CaptureClock || 0) / t.games, 1)}</td><td>${score(t.blue_score)}</td><td>${t.decided_median ? `ply ${Math.round(t.decided_median)}` : "–"}</td></tr>`).join("")}
      </tbody></table></div>` : "";
    const liveCards = live.length ? `<h4 class="sub">Being played now</h4><div class="live">${live.map((x) =>
      `<button class="live-game" type="button" data-live="${x.id}"><span class="mini" data-mini="${x.id}"></span>
        <span class="live-label">Game ${x.id}, ply ${x.plies}</span><span class="value-bar" title="the search's value for Blue"><span style="left:${50 + 50 * (x.value || 0)}%"></span></span></button>`).join("")}</div>` : "";
    const opts = (items, value) => items.map(([v, l]) => `<option value="${v}" ${v === value ? "selected" : ""}>${l}</option>`).join("");
    const browser = st.games ? `<h4 class="sub">Browse</h4>
      <div class="controls">
        <label class="field">Show <select id="games-end">${opts([["", "All games"], ["blue", "Blue wins"], ["red", "Red wins"], ["draw", "Draws"],
          ...Object.keys(st.ends || {}).map((e) => [e, `Ended by ${END_WORDS[e] || e}`])], g.end)}</select></label>
        <label class="field">Order <select id="games-sort">${opts([["id", "As played"], ["longest", "Longest first"], ["shortest", "Shortest first"],
          ["captures", "Most captures"], ["earliest_decided", "Decided earliest"]], g.sort)}</select></label>
        <label class="field">Per page <select id="games-limit">${opts([["10", "10"], ["25", "25"], ["50", "50"], ["100", "100"]], String(g.limit))}</select></label>
      </div>
      <div class="table-scroll"><table class="kv games-table"><thead><tr><th>Game</th><th>Result</th><th>How</th><th>Plies</th><th>Captures</th><th>Decided by</th><th>Pieces left</th></tr></thead><tbody>
      ${list.games.map((x) => `<tr data-game="${x.id}" tabindex="0"><th scope="row">${x.id}</th><td>${resultWord(x)}</td><td>${esc(END_WORDS[x.end] || x.end)}</td>
        <td>${x.plies}</td><td>${x.captures}</td><td>${x.decided ?? "–"}</td><td>${x.blue_left} : ${x.red_left}</td></tr>`).join("")}
      </tbody></table></div>
      <div class="pager"><button class="button" type="button" data-page="-1" ${g.offset ? "" : "disabled"}>Previous</button>
        <span>${list.total ? `${g.offset + 1} to ${Math.min(list.total, g.offset + g.limit)} of ${list.total.toLocaleString()}` : "no games"}</span>
        <button class="button" type="button" data-page="1" ${g.offset + g.limit < list.total ? "" : "disabled"}>Next</button></div>` : "";
    body.innerHTML = `${facts}<div class="games-charts">${st.games ? `<div><h4 class="sub">Game length</h4><div id="length-hist"></div></div>
      <div><h4 class="sub">How games end</h4>${endBar}</div>` : ""}</div>${trendTable}${liveCards}${browser}`;
    if (st.games) {
      const h = st.histogram;
      Charts.barChart($("#length-hist"), { values: h.counts, labels: h.counts.map((_, i) => String(i * h.width)), height: 170, color: "var(--blue)",
        xTitle: "plies", tip: (i) => `<b>${i * h.width} to ${(i + 1) * h.width - 1} plies</b><div class="row"><span>games</span><b>${h.counts[i]}</b></div>` });
    }
    live.forEach((x) => Charts.board(body.querySelector(`[data-mini="${x.id}"]`), { cells: x.board, move: x.last, small: true, label: `game ${x.id}` }));
    body.querySelectorAll("[data-goto]").forEach((b) => b.addEventListener("click", () => select(b.dataset.goto)));
    body.querySelectorAll("[data-live]").forEach((b) => b.addEventListener("click", () => openGame(run.name, +b.dataset.live)));
    body.querySelectorAll("tr[data-game]").forEach((tr) => {
      const open = () => openGame(run.name, +tr.dataset.game);
      tr.addEventListener("click", open);
      tr.addEventListener("keydown", (e) => { if (e.key === "Enter") open(); });
    });
    $("#games-end")?.addEventListener("change", (e) => { g.end = e.target.value; g.offset = 0; refreshGames(run); });
    $("#games-sort")?.addEventListener("change", (e) => { g.sort = e.target.value; g.offset = 0; refreshGames(run); });
    body.querySelectorAll("[data-page]").forEach((b) => b.addEventListener("click", () => { g.offset = Math.max(0, g.offset + g.limit * +b.dataset.page); refreshGames(run); }));
    $("#games-limit")?.addEventListener("change", (e) => { g.limit = +e.target.value; g.offset = 0; refreshGames(run); });
  }

  // the viewer: one game move by move
  const viewer = { run: null, id: null, game: null, boards: [], ply: 0, timer: null, poll: null, graph: null };

  async function openGame(run, id) {
    const dlg = $("#viewer");
    Object.assign(viewer, { run, id, ply: -1, game: null });
    dlg.innerHTML = `<div class="viewer"><p class="missing">Loading game ${id}…</p></div>`;
    if (!dlg.open) dlg.showModal();
    await loadGame(true);
  }

  async function loadGame(first) {
    const dlg = $("#viewer");
    if (!dlg.open) return;
    const res = await fetch(`/api/game?run=${encodeURIComponent(viewer.run)}&id=${viewer.id}`);
    if (!res.ok) {
      dlg.querySelector(".viewer").innerHTML = `<p class="missing">Game ${viewer.id} is not available right now (a game that just finished
        is published with its shard; open it again from the list in a moment).</p><button class="button" type="button" data-close-viewer>Close</button>`;
      bindViewer();
      return;
    }
    const game = await res.json();
    const atEnd = viewer.ply < 0 || viewer.ply >= viewer.boards.length - 1;
    viewer.game = game;
    const boards = [game.initial.slice()];
    for (const p of game.plies) {
      const b = boards[boards.length - 1].slice();
      b[p.to] = b[p.from];
      b[p.from] = 0;
      boards.push(b);
    }
    viewer.boards = boards;
    viewer.ply = first || (game.live && atEnd) ? boards.length - 1 : Math.min(viewer.ply, boards.length - 1);
    renderViewer();
    clearTimeout(viewer.poll);
    if (game.live) viewer.poll = setTimeout(() => loadGame(false), 2500);
  }

  function renderViewer() {
    const { game, boards } = viewer;
    const last = boards.length - 1;
    const result = game.live ? "being played now" : `${resultWord(game)}, ${END_WORDS[game.end] || game.end}, ${game.plies.length} plies`;
    $("#viewer").innerHTML = `<div class="viewer">
      <div class="viewer-head"><div><h2>Game ${game.id}</h2><p>${esc(viewer.run)}, ${esc(result)}</p></div>
        <button class="button quiet-button" type="button" data-close-viewer>Close</button></div>
      <div class="viewer-grid">
        <div class="viewer-board" id="viewer-board"></div>
        <div class="viewer-side" id="viewer-side"></div>
      </div>
      <div class="viewer-controls">
        <button class="button" type="button" data-step="first" aria-label="First position">⏮</button>
        <button class="button" type="button" data-step="-1" aria-label="Previous move">◀</button>
        <button class="button primary" type="button" data-play>${viewer.timer ? "Pause" : "Play"}</button>
        <button class="button" type="button" data-step="1" aria-label="Next move">▶</button>
        <button class="button" type="button" data-step="last" aria-label="Last position">⏭</button>
        <input type="range" min="0" max="${last}" value="${viewer.ply}" id="viewer-slider" aria-label="Ply">
      </div>
      <div id="viewer-graph"></div>
      <p class="small-note">The line is the search's value of each position for Blue: +1 a sure Blue win, −1 a sure Red win. Ochre marks
        are random moves; a dashed arrow is the search's own choice where another move was played. Arrow keys step, space plays.</p>
    </div>`;
    viewer.graph = Charts.evalGraph($("#viewer-graph"), { values: game.plies.map((p) => p.value ?? null).concat([null]), cursor: viewer.ply,
      marks: game.plies.map((p, i) => (p.random && i >= game.opening_plies ? i : null)).filter((x) => x !== null), onSeek: seek });
    bindViewer();
    drawPosition();
  }

  function drawPosition() {
    const { game, boards, ply } = viewer;
    if (!game) return;
    const move = ply > 0 ? game.plies[ply - 1] : null;
    const next = game.plies[ply] || move;  // the final position has no search of its own: show the last one
    Charts.board($("#viewer-board"), { cells: boards[ply], move, best: move?.best, label: `position after ply ${ply}` });
    const pieces = (red) => boards[ply].filter((c) => (red ? c >= 4 : c >= 1 && c <= 3)).length;
    const v = next?.value;
    $("#viewer-side").innerHTML = `<dl class="facts">
      <dt>Ply</dt><dd>${ply} <small>of ${boards.length - 1}</small></dd>
      <dt>To move</dt><dd>${ply % 2 ? "Red" : "Blue"}</dd>
      <dt>Last move</dt><dd>${move ? `${esc(move.token)}${move.taken ? " takes" : ""}` : "–"}${move?.random ? " <small>random</small>" : ""}${move?.alternative ? " <small>alternative</small>" : ""}</dd>
      ${move?.best ? `<dt>Search preferred</dt><dd>${esc(move.best.token)}</dd>` : ""}
      <dt>Value for Blue</dt><dd>${typeof v === "number" ? (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(2) : "–"}</dd>
      <dt>Depth, nodes</dt><dd>${next?.depth ?? "–"}, ${next?.nodes ? next.nodes.toLocaleString() : "–"}</dd>
      <dt>Since a capture</dt><dd>${next?.since ?? "–"} <small>of ${game.capture_clock || "–"}</small></dd>
      <dt>Pieces</dt><dd>${pieces(false)} <small>Blue</small> ${pieces(true)} <small>Red</small></dd>
    </dl>
    <div class="piece-key"><span><svg viewBox="0 0 12 12"><circle cx="6" cy="6" r="5"/></svg> rock</span>
      <span><svg viewBox="0 0 12 12"><rect x="1" y="1" width="10" height="10" rx="2"/></svg> paper</span>
      <span><svg viewBox="0 0 12 12"><path d="M6 0.5 11.5 10.5H0.5z"/></svg> scissors</span></div>`;
    const slider = $("#viewer-slider");
    if (slider) slider.value = ply;
    viewer.graph?.move(ply);
  }

  function seek(ply) {
    viewer.ply = Math.max(0, Math.min(viewer.boards.length - 1, ply));
    drawPosition();
  }

  function bindViewer() {
    const dlg = $("#viewer");
    dlg.querySelectorAll("[data-close-viewer]").forEach((b) => b.addEventListener("click", () => dlg.close()));
    dlg.querySelectorAll("[data-step]").forEach((b) => b.addEventListener("click", () => {
      const k = b.dataset.step;
      seek(k === "first" ? 0 : k === "last" ? viewer.boards.length - 1 : viewer.ply + +k);
    }));
    dlg.querySelector("[data-play]")?.addEventListener("click", togglePlay);
    dlg.querySelector("#viewer-slider")?.addEventListener("input", (e) => seek(+e.target.value));
  }

  function togglePlay() {
    if (viewer.timer) {
      clearInterval(viewer.timer);
      viewer.timer = null;
    } else {
      if (viewer.ply >= viewer.boards.length - 1) viewer.ply = 0;
      viewer.timer = setInterval(() => {
        if (viewer.ply >= viewer.boards.length - 1) {
          clearInterval(viewer.timer);
          viewer.timer = null;
          const b = $("#viewer [data-play]");
          if (b) b.textContent = "Play";
          return;
        }
        seek(viewer.ply + 1);
      }, 220);
    }
    const b = $("#viewer [data-play]");
    if (b) b.textContent = viewer.timer ? "Pause" : "Play";
  }

  // ------------------------------------------------------------------ training
  function metricOptions(run) {
    const cols = new Set(run.log?.columns || []);
    const sets = run.mixture.map((m) => m.set);
    const has = (m) => sets.some((s) => cols.has(`${s}.${m}`));
    const group = (label, items) => (items.length ? `<optgroup label="${label}">${items.join("")}</optgroup>` : "");
    const opt = (value, label) => `<option value="${value}" ${state.metric === value ? "selected" : ""}>${label}</option>`;
    const strata = Object.keys(state.data.strata_help);
    return [
      group("Whole run", RUN_METRICS.filter((m) => cols.has(m)).map((m) => opt(`run:${m}`, m))),
      group("Per batch", SET_METRICS.filter(has).map((m) => opt(`set:${m}`, m.replaceAll("_", " ")))),
      ...strata.map((st) => group(`Slice: ${state.data.strata_help[st]}`,
        SLICE_METRICS.filter((m) => has(`${st}.${m}`)).map((m) => opt(`set:${st}.${m}`, `${st.replace("_", " ")} ${m}`)))),
    ].join("");
  }

  function bestIndexOf(log) {
    let best = -1;
    log.rows.forEach((r, i) => {
      if (typeof r.objective === "number" && (best < 0 || r.objective < log.rows[best].objective)) best = i;
    });
    return best;
  }

  function training(run) {
    if (!run.log || !run.log.rows.length) {
      return `<section><h3>Training</h3><p class="missing">No <code>log.csv</code> yet. It gains a row at the end of every epoch.</p></section>`;
    }
    const sets = run.mixture.map((m) => m.set);
    if (!state.sets) state.sets = [...sets];
    const rowsDone = run.log.rows.length;
    const planned = (run.config || {}).epochs || rowsDone;
    const bi = bestIndexOf(run.log);
    const others = state.data.runs.filter((r) => r !== run && r.log && r.log.rows.length);
    const running = (state.control?.jobs || []).some((j) => j.run === run.name && j.state === "running");
    const retrain = controlOn() && run.config && !running
      ? `<button class="button" type="button" data-retrain>Retrain with other settings…</button>` : "";
    return `<section>
      <div class="section-head"><h3>Training</h3>${retrain}</div>
      ${earlierTrainings(run, running)}
      <p class="lede">Epoch ${rowsDone} of ${planned}${rowsDone < planned ? ", still running" : ""}.
        ${bi >= 0 ? `The best is epoch ${run.log.rows[bi].epoch}, objective ${num(run.log.rows[bi].objective, 5)}.` : ""}
        Compare epochs within one run; objective levels differ between runs on different data.</p>
      <div class="controls">
        <label class="field">Metric <select id="metric">${metricOptions(run)}</select></label>
        <div class="field" id="set-field"><span>Batches</span><div class="sets">${sets.map((s) =>
          `<label><input type="checkbox" value="${esc(s)}" ${state.sets.includes(s) ? "checked" : ""}>
           <span class="swatch" style="background:${colorFor(sets, s)}"></span>${esc(setLabel(s))}</label>`).join("")}</div></div>
        <label class="field">Overlay a run <select id="compare"><option value="">None</option>${others.map((r) =>
          `<option value="${esc(r.name)}" ${state.compare === r.name ? "selected" : ""}>${esc(r.name)}</option>`).join("")}</select></label>
      </div>
      <div class="chart-main"><div id="chart"></div></div>
      <div class="chart-help" id="chart-help"></div>
      <div class="smalls">
        <div class="small"><h4>Objective against loss</h4><p>Change since epoch 0. They should fall together; loss down while objective climbs is memorising.</p><div id="small-fit"></div><div class="legend" id="legend-fit"></div></div>
        <div class="small"><h4>Validation error by batch</h4><p>Change since epoch 0. An older batch whose error rises is being forgotten.</p><div id="small-drift"></div><div class="legend" id="legend-drift"></div></div>
        <div class="small"><h4>Bias by batch</h4><p>Signed error. The shaded band is ±0.005.</p><div id="small-bias"></div><div class="legend" id="legend-bias"></div></div>
      </div>
    </section>`;
  }

  function earlierTrainings(run, running) {
    const list = run.earlier || [];
    if (!list.length) return "";
    const live = (state.control?.jobs || []).some((j) => j.run === run.name && ["running", "queued", "paused"].includes(j.state));
    return `<details class="earlier" data-key="earlier-${esc(run.name)}" ${list.some((e) => e.direct) ? "open" : ""}>
      <summary>${list.length} earlier training${list.length === 1 ? "" : "s"} of this run</summary>
      <div class="table-scroll"><table class="kv"><thead><tr><th>Trained</th><th>Batch, epochs</th><th>Best objective</th><th>Own test</th><th>Current against it</th><th></th></tr></thead><tbody>
      ${list.map((e) => `<tr><th scope="row">${esc(e.label.split("(").pop().replace(")", ""))}</th>
        <td>${e.batch ? e.batch.toLocaleString() : "–"}, ${e.epochs ?? "–"} <small class="muted">x ${e.steps_per_epoch ?? "?"} steps</small></td>
        <td>${num(e.objective, 5)}${e.best_epoch !== undefined && e.best_epoch !== null ? ` <small class="muted">epoch ${e.best_epoch}</small>` : ""}</td>
        <td>${e.test ? `${score(e.test.score)}, ${elo(e.test.elo)} Elo` : "–"}</td>
        <td>${e.direct ? `<b>${score(e.direct.score)}</b> <small class="muted">${score(e.direct.score_interval[0])} to ${score(e.direct.score_interval[1])}, ${e.direct.pairs} pairs</small>` : "–"}</td>
        <td>${controlOn() && e.has_net && !live ? `<button class="small-button" type="button" data-restore="${esc(e.dir)}" data-label="${esc(e.label)}">Restore</button>` : ""}</td></tr>`).join("")}
      </tbody></table></div>
      <p class="small-note">"Current against it" is the current network's score in a direct match: below .50 means the earlier training played better.</p>
    </details>`;
  }

  function seriesFor(run, metric, sets, opts = {}) {
    const rows = run.log.rows;
    const [scope, name] = metric.split(":");
    if (scope === "run") return [{ name: opts.prefix ? run.name : name, color: opts.color || "var(--blue)", values: rows.map((r) => r[name]), dash: opts.dash }];
    const all = run.mixture.map((m) => m.set);
    return sets.filter((s) => run.log.columns.includes(`${s}.${name}`)).map((s) => ({
      name: opts.prefix ? `${run.name} ${setLabel(s)}` : setLabel(s),
      color: opts.color || colorFor(all.length ? all : sets, s), values: rows.map((r) => r[`${s}.${name}`]), dash: opts.dash,
    }));
  }

  const relative = (values) => {
    const base = values.find((v) => typeof v === "number" && v !== 0);
    return values.map((v) => (typeof v === "number" && base ? v / base - 1 : null));
  };

  function drawCharts(run) {
    if ($("#staircase")) Charts.staircase($("#staircase"), { start: byName(run.chain[0].name).start_label, steps: staircaseSteps(run) });
    const ev = activeEval(run);
    if (ev) {
      const clear = ev.kind === "sequential" ? ev.sprt_target : ev.sd_pair ? 0.5 + (1.96 * ev.sd_pair) / Math.sqrt(state.data.plan.decision_pairs) : null;
      Charts.ruler($("#ruler"), { rows: rulerRows(run), clear: ev.against_start === false ? null : clear, versus: ev.against_start === false ? ev.opponent : "its start",
        clearLabel: ev.kind === "sequential" ? "the test's target" : `clears at ${state.data.plan.decision_pairs} pairs` });
      if ($("#sprt-track")) Charts.sprtTrack($("#sprt-track"), { llr: ev.llr, bound: ev.llr_bound, pairs: ev.pairs, cap: ev.sprt_cap });
    }
    if (!run.log || !run.log.rows.length) return;
    const rows = run.log.rows;
    const epochs = rows.map((r) => r.epoch);
    const bestIndex = bestIndexOf(run.log);
    const planned = rows.length < ((run.config || {}).epochs || 0) ? run.config.epochs : 0;
    const [scope, name] = state.metric.split(":");
    const leaf = name.split(".").pop();
    $("#set-field").style.display = scope === "set" ? "" : "none";

    let series = seriesFor(run, state.metric, state.sets);
    const other = state.compare && byName(state.compare);
    if (other && other.log) {
      series = series.concat(seriesFor(other, state.metric, scope === "set" ? other.mixture.map((m) => m.set) : [],
        { dash: "5 4", prefix: true, color: "var(--grey)" }));
    }
    const x = Array.from({ length: Math.max(...series.map((s) => s.values.length), rows.length, planned) }, (_, i) => rows[i]?.epoch ?? i);
    series.forEach((s) => { s.values = Array.from({ length: x.length }, (_, i) => s.values[i]); });
    const fmt = leaf === "lr" ? (v) => v.toExponential(1) : leaf === "rows" || leaf === "seconds" ? (v) => num(v, 3) : (v) => +v.toPrecision(3) + "";
    if (bestIndex >= 0 && series[0]) series[0].dots = [bestIndex];
    Charts.lineChart($("#chart"), {
      series, x, height: 300, xTitle: "epoch", format: fmt, tipFormat: (v) => (leaf === "lr" ? v.toExponential(2) : +v.toPrecision(5) + ""),
      markers: bestIndex >= 0 ? [{ i: bestIndex, label: `best.nnue, epoch ${rows[bestIndex].epoch}` }] : [],
      zeroLine: leaf === "bias", band: leaf === "bias" ? { lo: -0.005, hi: 0.005 } : null, label: `${name} by epoch`,
    });
    const help = state.data.metric_help[leaf] || state.data.metric_help[name] || ["", ""];
    const slice = name.includes(".") ? ` Slice: ${state.data.strata_help[name.split(".")[0]]}.` : "";
    $("#chart-help").innerHTML = `<p><b>What it is.</b> ${esc(help[0])}${esc(slice)}</p><p><b>What to look for.</b> ${esc(help[1])}</p>`;

    const sets = run.mixture.map((m) => m.set);
    const legend = (items) => items.map(([n, c, dash]) => `<span><span class="swatch${dash ? " dashed" : ""}" style="background:${c}"></span>${esc(n)}</span>`).join("");
    const setLegend = legend(sets.map((s) => [setLabel(s), colorFor(sets, s)]));
    $("#legend-fit").innerHTML = legend([["objective", "var(--blue)"], ["loss", "var(--grey)", true]]);
    $("#legend-drift").innerHTML = setLegend;
    $("#legend-bias").innerHTML = setLegend;
    const pctFmt = (v) => (v * 100).toFixed(Math.abs(v) < 0.02 ? 1 : 0) + "%";
    Charts.lineChart($("#small-fit"), {
      height: 190, left: 46, x: epochs, format: pctFmt, zeroLine: true, markers: bestIndex >= 0 ? [{ i: bestIndex, label: "best" }] : [],
      series: [
        { name: "objective", color: "var(--blue)", values: relative(rows.map((r) => r.objective)) },
        { name: "loss", color: "var(--grey)", values: relative(rows.map((r) => r.loss)), dash: "4 3" },
      ],
    });
    Charts.lineChart($("#small-drift"), {
      height: 190, left: 46, x: epochs, format: pctFmt, zeroLine: true,
      series: sets.map((s) => ({ name: setLabel(s), color: colorFor(sets, s), values: relative(rows.map((r) => r[`${s}.mse`])) })),
    });
    Charts.lineChart($("#small-bias"), {
      height: 190, left: 52, x: epochs, format: (v) => v.toFixed(3), zeroLine: true, band: { lo: -0.005, hi: 0.005 },
      series: sets.map((s) => ({ name: setLabel(s), color: colorFor(sets, s), values: rows.map((r) => r[`${s}.bias`]) })),
      tipFormat: (v) => v.toFixed(4),
    });
  }

  // ------------------------------------------------------------------ checks
  function checks(run) {
    if (!run.checks.length) return "";
    const words = { ok: "fine", warn: "watch", bad: "problem", info: "note" };
    const table = (items) => {
      const groups = [...new Set(items.map((c) => c.group))];
      return `<div class="table-scroll"><table class="checks">
        <thead><tr><th>Status</th><th>Measure</th><th>This run</th><th>Looking for</th><th>Why</th></tr></thead>
        ${groups.map((g) => `<tbody><tr><th scope="rowgroup" colspan="5">${esc(g)}</th></tr>
          ${items.filter((c) => c.group === g).map((c) => `<tr>
            <td><span class="status ${c.status}"><i></i>${words[c.status]}</span></td>
            <th scope="row">${esc(c.label)}</th>
            <td class="value">${esc(c.value)}</td>
            <td class="target">${esc(c.target)}</td>
            <td class="note">${esc(c.note)}</td></tr>`).join("")}</tbody>`).join("")}
      </table></div>`;
    };
    const flagged = run.checks.filter((c) => c.status === "bad" || c.status === "warn");
    const rest = run.checks.filter((c) => !(c.status === "bad" || c.status === "warn"));
    return `<section><h3>Checks</h3>
      <p class="lede">Only the evaluation decides. The training and data checks explain a result and say what to fix before the next batch.
        ${flagged.length ? "" : "Nothing needs attention."}</p>
      ${flagged.length ? table(flagged) : ""}
      ${rest.length ? `<details class="passing" data-key="passing-${esc(run.name)}"><summary>${rest.length} more check${rest.length === 1 ? "" : "s"} that ${flagged.length ? "pass or are notes" : "pass"}</summary>${table(rest)}</details>` : ""}
    </section>`;
  }

  // ------------------------------------------------------------------ data and settings
  function data(run) {
    const sets = run.mixture.length ? run.mixture : [{ set: run.own_set, share: 1 }];
    const names = sets.map((s) => s.set);
    const mixture = `<div class="table-scroll"><table class="kv"><thead><tr><th>Batch</th><th>Share</th><th>Root rows</th><th>Games</th></tr></thead><tbody>
      ${sets.map((m) => {
        const d = run.datasets[m.set];
        return `<tr><th scope="row"><span class="swatch" style="background:${colorFor(names, m.set)}"></span> ${esc(setLabel(m.set))}</th>
          <td>${m.share}</td><td>${d ? num(d.root_rows) : "–"}${d && d.line_rows ? ` <small class="muted">+ ${num(d.line_rows)} line</small>` : ""}</td>
          <td>${d ? num(d.games) : "–"}</td></tr>`;
      }).join("")}</tbody></table></div>`;

    const gen = run.generation;
    const generation = gen ? `<table class="kv"><tbody>
        <tr><th>Games</th><td>${num(gen.games_done)} of ${num(gen.games_target)}${gen.complete ? "" : " (playing)"}</td></tr>
        <tr><th>Nodes per move</th><td>${num(gen.nodes)}</td></tr>
        <tr><th>Searched positions per game</th><td>${num(gen.roots_per_game, 3)}</td></tr>
        <tr><th>Games per hour</th><td>${num(gen.games_per_hour, 3)} <small class="muted">with ${gen.threads} in flight</small></td></tr>
        <tr><th>Censored games</th><td>${pct(gen.censored_share)}</td></tr>
        <tr><th>Seed</th><td>${esc(gen.seed)}</td></tr>
        <tr><th>Multi-PV, PV labels</th><td>${esc(gen.multipv ?? 1)}, ${esc(gen.pv_labels ?? 0)}</td></tr>
      </tbody></table>` : `<p class="missing">No <code>runs/nnue_selfplay/${esc(run.name)}/manifest.json</code> for this run.</p>`;

    const parent = run.parent && byName(run.parent);
    const keys = Object.keys(run.config || {}).filter((k) => !["resume", "stop_epoch"].includes(k));
    const config = keys.length ? `<table class="kv"><thead><tr><th>Setting</th><th>${esc(run.name)}</th>${parent ? `<th>${esc(parent.name)}</th>` : ""}</tr></thead><tbody>
      ${keys.map((k) => {
        const a = run.config[k], b = parent ? parent.config[k] : undefined;
        const diff = parent && JSON.stringify(a) !== JSON.stringify(b);
        return `<tr><th scope="row">${esc(k)}</th><td class="${diff ? "diff" : ""}">${esc(JSON.stringify(a))}</td>${parent ? `<td class="${diff ? "" : "same"}">${esc(JSON.stringify(b))}</td>` : ""}</tr>`;
      }).join("")}</tbody></table>` : `<p class="missing">No <code>config.json</code> yet: training has not started.</p>`;

    const best = run.best;
    const net = best ? `<table class="kv"><tbody>
        <tr><th>Epoch, objective</th><td>${best.epoch ?? "–"}, ${num(best.objective, 5)}</td></tr>
        <tr><th>Shape</th><td>format ${best.version}, ${num(best.features)} features, H${best.hidden}</td></tr>
        <tr><th>Size</th><td>${best.bytes ? (best.bytes / 1048576).toFixed(1) + " MiB" : "–"}</td></tr>
        <tr><th>sha256</th><td class="hash">${esc(best.sha256)}</td></tr>
        <tr><th>Start sha256</th><td class="hash">${esc(run.init_sha256)}</td></tr>
      </tbody></table>` : `<p class="missing">No best.nnue yet.</p>`;

    return `<section><h3>Data and settings</h3>
      <p class="lede">${parent ? `Settings in blue differ from ${esc(parent.name)}.` : "The start of this run is not in this runs directory, so there is nothing to compare settings against."}</p>
      <div class="two">
        <div><h4 class="sub">Training mixture</h4>${mixture}<h4 class="sub">Self-play batch</h4>${generation}<h4 class="sub">best.nnue</h4>${net}</div>
        <div><h4 class="sub">Settings</h4>${config}</div>
      </div></section>`;
  }

  // ------------------------------------------------------------------ events
  function bind(run) {
    document.querySelectorAll("#main [data-goto]").forEach((b) => b.addEventListener("click", () => select(b.dataset.goto)));
    document.querySelectorAll("#main [data-copy]").forEach((b) => b.addEventListener("click", async () => {
      const label = b.dataset.label || (b.dataset.label = b.textContent);
      try { await navigator.clipboard.writeText(b.dataset.copy); b.textContent = "Copied"; }
      catch { b.textContent = "Clipboard blocked: select the text"; }
      setTimeout(() => (b.textContent = label), 1600);
    }));
    const next = run.decision.next;
    $("#main [data-next]")?.addEventListener("click", async (evt) => {
      evt.target.disabled = true;
      try {
        if (next.kind === "eval") await act("/api/jobs/eval", { run: next.run, settings: next.settings }, "Sequential test queued");
        else {
          await act("/api/jobs/pipeline", { ...state.control.defaults, ...next.settings, threads: state.control.defaults.threads,
            eval: { ...state.control.defaults.eval, ...next.settings.eval } }, `Started ${next.run}`);
          select(next.run);
        }
      } catch { evt.target.disabled = false; }
    });
    $("#main [data-next-edit]")?.addEventListener("click", () => pipelineDialog({ ...state.control.defaults, ...next.settings,
      threads: state.control.defaults.threads, eval: { ...state.control.defaults.eval, ...next.settings.eval } },
      run.decision.verdict === "adopt" ? "Next generation" : "Another batch", "Start"));
    $("#main [data-auto-from]")?.addEventListener("click", () => autoDialog(run.name));
    $("#main [data-retrain]")?.addEventListener("click", () => retrainDialog(run.name));
    $("#main [data-measure]")?.addEventListener("click", (e) => evalDialog(run.name,
      { mode: "fixed", sims: 16, pairs: 200, reference: e.currentTarget.dataset.measure }, "Measure against the first network"));
    $("#main [data-eval-again]")?.addEventListener("click", () => evalDialog(run.name, activeEval(run)
      ? { mode: activeEval(run).kind === "sequential" ? "sprt" : "fixed", sims: activeEval(run).sims, pairs: activeEval(run).pairs } : {}));
    $("#metric")?.addEventListener("change", (e) => { state.metric = e.target.value; drawCharts(run); });
    $("#compare")?.addEventListener("change", (e) => { state.compare = e.target.value; drawCharts(run); });
    document.querySelectorAll("#main tr[data-eval]").forEach((tr) => {
      const pick = () => { state.evalFile = tr.dataset.eval; render(); };
      tr.addEventListener("click", pick);
      tr.addEventListener("keydown", (e) => { if (e.key === "Enter") pick(); });
    });
    document.querySelectorAll("#main [data-restore]").forEach((b) => b.addEventListener("click", async () => {
      if (!confirm(`Put ${b.dataset.label} back as ${run.name}? The current training moves aside and can be restored the same way.`)) return;
      b.disabled = true;
      try {
        const r = await act("/api/runs/restore", { run: run.name, earlier: b.dataset.restore }, `Restored ${b.dataset.label}`);
        state.evalFile = null;
      } catch { b.disabled = false; }
    }));
    document.querySelectorAll("#set-field input").forEach((box) => box.addEventListener("change", () => {
      state.sets = [...document.querySelectorAll("#set-field input:checked")].map((i) => i.value);
      drawCharts(run);
    }));
  }

  let resizeTimer;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => { const r = current(); if (r) drawCharts(r); }, 120);
  });
  window.addEventListener("hashchange", () => {
    const n = decodeURIComponent(location.hash.slice(1));
    if (n !== state.run && (byName(n) || pendingJob(n))) select(n);
  });

  const notifyButton = $("#notify");
  const notifyOn = () => { try { return localStorage.getItem("nnue-notify") === "on"; } catch { return false; } };
  function syncNotify() {
    const supported = "Notification" in window;
    notifyButton.hidden = !supported || !controlOn();
    if (supported) notifyButton.textContent = notifyOn() && Notification.permission === "granted" ? "Notifications on" : "Notify me";
  }
  notifyButton.addEventListener("click", async () => {
    const turningOn = !(notifyOn() && Notification.permission === "granted");
    if (turningOn && Notification.permission !== "granted") await Notification.requestPermission();
    const on = turningOn && Notification.permission === "granted";
    try { localStorage.setItem("nnue-notify", on ? "on" : "off"); } catch {}
    syncNotify();
    toast(on ? "You will be notified when a job finishes, fails or pauses while this tab is in the background"
      : Notification.permission === "denied" ? "Notifications are blocked for this page in the browser's settings" : "Notifications off");
  });
  syncNotify();

  document.addEventListener("keydown", (e) => {
    if (!$("#viewer").open || e.target.closest("input, select")) return;
    if (e.key === "ArrowRight") { seek(viewer.ply + 1); e.preventDefault(); }
    else if (e.key === "ArrowLeft") { seek(viewer.ply - 1); e.preventDefault(); }
    else if (e.key === "Home") seek(0);
    else if (e.key === "End") seek(viewer.boards.length - 1);
    else if (e.key === " ") { togglePlay(); e.preventDefault(); }
  });
  $("#viewer").addEventListener("close", () => { clearTimeout(viewer.poll); clearInterval(viewer.timer); viewer.timer = null; });

  const themeButton = $("#theme");
  const applyTheme = (t) => {
    if (t) document.documentElement.dataset.theme = t; else delete document.documentElement.dataset.theme;
    const dark = t ? t === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    themeButton.textContent = dark ? "Light theme" : "Dark theme";
  };
  let theme = null;
  try { theme = localStorage.getItem("nnue-theme"); } catch {}
  applyTheme(theme);
  themeButton.addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    const next = dark ? "light" : "dark";
    try { localStorage.setItem("nnue-theme", next); } catch {}
    applyTheme(next);
    const r = current(); if (r) drawCharts(r);
  });

  // job state every 2 s; the runs (charts, decisions) every 5 s while something runs, else every 20 s
  (async function loop() {
    await loadControl();
    if (!state.data) await loadRuns();
    let last = 0;
    setInterval(loadControl, 2000);
    setInterval(() => {
      const busy = state.control?.current || (state.data?.runs || []).some((r) => r.status.source === "files" && LIVE.includes(r.status.state));
      if (Date.now() - last > (busy ? 5000 : 20000) && !dialog().open) { last = Date.now(); loadRuns(); }
    }, 1000);
  })();
})();
