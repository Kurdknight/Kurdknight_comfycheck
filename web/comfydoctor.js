// ComfyDoctor — sidebar tab that scans and repairs a broken ComfyUI Python environment.
// Plain ES module, no bundler, no external resources. DOM is built with
// document.createElement + textContent only — server-provided strings are never
// passed through innerHTML.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// ---------------------------------------------------------------------------
// Small DOM helpers
// ---------------------------------------------------------------------------

/** Build an element. `opts.class` sets className, `opts.text` sets textContent,
 *  `onXxx` functions become listeners, everything else becomes an attribute. */
function el(tag, opts = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(opts)) {
    if (value === undefined || value === null) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on") && typeof value === "function") {
      node.addEventListener(key.slice(2).toLowerCase(), value);
    } else {
      node.setAttribute(key, value);
    }
  }
  for (const child of [].concat(children)) {
    if (child === null || child === undefined) continue;
    node.appendChild(typeof child === "string" ? document.createTextNode(child) : child);
  }
  return node;
}

function icon(piClasses) {
  return el("i", { class: `pi ${piClasses} cd-icon` });
}

/** Icon + caption button — never a bare icon, per the house style rule. */
function iconButton(piClass, label, onClick, extraClass = "") {
  const btn = el("button", { class: `cd-btn ${extraClass}`.trim(), type: "button" });
  btn.appendChild(icon(piClass));
  btn.appendChild(el("span", { class: "cd-btn-label", text: label }));
  btn.addEventListener("click", onClick);
  return btn;
}

/** Briefly swap a button's caption to `message`, then restore it. */
function flashButton(btn, message, duration = 2000) {
  const label = btn.querySelector(".cd-btn-label");
  if (!label) return;
  if (btn._cdFlashTimer) clearTimeout(btn._cdFlashTimer);
  if (btn.dataset.cdOriginal === undefined) btn.dataset.cdOriginal = label.textContent;
  label.textContent = message;
  btn.classList.add("cd-btn-flash");
  btn._cdFlashTimer = setTimeout(() => {
    label.textContent = btn.dataset.cdOriginal;
    btn.classList.remove("cd-btn-flash");
    btn._cdFlashTimer = null;
  }, duration);
}

async function copyText(text, btn) {
  try {
    await navigator.clipboard.writeText(text);
    flashButton(btn, "Copied!");
  } catch (err) {
    flashButton(btn, "Copy failed");
  }
}

/** Join an argv array into a shell-ish display string, quoting args with spaces. */
function formatCommand(argv) {
  return argv.map((a) => (String(a).includes(" ") ? `"${a}"` : String(a))).join(" ");
}

function apiUrl(path) {
  return typeof api.apiURL === "function" ? api.apiURL(path) : path;
}

// ---------------------------------------------------------------------------
// Severity + health metadata
// ---------------------------------------------------------------------------

const SEVERITY_LABEL = { critical: "Critical", error: "Error", warning: "Warning", info: "Info", ok: "OK" };
const SEVERITY_NOUN = {
  critical: ["critical", "criticals"],
  error: ["error", "errors"],
  warning: ["warning", "warnings"],
  info: ["info item", "info items"],
  ok: ["all-clear", "all-clears"],
};

function healthLabel(score) {
  if (score >= 100) return "Healthy";
  if (score >= 80) return "Minor issues";
  if (score >= 60) return "Needs attention";
  return "Broken";
}

function healthTier(score) {
  if (score >= 100) return "ok";
  if (score >= 80) return "minor";
  if (score >= 60) return "attention";
  return "broken";
}

function countsPhrase(counts) {
  if (!counts) return "";
  const order = ["critical", "error", "warning", "info", "ok"];
  const parts = [];
  for (const key of order) {
    const n = counts[key] || 0;
    if (n <= 0) continue;
    const [singular, plural] = SEVERITY_NOUN[key];
    parts.push(`${n} ${n === 1 ? singular : plural}`);
  }
  return parts.join(" · ");
}

// ---------------------------------------------------------------------------
// Remedy block — title/explain/commands + the fix -> confirm -> run flow
// ---------------------------------------------------------------------------

function buildRemedyBlock(finding, ctx) {
  const remedy = finding.remedy;
  const wrap = el("div", { class: "cd-remedy" });

  // view: idle | confirm | starting | running | success | failed | cancelled
  let view = "idle";
  let job = null; // { id, lines[], total_lines, elapsed, status, exit_code }
  let errorMsg = "";

  function render() {
    wrap.textContent = "";

    const titleRow = el("div", { class: "cd-remedy-title" });
    titleRow.appendChild(icon("pi-wrench"));
    titleRow.appendChild(el("span", { text: remedy.title || "Suggested fix" }));
    wrap.appendChild(titleRow);

    if (remedy.explain) {
      wrap.appendChild(el("div", { class: "cd-remedy-explain", text: remedy.explain }));
    }

    if (Array.isArray(remedy.commands) && remedy.commands.length) {
      const list = el("div", { class: "cd-cmd-list" });
      remedy.commands.forEach((argv) => {
        const cmdStr = formatCommand(argv);
        const row = el("div", { class: "cd-cmd-row" });
        const codeWrap = el("div", { class: "cd-cmd-code-wrap" });
        codeWrap.appendChild(el("code", { class: "cd-cmd-code", text: cmdStr }));
        row.appendChild(codeWrap);
        const copyBtn = iconButton("pi-copy", "Copy", () => copyText(cmdStr, copyBtn), "cd-btn-sm");
        row.appendChild(copyBtn);
        list.appendChild(row);
      });
      wrap.appendChild(list);
    }

    if (remedy.doc_url) {
      const link = el("a", { class: "cd-doc-link", href: remedy.doc_url, target: "_blank", rel: "noopener" });
      link.appendChild(icon("pi-external-link"));
      link.appendChild(el("span", { text: "Learn more" }));
      wrap.appendChild(link);
    }

    if (errorMsg) {
      wrap.appendChild(
        el("div", { class: "cd-inline-error" }, [icon("pi-exclamation-triangle"), el("span", { text: errorMsg })])
      );
    }

    if (view === "idle") {
      if (remedy.runnable) {
        wrap.appendChild(iconButton("pi-wrench", "Fix this", () => { view = "confirm"; render(); }, "cd-btn-primary"));
      }
      return;
    }

    if (view === "confirm") {
      const box = el("div", { class: "cd-confirm-box" });
      box.appendChild(el("div", { class: "cd-confirm-label", text: "This will run the command(s) above." }));
      if (remedy.restart_required) {
        box.appendChild(
          el("div", { class: "cd-restart-note" }, [icon("pi-info-circle"), el("span", { text: "ComfyUI will need a restart afterwards." })])
        );
      }
      if (remedy.danger) {
        box.appendChild(
          el("div", { class: "cd-danger-box" }, [icon("pi-exclamation-triangle"), el("span", { text: remedy.danger })])
        );
      }
      const btnRow = el("div", { class: "cd-btn-row" });
      btnRow.appendChild(iconButton("pi-check", "Run it", () => startFix(), "cd-btn-primary"));
      btnRow.appendChild(iconButton("pi-times", "Cancel", () => { view = "idle"; render(); }));
      box.appendChild(btnRow);
      wrap.appendChild(box);
      return;
    }

    if (view === "starting") {
      wrap.appendChild(el("div", { class: "cd-status-line" }, [icon("pi-spinner pi-spin"), el("span", { text: "Starting…" })]));
      return;
    }

    // running | success | failed | cancelled — all show the run panel + log.
    const statusRow = el("div", { class: "cd-run-status" });
    if (view === "running") {
      statusRow.appendChild(el("span", { class: "cd-run-badge cd-run-badge--running" }, [icon("pi-spinner pi-spin"), el("span", { text: "Running" })]));
      statusRow.appendChild(el("span", { class: "cd-elapsed", text: `${(job?.elapsed ?? 0).toFixed(1)}s` }));
      const stopBtn = iconButton("pi-times", "Stop", async () => { stopBtn.disabled = true; await cancelFix(); });
      statusRow.appendChild(stopBtn);
    } else if (view === "success") {
      statusRow.appendChild(
        el("div", { class: "cd-banner cd-banner--ok" }, [
          icon("pi-check"),
          el("span", { text: remedy.restart_required ? "Done — restart ComfyUI, then scan again." : "Done — scan again to confirm." }),
        ])
      );
    } else if (view === "failed") {
      statusRow.appendChild(
        el("div", { class: "cd-banner cd-banner--fail" }, [
          icon("pi-exclamation-triangle"),
          el("span", { text: `Failed — exit code ${job?.exit_code ?? "?"}` }),
        ])
      );
    } else if (view === "cancelled") {
      statusRow.appendChild(
        el("div", { class: "cd-banner cd-banner--warn" }, [icon("pi-times"), el("span", { text: "Cancelled." })])
      );
    }
    wrap.appendChild(statusRow);

    const logPre = el("pre", { class: "cd-log" });
    logPre.textContent = (job?.lines || []).join("\n");
    wrap.appendChild(logPre);
    logPre.scrollTop = logPre.scrollHeight;

    if (view === "success") {
      wrap.appendChild(iconButton("pi-refresh", "Scan again", () => ctx.scan(), "cd-btn-primary"));
    }
  }

  async function startFix() {
    errorMsg = "";
    view = "starting";
    render();
    try {
      const res = await api.fetchApi("/comfydoctor/fix", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ finding_id: finding.id }),
      });
      const body = await res.json().catch(() => ({}));
      if (res.status === 404) {
        errorMsg = body.error || "No runnable fix is available for this finding.";
        view = "idle";
        render();
        return;
      }
      if (res.status === 409) {
        errorMsg = body.error || "Another fix is already running.";
        view = "idle";
        render();
        return;
      }
      if (!res.ok || !body.job_id) {
        errorMsg = body.error || `Could not start the fix (HTTP ${res.status}).`;
        view = "idle";
        render();
        return;
      }
      job = { id: body.job_id, lines: [], total_lines: 0, elapsed: 0, status: "pending", exit_code: null, _pollToken: null };
      view = "running";
      render();
      poll();
    } catch (err) {
      errorMsg = "Network error while starting the fix.";
      view = "idle";
      render();
    }
  }

  function poll() {
    const token = setTimeout(async () => {
      ctx.timers.delete(token);
      try {
        const res = await api.fetchApi(`/comfydoctor/fix/${job.id}?since=${job.total_lines}`);
        const data = await res.json();
        job.lines = job.lines.concat(data.lines || []);
        job.total_lines = data.total_lines ?? job.total_lines;
        job.elapsed = data.elapsed ?? job.elapsed;
        job.status = data.status;
        job.exit_code = data.exit_code;
        if (data.status === "pending" || data.status === "running") {
          view = "running";
          render();
          poll();
        } else {
          view = data.status === "success" ? "success" : data.status === "cancelled" ? "cancelled" : "failed";
          render();
        }
      } catch (err) {
        // Transient network hiccup — keep polling rather than losing the job.
        poll();
      }
    }, 700);
    ctx.timers.add(token);
    job._pollToken = token;
  }

  async function cancelFix() {
    if (!job) return;
    try {
      const res = await api.fetchApi(`/comfydoctor/fix/${job.id}/cancel`, { method: "POST" });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        errorMsg = body.error || "Failed to cancel.";
        render();
      }
    } catch (err) {
      errorMsg = "Network error while cancelling.";
      render();
    }
  }

  render();
  return wrap;
}

// ---------------------------------------------------------------------------
// Findings list
// ---------------------------------------------------------------------------

function buildFindingRow(finding, ctx) {
  const severity = SEVERITY_LABEL[finding.severity] ? finding.severity : "info";
  let expanded = severity === "critical" || severity === "error";

  const header = el("button", { class: "cd-finding-header", type: "button" });
  header.setAttribute("aria-expanded", String(expanded));
  header.appendChild(el("span", { class: `cd-dot cd-dot--${severity}` }));
  header.appendChild(el("span", { class: `cd-chip cd-chip--${severity}`, text: SEVERITY_LABEL[severity] }));
  header.appendChild(el("span", { class: "cd-finding-title", text: finding.title || "" }));
  if (finding.remedy && finding.remedy.runnable) {
    header.appendChild(el("span", { class: "cd-pill-fix" }, [icon("pi-wrench"), el("span", { text: "Fix available" })]));
  }
  const caret = icon(expanded ? "pi-angle-down" : "pi-angle-right");
  caret.classList.add("cd-caret");
  header.appendChild(caret);

  const body = el("div", { class: "cd-finding-body" });
  if (!expanded) body.classList.add("cd-collapsed");

  if (finding.detail) {
    body.appendChild(el("div", { class: "cd-detail", text: finding.detail }));
  }
  if (finding.impact) {
    const impactBlock = el("div", { class: "cd-impact" });
    impactBlock.appendChild(el("div", { class: "cd-impact-label", text: "What this means for you" }));
    impactBlock.appendChild(el("div", { class: "cd-impact-text", text: finding.impact }));
    body.appendChild(impactBlock);
  }
  if (finding.remedy) {
    body.appendChild(buildRemedyBlock(finding, ctx));
  }

  header.addEventListener("click", () => {
    expanded = !expanded;
    header.setAttribute("aria-expanded", String(expanded));
    body.classList.toggle("cd-collapsed", !expanded);
    caret.className = `pi ${expanded ? "pi-angle-down" : "pi-angle-right"} cd-icon cd-caret`;
  });

  const card = el("div", { class: "cd-finding" });
  card.appendChild(header);
  card.appendChild(body);
  return card;
}

function buildFindingsList(data, ctx) {
  const container = el("div", { class: "cd-findings" });
  const findings = data.findings || [];
  if (!findings.length) {
    container.appendChild(el("div", { class: "cd-empty" }, [icon("pi-check"), el("span", { text: "No findings — nothing to report." })]));
    return container;
  }

  const groups = new Map(); // preserves first-appearance order of categories
  for (const finding of findings) {
    const cat = finding.category || "General";
    if (!groups.has(cat)) groups.set(cat, []);
    groups.get(cat).push(finding);
  }

  for (const [category, items] of groups) {
    const section = el("div", { class: "cd-category" });
    section.appendChild(el("div", { class: "cd-category-title", text: category }));
    const list = el("div", { class: "cd-category-list" });
    items.forEach((f) => list.appendChild(buildFindingRow(f, ctx)));
    section.appendChild(list);
    container.appendChild(section);
  }
  return container;
}

// ---------------------------------------------------------------------------
// Header, skeleton, error view
// ---------------------------------------------------------------------------

async function copyReport(btn) {
  try {
    const res = await api.fetchApi("/comfydoctor/report.md");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const text = await res.text();
    await navigator.clipboard.writeText(text);
    flashButton(btn, "Copied!");
  } catch (err) {
    flashButton(btn, "Copy failed");
  }
}

function downloadHtml() {
  const a = el("a", { href: apiUrl("/comfydoctor/report.html"), download: "comfydoctor-report.html" });
  document.body.appendChild(a);
  a.click();
  a.remove();
}

function buildHeader(data, ctx) {
  const header = el("div", { class: "cd-header" });

  const tier = healthTier(data.health);
  const scoreRow = el("div", { class: "cd-score-row" });
  const scoreCircle = el("div", { class: `cd-score cd-score--${tier}` });
  scoreCircle.appendChild(el("span", { class: "cd-score-num", text: String(Math.round(data.health)) }));
  scoreRow.appendChild(scoreCircle);

  const info = el("div", { class: "cd-score-info" });
  info.appendChild(el("div", { class: `cd-score-label cd-score-label--${tier}`, text: healthLabel(data.health) }));
  const counts = countsPhrase(data.counts);
  if (counts) info.appendChild(el("div", { class: "cd-counts-line", text: counts }));
  const scannedAt = data.scanned_at ? new Date(data.scanned_at) : null;
  const scannedText = scannedAt && !isNaN(scannedAt) ? scannedAt.toLocaleString() : "";
  const duration = typeof data.duration_ms === "number" ? `${(data.duration_ms / 1000).toFixed(1)}s` : "";
  const metaText = [scannedText, duration].filter(Boolean).join(" · ");
  if (metaText) info.appendChild(el("div", { class: "cd-meta-line", text: metaText }));
  scoreRow.appendChild(info);
  header.appendChild(scoreRow);

  if (data.comfy_runtime === false) {
    header.appendChild(
      el("div", { class: "cd-runtime-note" }, [
        icon("pi-info-circle"),
        el("span", { text: "Running outside the ComfyUI runtime — some checks may be skipped." }),
      ])
    );
  }

  const btnRow = el("div", { class: "cd-header-btns" });
  btnRow.appendChild(iconButton("pi-refresh", "Scan again", () => ctx.scan()));
  const copyBtn = iconButton("pi-copy", "Copy report", () => copyReport(copyBtn));
  btnRow.appendChild(copyBtn);
  btnRow.appendChild(iconButton("pi-download", "Download HTML", () => downloadHtml()));
  header.appendChild(btnRow);

  return header;
}

function buildSkeleton() {
  const wrap = el("div", { class: "cd-skeleton" });
  wrap.appendChild(icon("pi-spinner pi-spin"));
  wrap.appendChild(el("div", { class: "cd-skeleton-text", text: "Examining your environment…" }));
  return wrap;
}

function buildErrorView(message, ctx) {
  const wrap = el("div", { class: "cd-error-view" });
  wrap.appendChild(icon("pi-exclamation-triangle"));
  wrap.appendChild(el("div", { class: "cd-error-text", text: message }));
  wrap.appendChild(iconButton("pi-refresh", "Retry", () => ctx.scan(), "cd-btn-primary"));
  return wrap;
}

// ---------------------------------------------------------------------------
// Stylesheet injection (once per document)
// ---------------------------------------------------------------------------

function ensureStylesheet() {
  if (document.getElementById("comfydoctor-styles")) return;
  const link = document.createElement("link");
  link.id = "comfydoctor-styles";
  link.rel = "stylesheet";
  link.href = new URL("./comfydoctor.css", import.meta.url).href;
  document.head.appendChild(link);
}

// ---------------------------------------------------------------------------
// Extension registration
// ---------------------------------------------------------------------------

app.registerExtension({
  name: "Kurdknight.ComfyDoctor",
  async setup() {
    // The sidebar API is the ONE thing here coupled to ComfyUI's frontend, so it
    // is the one thing that a future frontend could take away. If it ever does,
    // lose the panel - not the tool. The scanner, the rules and the fixes are
    // plain Python with no ComfyUI dependency at all, and stay reachable through
    // `doctor.py` / comfydoctor.bat and through /comfydoctor/scan.
    //
    // Throwing here would be the worst outcome: an unhandled error in setup()
    // can take down other extensions registered after us. A diagnostic tool has
    // no business breaking the app it is meant to diagnose.
    if (!app.extensionManager?.registerSidebarTab) {
      console.warn(
        "[ComfyDoctor] This ComfyUI frontend has no sidebar-tab API, so the panel is " +
          "unavailable. Everything still works from a terminal:\n" +
          "    python custom_nodes/Kurdknight_comfycheck/doctor.py\n" +
          "(or double-click comfydoctor.bat in that folder)"
      );
      return;
    }

    app.extensionManager.registerSidebarTab({
      id: "comfydoctor",
      icon: "pi pi-heart",
      title: "Doctor",
      tooltip: "ComfyDoctor — diagnose your environment",
      type: "custom",
      render: (rootEl) => {
        ensureStylesheet();

        const ctx = { timers: new Set() };
        const state = { loading: false, error: null, data: null };

        const panelRoot = el("div", { class: "comfydoctor" });
        rootEl.textContent = "";
        rootEl.appendChild(panelRoot);

        function update() {
          panelRoot.textContent = "";
          if (state.loading && !state.data) {
            panelRoot.appendChild(buildSkeleton());
            return;
          }
          if (state.error) {
            panelRoot.appendChild(buildErrorView(state.error, ctx));
            return;
          }
          if (!state.data) return;
          panelRoot.appendChild(buildHeader(state.data, ctx));
          panelRoot.appendChild(buildFindingsList(state.data, ctx));
        }

        async function scan() {
          state.loading = true;
          state.error = null;
          update();
          try {
            const res = await api.fetchApi("/comfydoctor/scan");
            if (!res.ok) throw new Error(`Scan failed (HTTP ${res.status})`);
            state.data = await res.json();
          } catch (err) {
            state.error = (err && err.message) || "Failed to scan your environment.";
          } finally {
            state.loading = false;
            update();
          }
        }

        ctx.scan = scan;
        scan();

        return () => {
          for (const token of ctx.timers) clearTimeout(token);
          ctx.timers.clear();
        };
      },
    });
  },
});
