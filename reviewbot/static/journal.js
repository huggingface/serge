(() => {
  const tbody = document.getElementById("journal-tbody");
  const countEl = document.getElementById("journal-count");
  const banner = document.getElementById("error-banner");

  function showError(msg) {
    banner.textContent = msg;
    banner.classList.add("visible");
  }

  function relTime(epochSeconds) {
    const delta = Date.now() / 1000 - epochSeconds;
    if (delta < 60) return `${Math.floor(delta)}s ago`;
    if (delta < 3600) return `${Math.floor(delta / 60)}m ago`;
    if (delta < 86400) return `${Math.floor(delta / 3600)}h ago`;
    return `${Math.floor(delta / 86400)}d ago`;
  }

  function fmtTokens(n) {
    if (n === null || n === undefined || n === "") return "—";
    const v = Number(n);
    if (!Number.isFinite(v) || v <= 0) return "—";
    if (v >= 1_000_000) return `${(v / 1_000_000).toFixed(1)}M`;
    if (v >= 1000) return `${(v / 1000).toFixed(1)}k`;
    return String(v);
  }

  function cell(text, opts = {}) {
    const td = document.createElement("td");
    td.textContent = text == null || text === "" ? "—" : text;
    if (opts.cls) td.className = opts.cls;
    if (opts.title) td.title = opts.title;
    return td;
  }

  function renderEntries(entries) {
    tbody.replaceChildren();
    countEl.textContent = entries.length ? `(${entries.length})` : "";
    for (const e of entries) {
      const tr = document.createElement("tr");

      const tdWhen = cell(relTime(e.created_at), {
        cls: "ago",
        title: new Date(e.created_at * 1000).toLocaleString(),
      });

      const tdUser = cell(e.user || "—");
      // Webhook-triggered reviews have no logged-in submitter; tag them so
      // the journal makes clear they were kicked off by a GitHub comment.
      if (e.source === "webhook") {
        const tag = document.createElement("span");
        tag.className = "source-tag";
        tag.textContent = "webhook";
        tdUser.appendChild(tag);
      }

      const tdPr = document.createElement("td");
      const link = document.createElement("a");
      link.href = e.url;
      link.textContent = `${e.owner}/${e.repo}#${e.number}`;
      tdPr.appendChild(link);

      const tdProvider = cell(e.provider || "—");
      const tdModel = cell(e.model || "—");
      const tdIn = cell(fmtTokens(e.prompt_tokens));
      const tdOut = cell(fmtTokens(e.completion_tokens));

      const tdStatus = document.createElement("td");
      const badge = document.createElement("span");
      badge.className = `status-badge ${e.status}`;
      // A still-running review gets a small animated spinner so the
      // journal reads as live at a glance.
      if (e.status === "running") {
        const spinner = document.createElement("span");
        spinner.className = "spinner";
        spinner.setAttribute("aria-hidden", "true");
        badge.appendChild(spinner);
      }
      badge.appendChild(document.createTextNode(e.status));
      tdStatus.appendChild(badge);

      tr.appendChild(tdWhen);
      tr.appendChild(tdUser);
      tr.appendChild(tdPr);
      tr.appendChild(tdProvider);
      tr.appendChild(tdModel);
      tr.appendChild(tdIn);
      tr.appendChild(tdOut);
      tr.appendChild(tdStatus);
      tbody.appendChild(tr);
    }
  }

  async function load() {
    try {
      const r = await fetch("/journal/data");
      if (!r.ok) {
        showError(`Failed to load journal (${r.status})`);
        return;
      }
      const data = await r.json();
      renderEntries(data.entries || []);
    } catch (err) {
      showError(`Failed to load journal: ${err}`);
    }
  }

  load();
  // Soft-refresh every 5s so running reviews tick over to "done" without
  // requiring a manual page reload.
  setInterval(load, 5000);
})();
