(() => {
  const tbody = document.getElementById("pods-tbody");
  const countEl = document.getElementById("pods-count");
  const backendEl = document.getElementById("pods-backend");
  const emptyEl = document.getElementById("pods-empty");
  const banner = document.getElementById("error-banner");

  function showError(msg) {
    banner.textContent = msg;
    banner.classList.add("visible");
  }
  function clearError() {
    banner.textContent = "";
    banner.classList.remove("visible");
  }

  function relTime(epochSeconds) {
    if (!epochSeconds) return "—";
    const delta = Date.now() / 1000 - epochSeconds;
    if (delta < 60) return `${Math.floor(delta)}s ago`;
    if (delta < 3600) return `${Math.floor(delta / 60)}m ago`;
    if (delta < 86400) return `${Math.floor(delta / 3600)}h ago`;
    return `${Math.floor(delta / 86400)}d ago`;
  }

  function cell(text, opts = {}) {
    const td = document.createElement("td");
    td.textContent = text == null || text === "" ? "—" : text;
    if (opts.cls) td.className = opts.cls;
    if (opts.title) td.title = opts.title;
    return td;
  }

  async function killPod(row, btn) {
    if (!row.job_name) return;
    if (!window.confirm(`Kill task pod ${row.job_name}? This deletes the Job.`)) {
      return;
    }
    btn.disabled = true;
    btn.textContent = "killing…";
    try {
      const r = await fetch("/admin/pods/kill", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          job_name: row.job_name,
          namespace: row.namespace,
        }),
      });
      if (!r.ok) {
        showError(`Kill failed (${r.status})`);
        btn.disabled = false;
        btn.textContent = "kill";
        return;
      }
      load(); // refresh immediately
    } catch (err) {
      showError(`Kill failed: ${err}`);
      btn.disabled = false;
      btn.textContent = "kill";
    }
  }

  function render(data) {
    const rows = data.pods || [];
    tbody.replaceChildren();
    countEl.textContent = rows.length ? `(${rows.length})` : "";
    backendEl.textContent = data.backend ? `· backend: ${data.backend}` : "";
    emptyEl.hidden = rows.length > 0;
    if (data.error) showError(`Cluster query failed: ${data.error}`);
    else clearError();

    for (const row of rows) {
      const tr = document.createElement("tr");

      tr.appendChild(
        cell(relTime(row.start_epoch), {
          cls: "ago",
          title: row.start_epoch
            ? new Date(row.start_epoch * 1000).toLocaleString()
            : "",
        }),
      );

      const tdKind = document.createElement("td");
      const kindTag = document.createElement("span");
      kindTag.className = `source-tag type-${row.kind || "task"}`;
      kindTag.textContent = row.kind || "task";
      tdKind.appendChild(kindTag);
      tr.appendChild(tdKind);

      tr.appendChild(cell(row.repo || "—"));
      tr.appendChild(cell(row.user || "—"));

      const tdStatus = document.createElement("td");
      if (row.status) {
        const badge = document.createElement("span");
        badge.className = `status-badge ${row.status}`;
        if (row.status === "running") {
          const spinner = document.createElement("span");
          spinner.className = "spinner";
          spinner.setAttribute("aria-hidden", "true");
          badge.appendChild(spinner);
        }
        badge.appendChild(document.createTextNode(row.status));
        tdStatus.appendChild(badge);
      } else {
        tdStatus.textContent = "—";
      }
      tr.appendChild(tdStatus);

      tr.appendChild(cell(row.phase || "—"));
      tr.appendChild(cell(row.node || "—", { title: row.node }));
      tr.appendChild(cell(row.pod || row.job_name || "—", { title: row.job_name }));

      const tdAct = document.createElement("td");
      if (row.job_name) {
        const btn = document.createElement("button");
        btn.className = "secondary";
        btn.textContent = "kill";
        btn.addEventListener("click", () => killPod(row, btn));
        tdAct.appendChild(btn);
      }
      tr.appendChild(tdAct);

      tbody.appendChild(tr);
    }
  }

  async function load() {
    try {
      const r = await fetch("/admin/pods/data");
      if (!r.ok) {
        showError(`Failed to load pods (${r.status})`);
        return;
      }
      render(await r.json());
    } catch (err) {
      showError(`Failed to load pods: ${err}`);
    }
  }

  load();
  // Auto-refresh so pods appear/disappear and statuses tick over live.
  setInterval(load, 5000);
})();
