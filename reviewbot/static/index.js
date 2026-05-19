(() => {
  const form = document.getElementById("submit-form");
  const btn = document.getElementById("submit-btn");
  const banner = document.getElementById("error-banner");
  const commentEl = document.getElementById("comment");
  const providerEl = document.getElementById("llm-provider");
  const modelEl = document.getElementById("llm-model");
  const baseUrlEl = document.getElementById("llm-base-url");
  const customBaseRow = document.getElementById("custom-base-row");
  const jobsSection = document.getElementById("jobs-section");
  const jobsTbody = document.getElementById("jobs-tbody");
  const jobsCount = document.getElementById("jobs-count");
  const LLM_PREF_COOKIE = "serge_llm_prefs";
  const PROVIDER_DEFAULT_MODELS = {
    anthropic: "claude-opus-4-6",
  };

  const PRESETS = {
    "first-pass":
      "@serge first-pass review. Flag clear correctness, security, and " +
      "API-shape problems; skip style nits and speculative concerns. " +
      "Use the browse tools to verify any claim before flagging it.",
    "new-model":
      "@serge this PR adds a new model. Verify the modular file structure, " +
      "tokenizer/config/processor wiring, that tests exist and cover the " +
      "model meaningfully, and that the docs entry is in place. Be strict " +
      "about consistency with sibling model implementations — use the " +
      "browse tools to compare.",
    "bugfix":
      "@serge this is a bug fix. Verify the change actually addresses the " +
      "root cause (not just the symptom), confirm a regression test was " +
      "added (or call out clearly that one is missing), and flag any " +
      "unrelated changes that slipped in.",
    "docs":
      "@serge this is a documentation change. Focus on accuracy, clarity, " +
      "and whether code samples are runnable. Ignore changes outside of " +
      "docs and docstrings; do not nit on prose style unless it's actively " +
      "misleading.",
  };

  function showError(msg) {
    banner.textContent = msg;
    banner.classList.add("visible");
  }

  async function errorMessage(r) {
    const body = await r.text();
    try {
      const parsed = JSON.parse(body);
      if (typeof parsed.detail === "string") return `${r.status}: ${parsed.detail}`;
    } catch {
      // Fall through to the raw body.
    }
    return `${r.status}: ${body}`;
  }

  function cookieOptions() {
    const secure = window.location.protocol === "https:" ? "; Secure" : "";
    return `Path=/; Max-Age=31536000; SameSite=Lax${secure}`;
  }

  function readCookie(name) {
    const prefix = `${name}=`;
    for (const part of document.cookie.split(";")) {
      const item = part.trim();
      if (item.startsWith(prefix)) return item.slice(prefix.length);
    }
    return "";
  }

  function loadSavedLlmPrefs() {
    const raw = readCookie(LLM_PREF_COOKIE);
    if (!raw) return null;
    try {
      const parsed = JSON.parse(decodeURIComponent(raw));
      return parsed && typeof parsed === "object" ? parsed : null;
    } catch {
      return null;
    }
  }

  function saveLlmPrefs() {
    const prefs = {
      provider: providerEl.value,
      model: modelEl.value.trim(),
      base_url: baseUrlEl.value.trim(),
    };
    document.cookie = `${LLM_PREF_COOKIE}=${encodeURIComponent(JSON.stringify(prefs))}; ${cookieOptions()}`;
  }

  function applySavedLlmPrefs() {
    const prefs = loadSavedLlmPrefs();
    if (!prefs) return;
    if (["hf", "openai", "anthropic", "custom"].includes(prefs.provider)) {
      providerEl.value = prefs.provider;
    }
    if (typeof prefs.model === "string") modelEl.value = prefs.model;
    if (typeof prefs.base_url === "string") baseUrlEl.value = prefs.base_url;
  }

  function updateProviderFields() {
    const isCustom = providerEl.value === "custom";
    customBaseRow.style.display = isCustom ? "" : "none";
    baseUrlEl.required = isCustom;
  }

  function applyProviderModelDefault() {
    const defaultModel = PROVIDER_DEFAULT_MODELS[providerEl.value];
    if (defaultModel && !modelEl.value.trim()) {
      modelEl.value = defaultModel;
    }
  }

  async function loadLlmOptions() {
    try {
      const r = await fetch("/llm-options");
      if (!r.ok) return;
      const data = await r.json();
      providerEl.value = data.default_provider || "hf";
      modelEl.value = data.default_model || "";
      baseUrlEl.value = data.custom_base_url || "";
      applySavedLlmPrefs();
      applyProviderModelDefault();
      updateProviderFields();
    } catch {
      applySavedLlmPrefs();
      applyProviderModelDefault();
      updateProviderFields();
    }
  }

  function relTime(epochSeconds) {
    const delta = Date.now() / 1000 - epochSeconds;
    if (delta < 60) return `${Math.floor(delta)}s ago`;
    if (delta < 3600) return `${Math.floor(delta / 60)}m ago`;
    if (delta < 86400) return `${Math.floor(delta / 3600)}h ago`;
    return `${Math.floor(delta / 86400)}d ago`;
  }

  function renderJobs(jobs) {
    jobsTbody.replaceChildren();
    jobsCount.textContent = jobs.length ? `(${jobs.length})` : "";
    if (!jobs.length) {
      jobsSection.style.display = "none";
      return;
    }
    for (const j of jobs) {
      const tr = document.createElement("tr");

      const tdStatus = document.createElement("td");
      const badge = document.createElement("span");
      badge.className = `status-badge ${j.status}`;
      badge.textContent = j.status;
      tdStatus.appendChild(badge);

      const tdPr = document.createElement("td");
      const link = document.createElement("a");
      link.href = j.url;
      link.textContent = `${j.owner}/${j.repo}#${j.number}`;
      tdPr.appendChild(link);

      const tdAgo = document.createElement("td");
      tdAgo.className = "ago";
      tdAgo.textContent = relTime(j.created_at);
      tdAgo.title = new Date(j.created_at * 1000).toLocaleString();

      tr.appendChild(tdStatus);
      tr.appendChild(tdPr);
      tr.appendChild(tdAgo);
      jobsTbody.appendChild(tr);
    }
    jobsSection.style.display = "";
  }

  async function loadJobs() {
    try {
      const r = await fetch("/reviews");
      if (!r.ok) return;
      const data = await r.json();
      renderJobs(data.jobs || []);
    } catch {
      // Non-fatal — the form still works.
    }
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    btn.disabled = true;
    banner.classList.remove("visible");
    const pr = document.getElementById("pr").value.trim();
    const comment = document.getElementById("comment").value.trim();
    const llm_provider = providerEl.value;
    const llm_model = modelEl.value.trim();
    const llm_base_url = baseUrlEl.value.trim();
    try {
      saveLlmPrefs();
      const r = await fetch("/reviews", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pr, comment, llm_provider, llm_model, llm_base_url }),
      });
      if (!r.ok) {
        throw new Error(await errorMessage(r));
      }
      const { url } = await r.json();
      window.location.href = url;
    } catch (err) {
      showError(`Could not start review — ${err.message}`);
      btn.disabled = false;
    }
  });

  for (const b of document.querySelectorAll("button.preset")) {
    b.addEventListener("click", () => {
      const text = PRESETS[b.dataset.preset];
      if (!text) return;
      commentEl.value = text;
      commentEl.focus();
    });
  }

  providerEl.addEventListener("change", () => {
    updateProviderFields();
    applyProviderModelDefault();
    saveLlmPrefs();
  });
  modelEl.addEventListener("change", saveLlmPrefs);
  baseUrlEl.addEventListener("change", saveLlmPrefs);

  loadLlmOptions();
  loadJobs();
  // Soft-refresh every 5s so running reviews tick over to "done" /
  // "published" without the user having to reload.
  setInterval(loadJobs, 5000);
})();
