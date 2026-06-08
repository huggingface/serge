(() => {
  const form = document.getElementById("submit-form");
  const btn = document.getElementById("submit-btn");
  const banner = document.getElementById("error-banner");
  const prEl = document.getElementById("pr");
  const commentEl = document.getElementById("comment");
  const providerEl = document.getElementById("llm-provider");
  const modelEl = document.getElementById("llm-model");
  const modelSelectEl = document.getElementById("llm-model-select");
  const baseUrlEl = document.getElementById("llm-base-url");
  const customBaseRow = document.getElementById("custom-base-row");
  const providerHint = document.getElementById("provider-hint");
  const jobsSection = document.getElementById("jobs-section");
  const jobsTbody = document.getElementById("jobs-tbody");
  const jobsCount = document.getElementById("jobs-count");
  const LLM_PREF_COOKIE = "serge_llm_prefs";
  // Per-provider default model. Seeded with a hardcoded fallback for
  // Anthropic so the form still works if /llm-options can't be reached;
  // loadLlmOptions() overwrites entries with whatever the server reports
  // (which folds in cfg.llm_model for the system-default provider).
  const providerDefaultModels = {
    anthropic: "claude-opus-4-6",
  };

  const PRESETS = {
    "first-pass":
      "@askserge first-pass review. Flag clear correctness, security, and " +
      "API-shape problems; skip style nits and speculative concerns. " +
      "Use the browse tools to verify any claim before flagging it.",
    "new-model":
      "@askserge this PR adds a new model. Verify the modular file structure, " +
      "tokenizer/config/processor wiring, that tests exist and cover the " +
      "model meaningfully, and that the docs entry is in place. Be strict " +
      "about consistency with sibling model implementations — use the " +
      "browse tools to compare.",
    "bugfix":
      "@askserge this is a bug fix. Verify the change actually addresses the " +
      "root cause (not just the symptom), confirm a regression test was " +
      "added (or call out clearly that one is missing), and flag any " +
      "unrelated changes that slipped in.",
    "docs":
      "@askserge this is a documentation change. Focus on accuracy, clarity, " +
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
    const defaultModel = providerDefaultModels[providerEl.value];
    if (defaultModel && !modelEl.value.trim()) {
      modelEl.value = defaultModel;
    }
  }

  function resetModelToProviderDefault() {
    // Used when the user changes the provider dropdown: always overwrite
    // the model field, since the previous model name almost certainly
    // doesn't exist on the new provider. Clears the field when the new
    // provider has no registered default.
    modelEl.value = providerDefaultModels[providerEl.value] || "";
  }

  // HF Router model catalogue, lazily fetched once and cached. null until
  // loaded; [] when the endpoint is unreachable (we then fall back to the
  // free-text input rather than an empty dropdown).
  let hfModels = null;

  async function ensureHfModels() {
    if (hfModels !== null) return hfModels;
    try {
      const r = await fetch("/llm-options/hf-models");
      const data = r.ok ? await r.json() : {};
      hfModels = Array.isArray(data.models) ? data.models : [];
    } catch {
      hfModels = [];
    }
    return hfModels;
  }

  function populateModelSelect() {
    // The text input stays the source of truth for the submitted value;
    // the dropdown mirrors it. Keep the current value selectable even if
    // the router doesn't list it (e.g. a config default not yet live), so
    // switching to HF never silently drops a configured model.
    const current = modelEl.value.trim();
    const options = [...(hfModels || [])];
    if (current && !options.includes(current)) options.unshift(current);
    modelSelectEl.replaceChildren();
    for (const m of options) {
      const opt = document.createElement("option");
      opt.value = m;
      opt.textContent = m;
      modelSelectEl.appendChild(opt);
    }
    if (current && options.includes(current)) {
      modelSelectEl.value = current;
    } else if (options.length) {
      modelSelectEl.value = options[0];
      modelEl.value = options[0];
    }
  }

  // Show a dropdown of HF Router models when the HF provider is selected;
  // every other provider keeps the free-text input. Falls back to the text
  // input when the model list can't be fetched.
  async function updateModelControl() {
    if (providerEl.value === "hf") {
      const models = await ensureHfModels();
      // Guard against the provider changing while the fetch was in flight.
      if (providerEl.value === "hf" && models.length) {
        populateModelSelect();
        modelSelectEl.style.display = "";
        modelEl.style.display = "none";
        modelEl.required = false;
        return;
      }
    }
    modelSelectEl.style.display = "none";
    modelEl.style.display = "";
    modelEl.required = true;
  }

  function ingestProviderDefaults(providers) {
    if (!Array.isArray(providers)) return;
    for (const p of providers) {
      if (
        p &&
        typeof p.id === "string" &&
        typeof p.default_model === "string" &&
        p.default_model
      ) {
        providerDefaultModels[p.id] = p.default_model;
      }
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
      ingestProviderDefaults(data.providers);
      applySavedLlmPrefs();
      applyProviderModelDefault();
      updateProviderFields();
      await updateModelControl();
    } catch {
      applySavedLlmPrefs();
      applyProviderModelDefault();
      updateProviderFields();
      await updateModelControl();
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

  // Mirrors the server-side _parse_pr_ref shapes: full GitHub URL,
  // "owner/repo#NN", or "owner/repo/pull/NN". Returns null when the
  // string isn't yet a parseable PR reference (e.g. half-typed).
  function parseOwnerRepo(raw) {
    const s = (raw || "").trim();
    if (!s) return null;
    const name = "[A-Za-z0-9._-]+";
    let m;
    m = s.match(new RegExp(`github\\.com/(${name})/(${name})/(?:pull|pulls)/\\d+`));
    if (m) return { owner: m[1], repo: m[2] };
    m = s.match(new RegExp(`^(${name})/(${name})#\\d+`));
    if (m) return { owner: m[1], repo: m[2] };
    m = s.match(new RegExp(`^(${name})/(${name})/pull/\\d+`));
    if (m) return { owner: m[1], repo: m[2] };
    return null;
  }

  let lookupTimer = 0;
  let lastLookupKey = "";

  function scheduleProviderLookup() {
    clearTimeout(lookupTimer);
    lookupTimer = setTimeout(runProviderLookup, 250);
  }

  async function runProviderLookup() {
    const parsed = parseOwnerRepo(prEl.value);
    if (!parsed) {
      providerHint.textContent = "";
      lastLookupKey = "";
      return;
    }
    const key = `${parsed.owner}/${parsed.repo}`.toLowerCase();
    if (key === lastLookupKey) return;
    lastLookupKey = key;
    try {
      const qs = new URLSearchParams({ owner: parsed.owner, repo: parsed.repo });
      const r = await fetch(`/reviews/lookup-provider?${qs}`);
      if (!r.ok) {
        providerHint.textContent = "";
        return;
      }
      const data = await r.json();
      if (!data.match) {
        providerHint.textContent = `No provider config matches ${parsed.owner}/${parsed.repo}. Add one at /admin or your submission will be refused.`;
        return;
      }
      applyMatchedConfig(data.match, parsed);
    } catch {
      // Non-fatal — the user can still submit; the server will reject
      // with the same message if no config matches.
      providerHint.textContent = "";
    }
  }

  function applyMatchedConfig(match, parsed) {
    providerEl.value = match.provider;
    if (match.default_model) {
      modelEl.value = match.default_model;
    }
    if (match.provider === "custom" && match.api_base) {
      baseUrlEl.value = match.api_base;
    }
    updateProviderFields();
    updateModelControl();
    const modelPart = match.default_model ? ` · model ${match.default_model}` : "";
    const scope =
      match.repo_pattern && match.repo_pattern !== `${parsed.owner}/${parsed.repo}`
        ? ` (via ${match.repo_pattern})`
        : "";
    providerHint.textContent = `Using ${match.provider}${modelPart}${scope}.`;
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
    resetModelToProviderDefault();
    updateModelControl();
    saveLlmPrefs();
  });
  modelEl.addEventListener("change", saveLlmPrefs);
  modelSelectEl.addEventListener("change", () => {
    modelEl.value = modelSelectEl.value;
    saveLlmPrefs();
  });
  baseUrlEl.addEventListener("change", saveLlmPrefs);

  // Auto-fill provider/model from the matching DB config whenever the
  // PR field changes. Debounced so we don't hit the endpoint on every
  // keystroke, and short-circuited by lastLookupKey when the
  // owner/repo hasn't actually changed.
  prEl.addEventListener("input", scheduleProviderLookup);
  prEl.addEventListener("change", runProviderLookup);

  loadLlmOptions();
  loadJobs();
  // Catch the case where the input already has a value at load time
  // (e.g. browser autofill or a prefilled paste).
  runProviderLookup();
  // Soft-refresh every 5s so running reviews tick over to "done" /
  // "published" without the user having to reload.
  setInterval(loadJobs, 5000);
})();
