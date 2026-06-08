(() => {
  const form = document.getElementById("config-form");
  const banner = document.getElementById("error-banner");
  const tbody = document.getElementById("configs-tbody");
  const emptyHint = document.getElementById("configs-empty");
  const providerEl = document.getElementById("provider");
  const apiBaseRow = document.getElementById("api-base-row");
  const apiBaseEl = document.getElementById("api_base");
  const apiKeyEl = document.getElementById("api_key");
  const apiKeyHint = document.getElementById("api-key-hint");
  const defaultModelEl = document.getElementById("default_model");
  const defaultModelSelectEl = document.getElementById("default-model-select");
  const repoPatternEl = document.getElementById("repo_pattern");
  const allowedUsersEl = document.getElementById("allowed_users");
  const allowedOrgsEl = document.getElementById("allowed_orgs");
  const submitBtn = document.getElementById("submit-btn");
  const cancelBtn = document.getElementById("cancel-btn");
  const configIdEl = document.getElementById("config-id");
  const formTitle = document.getElementById("form-title");

  let defaultModels = {};
  let editing = null;

  function showError(msg) {
    banner.textContent = msg;
    banner.classList.add("visible");
  }

  function clearError() {
    banner.textContent = "";
    banner.classList.remove("visible");
  }

  async function errorMessage(r) {
    const body = await r.text();
    try {
      const parsed = JSON.parse(body);
      if (typeof parsed.detail === "string") return `${r.status}: ${parsed.detail}`;
    } catch {}
    return `${r.status}: ${body}`;
  }

  function updateProviderRow() {
    const isCustom = providerEl.value === "custom";
    apiBaseRow.style.display = isCustom ? "" : "none";
    apiBaseEl.required = isCustom;
  }

  function applyProviderDefaultModel() {
    if (defaultModelEl.value.trim()) return;
    const def = defaultModels[providerEl.value];
    if (def) defaultModelEl.value = def;
  }

  // HF Router model catalogue, lazily fetched once and cached. null until
  // loaded; [] when unreachable (we then keep the free-text input).
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
    // The text input stays the source of truth; the dropdown mirrors it.
    // A leading blank option keeps "no default" expressible (the submit
    // form's model then wins). Preserve any current value even if the
    // router doesn't list it.
    const current = defaultModelEl.value.trim();
    const models = [...(hfModels || [])];
    if (current && !models.includes(current)) models.unshift(current);
    defaultModelSelectEl.replaceChildren();
    const blank = document.createElement("option");
    blank.value = "";
    blank.textContent = "— none (use submit-form model) —";
    defaultModelSelectEl.appendChild(blank);
    for (const m of models) {
      const opt = document.createElement("option");
      opt.value = m;
      opt.textContent = m;
      defaultModelSelectEl.appendChild(opt);
    }
    defaultModelSelectEl.value = current;
  }

  // Show a dropdown of HF Router models when the HF provider is selected;
  // other providers keep the free-text input. Falls back to the text input
  // when the model list can't be fetched.
  async function updateModelControl() {
    if (providerEl.value === "hf") {
      const models = await ensureHfModels();
      if (providerEl.value === "hf" && models.length) {
        populateModelSelect();
        defaultModelSelectEl.style.display = "";
        defaultModelEl.style.display = "none";
        return;
      }
    }
    defaultModelSelectEl.style.display = "none";
    defaultModelEl.style.display = "";
  }

  function resetForm() {
    editing = null;
    configIdEl.value = "";
    providerEl.value = "hf";
    apiBaseEl.value = "";
    apiKeyEl.value = "";
    apiKeyEl.required = true;
    apiKeyEl.placeholder = "paste key here";
    apiKeyHint.textContent =
      "Stored as-is in the SQLite file. Never returned to the UI after saving.";
    defaultModelEl.value = "";
    repoPatternEl.value = "";
    allowedUsersEl.value = "";
    allowedOrgsEl.value = "";
    formTitle.textContent = "Add a provider config";
    submitBtn.textContent = "Save";
    cancelBtn.style.display = "none";
    updateProviderRow();
    updateModelControl();
  }

  function startEdit(cfg) {
    editing = cfg.id;
    configIdEl.value = cfg.id;
    providerEl.value = cfg.provider;
    apiBaseEl.value = cfg.api_base || "";
    apiKeyEl.value = "";
    apiKeyEl.required = false;
    apiKeyEl.placeholder = "leave blank to keep current key";
    apiKeyHint.textContent = `Current: ${cfg.api_key_status || "set"}. Leave blank to keep it; type a new value to replace.`;
    defaultModelEl.value = cfg.default_model || "";
    repoPatternEl.value = cfg.repo_pattern || "";
    allowedUsersEl.value = (cfg.allowed_users || []).join(", ");
    allowedOrgsEl.value = (cfg.allowed_orgs || []).join(", ");
    formTitle.textContent = `Edit config (${cfg.provider} · ${cfg.repo_pattern})`;
    submitBtn.textContent = "Save changes";
    cancelBtn.style.display = "";
    updateProviderRow();
    updateModelControl();
    window.scrollTo({ top: form.offsetTop - 16, behavior: "smooth" });
  }

  function relTime(epoch) {
    if (!epoch) return "";
    const delta = Date.now() / 1000 - epoch;
    if (delta < 60) return `${Math.floor(delta)}s ago`;
    if (delta < 3600) return `${Math.floor(delta / 60)}m ago`;
    if (delta < 86400) return `${Math.floor(delta / 3600)}h ago`;
    return `${Math.floor(delta / 86400)}d ago`;
  }

  function renderConfigs(configs) {
    tbody.replaceChildren();
    if (!configs.length) {
      emptyHint.style.display = "";
      return;
    }
    emptyHint.style.display = "none";
    for (const cfg of configs) {
      const tr = document.createElement("tr");

      const tdRepo = document.createElement("td");
      tdRepo.textContent = cfg.repo_pattern;
      tr.appendChild(tdRepo);

      const tdProvider = document.createElement("td");
      tdProvider.textContent = cfg.provider;
      tr.appendChild(tdProvider);

      const tdBase = document.createElement("td");
      tdBase.textContent = cfg.api_base || "—";
      tdBase.title = cfg.api_base || "";
      tr.appendChild(tdBase);

      const tdModel = document.createElement("td");
      tdModel.textContent = cfg.default_model || "—";
      tr.appendChild(tdModel);

      const tdAccess = document.createElement("td");
      const users = (cfg.allowed_users || []).join(", ");
      const orgs = (cfg.allowed_orgs || []).join(", ");
      const access = [];
      if (users) access.push(`users: ${users}`);
      if (orgs) access.push(`orgs: ${orgs}`);
      tdAccess.textContent = access.join(" · ") || "—";
      tr.appendChild(tdAccess);

      const tdKey = document.createElement("td");
      tdKey.textContent = cfg.api_key_status || "";
      tdKey.title = "API key is never displayed; edit the row to replace it.";
      tr.appendChild(tdKey);

      const tdUpdated = document.createElement("td");
      tdUpdated.className = "ago";
      tdUpdated.textContent = relTime(cfg.updated_at);
      if (cfg.updated_at) {
        tdUpdated.title = new Date(cfg.updated_at * 1000).toLocaleString();
      }
      tr.appendChild(tdUpdated);

      const tdActions = document.createElement("td");
      const editBtn = document.createElement("button");
      editBtn.type = "button";
      editBtn.className = "secondary";
      editBtn.textContent = "Edit";
      editBtn.addEventListener("click", () => startEdit(cfg));
      const delBtn = document.createElement("button");
      delBtn.type = "button";
      delBtn.className = "danger";
      delBtn.style.marginLeft = "8px";
      delBtn.textContent = "Delete";
      delBtn.addEventListener("click", () => deleteConfig(cfg));
      tdActions.appendChild(editBtn);
      tdActions.appendChild(delBtn);
      tr.appendChild(tdActions);

      tbody.appendChild(tr);
    }
  }

  async function loadConfigs() {
    clearError();
    try {
      const r = await fetch("/admin/providers");
      if (!r.ok) {
        showError(`Could not load configs — ${await errorMessage(r)}`);
        return;
      }
      const data = await r.json();
      defaultModels = data.default_models || {};
      applyProviderDefaultModel();
      renderConfigs(data.configs || []);
    } catch (err) {
      showError(`Could not load configs — ${err.message}`);
    }
  }

  async function deleteConfig(cfg) {
    const label = `${cfg.provider} · ${cfg.repo_pattern}`;
    if (!confirm(`Delete provider config "${label}"? This cannot be undone.`)) return;
    try {
      const r = await fetch(`/admin/providers/${encodeURIComponent(cfg.id)}`, {
        method: "DELETE",
      });
      if (!r.ok) {
        showError(`Could not delete — ${await errorMessage(r)}`);
        return;
      }
      if (editing === cfg.id) resetForm();
      await loadConfigs();
    } catch (err) {
      showError(`Could not delete — ${err.message}`);
    }
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    clearError();
    submitBtn.disabled = true;
    const body = {
      provider: providerEl.value,
      api_base: apiBaseEl.value.trim(),
      default_model: defaultModelEl.value.trim(),
      repo_pattern: repoPatternEl.value.trim(),
      allowed_users: allowedUsersEl.value.trim(),
      allowed_orgs: allowedOrgsEl.value.trim(),
    };
    const apiKey = apiKeyEl.value;
    // On create the key is required. On edit a blank key means "keep
    // the stored one"; we only attach the field when non-empty so the
    // backend can tell the two cases apart.
    if (!editing) {
      if (!apiKey.trim()) {
        showError("API key is required when creating a config.");
        submitBtn.disabled = false;
        return;
      }
      body.api_key = apiKey;
    } else if (apiKey.trim()) {
      body.api_key = apiKey;
    }
    const url = editing
      ? `/admin/providers/${encodeURIComponent(editing)}`
      : "/admin/providers";
    const method = editing ? "PATCH" : "POST";
    try {
      const r = await fetch(url, {
        method,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!r.ok) {
        showError(`Could not save — ${await errorMessage(r)}`);
        submitBtn.disabled = false;
        return;
      }
      resetForm();
      await loadConfigs();
    } catch (err) {
      showError(`Could not save — ${err.message}`);
    } finally {
      submitBtn.disabled = false;
    }
  });

  cancelBtn.addEventListener("click", () => {
    resetForm();
    clearError();
  });

  providerEl.addEventListener("change", () => {
    updateProviderRow();
    if (!editing) {
      defaultModelEl.value = defaultModels[providerEl.value] || "";
    }
    updateModelControl();
  });

  defaultModelSelectEl.addEventListener("change", () => {
    defaultModelEl.value = defaultModelSelectEl.value;
  });

  resetForm();
  loadConfigs();
})();
