const boot = window.__BOOT__ || {};

let cfg = boot.cfg || {
  model_key: "llama",
  max_length: 256,
  temperature: 0.7,
  top_p: 0.9,
  seed: 42,
};

let modalOpen = false;
let isProcessing = false;
let currentMode = boot.mode || "assistant";

const TRANSLATE_CHAR_LIMIT = 512;

/* Cached DOM Elements */
const modeTabs = document.querySelectorAll(".mode-tab");
const settingsRow = document.querySelector(".settings-row");
const suggestionRow = document.querySelector(".suggestion-row");
const modeSubtitle = document.getElementById("mode-subtitle");

const el = {
  // Chat
  chatLog: document.getElementById("chat-log"),
  input: document.getElementById("user-input"),
  sendBtn: document.getElementById("send-btn"),

  // Assistant settings
  temp: document.getElementById("assistant-temperature"),
  tempVal: document.getElementById("temperature-value"),
  topP: document.getElementById("assistant-top-p"),
  topPVal: document.getElementById("top-p-value"),
  maxTokens: document.getElementById("assistant-max-tokens"),
  seed: document.getElementById("assistant-seed"),
  clearHistory: document.getElementById("assistant-clear-history"),
  confirmSettings: document.getElementById("assistant-confirm-settings"),

  // Dialog
  dialog: document.getElementById("assistant-dialog"),
  dialogTitle: document.getElementById("assistant-dialog-title"),
  dialogText: document.getElementById("assistant-dialog-text"),
  dialogClear: document.getElementById("dialog-clear-history"),
  dialogSkip: document.getElementById("dialog-skip-reply"),

  // System usage
  cpuBar: document.getElementById("cpu-bar"),
  cpuText: document.getElementById("cpu-text"),
  gpuBar: document.getElementById("gpu-bar"),
  gpuText: document.getElementById("gpu-text"),

  // Character counter
  charCounter: document.getElementById("char-counter"),

  // Translate controls
  translateControls: document.getElementById("translate-controls"),
  translateSource: document.getElementById("translate-source"),
  translateTarget: document.getElementById("translate-target"),
  translateSwap: document.getElementById("translate-swap"),
  translateDirection: document.getElementById("translate-direction"),

  // Settings groups used in mode switching
  deviceGroup: document.getElementById("device-group"),
  modelGroup: document.getElementById("model-group"),
  slidersGroup: document.getElementById("sliders-group"),

  // Local guide controls
  localControls: document.getElementById("local-controls"),
  localOriginAirport: document.getElementById("local-origin-airport"),
  localDestinationAirport: document.getElementById("local-destination-airport"),
  localSearchFlight: document.getElementById("local-search-flight"),
};

/* Footer Suggestions per Mode */
const suggestionPresets = {
  assistant: [
    { label: "Ask a question?", value: "What is photosynthesis?" },
    { label: "Tell me about something.", value: "Tell me about Einstein." },
    { label: "Display entire chat history.", value: "show_chat_history()" },
    { label: "Clear n rounds conversation from beginning.", value: "Clear 2" },
  ],

  math: [
    { label: "Solve Equation", value: "Solve Equation: 4x + 2y = 10 and x + y = 2" },
    { label: "Word Problems", value: "A pen costs €2 and a notebook costs €3. How much for 5 pens and 4 notebook?" },
    { label: "Geometry", value: "In a triangle, two angles measure 50° and 60°. What is the third angle?" },
    { label: "Statistics", value: "Find the mean of the numbers: 6, 8, 10, 12, 14, 9" },
    { label: "Pattern Recognition", value: "Find the next number: 2, 4, 6, 8, 10, ?" },
    { label: "Probability", value: "A bag has 5 red balls and 3 blue balls. What is the probability of picking a red ball?" },
  ],

  translate: [
    { label: "Word", value: "Water" },
    { label: "Question", value: "What is photosynthesis?" },
    { label: "Sentence", value: "This is a lab from the Hardware for Artificial Intelligence Group where I am deploying an LLM." },
    { label: "Passage", value: "Europe is a continent located entirely in the Northern Hemisphere and mostly in the Eastern Hemisphere. It is bordered by the Arctic Ocean to the north, the Atlantic Ocean to the west, the Mediterranean Sea to the south and Asia to the east." },
  ],

  local: [
    { label: "Weather", value: "How’s the weather today in Darmstadt?" },
    { label: "News", value: "Tell me the latest news about Frankfurt." },
    { label: "Tourist Places", value: "Suggest me some tourist places in Munich." },
    { label: "Supermarkets", value: "I want to buy groceries in Darmstadt." },
    { label: "Hotels", value: "Where can I stay overnight in Hamburg?" },
    { label: "Flights", value: "I want to travel New York." },
  ],
};

/* Small UI Helpers */
function scrollToBottom() {
  el.chatLog.scrollTop = el.chatLog.scrollHeight;
}

function addBubble(role, text) {
  const msg = document.createElement("div");
  msg.className = `message ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;

  msg.appendChild(bubble);
  el.chatLog.appendChild(msg);
  scrollToBottom();
}

function updateSuggestionRow(mode) {
  if (!suggestionRow) return;

  const chips = Array.from(suggestionRow.querySelectorAll(".chip"));
  const preset = suggestionPresets[mode] || suggestionPresets.assistant;

  chips.forEach((chip, index) => {
    const item = preset[index];
    if (item) {
      chip.classList.remove("hidden");
      chip.textContent = item.label;
      chip.dataset.suggest = item.value;
    } else {
      chip.classList.add("hidden");
    }
  });
}

function setSliderLabels() {
  if (el.tempVal) el.tempVal.textContent = Number(el.temp.value).toFixed(2);
  if (el.topPVal) el.topPVal.textContent = Number(el.topP.value).toFixed(2);
}

function autoGrowTextarea() {
  el.input.style.height = "auto";
  el.input.style.height = Math.min(el.input.scrollHeight, 90) + "px";
}

function updateTranslateDirection() {
  if (!el.translateSource || !el.translateTarget || !el.translateDirection) return;
  el.translateDirection.textContent = `${el.translateSource.value} → ${el.translateTarget.value}`;
}

function extractAirportCode(text) {
  const value = (text || "").trim();
  const match = value.match(/^([A-Za-z]{3})\s*-/);
  return match ? match[1].toUpperCase() : "";
}

function updateCharCounter() {
  if (!el.charCounter) return;

  if (currentMode === "translate") {
    const count = (el.input.value || "").length;
    el.charCounter.classList.remove("hidden");
    el.charCounter.textContent = `${count} / ${TRANSLATE_CHAR_LIMIT}`;
    el.charCounter.classList.toggle("warning", count > TRANSLATE_CHAR_LIMIT);
  } else {
    el.charCounter.classList.add("hidden");
    el.charCounter.classList.remove("warning");
    el.charCounter.textContent = `0 / ${TRANSLATE_CHAR_LIMIT}`;
  }
}

/* Disable input/button during LLM response */
function setProcessingState(processing) {
  isProcessing = processing;

  if (el.sendBtn) {
    el.sendBtn.disabled = processing;

    if (processing) {
      el.sendBtn.innerHTML = '<div class="square-spinner"></div>';
    } else {
      el.sendBtn.textContent = "➤";
    }

    el.sendBtn.style.opacity = processing ? "0.9" : "1";
    el.sendBtn.style.cursor = processing ? "not-allowed" : "pointer";
  }

  if (el.input) {
    el.input.disabled = processing;
  }
}

/* API Helper */
async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });

  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`HTTP ${res.status}: ${txt}`);
  }

  return res.json();
}

/* Dialog Helpers */
function openDialog(title, text) {
  if (title) el.dialogTitle.textContent = title;
  if (text) el.dialogText.textContent = text;

  if (el.dialogClear) el.dialogClear.classList.remove("hidden");

  el.dialog.classList.remove("hidden");
  modalOpen = true;
}

function closeDialog() {
  el.dialog.classList.add("hidden");
  modalOpen = false;
}

/* Mode UI Switching */
function applyModeUI(mode) {
  document.body.classList.remove("math-mode", "translate-mode", "local-mode");

  if (el.translateControls) el.translateControls.classList.add("hidden");
  if (el.localControls) el.localControls.classList.add("hidden");
  if (el.confirmSettings) el.confirmSettings.classList.remove("hidden");
  if (el.modelGroup) el.modelGroup.classList.remove("hidden");
  if (el.slidersGroup) el.slidersGroup.classList.remove("hidden");
  if (el.maxTokens) el.maxTokens.classList.remove("hidden");
  if (el.seed) el.seed.classList.remove("hidden");
  if (el.deviceGroup) el.deviceGroup.classList.remove("hidden");

  if (mode === "assistant") {
    if (settingsRow) settingsRow.classList.remove("hidden");
    if (modeSubtitle) modeSubtitle.textContent = "Normal Assistant · Ask anything";
  } else if (mode === "math") {
    if (settingsRow) settingsRow.classList.remove("hidden");
    document.body.classList.add("math-mode");
    if (modeSubtitle) modeSubtitle.textContent = "Math Mode · Solve Math Problems";
  } else if (mode === "translate") {
    if (settingsRow) settingsRow.classList.remove("hidden");
    document.body.classList.add("translate-mode");
    if (modeSubtitle) modeSubtitle.textContent = "Translation Mode · Instant Language Translator";
    if (el.translateControls) el.translateControls.classList.remove("hidden");
  } else if (mode === "local") {
    if (settingsRow) settingsRow.classList.remove("hidden");
    document.body.classList.add("local-mode");
    if (modeSubtitle) modeSubtitle.textContent = "Local Guide · Your Smart Travel Assistant";
    if (el.localControls) el.localControls.classList.remove("hidden");
  }

  updateSuggestionRow(mode);
  updateCharCounter();
  updateTranslateDirection();
}

/* Chat Message Sending */
async function sendMessage(text) {
  if (isProcessing || modalOpen) return;

  const cleanText = (text || "").trim();
  if (!cleanText) return;

  setProcessingState(true);
  addBubble("user", cleanText);

  const typing = document.createElement("div");
  typing.className = "message assistant";
  typing.innerHTML = `<div class="bubble">…</div>`;
  el.chatLog.appendChild(typing);
  scrollToBottom();

  let endpoint = "/api/chat";
  let payload = { message: cleanText };

  if (currentMode === "math") {
    endpoint = "/api/math";
  } else if (currentMode === "translate") {
    endpoint = "/api/translate";
    payload = {
      message: cleanText,
      source_lang: el.translateSource?.value || "English",
      target_lang: el.translateTarget?.value || "German",
    };
  } else if (currentMode === "local") {
    endpoint = "/api/guide";
    payload = { message: cleanText };
  }

  try {
    const out = await postJSON(endpoint, payload);

    if (typing.parentNode) typing.parentNode.removeChild(typing);

    if (out.action_required) {
      openDialog("Generation blocked", out.message || "");
      return;
    }

    if (out.reply) {
      addBubble("assistant", out.reply);
    }
  } catch (err) {
    if (typing.parentNode) typing.parentNode.removeChild(typing);
    addBubble("assistant", `Error: ${err.message}`);
  } finally {
    setProcessingState(false);
  }
}

/* Flight Search */
if (el.localSearchFlight) {
  el.localSearchFlight.addEventListener("click", async () => {
    if (currentMode !== "local" || isProcessing) return;

    const originText = el.localOriginAirport?.value || "";
    const destinationText = el.localDestinationAirport?.value || "";

    const origin = extractAirportCode(originText);
    const destination = extractAirportCode(destinationText);

    if (!origin || !destination) {
      addBubble("assistant", "Please select valid Departure and Arrival Airports from the list.");
      return;
    }

    setProcessingState(true);
    addBubble("user", `Search flight: ${origin} → ${destination}`);

    const typing = document.createElement("div");
    typing.className = "message assistant";
    typing.innerHTML = `<div class="bubble">…</div>`;
    el.chatLog.appendChild(typing);
    scrollToBottom();

    try {
      const out = await postJSON("/api/flights", {
        origin,
        destination,
      });

      if (typing.parentNode) typing.parentNode.removeChild(typing);

      if (out.reply) {
        addBubble("assistant", out.reply);
      }
    } catch (err) {
      if (typing.parentNode) typing.parentNode.removeChild(typing);
      addBubble("assistant", `Error: ${err.message}`);
    } finally {
      setProcessingState(false);
    }
  });
}

/* Dialog Actions */
el.dialogSkip.addEventListener("click", async () => {
  try {
    if (currentMode === "assistant") {
      const out = await postJSON("/api/resolve", { action: "skip" });
      closeDialog();
      if (out.reply) addBubble("assistant", out.reply);
    } else if (currentMode === "math") {
      closeDialog();
      addBubble(
        "assistant",
        "Chat history is too large. This response was skipped. Please clear or truncate your history."
      );
    } else if (currentMode === "translate") {
      closeDialog();
      addBubble(
        "assistant",
        "The translation input is too large for the model context. Please shorten your text and try again."
      );
    } else if (currentMode === "local") {
      closeDialog();
      addBubble(
        "assistant",
        "The Local Guide message and history are too large for the model context. Please shorten your text or clear the history."
      );
    }
  } catch (err) {
    closeDialog();
    addBubble("assistant", `Error: ${err.message}`);
  }
});

el.dialogClear.addEventListener("click", async () => {
  try {
    if (currentMode === "assistant") {
      await postJSON("/api/clear", {});
      closeDialog();
      el.chatLog.innerHTML = "";
      addBubble("assistant", "History cleared. Please ask your question again.");
    } else if (currentMode === "math") {
      await postJSON("/api/clear_mode", { mode: "math" });
      closeDialog();
      el.chatLog.innerHTML = "";
      addBubble("assistant", "Math history cleared. Please ask your question again.");
    } else if (currentMode === "translate") {
      await postJSON("/api/clear_mode", { mode: "translate" });
      closeDialog();
      el.chatLog.innerHTML = "";
      addBubble("assistant", "Translation history cleared. Please enter text again.");
    } else if (currentMode === "local") {
      await postJSON("/api/clear_mode", { mode: "local" });
      closeDialog();
      el.chatLog.innerHTML = "";
      addBubble("assistant", "Local Guide history cleared. Please enter text again.");
    }
  } catch (err) {
    closeDialog();
    addBubble("assistant", `Error: ${err.message}`);
  }
});

/* Main Buttons */
el.clearHistory.addEventListener("click", async () => {
  if (currentMode === "assistant") {
    await postJSON("/api/clear", {});
  } else if (currentMode === "math") {
    await postJSON("/api/clear_mode", { mode: "math" });
  } else if (currentMode === "translate") {
    await postJSON("/api/clear_mode", { mode: "translate" });
  } else if (currentMode === "local") {
    await postJSON("/api/clear_mode", { mode: "local" });
  }

  el.chatLog.innerHTML = "";
});

el.confirmSettings.addEventListener("click", async () => {
  if (currentMode !== "assistant") return;

  const modelKey = document.querySelector('input[name="assistant-model"]:checked')?.value || "llama";
  const devicePref = document.querySelector('input[name="assistant-device"]:checked')?.value || "auto";

  const payload = {
    model_key: modelKey,
    device: devicePref,
    max_length: Number(el.maxTokens.value),
    temperature: Number(el.temp.value),
    top_p: Number(el.topP.value),
    seed: Number(el.seed.value),
    mode: "assistant",
  };

  const out = await postJSON("/api/config", payload);
  cfg = out.cfg;

  if (out.model_changed || out.mode_changed) {
    el.chatLog.innerHTML = "";
    addBubble("assistant", "New session started.");
  } else {
    addBubble("assistant", "Settings applied.");
  }
});

el.sendBtn.addEventListener("click", async () => {
  if (modalOpen || isProcessing) return;

  const text = (el.input.value || "").trim();
  if (!text) return;

  el.input.value = "";
  autoGrowTextarea();
  updateCharCounter();
  await sendMessage(text);
});

el.input.addEventListener("keydown", async (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();

    if (modalOpen || isProcessing) return;

    const text = (el.input.value || "").trim();
    if (!text) return;

    el.input.value = "";
    autoGrowTextarea();
    updateCharCounter();
    await sendMessage(text);
  }
});

/* Misc UI Events */
document.querySelectorAll(".chip").forEach((btn) => {
  btn.addEventListener("click", () => {
    const t = btn.getAttribute("data-suggest") || btn.textContent || "";
    el.input.value = t;
    autoGrowTextarea();
    el.input.focus();
  });
});

if (el.translateSwap) {
  el.translateSwap.addEventListener("click", () => {
    const oldSource = el.translateSource.value;
    el.translateSource.value = el.translateTarget.value;
    el.translateTarget.value = oldSource;
    updateTranslateDirection();
  });
}

if (el.translateSource) {
  el.translateSource.addEventListener("change", updateTranslateDirection);
}

if (el.translateTarget) {
  el.translateTarget.addEventListener("change", updateTranslateDirection);
}

el.input.addEventListener("input", () => {
  autoGrowTextarea();
  updateCharCounter();
});

modeTabs.forEach((tab) => {
  tab.addEventListener("click", async () => {
    const mode = tab.dataset.mode;
    if (!mode || tab.disabled) return;

    currentMode = mode;
    modeTabs.forEach((t) => t.classList.toggle("active", t === tab));
    applyModeUI(mode);

    try {
      await postJSON("/api/clear_mode", { mode });
    } catch (err) {
      console.error("Failed to clear mode history:", err);
    }

    el.chatLog.innerHTML = "";
  });
});

/* System Usage Info */
async function refreshSystemUsage() {
  try {
    const out = await postJSON("/api/system", {});
    const cpu = Number(out.cpu_percent || 0);
    const gpu = out.gpu_percent == null ? null : Number(out.gpu_percent);

    if (el.cpuText) el.cpuText.textContent = `${cpu.toFixed(0)}%`;
    if (el.cpuBar) el.cpuBar.style.width = `${Math.max(0, Math.min(100, cpu))}%`;

    if (gpu == null) {
      if (el.gpuText) el.gpuText.textContent = "--%";
      if (el.gpuBar) el.gpuBar.style.width = "0%";
    } else {
      if (el.gpuText) el.gpuText.textContent = `${gpu.toFixed(0)}%`;
      if (el.gpuBar) el.gpuBar.style.width = `${Math.max(0, Math.min(100, gpu))}%`;
    }
  } catch (err) {
    console.error("System usage update failed:", err);
  }
}

setSliderLabels();
autoGrowTextarea();
updateCharCounter();
applyModeUI(currentMode);
refreshSystemUsage();
setInterval(refreshSystemUsage, 3000);