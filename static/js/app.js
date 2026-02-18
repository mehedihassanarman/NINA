const boot = window.__BOOT__ || {};
let cfg = boot.cfg || { model_key: "llama", max_length: 256, temperature: 0.7, top_p: 0.9, seed: 42 };

const el = {
  chatLog: document.getElementById("chat-log"),
  input: document.getElementById("user-input"),
  sendBtn: document.getElementById("send-btn"),

  temp: document.getElementById("assistant-temperature"),
  tempVal: document.getElementById("temperature-value"),
  topP: document.getElementById("assistant-top-p"),
  topPVal: document.getElementById("top-p-value"),
  maxTokens: document.getElementById("assistant-max-tokens"),
  seed: document.getElementById("assistant-seed"),

  clearHistory: document.getElementById("assistant-clear-history"),
  confirmSettings: document.getElementById("assistant-confirm-settings"),

  dialog: document.getElementById("assistant-dialog"),
  dialogTitle: document.getElementById("assistant-dialog-title"),
  dialogText: document.getElementById("assistant-dialog-text"),
  dialogClear: document.getElementById("dialog-clear-history"),
  dialogSkip: document.getElementById("dialog-skip-reply"),

  deviceRadios: document.querySelectorAll('input[name="assistant-device"]'),
  cpuBar: document.getElementById("cpu-bar"),
  cpuText: document.getElementById("cpu-text"),
  gpuBar: document.getElementById("gpu-bar"),
  gpuText: document.getElementById("gpu-text"),

};

let modalOpen = false;

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

function setSliderLabels() {
  el.tempVal.textContent = Number(el.temp.value).toFixed(2);
  el.topPVal.textContent = Number(el.topP.value).toFixed(2);
}
setSliderLabels();
el.temp.addEventListener("input", setSliderLabels);
el.topP.addEventListener("input", setSliderLabels);

function autoGrowTextarea() {
  el.input.style.height = "auto";
  el.input.style.height = Math.min(el.input.scrollHeight, 90) + "px";
}
el.input.addEventListener("input", autoGrowTextarea);
autoGrowTextarea();

document.querySelectorAll(".chip").forEach(btn => {
  btn.addEventListener("click", () => {
    const t = btn.getAttribute("data-suggest") || btn.textContent || "";
    el.input.value = t;
    autoGrowTextarea();
    el.input.focus();
  });
});

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

/* Dialog helpers */
function openDialog(title, text) {
  if (title) el.dialogTitle.textContent = title;
  if (text) el.dialogText.textContent = text;
  el.dialog.classList.remove("hidden");
  modalOpen = true;
}
function closeDialog() {
  el.dialog.classList.add("hidden");
  modalOpen = false;
}

el.dialogSkip.addEventListener("click", async () => {
  const out = await postJSON("/api/resolve", { action: "skip" });
  closeDialog();
  if (out.reply) addBubble("assistant", out.reply);
});

el.dialogClear.addEventListener("click", async () => {
  const out = await postJSON("/api/resolve", { action: "clear" });
  closeDialog();
  if (out.reply) addBubble("assistant", out.reply);
});

/* Clear history */
el.clearHistory.addEventListener("click", async () => {
  await postJSON("/api/clear", {});
  el.chatLog.innerHTML = "";
});

/* Confirm settings (clears history if model changed, backend handles that) */
el.confirmSettings.addEventListener("click", async () => {
  const modelKey = document.querySelector('input[name="assistant-model"]:checked')?.value || "llama";

  const devicePref =
    document.querySelector('input[name="assistant-device"]:checked')?.value || "auto";

  const payload = {
    model_key: modelKey,
    device: devicePref,     // ✅ NEW
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

/* Send message */
async function sendMessage(text) {
  addBubble("user", text);

  // tiny typing bubble
  const typing = document.createElement("div");
  typing.className = "message assistant";
  typing.innerHTML = `<div class="bubble">…</div>`;
  el.chatLog.appendChild(typing);
  scrollToBottom();

  try {
    const out = await postJSON("/api/chat", { message: text });

    if (typing && typing.parentNode) typing.parentNode.removeChild(typing);

    if (out.action_required) {
      openDialog("Generation blocked", out.message || "Choose an action.");
      return;
    }

    if (out.reply) addBubble("assistant", out.reply);
  } catch (err) {
    if (typing && typing.parentNode) typing.parentNode.removeChild(typing);
    addBubble("assistant", `Error: ${err.message}`);
  }
}

el.sendBtn.addEventListener("click", async () => {
  if (modalOpen) return;
  const text = (el.input.value || "").trim();
  if (!text) return;
  el.input.value = "";
  autoGrowTextarea();
  await sendMessage(text);
});

el.input.addEventListener("keydown", async (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    if (modalOpen) return;
    const text = (el.input.value || "").trim();
    if (!text) return;
    el.input.value = "";
    autoGrowTextarea();
    await sendMessage(text);
  }
});


function clampPct(x){
  x = Number(x);
  if (Number.isNaN(x)) return 0;
  return Math.max(0, Math.min(100, x));
}

async function refreshSystemUsage(){
  try{
    const out = await postJSON("/api/system", {});
    const cpu = clampPct(out.cpu_percent);
    const gpu = out.gpu_percent == null ? null : clampPct(out.gpu_percent);

    if (el.cpuBar) el.cpuBar.style.width = cpu + "%";
    if (el.cpuText) el.cpuText.textContent = cpu.toFixed(1) + "%";

    if (gpu === null){
      if (el.gpuBar) el.gpuBar.style.width = "0%";
      if (el.gpuText) el.gpuText.textContent = "N/A";
    } else {
      if (el.gpuBar) el.gpuBar.style.width = gpu + "%";
      if (el.gpuText) el.gpuText.textContent = gpu.toFixed(1) + "%";
    }
  } catch (e){
    // silent failure
  }
}

// initial + interval
refreshSystemUsage();
setInterval(refreshSystemUsage, 2000);
