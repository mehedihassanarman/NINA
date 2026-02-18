from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import torch
from flask import Flask, jsonify, render_template, request, session

# IMPORTANT: file name must match your module
from normal_mode import normal_mode_load_model, normal_chat_turn

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key-change-me")

# -------------------------
# Models available in UI
# -------------------------
MODEL_PATHS = {
    "llama": "models/Llama-3.2-1B-Instruct",
    "qwen": "models/Qwen1.5-0.5B-Chat",
}

# -------------------------
# In-memory model cache
# -------------------------
MODEL_CACHE: Dict[str, Dict[str, Any]] = {}  # key -> {"model":..., "tokenizer":..., "device":...}


@dataclass
class ChatConfig:
    model_key: str = "llama"
    device: str = "auto"      # ✅ NEW: auto | cpu | gpu
    max_length: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    seed: int = 42



def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_config() -> ChatConfig:
    cfg = session.get("cfg")
    if not cfg:
        cfg = asdict(ChatConfig())
        session["cfg"] = cfg
    return ChatConfig(**cfg)


def set_config(cfg: ChatConfig) -> None:
    session["cfg"] = asdict(cfg)


def get_history() -> list[str]:
    return session.get("history", [])


def set_history(history: list[str]) -> None:
    session["history"] = history


def clear_history() -> None:
    session["history"] = []


def resolve_device(device_pref: str) -> str:
    pref = (device_pref or "auto").lower()
    if pref == "cpu":
        return "cpu"
    if pref in {"gpu", "cuda"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    # auto
    return "cuda" if torch.cuda.is_available() else "cpu"


def cache_key(model_key: str, device: str) -> str:
    return f"{model_key}::{device}"


def load_model_for_key(model_key: str, device_pref: str) -> None:
    key_device = resolve_device(device_pref)
    key = cache_key(model_key, key_device)

    if key in MODEL_CACHE:
        return

    model_path = MODEL_PATHS[model_key]
    model, tokenizer = normal_mode_load_model(device=key_device, model_path=model_path)
    MODEL_CACHE[key] = {"model": model, "tokenizer": tokenizer, "device": key_device}


def get_loaded_model(model_key: str, device_pref: str):
    load_model_for_key(model_key, device_pref)
    key_device = resolve_device(device_pref)
    cached = MODEL_CACHE[cache_key(model_key, key_device)]
    return cached["model"], cached["tokenizer"], cached["device"]



# -------------------------
# Routes
# -------------------------

import subprocess
import os

@app.post("/api/system")
def api_system():
    # CPU percent
    cpu_percent = None
    try:
        import psutil  # type: ignore
        cpu_percent = psutil.cpu_percent(interval=0.1)
    except Exception:
        # fallback: rough load estimate
        try:
            load1 = os.getloadavg()[0]
            cpu_percent = min(100.0, (load1 / (os.cpu_count() or 1)) * 100.0)
        except Exception:
            cpu_percent = 0.0

    # GPU utilization percent (NVIDIA)
    gpu_percent = None
    try:
        if torch.cuda.is_available():
            result = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=1.0,
            ).strip()
            # if multiple GPUs, take first line
            first = result.splitlines()[0].strip()
            gpu_percent = float(first)
    except Exception:
        gpu_percent = None

    return jsonify(ok=True, cpu_percent=cpu_percent, gpu_percent=gpu_percent)




@app.get("/")
def home():
    cfg = get_config()
    # ensure model is loaded so UI can show "Online"
    try:
        get_loaded_model(cfg.model_key, cfg.device)
        online = True
    except Exception:
        online = False

    return render_template(
        "home.html",
        cfg=asdict(cfg),
        online=online,
        mode="Assistant",
    )


@app.post("/api/config")
def api_config():
    """
    Called when user clicks "Confirm settings".
    If model changed => clear history and start new session (as requested).
    """
    data = request.get_json(force=True) or {}
    old_cfg = get_config()

    new_cfg = ChatConfig(
        model_key=data.get("model_key", old_cfg.model_key),
        device=data.get("device", old_cfg.device),   # ✅ NEW
        max_length=int(data.get("max_length", old_cfg.max_length)),
        temperature=float(data.get("temperature", old_cfg.temperature)),
        top_p=float(data.get("top_p", old_cfg.top_p)),
        seed=int(data.get("seed", old_cfg.seed)),
    )

    model_changed = (new_cfg.model_key != old_cfg.model_key)
    device_changed = (new_cfg.device != old_cfg.device)

    set_config(new_cfg)

    # load model with selected device
    get_loaded_model(new_cfg.model_key, new_cfg.device)

    if model_changed or device_changed:
        clear_history()

    return jsonify(
        ok=True,
        model_changed=model_changed,
        device_changed=device_changed,  # optional
        cfg=asdict(new_cfg),
        history=get_history(),
    )



@app.post("/api/clear")
def api_clear():
    clear_history()
    # clear any pending
    session.pop("pending_user_input", None)
    session.pop("pending_reason", None)
    session.pop("pending_message", None)
    return jsonify(ok=True, history=[])


@app.post("/api/chat")
def api_chat():
    """
    Main chat endpoint.
    If backend detects context too long / prompt too long / OOM:
      - return action_required=True and show modal on UI
      - store pending_user_input in session
    """
    cfg = get_config()
    data = request.get_json(force=True) or {}
    user_input = (data.get("message") or "").strip()

    if not user_input:
        return jsonify(ok=True, reply="", history=get_history())

    model, tokenizer, device = get_loaded_model(cfg.model_key, cfg.device)
    history = get_history()

    out = normal_chat_turn(
        model=model,
        tokenizer=tokenizer,
        device=device,
        history=history,
        user_input=user_input,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_length=cfg.max_length,
        seed=cfg.seed,
    )

    # If generation skipped for safety (prompt/context/oom), require user choice
    if out.get("skipped") and out.get("reason") in {"prompt_too_long", "context_too_long", "oom"}:
        session["pending_user_input"] = user_input
        session["pending_reason"] = out.get("reason")
        session["pending_message"] = out.get("reply")

        return jsonify(
            ok=True,
            action_required=True,
            action_type=out.get("reason"),
            message=out.get("reply"),
            history=get_history(),  # unchanged
        )

    # Normal path: update history stored
    set_history(out.get("history", history))
    return jsonify(
        ok=True,
        action_required=False,
        reply=out.get("reply", ""),
        history=get_history(),
    )


@app.post("/api/resolve")
def api_resolve():
    """
    Resolve the modal choice for context-too-long / OOM:
      - action=clear  => clear history and re-run pending input
      - action=skip   => show warning as assistant message (but do NOT add to history)
    """
    data = request.get_json(force=True) or {}
    action = (data.get("action") or "").strip().lower()

    pending_user_input: Optional[str] = session.get("pending_user_input")
    pending_message: str = session.get("pending_message", "")

    # no pending => nothing to do
    if not pending_user_input:
        return jsonify(ok=True, action_required=False, reply="", history=get_history())

    cfg = get_config()

    if action == "skip":
        # discard pending, return the warning message as a one-off assistant bubble
        session.pop("pending_user_input", None)
        session.pop("pending_reason", None)
        session.pop("pending_message", None)

        return jsonify(
            ok=True,
            action_required=False,
            reply=pending_message,
            history=get_history(),  # unchanged
            skipped=True,
        )

    if action == "clear":
        clear_history()

        # rerun with empty history
        model, tokenizer, device = get_loaded_model(cfg.model_key, cfg.device)

        out = normal_chat_turn(
            model=model,
            tokenizer=tokenizer,
            device=device,
            history=[],
            user_input=pending_user_input,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_length=cfg.max_length,
            seed=cfg.seed,
        )

        session.pop("pending_user_input", None)
        session.pop("pending_reason", None)
        session.pop("pending_message", None)

        # If STILL skipped (rare), just return message
        if out.get("skipped"):
            return jsonify(
                ok=True,
                action_required=False,
                reply=out.get("reply", ""),
                history=get_history(),
                skipped=True,
            )

        set_history(out.get("history", []))
        return jsonify(ok=True, action_required=False, reply=out.get("reply", ""), history=get_history())

    return jsonify(ok=False, error="Invalid action. Use 'clear' or 'skip'."), 400


if __name__ == "__main__":
    app.run(debug=True)
