from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional
import uuid

import torch
from flask import Flask, jsonify, render_template, request, session

# IMPORTANT: file name must match your module
from normal_mode import normal_mode_load_model, normal_chat_turn
from multi_modes_mini import get_device as mm_get_device, load_llama_model as mm_load_llama_model, math_mode, translator_mode, SUPPORTED_LANGUAGES

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

# -------------------------
# Per-session chat state (server-side)
# -------------------------
# We keep only a tiny "sid" in the browser cookie.
# All heavy data (cfg, history, pending_*) lives here in memory.
CHAT_STATE: Dict[str, Dict[str, Any]] = {}


@dataclass
class ChatConfig:
    model_key: str = "llama"
    device: str = "cpu"      # auto | cpu | gpu
    max_length: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    seed: int = 42


def get_sid() -> str:
    """Get or create a small per-user session id stored in the cookie."""
    sid = session.get("sid")
    if not sid:
        sid = uuid.uuid4().hex
        session["sid"] = sid
    if sid not in CHAT_STATE:
        CHAT_STATE[sid] = {
            "cfg": ChatConfig(),
            "history": [],          # assistant history
            "math_history": [],      # math mode history
            "translate_history": [], # translator mode history
            "pending_user_input": None,
            "pending_reason": None,
            "pending_message": None,
        }
    return sid


def get_state() -> Dict[str, Any]:
    sid = get_sid()
    return CHAT_STATE[sid]


def get_config() -> ChatConfig:
    state = get_state()
    cfg = state.get("cfg")
    if not isinstance(cfg, ChatConfig):
        cfg = ChatConfig(**cfg)  # fallback if something old sneaks in
        state["cfg"] = cfg
    return cfg


def set_config(cfg: ChatConfig) -> None:
    state = get_state()
    state["cfg"] = cfg


def get_history() -> list[str]:
    state = get_state()
    return state.get("history", [])


def set_history(history: list[str]) -> None:
    state = get_state()
    state["history"] = history


def get_math_history() -> list[str]:
    state = get_state()
    return state.get("math_history", [])


def set_math_history(history: list[str]) -> None:
    state = get_state()
    state["math_history"] = history

def clear_math_history() -> None:
    state = get_state()
    state["math_history"] = []

def clear_history() -> None:
    state = get_state()
    state["history"] = []


def get_pending() -> tuple[Optional[str], Optional[str], str]:
    state = get_state()
    return (
        state.get("pending_user_input"),
        state.get("pending_reason"),
        state.get("pending_message", ""),
    )


def set_pending(user_input: Optional[str], reason: Optional[str], message: str) -> None:
    state = get_state()
    state["pending_user_input"] = user_input
    state["pending_reason"] = reason
    state["pending_message"] = message


def clear_pending() -> None:
    state = get_state()
    state["pending_user_input"] = None
    state["pending_reason"] = None
    state["pending_message"] = ""


#def get_translate_history() -> list[str]:
#    state = get_state()
#    return state.get("translate_history", [])


#def set_translate_history(history: list[str]) -> None:
#    state = get_state()
#    state["translate_history"] = history


def clear_translate_history() -> None:
    state = get_state()
    state["translate_history"] = []




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


# Shared LLaMA for math
MATH_MODEL = None
MATH_TOKENIZER = None
MATH_DEVICE = None

# Separate LLaMA for translator
TRANSLATE_MODEL = None
TRANSLATE_TOKENIZER = None
TRANSLATE_DEVICE = None


def get_math_model():
    global MATH_MODEL, MATH_TOKENIZER, MATH_DEVICE
    if MATH_MODEL is None or MATH_TOKENIZER is None or MATH_DEVICE is None:
        MATH_DEVICE = mm_get_device()
        MATH_MODEL, MATH_TOKENIZER = mm_load_llama_model(MATH_DEVICE)
    return MATH_MODEL, MATH_TOKENIZER, MATH_DEVICE


def get_translate_model():
    global TRANSLATE_MODEL, TRANSLATE_TOKENIZER, TRANSLATE_DEVICE
    if TRANSLATE_MODEL is None or TRANSLATE_TOKENIZER is None or TRANSLATE_DEVICE is None:
        TRANSLATE_DEVICE = mm_get_device()
        TRANSLATE_MODEL, TRANSLATE_TOKENIZER = mm_load_llama_model(TRANSLATE_DEVICE)
    return TRANSLATE_MODEL, TRANSLATE_TOKENIZER, TRANSLATE_DEVICE

# -------------------------
# Routes
# -------------------------


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
        supported_languages=SUPPORTED_LANGUAGES,
    )


@app.post("/api/config")
def api_config():
    """
    Called when user clicks "Confirm settings".
    If model or device changed => clear history and start new session (by design).
    """
    data = request.get_json(force=True) or {}
    old_cfg = get_config()

    new_cfg = ChatConfig(
        model_key=data.get("model_key", old_cfg.model_key),
        device=data.get("device", old_cfg.device),
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
        device_changed=device_changed,
        cfg=asdict(new_cfg),
        history=get_history(),
    )


@app.post("/api/clear")
def api_clear():
    clear_history()
    clear_pending()
    return jsonify(ok=True, history=[])


@app.post("/api/clear_mode")
def api_clear_mode():
    """
    Clear conversation history for a specific mode when switching tabs.
    Does NOT touch settings (cfg).
    """
    data = request.get_json(force=True) or {}
    mode = (data.get("mode") or "").strip().lower()

    if mode == "assistant":
        clear_history()
        clear_pending()
        return jsonify(ok=True, mode="assistant")

    if mode == "math":
        clear_math_history()
        return jsonify(ok=True, mode="math")

    if mode == "translate":
        clear_translate_history()
        return jsonify(ok=True, mode="translate")

    return jsonify(ok=False, error=f"Unsupported mode: {mode}"), 400


@app.post("/api/chat")
def api_chat():
    """
    Main chat endpoint.
    If backend detects context too long / prompt too long / OOM:
      - return action_required=True and show modal on UI
      - store pending_user_input/pending_message in server-side state
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
        set_pending(
            user_input=user_input,
            reason=out.get("reason"),
            message=out.get("reply", ""),
        )

        return jsonify(
            ok=True,
            action_required=True,
            action_type=out.get("reason"),
            message=out.get("reply"),
            history=get_history(),  # unchanged
        )

    # Normal path: update history stored
    set_history(out.get("history", history))
    clear_pending()  # clean up any old pending state
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

    pending_user_input, pending_reason, pending_message = get_pending()

    # no pending => nothing to do
    if not pending_user_input:
        return jsonify(ok=True, action_required=False, reply="", history=get_history())

    cfg = get_config()

    if action == "skip":
        # discard pending, return the warning message as a one-off assistant bubble
        clear_pending()
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

        clear_pending()

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


# Math mode

@app.post("/api/math")
def api_math():
    data = request.get_json(force=True) or {}
    user_input = (data.get("message") or "").strip()
    history = get_math_history()

    if not user_input:
        return jsonify(ok=True, reply="", history=history)

    model, tokenizer, device = get_math_model()

    out = math_mode(
        model=model,
        tokenizer=tokenizer,
        device=device,
        history=history,
        user_input=user_input,
    )

    # 👉 If no tokens left / context too long:
    if out.get("skipped") and out.get("reason") in {
        "prompt_too_long",
        "no_space_for_generation",
    }:
        # ⬇️ history returned from math_mode is already trimmed, so SAVE it
        trimmed = out.get("history", history)
        set_math_history(trimmed)

        return jsonify(
            ok=True,
            action_required=True,
            action_type=out.get("reason"),
            message=out.get("reply", ""),
            history=trimmed,  # send trimmed history to frontend
        )

    # Normal path: save updated history
    new_history = out.get("history", history)
    set_math_history(new_history)

    return jsonify(
        ok=True,
        action_required=False,
        reply=out.get("reply", ""),
        history=new_history,
        skipped=out.get("skipped", False),
        reason=out.get("reason"),
    )


# Translator mode

@app.post("/api/translate")
def api_translate():
    data = request.get_json(force=True) or {}
    user_input = (data.get("message") or "").strip()
    source_lang = (data.get("source_lang") or "English").strip()
    target_lang = (data.get("target_lang") or "German").strip()

    if not user_input:
        #return jsonify(ok=True, reply="", history=get_translate_history())
        return jsonify(ok=True, reply="", history=[])

    #history = get_translate_history()
    model, tokenizer, device = get_translate_model()

    out = translator_mode(
        model=model,
        tokenizer=tokenizer,
        device=device,
        history=None,
        user_input=user_input,
        source_lang=source_lang,
        target_lang=target_lang,
        temperature=None,
        top_p=None,
    )

    #set_translate_history(out.get("history", history))

    reason = out.get("reason")
    if reason in {"no_space_for_generation", "prompt_too_long"}:
        return jsonify(
            ok=True,
            action_required=True,
            message=out.get("reply", "Translation could not be generated."),
            reason=reason,
            #history=out.get("history", history),
            history=[],
        )

    return jsonify(
        ok=True,
        reply=out.get("reply", ""),
        #history=out.get("history", history),
        history=[],
        skipped=out.get("skipped", False),
        reason=reason,
    )






if __name__ == "__main__":
    app.run(debug=True)