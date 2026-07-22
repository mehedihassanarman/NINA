from __future__ import annotations
import os
import subprocess
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional
import uuid
import torch
from flask import Flask, jsonify, render_template, request, session

# Loading modules
from normal_mode import normal_mode_load_model, normal_chat_turn
from multi_modes import get_device as mm_get_device, load_llama_model as mm_load_llama_model, math_mode, translator_mode, programming_mode, SUPPORTED_LANGUAGES, guide_mode, load_airports,AIRPORTS,fetch_flights
from data_analysis import build_dataset_context, dataframe_metadata, data_analysis_chat, load_dataset, make_upload_reply

# Flask app setup
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key-change-me")

# Load airport data once at startup
load_airports()

# Models Paths
MODEL_PATHS = {
    "llama": "models/Llama-3.2-1B-Instruct",
    "qwen": "models/Qwen1.5-0.5B-Chat",
}
 
# In-memory model cache
MODEL_CACHE: Dict[str, Dict[str, Any]] = {}  
CHAT_STATE: Dict[str, Dict[str, Any]] = {}

# Chat configuration
@dataclass
class ChatConfig:
    model_key: str = "llama"
    device: str = "cpu"  
    max_length: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    seed: int = 42


# Get or create a small per-user session id stored in the cookie.
def get_sid() -> str:
    sid = session.get("sid")

    if not sid:
        sid = uuid.uuid4().hex
        session["sid"] = sid

    if sid not in CHAT_STATE:
        CHAT_STATE[sid] = {
            "cfg": ChatConfig(),
            "history": [],              # assistant mode history
            "math_history": [],         # math mode history
            "translate_history": [],    # translator mode history
            "guide_history": [],        # local guide mode history
            "programming_history": [],  # programming mode history
            "data_history": [],
            "data_context": None,
            "data_filename": None,
            "guide_current_city": None,
            "pending_user_input": None,
            "pending_reason": None,
            "pending_message": None,
        }
    return sid

def get_state() -> Dict[str, Any]:
    sid = get_sid()
    return CHAT_STATE[sid]


# Fetch ChatConfig from session state.
def get_config() -> ChatConfig:
    state = get_state()
    cfg = state.get("cfg")

    if not isinstance(cfg, ChatConfig):
        cfg = ChatConfig(**cfg)
        state["cfg"] = cfg
    return cfg


# Assistant history helpers
def set_config(cfg: ChatConfig) -> None:
    state = get_state()
    state["cfg"] = cfg

def get_history() -> list[str]:
    state = get_state()
    return state.get("history", [])

def set_history(history: list[str]) -> None:
    state = get_state()
    state["history"] = history

def clear_history() -> None:
    state = get_state()
    state["history"] = []


# Math history helpers
def get_math_history() -> list[str]:
    state = get_state()
    return state.get("math_history", [])

def set_math_history(history: list[str]) -> None:
    state = get_state()
    state["math_history"] = history

def clear_math_history() -> None:
    state = get_state()
    state["math_history"] = []


# Translator history helpers
def clear_translate_history() -> None:
    state = get_state()
    state["translate_history"] = []


# Local Guide helpers
def get_guide_history() -> list[str]:
    state = get_state()
    return state.get("guide_history", [])

def set_guide_history(history: list[str]) -> None:
    state = get_state()
    state["guide_history"] = history

def clear_guide_history() -> None:
    state = get_state()
    state["guide_history"] = []

def get_guide_current_city() -> Optional[str]:
    state = get_state()
    return state.get("guide_current_city")

def set_guide_current_city(city: Optional[str]) -> None:
    state = get_state()
    state["guide_current_city"] = city


# Programming Mode history helpers
def get_programming_history():
    state = get_state()
    return state.get("programming_history", [])


def set_programming_history(history):
    state = get_state()
    state["programming_history"] = history


def clear_programming_history():
    state = get_state()
    state["programming_history"] = []


# Data Analysis history and dataset helpers
def get_data_history() -> list[str]:
    state = get_state()
    return state.get("data_history", [])


def set_data_history(history: list[str]) -> None:
    state = get_state()
    state["data_history"] = history


def get_data_context() -> Optional[str]:
    state = get_state()
    return state.get("data_context")


def set_data_context(context: Optional[str]) -> None:
    state = get_state()
    state["data_context"] = context


def get_data_filename() -> Optional[str]:
    state = get_state()
    return state.get("data_filename")


def set_data_filename(filename: Optional[str]) -> None:
    state = get_state()
    state["data_filename"] = filename


def clear_data_analysis() -> None:
    state = get_state()
    state["data_history"] = []
    state["data_context"] = None
    state["data_filename"] = None


# Pending assistant modal helpers. Used when generation is blocked due to prompt or context limits.
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


# Convert UI preference into an actual runtime device.
def resolve_device(device_pref: str) -> str:
    pref = (device_pref or "auto").lower()

    if pref == "cpu":
        return "cpu"
    
    if pref in {"gpu", "cuda"}:
        return "cuda" if torch.cuda.is_available() else "cpu"

    return "cuda" if torch.cuda.is_available() else "cpu"

def cache_key(model_key: str, device: str) -> str:
    return f"{model_key}::{device}"


# Load Assistant model into shared cache if not already loaded.
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


# Load a separate model for Math mode
MATH_MODEL = None
MATH_TOKENIZER = None
MATH_DEVICE = None

def get_math_model():
    global MATH_MODEL, MATH_TOKENIZER, MATH_DEVICE
    if MATH_MODEL is None or MATH_TOKENIZER is None or MATH_DEVICE is None:
        MATH_DEVICE = mm_get_device()
        MATH_MODEL, MATH_TOKENIZER = mm_load_llama_model(MATH_DEVICE)
    return MATH_MODEL, MATH_TOKENIZER, MATH_DEVICE


# Load a separate model for Translator mode
TRANSLATE_MODEL = None
TRANSLATE_TOKENIZER = None
TRANSLATE_DEVICE = None

def get_translate_model():
    global TRANSLATE_MODEL, TRANSLATE_TOKENIZER, TRANSLATE_DEVICE
    if TRANSLATE_MODEL is None or TRANSLATE_TOKENIZER is None or TRANSLATE_DEVICE is None:
        TRANSLATE_DEVICE = mm_get_device()
        TRANSLATE_MODEL, TRANSLATE_TOKENIZER = mm_load_llama_model(TRANSLATE_DEVICE)
    return TRANSLATE_MODEL, TRANSLATE_TOKENIZER, TRANSLATE_DEVICE


# Load a separate model for Local Guide mode
GUIDE_MODEL = None
GUIDE_TOKENIZER = None
GUIDE_DEVICE = None

def get_guide_model():
    global GUIDE_MODEL, GUIDE_TOKENIZER, GUIDE_DEVICE
    if GUIDE_MODEL is None or GUIDE_TOKENIZER is None or GUIDE_DEVICE is None:
        GUIDE_DEVICE = mm_get_device()
        GUIDE_MODEL, GUIDE_TOKENIZER = mm_load_llama_model(GUIDE_DEVICE)
    return GUIDE_MODEL, GUIDE_TOKENIZER, GUIDE_DEVICE


def get_programming_model():
    return get_math_model()


def get_data_model():
    return get_math_model()

# Routes : System Usage
@app.post("/api/system")
def api_system():
    # CPU Usage
    cpu_percent = None
    try:
        import psutil  
        cpu_percent = psutil.cpu_percent(interval=0.1)
    except Exception:
        try:
            load1 = os.getloadavg()[0]
            cpu_percent = min(100.0, (load1 / (os.cpu_count() or 1)) * 100.0)
        except Exception:
            cpu_percent = 0.0

    # GPU Usage (NVIDIA)
    gpu_percent = None
    try:
        if torch.cuda.is_available():
            result = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=1.0,
            ).strip()
            # In case of multiple GPUs, take first line
            first = result.splitlines()[0].strip()
            gpu_percent = float(first)
    except Exception:
        gpu_percent = None

    return jsonify(ok=True, cpu_percent=cpu_percent, gpu_percent=gpu_percent)


# Routes: Home Page
@app.get("/")
def home():
    cfg = get_config()
    try:
        get_loaded_model(cfg.model_key, cfg.device)
        online = True
    except Exception:
        online = False

    sorted_airports = sorted(
        AIRPORTS,
        key=lambda a: (
            (a.get("iata") or "").upper(),
            (a.get("name") or "").lower(),
        ),
    )

    return render_template(
        "home.html",
        cfg=asdict(cfg),
        online=online,
        mode="Assistant",
        supported_languages=SUPPORTED_LANGUAGES,
        airports=sorted_airports,
    )


# Routes: Assistant configuration
@app.post("/api/config")
def api_config():
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

# Routes: Clear Assistant mode history and pending modal state.
@app.post("/api/clear")
def api_clear():
    clear_history()
    clear_pending()
    return jsonify(ok=True, history=[])


# Routes: Clear conversation history for a specific mode when switching tabs.
@app.post("/api/clear_mode")
def api_clear_mode():
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
    
    if mode == "local":
        clear_guide_history()
        set_guide_current_city(None)
        return jsonify(ok=True, mode="local")
    
    if mode == "programming":
        clear_programming_history()
        return jsonify(ok=True, mode="programming")
    
    if mode == "data":
        clear_data_analysis()
        return jsonify(ok=True, mode="data")

    return jsonify(ok=False, error=f"Unsupported mode: {mode}"), 400


# Routes: Normal Assistant Mode
@app.post("/api/chat")
def api_chat():
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

    # If generation skipped for safety (prompt/context/OOM), asks for user's choice
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
            history=get_history(), 
        )
  
    set_history(out.get("history", history))
    clear_pending()  

    return jsonify(
        ok=True,
        action_required=False,
        reply=out.get("reply", ""),
        history=get_history(),
    )


# Resolve Assistant modal action after blocked generation.
@app.post("/api/resolve")
def api_resolve():
    data = request.get_json(force=True) or {}
    action = (data.get("action") or "").strip().lower()
    pending_user_input, pending_reason, pending_message = get_pending()

    if not pending_user_input:
        return jsonify(ok=True, action_required=False, reply="", history=get_history())

    cfg = get_config()

    if action == "skip":
        clear_pending()
        return jsonify(
            ok=True,
            action_required=False,
            reply=pending_message,
            history=get_history(),
            skipped=True,
        )

    if action == "clear":
        clear_history()
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


# Routes: Math Mode
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

    if out.get("skipped") and out.get("reason") in {"prompt_too_long","no_space_for_generation"}:
        trimmed = out.get("history", history)
        set_math_history(trimmed)

        return jsonify(
            ok=True,
            action_required=True,
            action_type=out.get("reason"),
            message=out.get("reply", ""),
            history=trimmed,  
        )
    
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


# Routes: Translator Mode
@app.post("/api/translate")
def api_translate():
    data = request.get_json(force=True) or {}
    user_input = (data.get("message") or "").strip()
    source_lang = (data.get("source_lang") or "English").strip()
    target_lang = (data.get("target_lang") or "German").strip()

    if not user_input:
        return jsonify(ok=True, reply="", history=[])

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

    reason = out.get("reason")

    if reason in {"no_space_for_generation", "prompt_too_long"}:
        return jsonify(
            ok=True,
            action_required=True,
            message=out.get("reply", "Translation could not be generated."),
            reason=reason,
            history=[],
        )

    return jsonify(
        ok=True,
        reply=out.get("reply", ""),
        history=[],
        skipped=out.get("skipped", False),
        reason=reason,
    )


# Routes: Local Guide Mode
@app.post("/api/guide")
def api_guide():
    data = request.get_json(force=True) or {}
    user_input = (data.get("message") or "").strip()

    if not user_input:
        return jsonify(ok=True, reply="", history=get_guide_history())

    history = get_guide_history()
    current_city = get_guide_current_city()
    model, tokenizer, device = get_guide_model()
    out = guide_mode(
        model=model,
        tokenizer=tokenizer,
        device=device,
        history=history,
        user_input=user_input,
        current_city=current_city,
    )

    set_guide_history(out.get("history", history))
    set_guide_current_city(out.get("current_city", current_city))
    reason = out.get("reason")

    if reason in {"no_space_for_generation", "prompt_too_long"}:
        return jsonify(
            ok=True,
            action_required=True,
            message=out.get("reply", "Local guide response could not be generated."),
            reason=reason,
            history=out.get("history", history),
        )

    return jsonify(
        ok=True,
        reply=out.get("reply", ""),
        history=out.get("history", history),
        skipped=out.get("skipped", False),
        reason=reason,
        current_city=out.get("current_city"),
        intent=out.get("intent"),
    )

# Flight search
@app.post("/api/flights")
def api_flights():
    data = request.get_json(force=True) or {}
    origin = (data.get("origin") or "").strip()
    destination = (data.get("destination") or "").strip()
    reply = fetch_flights(origin, destination)
    return jsonify(ok=True, reply=reply)



# Routes: Programming Assisteant Mode
@app.post("/api/programming")
def api_programming():

    data = request.get_json(force=True) or {}

    prompt = ( data.get("prompt") or data.get("message") or "" ).strip()

    model, tokenizer, device = get_programming_model()

    result = programming_mode(
        model=model,
        tokenizer=tokenizer,
        device=device,
        history=get_programming_history(),
        user_input=prompt,
    )

    set_programming_history(result["history"])

    return jsonify(
        ok=True,
        reply=result["reply"],
        skipped=result.get("skipped", False),
        reason=result.get("reason"),
    )



@app.post("/api/data/upload")
def api_data_upload():
    uploaded_file = request.files.get("file")

    try:
        dataframe = load_dataset(uploaded_file)

        filename = (
            uploaded_file.filename
            if uploaded_file is not None
            else "dataset"
        )

        context = build_dataset_context(
            dataframe=dataframe,
            filename=filename,
        )

        metadata = dataframe_metadata(
            dataframe=dataframe,
            filename=filename,
        )

        set_data_context(context)
        set_data_filename(filename)
        set_data_history([])

        return jsonify(
            ok=True,
            **metadata,
            reply=make_upload_reply(
                dataframe=dataframe,
                filename=filename,
            ),
        )

    except ValueError as exc:
        return jsonify(
            ok=False,
            error=str(exc),
        ), 400

    except Exception as exc:
        app.logger.exception(
            "Unexpected Data Analysis upload failure"
        )

        return jsonify(
            ok=False,
            error=f"Unexpected dataset error: {exc}",
        ), 500


@app.post("/api/data")
def api_data_analysis():
    data = request.get_json(force=True) or {}

    user_input = (
        data.get("message")
        or data.get("prompt")
        or ""
    ).strip()

    history = get_data_history()
    dataset_context = get_data_context()

    if not user_input:
        return jsonify(
            ok=True,
            reply="",
            history=history,
            skipped=True,
            reason="empty_input",
        )

    if not dataset_context:
        return jsonify(
            ok=True,
            action_required=False,
            reply=(
                "Please upload a CSV or Excel dataset before "
                "asking data-analysis questions."
            ),
            history=history,
            skipped=True,
            reason="no_dataset",
        )

    try:
        model, tokenizer, device = get_data_model()

        result = data_analysis_chat(
            model=model,
            tokenizer=tokenizer,
            device=device,
            history=history,
            user_input=user_input,
            dataset_context=dataset_context,
        )

        updated_history = result.get(
            "history",
            history,
        )

        set_data_history(updated_history)

        blocked_reasons = {
            "prompt_too_long",
            "no_space_for_generation",
            "oom",
        }

        if (
            result.get("skipped")
            and result.get("reason") in blocked_reasons
        ):
            return jsonify(
                ok=True,
                action_required=True,
                action_type=result.get("reason"),
                message=result.get("reply", ""),
                history=updated_history,
            )

        return jsonify(
            ok=True,
            action_required=False,
            reply=result.get("reply", ""),
            history=updated_history,
            skipped=result.get("skipped", False),
            reason=result.get("reason"),
        )

    except Exception as exc:
        app.logger.exception(
            "Unexpected Data Analysis generation failure"
        )

        return jsonify(
            ok=False,
            error=f"Data Analysis failed: {exc}",
        ), 500


if __name__ == "__main__":
    app.run(debug=True)