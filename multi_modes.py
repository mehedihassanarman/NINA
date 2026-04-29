# multi_modes.py. Combined for web version of math, translator and local_guide.py

import os
import json
from typing import Optional, List, Dict, Any

import torch
import requests
from dotenv import load_dotenv

from basefunctions import load_model_and_tokenizer, set_seed, print_gpu_memory, is_oom_error, build_chat_prompt, print_status, get_max_context, compute_hard_prompt_cap,print_gpu_info

# ============================================================
# Global Configuration (shared)
# ============================================================

HARD_PROMPT_CAP = compute_hard_prompt_cap()
LLAMA_MODEL_PATH = "models/Llama-3.2-1B-Instruct"
PROMPT_FILE = "system_prompts.json"

DEFAULT_SEED = 42
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
MAX_NEW_TOKENS_CAP = 512  # still adapted dynamically

load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")
GEOAPIFY_API_KEY = os.getenv("GEOAPIFY_API_KEY")
AVIATIONSTACK_API_KEY = os.getenv("AVIATIONSTACK_API_KEY")

AIRPORTS: List[Dict[str, Any]] = []  # optional, used for flights helper


# ============================================================
# System Prompt Handling (shared)
# ============================================================

def _load_system_prompts(path: str = PROMPT_FILE) -> Dict[str, Any]:
    if not os.path.exists(path):
        print(f"[WARN] System prompt file '{path}' not found. Using built-in fallback prompts.")
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("JSON root must be an object/dict.")
        return data
    except Exception as e:
        print(f"[WARN] Failed to load system prompts from '{path}': {e}")
        return {}


_SYSTEM_PROMPTS = _load_system_prompts()


# ---------------- Math system prompt ----------------

def get_math_system_prompt() -> str:
    base = _SYSTEM_PROMPTS.get("math")
    if isinstance(base, str) and base.strip():
        return base

    # Fallback (from original math_solver.py idea)
    return (
        "You are a friendly math tutor for school-level math.\n"
        "Always explain step by step in simple language and clearly state the final answer."
    )


# ---------------- Translator system prompt ----------------

def get_translator_system_prompt(source_lang: str, target_lang: str) -> str:
    base = _SYSTEM_PROMPTS.get("translator")
    if isinstance(base, str) and base.strip():
        return (
            base
            + f"\n\nUser has chosen translation direction:\n"
              f"- Source language: {source_lang}\n"
              f"- Target language: {target_lang}\n"
              f"Your ONLY job is to translate from {source_lang} to {target_lang}.\n"
              f"Never answer the question itself. Only translate the text the user gives.\n"
              f"Output ONLY the translation, with no explanation or extra sentences.\n"
        )

    # Fallback strict translator prompt
    return (
        "You are a strict multilingual translation engine.\n"
        f"- Always translate ONLY from {source_lang} to {target_lang}.\n"
        "- The user will type a sentence or paragraph.\n"
        "- You MUST NOT answer questions or add explanations.\n"
        "- You MUST NOT change meaning or tone.\n"
        "- You MUST ONLY output the translation, nothing else.\n"
    )


# ---------------- Local guide system prompt ----------------

def get_local_guide_system_prompt(city: Optional[str] = None) -> str:
    base = _SYSTEM_PROMPTS.get("local_guide")

    extra = ""
    if city:
        extra = f"\nThe current city the user is asking about is: {city}.\n"

    if isinstance(base, str) and base.strip():
        return base + extra

    # Fallback (from original local_guide.py)
    return (
        "You are a polite and practical local guide assistant. "
        "You can suggest tourist places, talk about general news, explain weather data that the app provides, "
        "give tips for checking plane fares, checking hotels, and finding supermarkets in a city. "
        "You do NOT have real-time internet access yourself, but the app may give you up-to-date data from APIs. "
        "When data is provided (weather, news, prices), summarize and explain it clearly in simple language. "
        "If the user needs exact or live information, tell them to check a dedicated website or app.\n"
    ) + extra


# ============================================================
# Device & Model Loading (shared Llama)
# ============================================================

def get_device() -> str:
    if torch.cuda.is_available():
        print("[INFO] GPU is available. Using GPU (cuda).")
        return "cuda"
    print("[INFO] GPU not available. Using CPU.")
    return "cpu"


def load_llama_model(device: str):
    """
    Load a single Llama-3.2-1B-Instruct model instance for:
    - math solver
    - translator
    - local guide
    """
    model, tokenizer = load_model_and_tokenizer(
        model_path=LLAMA_MODEL_PATH,
        device=device,
        trust_remote_code=False,
        quantize_4bit_if_cuda=True,
    )
    return model, tokenizer


def choose_max_new_tokens(input_len: int, max_context: int) -> int:
    remaining = max_context - input_len - 10  # safety margin
    if remaining <= 0:
        return 0
    return min(MAX_NEW_TOKENS_CAP, remaining)


# ============================================================
# Mode 2: Math Solver (single-turn, web-friendly)
# ============================================================

def math_mode(model, tokenizer, device: str, history: Optional[List[str]], user_input: str, temperature: float = DEFAULT_TEMPERATURE, top_p: float = DEFAULT_TOP_P, max_new_tokens: Optional[int] = None, ) -> Dict[str, Any]:
    """
    Single-turn math solver for web usage.

    Arguments:
        model, tokenizer, device: shared Llama instance and device.
        history: list of strings ["User: ...", "Assistant: ...", ...].
        user_input: the current math question.

    Returns a dict suitable to jsonify in Flask.
    """
    if history is None:
        history = []
    else:
        history = list(history)

    user_input = (user_input or "").strip()
    if not user_input:
        return {
            "reply": "",
            "history": history,
            "skipped": True,
            "reason": "empty_input",
        }

    system_prompt = get_math_system_prompt()
    max_context = get_max_context(model)

    prompt = build_chat_prompt(
        history=history,
        user_input=user_input,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
    )
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_len = enc.input_ids.shape[1]

    if input_len > HARD_PROMPT_CAP:
        msg = (
            f" Prompt too long ({input_len} tokens > {HARD_PROMPT_CAP}).\n"
            "To avoid OOM, this response has been skipped.\n"
            "Tips: shorten your question or clear the chat history."
        )
        if history:
            history = history[6:] if len(history) >= 6 else []
            print("[INFO] History truncated by 3 rounds due to hard safety cap.\n")

        return {
            "reply": msg,
            "history": history,
            "input_len": input_len,
            "max_new_tokens": 0,
            "max_context": max_context,
            "skipped": True,
            "reason": "prompt_too_long",
        }

    if max_new_tokens is None:
        max_new_tokens = choose_max_new_tokens(input_len, max_context)

    if max_new_tokens <= 0 or (input_len + max_new_tokens > max_context):
        msg = (
            " The question + history are too long for the model's context window.\n"
            "Please clear some history or shorten the question."
        )
        if history:
            history = history[6:] if len(history) >= 6 else []
            print("[INFO] History truncated by 3 rounds due to hard safety cap.\n")
        
        return {
            "reply": msg,
            "history": history,
            "input_len": input_len,
            "max_new_tokens": 0,
            "max_context": max_context,
            "skipped": True,
            "reason": "no_space_for_generation",
        }

    try:
        inputs = enc.to(device)
        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=1,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            use_cache=True,
        )

        new_tokens = output[0, inputs.input_ids.shape[1]:]
        reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if not reply:
            # debugging fallback
            reply = tokenizer.decode(new_tokens, skip_special_tokens=False).strip()

    except Exception as e:
        if is_oom_error(e):
            print("\n[WARN] OOM detected during math generation.")
            print(
                f"prompt_tokens={input_len}, max_new_tokens={max_new_tokens}, "
                f"history_turns={len(history)//2}"
            )
            print_gpu_memory()
            if device == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            msg = (
                " The model ran out of memory while solving this problem.\n"
                "Try clearing history or asking a smaller sub-problem."
            )
            return {
                "reply": msg,
                "history": history,
                "input_len": input_len,
                "max_new_tokens": 0,
                "max_context": max_context,
                "skipped": True,
                "reason": "oom",
            }
        # For other errors, raise so Flask logs it
        raise

    history.append(f"User: {user_input}")
    history.append(f"Assistant: {reply}")

    # Optional: can log status server-side via print_status if desired
    print_status(input_len=input_len,max_length=max_new_tokens,history=history,max_context=max_context,device=device,generated_tokens=new_tokens.shape[0],)

    return {
        "reply": reply,
        "history": history,
        "input_len": input_len,
        "max_new_tokens": max_new_tokens,
        "max_context": max_context,
        "skipped": False,
        "reason": None,
    }


# ============================================================
# Mode 3: Translator (single-turn, web-friendly)
# ============================================================

SUPPORTED_LANGUAGES = [
    "English",
    "Spanish",
    "French",
    "German",
    "Italian",
    "Portuguese",
    "Dutch",
    "Russian",
]


def translator_mode(model, tokenizer, device: str, history: Optional[List[str]], user_input: str, source_lang: str, target_lang: str, temperature: float = DEFAULT_TEMPERATURE, top_p: float = DEFAULT_TOP_P, max_new_tokens: Optional[int] = None, ) -> Dict[str, Any]:
    """
    Single-turn translator for web apps.

    NOTE:
    - Enforces the 256-character limit from the CLI version.
    - Always wraps the user text in a translation task instruction.
    """
    if history is None:
        history = []
    else:
        history = list(history)

    raw_user_input = (user_input or "").strip()
    if not raw_user_input:
        return {
            "reply": "",
            "history": history,
            "skipped": True,
            "reason": "empty_input",
        }

    # Character limit from original translator code
    if len(raw_user_input) > 512:
        msg = (
            f" Your input is {len(raw_user_input)} characters long.\n"
            "Maximum allowed is 512 characters.\n"
            "Please shorten your text and try again."
        )
        return {
            "reply": msg,
            "history": history,
            "skipped": True,
            "reason": "input_too_long",
        }

    system_prompt = get_translator_system_prompt(source_lang, target_lang)
    max_context = get_max_context(model)

    translation_task = (
        f"Translate the following text from {source_lang} to {target_lang}.\n"
        f"Output ONLY the translation, with no explanation.\n\n"
        f"Text: {raw_user_input}"
    )

    prompt = build_chat_prompt(
        history=history,
        user_input=translation_task,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
    )
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_len = enc.input_ids.shape[1]

    if input_len > HARD_PROMPT_CAP:
        msg = (
            f" Prompt too long ({input_len} tokens > {HARD_PROMPT_CAP}).\n"
            "To avoid OOM, this translation has been skipped.\n"
            "Tips: shorten your text or clear the history."
        )
        # Trim some rounds to help future turns, then skip this one
        if history:
            history = history[6:] if len(history) >= 6 else []
            print("[INFO] History truncated by 3 rounds due to hard safety cap.\n")

        return {
            "reply": msg,
            "history": history,
            "input_len": input_len,
            "max_new_tokens": 0,
            "max_context": max_context,
            "skipped": True,
            "reason": "prompt_too_long",
        }

    if max_new_tokens is None:
        max_new_tokens = choose_max_new_tokens(input_len, max_context)

    if max_new_tokens <= 0 or (input_len + max_new_tokens > max_context):
        msg = (
            " Your text + history are too long for the model's context window.\n"
            "Please clear history or shorten the text."
        )
        # Trim some rounds to help future turns, then skip this one
        if history:
            history = history[6:] if len(history) >= 6 else []
            print("[INFO] History truncated by 3 rounds due to hard safety cap.\n")

        return {
            "reply": msg,
            "history": history,
            "input_len": input_len,
            "max_new_tokens": 0,
            "max_context": max_context,
            "skipped": True,
            "reason": "no_space_for_generation",
        }

    try:
        inputs = enc.to(device)
        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=1,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            use_cache=True,
        )

        new_tokens = output[0, inputs.input_ids.shape[1]:]
        reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if not reply:
            reply = tokenizer.decode(new_tokens, skip_special_tokens=False).strip()

    except Exception as e:
        if is_oom_error(e):
            print("\n[WARN] OOM detected during translation.")
            print(
                f"prompt_tokens={input_len}, max_new_tokens={max_new_tokens}, "
                f"history_turns={len(history)//2}"
            )
            print_gpu_memory()
            if device == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            msg = (
                " The model ran out of memory while translating.\n"
                "Try clearing history or sending a shorter text."
            )
            return {
                "reply": msg,
                "history": history,
                "input_len": input_len,
                "max_new_tokens": 0,
                "max_context": max_context,
                "skipped": True,
                "reason": "oom",
            }
        raise

    # Store original text in history, like the CLI version
    history.append(f"User: {raw_user_input}")
    history.append(f"Assistant: {reply}")

    print_status(input_len=input_len,max_length=max_new_tokens,history=history,max_context=max_context,device=device,generated_tokens=new_tokens.shape[0],)

    return {
        "reply": reply,
        "history": history,
        "input_len": input_len,
        "max_new_tokens": max_new_tokens,
        "max_context": max_context,
        "skipped": False,
        "reason": None,
    }


# ============================================================
# Mode 4: Local Guide (single-turn, web-friendly)
# ============================================================

def extract_city_from_text(text: str) -> Optional[str]:
    """
    Heuristic from original local_guide.py:
    Try to extract a city name from phrases like:
    - 'I want to visit places in Frankfurt'
    - 'I want to buy groceries in Darmstadt.'
    - 'I want to stay in Berlin'
    """
    lower = text.lower()
    preps = [" in ", " at ", " near ", " around ", " about "]

    best_idx = -1
    best_p = None
    for p in preps:
        idx = lower.rfind(p)
        if idx != -1 and idx > best_idx:
            best_idx = idx
            best_p = p

    if best_idx == -1 or best_p is None:
        return None

    start = best_idx + len(best_p)
    city_part = text[start:]
    city = city_part.strip(" .,!?:;\n\t")
    if not city:
        return None

    bad_tokens = {"in", "at", "near", "around", "about"}
    if city.lower() in bad_tokens:
        return None

    words = city.split()
    if len(words) > 3:
        return None

    return city


def llm_extract_city( model, tokenizer, device: str, user_input: str, ) -> Optional[str]:
    """
    Use the LLM itself to extract a city from user input.
    Returns None if no valid city is found.
    """
    system = (
        "You are a city extraction assistant.\n"
        "Given a single user message, extract the NAME OF THE CITY the user is referring to.\n"
        "Rules:\n"
        "- If the message mentions a city (e.g. 'I want to visit places in Frankfurt'), "
        "  respond with exactly that city name (e.g. 'Frankfurt').\n"
        "- If multiple cities are mentioned, pick the MAIN one.\n"
        "- If no city is mentioned, respond with 'NONE'.\n"
        "Respond with ONLY the city name or the word NONE. No extra words.\n"
    )

    prompt = f"{system}\nUser message: {user_input}\nCity:"
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(device) for k, v in enc.items()}

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=16,
            min_new_tokens=1,
            temperature=0.2,
            top_p=0.9,
            do_sample=True,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            use_cache=True,
        )

    new_tokens = output[0, inputs["input_ids"].shape[1]:]
    raw_reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    if not raw_reply:
        return None

    city = raw_reply.strip()
    if city.upper() == "NONE":
        return None

    if ":" in city:
        city = city.split(":", 1)[-1].strip()

    city = city.strip(" .,!?:;\n\t")
    if not city:
        return None

    words = city.split()
    if len(words) == 0 or len(words) > 3:
        return None
    if any(any(ch.isdigit() for ch in w) for w in words):
        return None

    bad_tokens = {"in", "at", "near", "around", "about", "city"}
    if city.lower() in bad_tokens:
        return None

    return city


def detect_intent(user_input: str) -> Optional[str]:
    """
    Simple rule-based intent detector from original local_guide.py.
    """
    text = user_input.lower()

    # Weather intent
    if any(kw in text for kw in [
        "weather", "temperature", "forecast", "rain today", "hot today", "cold today",
        "how is the weather", "how's the weather"
    ]):
        return "weather"

    # News intent
    if any(kw in text for kw in [
        "news", "headlines", "what's happening", "whats happening", "latest about",
        "recent events", "current events", "updates about"
    ]):
        return "news"

    # Tourist places / sightseeing intent
    if any(kw in text for kw in [
        "visit", "sightseeing", "tourist place", "tourist places", "attractions",
        "places to see", "things to see", "things to do", "landmarks",
        "monuments", "museums", "parks", "i want to visit", "want to visit"
    ]):
        return "tourist"

    # Supermarkets / groceries intent
    if any(kw in text for kw in [
        "supermarket", "supermarkets", "grocery", "groceries", "buy groceries",
        "buy food", "food shop", "grocery store", "i want to buy groceries",
        "need groceries"
    ]):
        return "supermarkets"

    # Hotels / accommodation intent
    if any(kw in text for kw in [
        "hotel", "hotels", "accommodation", "place to stay", "stay overnight",
        "where can i stay", "book a hotel", "need a hotel"
    ]):
        return "hotels"

    # Flights / travel intent
    if any(kw in text for kw in [
        "flight", "flights", "fly to", "fly from", "by plane", "by air",
        "airport", "travel by air", "i want to travel", "want to travel by plane"
    ]):
        return "flights"

    return None


# ---------------- API Helpers (same as original, but no input()) ----------------

def fetch_weather(city: str) -> str:
    if not OPENWEATHER_API_KEY:
        return "Weather service is not configured (missing OPENWEATHER_API_KEY)."

    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"q": city, "appid": OPENWEATHER_API_KEY, "units": "metric", "lang": "en"}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        desc = data["weather"][0]["description"].capitalize()
        temp = data["main"]["temp"]
        feels = data["main"].get("feels_like", temp)
        humidity = data["main"]["humidity"]
        wind = data["wind"]["speed"]

        return (
            f"Current weather in {city}:\n"
            f"- {desc}\n"
            f"- Temperature: {temp:.1f}°C (feels like {feels:.1f}°C)\n"
            f"- Humidity: {humidity}%\n"
            f"- Wind speed: {wind} m/s"
        )
    except Exception as e:
        return f"Could not fetch weather for {city}. Error: {e}"


def fetch_news(city_or_country: str) -> str:
    if not GNEWS_API_KEY:
        return "News service is not configured (missing GNEWS_API_KEY)."

    try:
        url = "https://gnews.io/api/v4/search"
        params = {
            "q": city_or_country,
            "lang": "en",
            "max": 5,
            "token": GNEWS_API_KEY,
        }

        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        articles = data.get("articles", [])
        if not articles:
            return f"No recent news found for {city_or_country}."

        lines = [f"Top news for {city_or_country}:"]

        for i, a in enumerate(articles, start=1):
            title = a.get("title") or "No title"
            source = a.get("source", {}).get("name", "Unknown source")
            url_link = a.get("url", "")
            lines.append(f"{i}. {title} — {source}\n   {url_link}")

        return "\n".join(lines)

    except Exception as e:
        return f"Could not fetch news for {city_or_country}. Error: {e}"


def _geoapify_geocode(city: str) -> Optional[tuple]:
    if not GEOAPIFY_API_KEY:
        return None

    geocode_url = "https://api.geoapify.com/v1/geocode/search"
    geocode_params = {"text": city, "apiKey": GEOAPIFY_API_KEY}

    geo_resp = requests.get(geocode_url, params=geocode_params, timeout=10)
    geo_resp.raise_for_status()
    features = geo_resp.json().get("features", [])
    if not features:
        return None

    lon = features[0]["geometry"]["coordinates"][0]
    lat = features[0]["geometry"]["coordinates"][1]
    return lon, lat


def fetch_tourist_places(city: str) -> str:
    if not GEOAPIFY_API_KEY:
        return "Tourist places search requires GEOAPIFY_API_KEY. Please set it in your .env file."

    try:
        coords = _geoapify_geocode(city)
        if not coords:
            return f"Could not determine coordinates for {city}."
        lon, lat = coords

        places_url = "https://api.geoapify.com/v2/places"
        places_params = {
            "categories": "tourism,tourism.attraction,tourism.sights,entertainment.museum,leisure.park,natural",
            "filter": f"circle:{lon},{lat},10000",
            "limit": 10,
            "apiKey": GEOAPIFY_API_KEY,
        }

        places_resp = requests.get(places_url, params=places_params, timeout=20)
        places_resp.raise_for_status()
        places = places_resp.json().get("features", [])

        if not places:
            return f"No major tourist attractions found within 10 km of {city}."

        lines = [f"Popular tourist attractions around {city}:"]
        for i, p in enumerate(places, start=1):
            props = p.get("properties", {})
            name = props.get("name") or "Unnamed place"
            address = props.get("formatted", "")
            lines.append(f"{i}. {name} — {address}")

        return "\n".join(lines)

    except requests.exceptions.Timeout:
        return (
            f"Geoapify took too long to respond while searching tourist places around {city}. "
            "Please try again in a moment."
        )
    except Exception as e:
        return f"Could not fetch tourist places for {city}. Error: {e}"


def fetch_supermarkets(city: str) -> str:
    if not GEOAPIFY_API_KEY:
        return "Missing GEOAPIFY_API_KEY for supermarket search."

    try:
        coords = _geoapify_geocode(city)
        if not coords:
            return f"Could not determine coordinates for {city}."
        lon, lat = coords

        places_url = "https://api.geoapify.com/v2/places"
        places_params = {
            "categories": "commercial.supermarket",
            "filter": f"circle:{lon},{lat},5000",
            "limit": 10,
            "apiKey": GEOAPIFY_API_KEY,
        }

        places_resp = requests.get(places_url, params=places_params, timeout=10)
        places_resp.raise_for_status()
        places = places_resp.json().get("features", [])

        if not places:
            return f"No supermarkets found within 5 km of {city}."

        lines = [f"Supermarkets near {city}:"]

        for i, p in enumerate(places, start=1):
            props = p.get("properties", {})
            name = props.get("name") or "Unnamed shop"
            address = props.get("formatted", "")
            lines.append(f"{i}. {name} — {address}")

        return "\n".join(lines)

    except Exception as e:
        return f"Could not fetch supermarkets for {city}. Error: {e}"


def fetch_hotels(city: str) -> str:
    if not GEOAPIFY_API_KEY:
        return "Hotel search needs GEOAPIFY_API_KEY. Please set it in your .env."

    try:
        coords = _geoapify_geocode(city)
        if not coords:
            return f"Could not find coordinates for {city}."
        lon, lat = coords

        places_url = "https://api.geoapify.com/v2/places"
        places_params = {
            "categories": "accommodation.hotel",
            "filter": f"circle:{lon},{lat},8000",
            "limit": 10,
            "apiKey": GEOAPIFY_API_KEY,
        }

        places_resp = requests.get(places_url, params=places_params, timeout=10)
        places_resp.raise_for_status()
        places = places_resp.json().get("features", [])

        if not places:
            return f"No hotels found within 8 km of {city}."

        lines = [f"Hotels near {city}:"]

        for i, p in enumerate(places, start=1):
            props = p.get("properties", {})
            name = props.get("name") or "Unnamed Hotel"
            address = props.get("formatted", "")
            lines.append(f"{i}. {name} — {address}")

        return "\n".join(lines)

    except Exception as e:
        return f"Could not fetch hotel data for {city}. Error: {e}"


def load_airports(path: str = "airports.json") -> None:
    """Optional helper if you want airport data for a UI."""
    global AIRPORTS
    try:
        with open(path, "r", encoding="utf-8") as f:
            AIRPORTS = json.load(f)
        print(f"[INFO] Loaded {len(AIRPORTS)} airports from {path}")
    except Exception as e:
        AIRPORTS = []
        print(f"[WARN] Could not load airport data from {path}: {e}")


def fetch_flights(origin: str, destination: str) -> str:
    """
    Flight API helper. For webapp, better to call this from a dedicated
    /api/flights endpoint with explicit 'origin' and 'destination' IATA codes.
    """
    if not AVIATIONSTACK_API_KEY:
        return "Flight search requires AVIATIONSTACK_API_KEY (AviationStack). Please set it in your .env."

    origin = origin.strip().upper()
    destination = destination.strip().upper()

    if len(origin) != 3 or not origin.isalpha():
        return f"Invalid origin code '{origin}'. Please enter a 3-letter IATA code like 'FRA'."
    if len(destination) != 3 or not destination.isalpha():
        return f"Invalid destination code '{destination}'. Please enter a 3-letter IATA code like 'BER'."

    try:
        url = "http://api.aviationstack.com/v1/flights"
        params = {
            "access_key": AVIATIONSTACK_API_KEY,
            "dep_iata": origin,
            "arr_iata": destination,
            "limit": 5,
        }

        resp = requests.get(url, params=params, timeout=12)
        resp.raise_for_status()
        data = resp.json()

        flights = data.get("data", [])
        if not flights:
            return f"No scheduled flights found from {origin} to {destination}."

        lines = [f"Flights from {origin} → {destination}:"]

        for i, f in enumerate(flights[:5], start=1):
            airline = f.get("airline", {}).get("name", "Unknown Airline")
            flight_num = f.get("flight", {}).get("iata", "N/A")

            dep_info = f.get("departure", {})
            arr_info = f.get("arrival", {})

            dep_airport = dep_info.get("airport", "Unknown Airport")
            dep_time = dep_info.get("scheduled", "N/A")

            arr_airport = arr_info.get("airport", "Unknown Airport")
            arr_time = arr_info.get("scheduled", "N/A")

            lines.append(
                f"{i}. {airline} {flight_num}\n"
                f"   From: {dep_airport} ({origin}) at {dep_time}\n"
                f"   To:   {arr_airport} ({destination}) at {arr_time}"
            )

        return "\n".join(lines)

    except Exception as e:
        return f"Could not fetch flight information. Error: {e}"


def guide_mode( model, tokenizer, device: str, history: Optional[List[str]], user_input: str, current_city: Optional[str] = None, temperature: float = DEFAULT_TEMPERATURE, top_p: float = DEFAULT_TOP_P, max_new_tokens: Optional[int] = None, ) -> Dict[str, Any]:
    """
    Single-turn Local Guide for web apps.

    - Tries to infer city from text (rule-based + LLM).
    - Uses APIs for weather/news/tourist/supermarkets/hotels when intent is clear.
    - Otherwise falls back to a normal LLM conversation about travel/city topics.
    """
    if history is None:
        history = []
    else:
        history = list(history)

    user_input = (user_input or "").strip()
    if not user_input:
        return {
            "reply": "",
            "history": history,
            "current_city": current_city,
            "intent": None,
            "skipped": True,
            "reason": "empty_input",
        }

    intent = detect_intent(user_input)

    # Try rule-based city extraction
    city_from_text = extract_city_from_text(user_input)
    # If rule-based fails, try LLM-based extraction
    if not city_from_text:
        city_from_text = llm_extract_city(model, tokenizer, device, user_input)

    if city_from_text:
        current_city = city_from_text

    # ========== API-based behavior ==========

    if intent == "weather" and current_city:
        info = fetch_weather(current_city)
        return {
            "reply": info,
            "history": history,
            "current_city": current_city,
            "intent": "weather",
            "skipped": True,
            "reason": "api_weather",
            "input_len": None,
            "max_new_tokens": None,
            "max_context": None,
        }

    if intent == "news" and current_city:
        info = fetch_news(current_city)
        return {
            "reply": info,
            "history": history,
            "current_city": current_city,
            "intent": "news",
            "skipped": True,
            "reason": "api_news",
            "input_len": None,
            "max_new_tokens": None,
            "max_context": None,
        }

    if intent == "tourist" and current_city:
        info = fetch_tourist_places(current_city)
        return {
            "reply": info,
            "history": history,
            "current_city": current_city,
            "intent": "tourist",
            "skipped": True,
            "reason": "api_tourist",
            "input_len": None,
            "max_new_tokens": None,
            "max_context": None,
        }

    if intent == "supermarkets" and current_city:
        info = fetch_supermarkets(current_city)
        return {
            "reply": info,
            "history": history,
            "current_city": current_city,
            "intent": "supermarkets",
            "skipped": True,
            "reason": "api_supermarkets",
            "input_len": None,
            "max_new_tokens": None,
            "max_context": None,
        }

    if intent == "hotels" and current_city:
        info = fetch_hotels(current_city)
        return {
            "reply": info,
            "history": history,
            "current_city": current_city,
            "intent": "hotels",
            "skipped": True,
            "reason": "api_hotels",
            "input_len": None,
            "max_new_tokens": None,
            "max_context": None,
        }

    # For flights, better to handle via a dedicated /api/flights endpoint
    if intent == "flights":
        msg = (
            "To search for flights, please use the flight form in the app and "
            "provide departure and arrival airport codes (e.g. FRA → BER)."
        )
        return {
            "reply": msg,
            "history": history,
            "current_city": current_city,
            "intent": "flights",
            "skipped": True,
            "reason": "needs_flight_form",
            "input_len": None,
            "max_new_tokens": None,
            "max_context": None,
        }

    # ========== Fallback: LLM conversation as local guide ==========

    system_prompt = get_local_guide_system_prompt(current_city)
    max_context = get_max_context(model)

    prompt = build_chat_prompt(
        history=history,
        user_input=user_input,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
    )
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_len = enc.input_ids.shape[1]

    if input_len > HARD_PROMPT_CAP:
        msg = (
            f" This conversation is too long for the model.\n"
            f"- Input tokens: {input_len}\n"
            f"- Max context:  {max_context}\n\n"
            "Please clear chat history or shorten your message."
        )

        if history:
            history = history[6:] if len(history) >= 6 else []
            print("[INFO] Removed 3 oldest rounds from history.\n")

        return {
            "reply": msg,
            "history": history,
            "current_city": current_city,
            "intent": intent,
            "input_len": input_len,
            "max_new_tokens": 0,
            "max_context": max_context,
            "skipped": True,
            "reason": "prompt_too_long",
        }

    if max_new_tokens is None:
        max_new_tokens = choose_max_new_tokens(input_len, max_context)

    if max_new_tokens <= 0 or (input_len + max_new_tokens > max_context):
        msg = (
            " Not enough remaining context to generate a reply.\n"
            "Please clear some history or shorten your message."
        )

        if history:
            history = history[6:] if len(history) >= 6 else []
            print("[INFO] Removed 3 oldest rounds from history.\n")
        
        return {
            "reply": msg,
            "history": history,
            "current_city": current_city,
            "intent": intent,
            "input_len": input_len,
            "max_new_tokens": 0,
            "max_context": max_context,
            "skipped": True,
            "reason": "no_space_for_generation",
        }

    try:
        inputs = enc.to(device)
        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=1,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            use_cache=True,
        )

        new_tokens = output[0, inputs.input_ids.shape[1]:]
        reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if not reply:
            reply = tokenizer.decode(new_tokens, skip_special_tokens=False).strip()

    except Exception as e:
        if is_oom_error(e):
            print("\n[WARN] OOM detected during local guide generation.")
            print(
                f"prompt_tokens={input_len}, max_new_tokens={max_new_tokens}, "
                f"history_turns={len(history)//2}"
            )
            print_gpu_memory()
            if device == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            msg = (
                " The model ran out of memory while answering.\n"
                "Try clearing history or sending a shorter message."
            )
            return {
                "reply": msg,
                "history": history,
                "current_city": current_city,
                "intent": intent,
                "input_len": input_len,
                "max_new_tokens": 0,
                "max_context": max_context,
                "skipped": True,
                "reason": "oom",
            }
        raise

    history.append(f"User: {user_input}")
    history.append(f"Assistant: {reply}")

    print_status(input_len=input_len,max_length=max_new_tokens,history=history,max_context=max_context,device=device,generated_tokens=new_tokens.shape[0],)

    return {
        "reply": reply,
        "history": history,
        "current_city": current_city,
        "intent": intent,
        "input_len": input_len,
        "max_new_tokens": max_new_tokens,
        "max_context": max_context,
        "skipped": False,
        "reason": None,
    }

if __name__ == "__main__":
    """
    Simple CLI tester for multi_modes.py

    Run in terminal:
        python multi_modes.py
    """

    #from pprint import pprint  # optional, just for nicer printing

    # 1) Init device + model
    print_gpu_info()
    device = get_device()
    set_seed(DEFAULT_SEED, device)
    model, tokenizer = load_llama_model(device)

    # 2) Choose mode
    print("=== multi_modes CLI Test ===")
    print("Choose a mode to test:")
    print("  1) Math solver")
    print("  2) Translator")
    print("  3) Local guide")
    mode = input("Enter mode number: ").strip()

    if mode == "1":
        mode_name = "math"
    elif mode == "2":
        mode_name = "translator"
    elif mode == "3":
        mode_name = "guide"
    else:
        print("Invalid mode. Exiting.")
        raise SystemExit

    print(f"\n[Mode selected] {mode_name}\n")

    # 3) Extra config for some modes
    history: list[str] = []
    current_city: str | None = None

    # Translator-specific settings
    source_lang = "English"
    target_lang = "German"

    if mode_name == "translator":
        print("Supported languages:")
        for lang in SUPPORTED_LANGUAGES:
            print(" -", lang)

        src = input("\nSource language : ").strip()
        tgt = input("Target language : ").strip()

        if src:
            source_lang = src
        if tgt:
            target_lang = tgt

    print("\nType 'exit' to quit.\n")

    # 4) Chat loop for the chosen mode
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Exiting.")
            break

        if mode_name == "math":
            out = math_mode(
                model=model,
                tokenizer=tokenizer,
                device=device,
                history=history,
                user_input=user_input,
            )

        elif mode_name == "translator":
            out = translator_mode(
                model=model,
                tokenizer=tokenizer,
                device=device,
                history=history,
                user_input=user_input,
                source_lang=source_lang,
                target_lang=target_lang,
            )

        elif mode_name == "guide":
            out = guide_mode(
                model=model,
                tokenizer=tokenizer,
                device=device,
                history=history,
                user_input=user_input,
                current_city=current_city,
            )
            # guide_mode may update current_city
            current_city = out.get("current_city", current_city)

        else:
            print("Unknown mode, something went wrong.")
            break

        # Update history for next turn
        history = out.get("history", history)

        # Print reply
        print("\nAssistant:")
        print(out.get("reply", ""))
        print()
