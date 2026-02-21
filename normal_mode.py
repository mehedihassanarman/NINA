# normal_assistant.py
#
# Web-oriented Normal Assistant module.
# - normal_mode_load_model: load a specific model (Llama/Qwen) for webapp
# - normal_chat_turn: single-turn function for webapp
# - __main__ block: small REPL to test normal_chat_turn from terminal

from typing import Any, Dict, List, Optional

import torch
from basefunctions import load_model_and_tokenizer, set_seed, print_gpu_memory, is_oom_error, build_chat_prompt, print_status, print_gpu_info

ASSISTANT_SYSTEM_PROMPT = "You are a helpful, concise assistant."
HARD_PROMPT_CAP = 3500


# =====================================================================
#  WEB-FRIENDLY MODEL LOADER (no CLI logic)
# =====================================================================

def normal_mode_load_model(device: str, model_path: str):

    needs_trust = "qwen" in model_path.lower()
    model, tokenizer = load_model_and_tokenizer(
        model_path=model_path,
        device=device,
        trust_remote_code=needs_trust,
        quantize_4bit_if_cuda=True,
    )
    return model, tokenizer


# =====================================================================
#  SINGLE-TURN WEB FUNCTION (used by Flask)
# =====================================================================

def normal_chat_turn(model,tokenizer,device: str,history: Optional[List[str]],user_input: str,temperature: float,top_p: float,max_length: int, seed: int,) -> Dict[str, Any]:
    """
    Single-turn chat function for the Normal Assistant.

    For webapp:
    - Parameters (model_path, temperature, top_p, max_length) are chosen by user in UI.
    - This function does NOT change model internally.
      (No 'change_chat_model' command here — model is chosen via radio in the webapp.)

    Still supports a few text commands for convenience:
        - "show_chat_history"  -> returns the full history as reply
        - "clear"              -> clears entire history
        - "clear n"            -> removes first n rounds (User+Assistant pairs)

    Returns a dict that you can jsonify in Flask:
        {
          "reply": str,
          "history": list[str],
          "input_len": Optional[int],
          "max_length": Optional[int],
          "max_context": Optional[int],
          "skipped": bool,
          "reason": Optional[str]
        }
    """
    #set_seed(seed, device)
    if history is None:
        history = []
    else:
        history = list(history)

    raw_input = (user_input or "").strip()
    if not raw_input:
        return {
            "reply": "",
            "history": history,
            "input_len": None,
            "max_length": None,
            "max_context": None,
            "skipped": True,
            "reason": "empty_input",
        }

    lower_input = raw_input.lower()

    # ---------------- COMMAND: show_chat_history ----------------
    if lower_input == "show_chat_history()":

        # Print to terminal (CLI style)
        #show_history(history)

        # Build identical multi-line text for web reply
        if not history:
            formatted = (
                "****** Conversation History (0 turns) ******\n"
                "No history is found.\n\n"
                "****** End of Conversation History ******"
            )
        else:
            turns = len(history) // 2
            formatted = f"****** Conversation History ({turns} turns) ******\n\n"

            for i in range(turns):
                user_msg = history[2 * i].replace("User:", "").strip()
                asst_msg = history[2 * i + 1].replace("Assistant:", "").strip()

                formatted += (
                    f"Turn {i+1}:\n"
                    f"§ User's Query : {user_msg}\n"
                    f"§ Assistant's Reply : {asst_msg}\n\n"
                )

            formatted += "****** End of Conversation History ******"

        return {
            "reply": formatted,
            "history": history,
            "input_len": None,
            "max_length": None,
            "max_context": None,
            "skipped": False,
            "reason": "show_chat_history",
        }


    # ---------------- COMMAND: clear ----------------
    if lower_input == "clear":
        return {
            "reply": "Conversation cleared.",
            "history": [],
            "input_len": None,
            "max_length": None,
            "max_context": None,
            "skipped": False,
            "reason": "clear",
        }

    # ---------------- COMMAND: clear n ----------------
    if lower_input.startswith("clear "):
        try:
            n = int(lower_input.split()[1])
            new_history = history[2 * n:] if len(history) >= 2 * n else []
            return {
                "reply": f"Cleared first {n} rounds from beginning.",
                "history": new_history,
                "input_len": None,
                "max_length": None,
                "max_context": None,
                "skipped": False,
                "reason": "clear_n",
            }
        except ValueError:
            return {
                "reply": "Invalid command. Usage: clear n",
                "history": history,
                "input_len": None,
                "max_length": None,
                "max_context": None,
                "skipped": True,
                "reason": "bad_clear_n",
            }

    # ---------------- Normal generation path ----------------

    max_context = getattr(model.config, "max_position_embeddings", 2048)

    prompt = build_chat_prompt(
        history=history,
        user_input=raw_input,
        tokenizer=tokenizer,
        system_prompt=ASSISTANT_SYSTEM_PROMPT,
    )
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_len = enc.input_ids.shape[1]

    if input_len > HARD_PROMPT_CAP:
        msg = (
            f"Prompt too long ({input_len} tokens > {HARD_PROMPT_CAP}). "
            "To avoid OOM, this response has been skipped.\n"
            "Tips: clear chat history or reduce your question length."
        )
        print_status(input_len, max_length, history, max_context, device, generated_tokens=0)

        return {
            "reply": msg,
            "history": history,
            "input_len": input_len,
            "max_length": max_length,
            "max_context": max_context,
            "skipped": True,
            "reason": "prompt_too_long",
        }

    if input_len + max_length > max_context:
        msg = (
            f"Context too long ({input_len} + {max_length} > {max_context}).\n"
            "Please clear chat history or reduce the max_length."
        )
        return {
            "reply": msg,
            "history": history,
            "input_len": input_len,
            "max_length": max_length,
            "max_context": max_context,
            "skipped": True,
            "reason": "context_too_long",
        }

    try:
        inputs = enc.to(device)
        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

        output = model.generate(
            **inputs,
            max_new_tokens=max_length,
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
        generated_tokens = new_tokens.shape[0]

        if not reply:
            raw = tokenizer.decode(new_tokens, skip_special_tokens=False)
            print(f"[DEBUG] Model returned empty output. Raw: {raw!r}")

    except Exception as e:
        if is_oom_error(e):
            print("\n!!!Warning!!! Out of Memory Detected (web mode)...")
            print(
                f"prompt_tokens={input_len}, output_tokens={max_length}, "
                f"history_turns={len(history)//2}"
            )
            print_gpu_memory()

            if device == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

            msg = (
                "The model ran out of memory while generating this reply.\n"
                "Try clearing the chat history or using a smaller max_length."
            )
            return {
                "reply": msg,
                "history": history,
                "input_len": input_len,
                "max_length": max_length,
                "max_context": max_context,
                "skipped": True,
                "reason": "oom",
            }
        # Non-OOM errors: let the web framework handle/log them
        raise

    # Update history for next turn
    history.append(f"User: {raw_input}")
    history.append(f"Assistant: {reply}")

    # Optional server-side logging
    print_status(input_len=input_len,max_length=max_length,history=history,max_context=max_context,device=device,generated_tokens=generated_tokens,)
    
    return {
        "reply": reply,
        "history": history,
        "input_len": input_len,
        "max_length": max_length,
        "max_context": max_context,
        "skipped": False,
        "reason": None,
    }


# =====================================================================
#  SIMPLE TERMINAL TEST HARNESS (for VS Code terminal)
# =====================================================================

if __name__ == "__main__":
    """
    Minimal REPL to test normal_chat_turn from the terminal.

    This is ONLY for debugging. The real webapp will call
    normal_chat_turn() from Flask.
    """

    print_gpu_info()

    # 1) Choose device
    if torch.cuda.is_available():
        print("[INFO] GPU detected, using cuda.")
    else:
        print("[INFO] GPU not available, using cpu.")

    device_choice = input("Do you want to use GPU or CPU? : ").strip().lower()
    device = "cuda" if (device_choice == "gpu" and torch.cuda.is_available()) else "cpu"

    # 2) Choose model path (same options as your web radio button)
    print("\nSelect model:")
    print("1. Llama-3.2-1B-Instruct")
    print("2. Qwen1.5-0.5B-Chat")
    choice = input("Type 1 or 2 (default=1): ").strip()

    if choice == "2":
        model_path = "models/Qwen1.5-0.5B-Chat"
    else:
        model_path = "models/Llama-3.2-1B-Instruct"

    # 3) Ask for generation parameters (like your web controls)
    def ask_float(msg, default):
        val = input(f"{msg} (default={default}): ").strip()
        return float(val) if val else default

    def ask_int(msg, default):
        val = input(f"{msg} (default={default}): ").strip()
        return int(val) if val else default

    temperature = ask_float("Temperature", 0.7)
    top_p = ask_float("top_p", 0.9)
    max_length = ask_int("max_length", 256)
    seed = ask_int("Enter the value of random seed -", 42)
 

    # 4) Load model once
    print(f"\n[INFO] Loading model: {model_path}")
    model, tokenizer = normal_mode_load_model(device, model_path)
    print("[INFO] Model loaded. You can now chat.")
    print("Type 'exit' to quit, 'show_chat_history', 'clear', or 'clear n'.\n")

    history: List[str] = []

    # 5) Simple loop that calls normal_chat_turn each time
    while True:
        msg = input("You: ").strip()
        if msg.lower() in {"exit", "quit"}:
            print("Bye!")
            break

        out = normal_chat_turn(
            model=model,
            tokenizer=tokenizer,
            device=device,
            history=history,
            user_input=msg,
            temperature=temperature,
            top_p=top_p,
            max_length=max_length,
            seed=seed,
        )

        history = out.get("history", history)
        print("\nAssistant:", out.get("reply", ""))
        print()
