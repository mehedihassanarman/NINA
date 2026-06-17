from typing import Any, Dict, List, Optional
import torch
from basefunctions import load_model_and_tokenizer, print_gpu_memory, is_oom_error, build_chat_prompt, print_status

# Default system instruction used for every prompt
ASSISTANT_SYSTEM_PROMPT = "You are a helpful, concise assistant."

# Hard safety cap for prompt tokens to prevent OOM for Normal Mode
HARD_PROMPT_CAP = 3000


# Load model for Normal Mode
def normal_mode_load_model(device: str, model_path: str):
    needs_trust = "qwen" in model_path.lower()
    model, tokenizer = load_model_and_tokenizer(
        model_path=model_path,
        device=device,
        trust_remote_code=needs_trust,
        quantize_4bit_if_cuda=True,
    )
    return model, tokenizer


#  Single-turn chat function for the Normal Mode.
def normal_chat_turn(model,tokenizer,device: str,history: Optional[List[str]],user_input: str,temperature: float,top_p: float,max_length: int, seed: int,) -> Dict[str, Any]:
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

    # Show Chat History
    if lower_input == "show_chat_history()":
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

    # Clear chat history
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

    # Clear n rounds of history from beginning.
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

    # Normal generation path 
    max_context = getattr(model.config, "max_position_embeddings", 2048)
    prompt = build_chat_prompt(
        history=history,
        user_input=raw_input,
        tokenizer=tokenizer,
        system_prompt=ASSISTANT_SYSTEM_PROMPT,
    )

    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_len = enc.input_ids.shape[1]

    # Safety checks to prevent OOM
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

    # LLM Generation
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
        raise

    # Update chat history
    history.append(f"User: {raw_input}")
    history.append(f"Assistant: {reply}")
    print_status(input_len=input_len,max_length=max_length,history=history,max_context=max_context,device=device,generated_tokens=generated_tokens) 
    
    return {
        "reply": reply,
        "history": history,
        "input_len": input_len,
        "max_length": max_length,
        "max_context": max_context,
        "skipped": False,
        "reason": None,
    }