# basefunctions.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# ---------- Model loading / device utilities ----------

def load_model_and_tokenizer(
    model_path: str,
    device: str,
    trust_remote_code: bool = False,
    quantize_4bit_if_cuda: bool = True,
):
    """
    Load a causal LM + tokenizer from a local directory.

    - model_path: local folder with the model
    - device: "cuda" or "cpu"
    - trust_remote_code: pass through to HF transformers
    - quantize_4bit_if_cuda: if True, use 4-bit quantization on GPU
    """
    print(f"\n[INFO] Loading model from LOCAL DIRECTORY: {model_path}\n")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        )
    except Exception:
        print("\n❌ ERROR: Model tokenizer not found offline!")
        print(f"Make sure folder exists: {model_path}")
        raise

    # GPU → 4-bit quantization (optional)
    if device == "cuda" and quantize_4bit_if_cuda:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=trust_remote_code,
                local_files_only=True,
            )
        except Exception:
            print("\n❌ ERROR: Model weights not found offline!")
            raise

        print("[INFO] Model loaded locally in 4-bit on GPU!\n")

    else:  # CPU or non-quantized GPU
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                trust_remote_code=trust_remote_code,
                local_files_only=True,
            )
        except Exception:
            print("\n❌ ERROR: Model weights not found offline!")
            raise

        model.to(device)
        print(f"[INFO] Model loaded locally on {device}!\n")

    return model, tokenizer


def set_seed(seed: int, device: str):
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    print(f"[INFO] Random seed set to {seed}")


def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e6
        reserved = torch.cuda.memory_reserved() / 1e6
        print(f"[GPU Memory] Allocated: {allocated:.1f} MB | Reserved: {reserved:.1f} MB")


def is_oom_error(e: Exception) -> bool:
    if isinstance(e, torch.cuda.OutOfMemoryError):
        return True
    msg = str(e).lower()
    return "out of memory" in msg or "cuda error: out of memory"


def get_max_context(model) -> int:
    return getattr(model.config, "max_position_embeddings", 2048)


# ---------- Prompt construction ----------

def build_chat_prompt(history, user_input: str, tokenizer, system_prompt: str | None = None) -> str:
    """
    Generic prompt builder for all modes.
    Uses chat template if available, otherwise falls back to plain text.
    """
    # Chat-template capable model (Qwen, some Llama variants)
    if hasattr(tokenizer, "apply_chat_template"):
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Add conversation history
        for i in range(0, len(history), 2):
            user_msg = history[i].replace("User:", "").strip()
            asst_msg = history[i + 1].replace("Assistant:", "").strip()

            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if asst_msg:
                messages.append({"role": "assistant", "content": asst_msg})

        # Latest user input
        messages.append({"role": "user", "content": user_input})

        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # Plain text fallback (standard Llama format)
    system_part = f"System: {system_prompt}\n" if system_prompt else ""
    conv = "\n".join(history + [f"User: {user_input}", "Assistant:"])
    return system_part + conv


# ----- Status / history utilities (CLI only) -----

def print_status(input_len, max_length, history, max_context, device, generated_tokens=None, minimal=False):
    # Count user turns
    user_messages = (len(history) + 1) // 2
    turns = 0 if len(history) == 0 else user_messages

    gen = generated_tokens if generated_tokens is not None else 0
    total = input_len + gen

    if minimal:
        print(
            f"[STATUS] History_Turns : {turns}, "
            f"Context_Window_Size : {max_context}, "
            f"Maximum_Output_Size : {max_length}, "
            f"Prompt_Tokens : {input_len}"
        )
    else:
        print(
            f"[STATUS] History_Turns : {turns}, "
            f"Context_Window_Size : {max_context}, "
            f"Maximum_Output_Size : {max_length}, "
            f"Prompt_Tokens : {input_len}, "
            f"Generated_Tokens : {gen}, "
            f"Total_Tokens : {total}"
        )

    if device == "cuda" and torch.cuda.is_available():
        print_gpu_memory()
    print("")


def show_history(history):
    if not history:
        print("****** Conversation History (0 turns) ******")
        print("No history is found.\n")
        return

    turns = len(history) // 2
    print(f"****** Conversation History ({turns} turns) ******\n")

    for i in range(turns):
        user_msg = history[2 * i].replace("User:", "").strip()
        asst_msg = history[2 * i + 1].replace("Assistant:", "").strip()
        print(f"Turn {i+1}:")
        print(f"§ User's Query : {user_msg}")
        print(f"§ Assistant's Reply : {asst_msg}\n")

    print("****** End of Conversation History ******\n")

    # ============================================================
#                PROMPT CAP LOGIC (from prompt_cap)
# ============================================================

DEFAULT_MAX_CONTEXT = 131072 # For "Llama-3.2-1B-Instruct" model


def _infer_device(device: str | None) -> str:
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    device = device.lower()
    if device.startswith("cuda"):
        return "cuda"
    return "cpu"


def _infer_max_context(max_context: int | None) -> int:
    if max_context is None or max_context <= 0:
        return DEFAULT_MAX_CONTEXT
    return max_context


def print_gpu_info():
    if not torch.cuda.is_available():
        print("No GPU available. Using CPU.")
        return

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)

    total = props.total_memory / (1024 ** 3)
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    free = (props.total_memory - torch.cuda.memory_reserved(device)) / (1024 ** 3)

    print(f"GPU Name: {props.name}")
    print(f"Total VRAM: {total:.2f} GB")
    print(f"Allocated: {allocated:.2f} GB")
    print(f"Reserved: {reserved:.2f} GB")
    print(f"Free (approx): {free:.2f} GB")

#print_gpu_info()


def compute_hard_prompt_cap(device: str | None = None,
                            max_context: int | None = None) -> int:

    device = _infer_device(device)
    max_context = _infer_max_context(max_context)

    cap = min(2500, max_context // 2)

    if device == "cuda" and torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            total_gb = props.total_memory / (1024 ** 3)

            if total_gb <= 4:
                cap = min(2000, max_context // 2)
            elif total_gb <= 6:
                cap = min(2300, max_context // 2)
            elif total_gb <= 8:
                cap = min(2700, max_context // 2)
            elif total_gb <= 12:
                cap = min(3000, max_context // 2)
            else:
                cap = min(4000, max_context // 2)

        except Exception:
            cap = min(2500, max_context // 2)
    else:
        cap = min(2500, max_context // 2)

    cap = min(cap, max_context - 512)

    if cap <= 0:
        cap = max_context // 2

    print(f"[INFO] HARD_PROMPT_CAP computed as {cap} tokens (device={device}, max_context={max_context}).")
    return cap


# Global default
#HARD_PROMPT_CAP = compute_hard_prompt_cap()
