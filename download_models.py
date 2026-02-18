from huggingface_hub import snapshot_download

print("Downloading Llama-3.2-1B-Instruct...")
snapshot_download("meta-llama/Llama-3.2-1B-Instruct", local_dir="models/Llama-3.2-1B-Instruct")

print("Downloading Qwen1.5-0.5B-Chat...")
snapshot_download("Qwen/Qwen1.5-0.5B-Chat", local_dir="models/Qwen1.5-0.5B-Chat")

print("All models downloaded successfully...")
