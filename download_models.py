from huggingface_hub import snapshot_download

# Download Llama-3.2-1B-Instruct model
print("Downloading Llama-3.2-1B-Instruct...")
snapshot_download("meta-llama/Llama-3.2-1B-Instruct", local_dir="models/Llama-3.2-1B-Instruct")

# Download Qwen1.5-0.5B-Chat model
print("Downloading Qwen1.5-0.5B-Chat...")
snapshot_download("Qwen/Qwen1.5-0.5B-Chat", local_dir="models/Qwen1.5-0.5B-Chat")

print("All models downloaded successfully...")