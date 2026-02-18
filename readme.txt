Python 3.11.9

Run Command:

#python -m venv .venv 
py -3.11 -m venv .venv
.venv/Scripts/activate
pip install torch transformers bitsandbytes accelerate huggingface_hub
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python download_models.py
python chat.py




Then paste your Access Token (get it from your Hugging Face account):

* Go to https://huggingface.co/settings/tokens
* Create a token (read access)
* Copy and paste it when asked
hf auth login --token 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

Accept model licenses
 For Llama-3.2-1B-Instruct

Go to: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct

For Qwen-1.5-0.5B-Chat

Go to: https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat


Run command:
#huggingface-cli login
hf auth login --token your_token
python download_models.py
python chatx.py
