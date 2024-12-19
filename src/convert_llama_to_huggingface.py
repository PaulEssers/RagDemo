from transformers import LlamaForCausalLM, LlamaConfig
from accelerate import init_empty_weights
import torch
import json

model_path = "/root/.llama/checkpoints/Llama3.2-3B-Instruct-int4-qlora-eo8/"  

# Load the Meta Llama model
checkpoint = torch.load(f"{model_path}/consolidated.00.pth", map_location="cpu")

# Load the tokenizer configuration
with open(f"{model_path}/params.json", "r") as f:
    params = json.load(f)

config = LlamaConfig(
    hidden_size=params["dim"],
    num_attention_heads=params["n_heads"],
    num_hidden_layers=params["n_layers"],
    intermediate_size=4 * params["dim"],  # Usually 4x hidden size
    vocab_size=params["vocab_size"],
    pad_token_id=params.get("pad_token_id", 0),
    eos_token_id=params.get("eos_token_id", 2),
    bos_token_id=params.get("bos_token_id", 1),
)

with init_empty_weights():
    model = LlamaForCausalLM(config)


state_dict = checkpoint.get("model", checkpoint)  # Use "model" key if available
model.load_state_dict(state_dict, strict=False)

# Save the model in Hugging Face format
model.save_pretrained("/root/huggingface/llama-3b-in4-exported", config=config, safe_serialization=False)