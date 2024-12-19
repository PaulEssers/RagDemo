import os
import sys
import logging
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import LlamaForCausalLM
from transformers import PreTrainedTokenizerFast, BitsAndBytesConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Initialize FastAPI app
app = FastAPI()

# Load the model and tokenizer
model_dir = os.path.join("models", os.environ.get("MODEL_DIR"))
if not os.path.exists(model_dir):
    found = os.listdir("models")
    raise ValueError(f"File '{model_dir}' not found! Found files: {found}")

# tokenizer = LlamaTokenizer.from_pretrained(model_dir)
logging.info("Loading tokenizer")
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

logging.info("Loading model")
# Set up the BitsAndBytesConfig for 8-bit or 4-bit loading
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = LlamaForCausalLM.from_pretrained(
    model_dir, torch_dtype=torch.float16, device_map='auto', quantization_config=quantization_config,
)

def generate_output(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    outputs = model.generate(inputs.input_ids.to('cuda'), max_length=500, attention_mask=inputs.attention_mask)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result


# quick test at startup:
logging.info("Running simple test:")
output = generate_output("hello! Are you ready?")
logging.info(f"Result: {output}")

class TextInput(BaseModel):
    text: str

# Define the API endpoint
@app.post("/predict/")
async def predict(input: TextInput):
    logging.info(f"Generating response for prompt: {input.text}")
    return {"result": generate_output(input.text)}

