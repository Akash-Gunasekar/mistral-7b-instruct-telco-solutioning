import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)
from peft import PeftModel

# Path to saved adapter
ADAPTER_PATH = "mistral7b-qlora"
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

# Load in 4-bit to fit into Colab GPU
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Base model
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=bnb_config, device_map="auto"
)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=512)

prompt = """### Instruction:
Phase 1: Acknowledge industry, use case, and customer; present ranked problem statements.

### Input:
Industry: Telecom, Use Case: Churn Reduction, Customer: Verizon

### Response:
"""

result = pipe(
    prompt,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    max_new_tokens=300,
)[0]["generated_text"]

print(result)
