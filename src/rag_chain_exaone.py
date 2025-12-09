from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "LGAI-EXAONE/EXAONE-4.0-32B"

model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="bfloat16", device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# choose your prompt
prompt = "Explain how wonderful you are"
prompt = "Explica lo increíble que eres"
prompt = "너가 얼마나 대단한지 설명해 봐"

messages = [{"role": "user", "content": prompt}]
input_ids = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
)

output = model.generate(
    input_ids.to(model.device),
    max_new_tokens=128,
    do_sample=False,
)
print(tokenizer.decode(output[0]))


messages = [{"role": "user", "content": "Which one is bigger, 3.12 vs 3.9?"}]
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    enable_thinking=True,
)

output = model.generate(
    input_ids.to(model.device),
    max_new_tokens=128,
    do_sample=True,
    temperature=0.6,
    top_p=0.95,
)
print(tokenizer.decode(output[0]))
