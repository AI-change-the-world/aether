import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

mode_path = "/root/models/qwen25_7b"
lora_path = "./output/checkpoint-6"

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(mode_path, trust_remote_code=True)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    mode_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
).eval()

# 加载lora权重
model = PeftModel.from_pretrained(model, tool_id=lora_path)

prompt = "你是谁？"
inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt",
    return_dict=True,
).to("cuda")


gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
with torch.no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)
    outputs = outputs[:, inputs["input_ids"].shape[1] :]
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
