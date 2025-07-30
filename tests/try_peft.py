import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

df = pd.read_json("try_peft_dataset.json")
ds = Dataset.from_pandas(df)

model_path = "/root/models/qwen25_7b"


tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map="auto"
)
model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法
# 构造 LoRA 结构
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32, lora_dropout=0.05
)
model = get_peft_model(model, peft_config)


def process_func(example):
    MAX_LENGTH = 384  # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
        f"<|im_start|>system\n现在你是一个程序员xiaoshuyui<|im_end|>\n<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )  # add_special_tokens 不在开头加 special_tokens
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = (
        instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    )
    attention_mask = (
        instruction["attention_mask"] + response["attention_mask"] + [1]
    )  # 因为eos token咱们也是要关注的所以 补充为1
    labels = (
        [-100] * len(instruction["input_ids"])
        + response["input_ids"]
        + [tokenizer.pad_token_id]
    )
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


tokenized_id = ds.map(process_func, remove_columns=ds.column_names)

model.print_trainable_parameters()

args = TrainingArguments(
    output_dir="./output/",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=3,
    fp16=True,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
)


# 用 Trainer 微调
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

trainer.train()
