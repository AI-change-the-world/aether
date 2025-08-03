import os
import time
import uuid
from typing import List, Optional

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


def model_size_to_number(size_str: str) -> float:
    size_str = size_str.upper().strip()
    if size_str.endswith("B"):
        return float(size_str[:-1])
    else:
        raise ValueError(f"Unsupported model size format: {size_str}")


class TrainingConfig(BaseModel):
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    max_seq_length: int = 2048
    num_train_epochs: int
    logging_steps: int = 10
    save_steps: int = 100
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True
    save_on_each_node: bool = True
    lr_scheduler_type: str
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: List[str]
    task_type: TaskType
    output_dir: str = str(uuid.uuid4())


def get_recommended_config_by_size(size_str: str) -> TrainingConfig:
    size = model_size_to_number(size_str)

    if size <= 7:
        return TrainingConfig(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=1e-4,
            max_seq_length=2048,
            num_train_epochs=3,
            logging_steps=10,
            save_steps=100,
            fp16=False,
            bf16=True,
            gradient_checkpointing=True,
            save_on_each_node=True,
            lr_scheduler_type="cosine",
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            task_type=TaskType.CAUSAL_LM,
        )
    elif size <= 13:
        return TrainingConfig(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=1e-4,
            max_seq_length=2048,
            num_train_epochs=3,
            logging_steps=10,
            save_steps=100,
            fp16=False,
            bf16=True,
            gradient_checkpointing=True,
            save_on_each_node=True,
            lr_scheduler_type="cosine",
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            task_type=TaskType.CAUSAL_LM,
        )
    elif size <= 32:
        return TrainingConfig(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=1e-4,
            max_seq_length=2048,
            num_train_epochs=8,
            logging_steps=10,
            save_steps=100,
            fp16=False,
            bf16=True,
            gradient_checkpointing=True,
            save_on_each_node=True,
            lr_scheduler_type="cosine",
            lora_r=16,
            lora_alpha=64,
            lora_dropout=0.05,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "down_proj",
                "up_proj",
            ],
            task_type=TaskType.CAUSAL_LM,
        )
    else:
        return TrainingConfig(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=1e-4,
            max_seq_length=2048,
            num_train_epochs=16,
            logging_steps=10,
            save_steps=100,
            fp16=False,
            bf16=True,
            gradient_checkpointing=True,
            save_on_each_node=True,
            lr_scheduler_type="cosine",
            lora_r=16,
            lora_alpha=64,
            lora_dropout=0.05,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "down_proj",
                "up_proj",
            ],
            task_type=TaskType.CAUSAL_LM,
        )


def training_config_to_lora_config(c: TrainingConfig) -> LoraConfig:
    return LoraConfig(
        task_type=c.task_type,
        r=c.lora_r,
        alpha=c.lora_alpha,
        dropout=c.lora_dropout,
        target_modules=c.target_modules,
        bias="none",
    )


def training_config_to_training_args(c: TrainingConfig) -> TrainingArguments:
    return TrainingArguments(
        output_dir=c.output_dir,
        per_device_train_batch_size=c.per_device_train_batch_size,
        gradient_accumulation_steps=c.gradient_accumulation_steps,
        logging_steps=10,
        num_train_epochs=8,
        # fp16=True,
        bf16=True,
        save_steps=100,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True,
        report_to="none",
    )


class TrainingStatus(BaseModel):
    epoch: int
    total_epochs: int
    loss: Optional[float] = None
    save_path: Optional[str] = None
    eta_seconds: Optional[float] = None
    eta_minutes: Optional[float] = None


df = pd.read_json("try_peft_dataset.json")
ds = Dataset.from_pandas(df)

model_path = "/root/models/qwen25_7b"


tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map="auto"
)
model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法
# 构造 LoRA 结构

# 7B
# peft_config = LoraConfig(
#     task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32, lora_dropout=0.05
# )

# 32B
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,  # ✅ 增加 rank
    lora_alpha=64,  # 保持 α/r ≈ 4
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
        "down_proj",
        "up_proj",
    ],  # ✅ 对 attention + FFN 均注入 LoRA
    bias="none",
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
    num_train_epochs=8,
    # fp16=True,
    bf16=True,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
)


class TrainingMonitorCallback(TrainerCallback):
    def __init__(self):
        self.start_time = None
        self.total_epochs = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        self.total_epochs = int(args.num_train_epochs)
        print(f"[Monitor] Training started. Total epochs: {self.total_epochs}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return

        # 计算 ETA
        elapsed = time.time() - self.start_time
        current_step = state.global_step
        total_steps = state.max_steps or (self.total_epochs * state.max_steps_per_epoch)
        eta = (
            (elapsed / current_step) * (total_steps - current_step)
            if current_step > 0
            else None
        )

        # 构建状态
        status = TrainingStatus(
            epoch=int(state.epoch or 0),
            total_epochs=self.total_epochs,
            loss=logs.get("loss"),
            eta_seconds=eta,
            eta_minutes=(eta / 60) if eta else None,
        )
        print(f"[Monitor] {status.model_dump_json(indent=2)}")

    def on_save(self, args, state, control, **kwargs):
        ckpt_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        status = TrainingStatus(
            epoch=int(state.epoch or 0),
            total_epochs=self.total_epochs,
            save_path=ckpt_path,
        )
        print(f"[Monitor] Model saved:\n{status.model_dump_json(indent=2)}")


# 用 Trainer 微调
trainer = Trainer(
    callbacks=[TrainingMonitorCallback()],
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

trainer.train()
