from typing import Any, Optional, Union

import pandas as pd
import torch
from datasets import Dataset
from peft import get_peft_model
from sqlalchemy.orm import Session
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
)

from aether.call.finetune import __IS_PEFT_AVAILABLE__
from aether.call.finetune.callback import TrainingMonitorCallback, TrainingStatus
from aether.call.finetune.peft_config import *


def process_func(example, tokenizer: Any, prompt: Optional[str] = None):
    MAX_LENGTH = 384  # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    if prompt:
        instruction = tokenizer(
            f"<|im_start|>system\n{prompt}<|im_end|>\n<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n<|im_start|>assistant\n",
            add_special_tokens=False,
        )  # add_special_tokens 不在开头加 special_tokens
    else:
        instruction = tokenizer(
            f"<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n<|im_start|>assistant\n",
            add_special_tokens=False,
        )
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


def qwen_lora_finetune(
    size: Union[int, str],
    data_path: str,
    model_path: str,
    prompt: Optional[str],
    session: Session,
    task_id: int,
):
    if not __IS_PEFT_AVAILABLE__:
        raise ImportError("Please install peft to use qwen_lora_finetune")

    if isinstance(size, str):
        cfg = get_recommended_config_by_size(size)
    else:
        cfg = get_recommended_config_by_size(str(size) + "B")

    train_args = training_config_to_training_args(cfg)
    lora_config = training_config_to_lora_config(cfg)

    df = pd.read_json(data_path)
    ds = Dataset.from_pandas(df)

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    )
    model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

    model = get_peft_model(model, lora_config)
    tokenized_id = ds.map(
        process_func,
        remove_columns=ds.column_names,
        fn_kwargs={"tokenizer": tokenizer, "prompt": prompt},
    )

    model.print_trainable_parameters()

    def save_to_db(status: TrainingStatus):
        # TODO implement save to db
        pass

    callback = TrainingMonitorCallback(report_func=lambda status: save_to_db(status))

    trainer = Trainer(
        callbacks=[callback],
        model=model,
        args=train_args,
        train_dataset=tokenized_id,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    trainer.train()


def qwen_lora_eval(model_path: str, system_prompt: Optional[str], prompt: str):
    ...
