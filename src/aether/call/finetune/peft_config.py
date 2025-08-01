import uuid
from typing import List

from peft import LoraConfig, TaskType
from pydantic import BaseModel
from transformers import TrainingArguments


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


def model_size_to_number(size_str: str) -> float:
    size_str = size_str.upper().strip()
    if size_str.endswith("B"):
        return float(size_str[:-1])
    else:
        raise ValueError(f"Unsupported model size format: {size_str}")


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
