import os
import time
from typing import Any, Callable, Optional

from pydantic import BaseModel
from transformers import TrainerCallback

from aether.common.logger import logger


class TrainingStatus(BaseModel):
    epoch: int
    total_epochs: int
    loss: Optional[float] = None
    save_path: Optional[str] = None
    eta_seconds: Optional[float] = None
    eta_minutes: Optional[float] = None


class TrainingMonitorCallback(TrainerCallback):
    def __init__(self, report_func: Callable[[TrainingStatus], Any] = None):
        self.start_time = None
        self.total_epochs = None
        self.report_func = report_func

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        self.total_epochs = int(args.num_train_epochs)
        logger.info(f"[Monitor] Training started. Total epochs: {self.total_epochs}")

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
        logger.info(f"[Monitor] {status.model_dump_json(indent=2)}")
        if self.report_func is not None:
            self.report_func(status)

    def on_save(self, args, state, control, **kwargs):
        ckpt_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        status = TrainingStatus(
            epoch=int(state.epoch or 0),
            total_epochs=self.total_epochs,
            save_path=ckpt_path,
        )
        logger.info(f"[Monitor] Model saved:\n{status.model_dump_json(indent=2)}")
        if self.report_func is not None:
            self.report_func(status)
