import json
from pathlib import Path
from typing import Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import TrainingConfig
from .model.architecture import DeepSeekForCausalLM


class Trainer:
    """训练器"""

    def __init__(
        self,
        model: DeepSeekForCausalLM,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: TrainingConfig = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # 设备
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        lr = config.learning_rate if config else 5e-5
        weight_decay = config.weight_decay if config else 0.01
        if isinstance(lr, str):
            lr = float(lr)
        if isinstance(weight_decay, str):
            weight_decay = float(weight_decay)

        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )

        # 学习率调度器
        num_epochs = config.num_epochs if config else 3
        min_lr = config.min_lr if config else 1e-6
        if isinstance(num_epochs, str):
            num_epochs = int(num_epochs)
        if isinstance(min_lr, str):
            min_lr = float(min_lr)

        total_steps = len(train_loader) * num_epochs
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=total_steps, eta_min=min_lr
        )

        # 训练状态
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")

        # 混合精度训练
        self.scaler = (
            torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        )

    def train_epoch(self):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}")

        for batch in pbar:
            # 移动到设备
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            # 创建注意力掩码
            attention_mask = (input_ids != 0).long().to(self.device)

            # 前向传播
            self.optimizer.zero_grad()

            if self.scaler is not None:
                # 混合精度训练
                with torch.cuda.amp.autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs[0]

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs[0]

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            self.scheduler.step()

            total_loss += loss.item()
            self.global_step += 1

            # 更新进度条
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
                }
            )

            # 保存检查点
            if self.config and self.global_step % self.config.save_steps == 0:
                self.save_checkpoint()

            # 验证
            if (
                self.config
                and self.val_loader
                and self.global_step % self.config.eval_steps == 0
            ):
                val_loss = self.validate()
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(is_best=True)

        return total_loss / len(self.train_loader)

    def validate(self) -> float:
        """验证"""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                attention_mask = (input_ids != 0).long().to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs[0]
                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        print(f"Validation Loss: {avg_loss:.4f}")

        self.model.train()
        return avg_loss

    def train(self):
        """完整训练流程"""
        num_epochs = self.config.num_epochs if self.config else 3

        for epoch in range(num_epochs):
            self.epoch = epoch
            train_loss = self.train_epoch()
            print(
                f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}"  # noqa: E501
            )

            # 每个 epoch 后验证
            if self.val_loader:
                val_loss = self.validate()
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(is_best=True)

        # 保存最终模型
        self.save_checkpoint(is_final=True)

    def save_checkpoint(self, is_best: bool = False, is_final: bool = False):
        """保存检查点"""
        if not self.config:
            return

        checkpoint_dir = Path(self.config.output_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if is_final:
            checkpoint_path = checkpoint_dir / "final"
        elif is_best:
            checkpoint_path = checkpoint_dir / "best"
        else:
            checkpoint_path = checkpoint_dir / f"checkpoint-{self.global_step}"

        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # 保存模型
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "global_step": self.global_step,
                "epoch": self.epoch,
                "best_val_loss": self.best_val_loss,
            },
            checkpoint_path / "pytorch_model.bin",
        )

        # 保存配置
        config_dict = {
            "model_config": self.model.config.__dict__,
            "training_config": self.config.__dict__ if self.config else {},
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_val_loss": self.best_val_loss,
        }

        with open(
            checkpoint_path / "training_config.json", "w", encoding="utf-8"
        ) as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

        print(f"Checkpoint saved to {checkpoint_path}")
