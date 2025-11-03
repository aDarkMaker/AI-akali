from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import yaml

from .model.architecture import ModelConfig


@dataclass
class TrainingConfig:
    """训练配置"""

    # 数据路径
    data_dir: str = "data/datasets"
    train_file: Optional[str] = None
    val_file: Optional[str] = None
    test_file: Optional[str] = None

    # Tokenizer
    tokenizer_path: Optional[str] = None
    vocab_size: int = 50000
    train_tokenizer: bool = True

    # 模型配置
    model_config: Optional[ModelConfig] = None
    # 或者直接指定参数
    hidden_size: int = 2048
    num_layers: int = 24
    num_heads: int = 16
    intermediate_size: int = 5632
    max_position_embeddings: int = 8192

    # 训练参数
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-5
    min_lr: float = 1e-6
    weight_decay: float = 0.01
    num_epochs: int = 3
    max_length: int = 2048

    # 输出
    output_dir: str = "data/models"
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100

    # 其他
    seed: int = 42
    fp16: bool = True
    dataloader_num_workers: int = 4

    def to_dict(self) -> dict:
        """转换为字典"""
        data = asdict(self)
        if self.model_config:
            data["model_config"] = self.model_config.__dict__
        return data

    def save(self, path: Path):
        """保存配置"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, allow_unicode=True)

    @classmethod
    def load(cls, path: Path) -> "TrainingConfig":
        """加载配置"""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # 处理 model_config
        if "model_config" in data and isinstance(data["model_config"], dict):
            data["model_config"] = ModelConfig(**data["model_config"])

        float_fields = ["learning_rate", "min_lr", "weight_decay"]
        int_fields = [
            "vocab_size",
            "hidden_size",
            "num_layers",
            "num_heads",
            "intermediate_size",
            "max_position_embeddings",
            "batch_size",
            "gradient_accumulation_steps",
            "num_epochs",
            "max_length",
            "save_steps",
            "eval_steps",
            "logging_steps",
            "seed",
            "dataloader_num_workers",
        ]
        bool_fields = ["train_tokenizer", "fp16"]

        for field in float_fields:
            if field in data and isinstance(data[field], str):
                data[field] = float(data[field])

        for field in int_fields:
            if field in data and isinstance(data[field], str):
                data[field] = int(data[field])
            elif field in data and data[field] is not None:
                data[field] = int(data[field])

        for field in bool_fields:
            if field in data and isinstance(data[field], str):
                data[field] = data[field].lower() in ("true", "1", "yes", "on")

        return cls(**data)
