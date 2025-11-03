import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from torch.utils.data import DataLoader  # noqa: E402

from src.training.config import TrainingConfig  # noqa: E402
from src.training.data.dataset import TextDataset  # noqa: E402
from src.training.data.tokenizer import BPETokenizer  # noqa: E402
from src.training.model.architecture import DeepSeekForCausalLM  # noqa: E402
from src.training.model.architecture import ModelConfig  # noqa: E402
from src.training.trainer import Trainer  # noqa: E402


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    """主函数"""
    # 加载配置
    config_path = project_root / "config" / "training.yaml"
    if config_path.exists():
        config = TrainingConfig.load(config_path)
    else:
        config = TrainingConfig()
        config.save(config_path)
        print(f"已创建默认配置文件: {config_path}")
        print("请编辑配置文件后重新运行")
        return

    # 设置随机种子
    set_seed(config.seed)

    # 准备路径
    data_dir = project_root / config.data_dir
    output_dir = project_root / config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 训练或加载 Tokenizer
    print("=" * 50)
    print("步骤 1: 准备 Tokenizer")
    print("=" * 50)

    tokenizer_dir = output_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    if (
        config.train_tokenizer
        or not (tokenizer_dir / "tokenizer.json").exists()
    ):
        print("训练 Tokenizer...")
        tokenizer = BPETokenizer(vocab_size=config.vocab_size, min_frequency=2)

        # 准备训练文件
        if config.train_file:
            train_files = [project_root / config.train_file]
        else:
            # 从数据目录查找所有文本文件
            train_files = list(data_dir.glob("*.txt"))

        if not train_files:
            raise ValueError("未找到训练数据文件")

        train_files_str = [str(f) for f in train_files]
        tokenizer.train(train_files_str)
        tokenizer.save(tokenizer_dir)
        print(f"Tokenizer 已保存到: {tokenizer_dir}")
    else:
        print("加载已有 Tokenizer...")
        tokenizer = BPETokenizer.load(tokenizer_dir)
        print(f"Tokenizer 已加载，词汇表大小: {tokenizer.vocab_size}")

    # 2. 创建模型配置
    print("\n" + "=" * 50)
    print("步骤 2: 初始化模型")
    print("=" * 50)

    if config.model_config:
        model_config = config.model_config
    else:
        model_config = ModelConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            intermediate_size=config.intermediate_size,
            max_position_embeddings=config.max_position_embeddings,
        )

    model = DeepSeekForCausalLM(model_config)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"模型参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    # 3. 准备数据集
    print("\n" + "=" * 50)
    print("步骤 3: 准备数据集")
    print("=" * 50)

    if config.train_file:
        train_path = project_root / config.train_file
    else:
        train_path = data_dir / "train.txt"

    train_dataset = TextDataset(
        train_path, tokenizer, max_length=config.max_length
    )
    print(f"训练集大小: {len(train_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.dataloader_num_workers,
        pin_memory=True,
    )

    val_loader = None
    if config.val_file:
        val_path = project_root / config.val_file
        val_dataset = TextDataset(
            val_path, tokenizer, max_length=config.max_length
        )
        print(f"验证集大小: {len(val_dataset)}")

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.dataloader_num_workers,
            pin_memory=True,
        )

    # 4. 训练
    print("\n" + "=" * 50)
    print("步骤 4: 开始训练")
    print("=" * 50)

    trainer = Trainer(model, train_loader, val_loader, config)
    trainer.train()

    print("\n训练完成！")


if __name__ == "__main__":
    main()
