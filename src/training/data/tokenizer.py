import json
from pathlib import Path
from typing import List, Optional

from tokenizers import Tokenizer, models, pre_tokenizers, processors, trainers
from tokenizers.normalizers import NFD, Lowercase, Sequence, StripAccents


class BPETokenizer:
    """BPE Tokenizer"""

    def __init__(
        self,
        vocab_size: int = 50000,
        min_frequency: int = 2,
        special_tokens: Optional[List[str]] = None,
    ):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.special_tokens = special_tokens or [
            "<pad>",
            "<unk>",
            "<bos>",
            "<eos>",
        ]

        self.tokenizer = None
        self.is_trained = False

    def train(self, files: List[str] or List[Path]):
        """训练 Tokenizer"""
        # 初始化 BPE 模型
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

        # 设置预处理器
        tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        # 训练
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=self.special_tokens,
            show_progress=True,
        )

        tokenizer.train(files, trainer=trainer)

        # 设置后处理器（添加特殊 token）
        tokenizer.post_processor = processors.ByteLevel(add_prefix_space=False)

        self.tokenizer = tokenizer
        self.is_trained = True

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """编码文本"""
        if not self.is_trained:
            raise ValueError("Tokenizer 尚未训练")

        encoding = self.tokenizer.encode(text)
        return encoding.ids

    def decode(
        self, token_ids: List[int], skip_special_tokens: bool = True
    ) -> str:
        """解码 token IDs"""
        if not self.is_trained:
            raise ValueError("Tokenizer 尚未训练")

        return self.tokenizer.decode(
            token_ids, skip_special_tokens=skip_special_tokens
        )

    def save(self, path: Path):
        """保存 Tokenizer"""
        if not self.is_trained:
            raise ValueError("Tokenizer 尚未训练")

        path.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(path / "tokenizer.json"))

        # 保存配置
        config = {
            "vocab_size": self.vocab_size,
            "min_frequency": self.min_frequency,
            "special_tokens": self.special_tokens,
        }
        with open(path / "tokenizer_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        """加载 Tokenizer"""
        tokenizer_file = path / "tokenizer.json"
        config_file = path / "tokenizer_config.json"

        if not tokenizer_file.exists():
            raise FileNotFoundError(f"Tokenizer 文件不存在: {tokenizer_file}")

        # 加载配置
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            config = {}

        instance = cls(
            vocab_size=config.get("vocab_size", 50000),
            min_frequency=config.get("min_frequency", 2),
            special_tokens=config.get(
                "special_tokens", ["<pad>", "<unk>", "<bos>", "<eos>"]
            ),
        )

        instance.tokenizer = Tokenizer.from_file(str(tokenizer_file))
        instance.is_trained = True

        return instance
