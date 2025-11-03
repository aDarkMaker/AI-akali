import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from .tokenizer import BPETokenizer


class TextDataset(Dataset):
    """文本数据集"""

    def __init__(
        self,
        data_path: Path,
        tokenizer: BPETokenizer,
        max_length: int = 2048,
        text_key: str = "text",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_key = text_key

        # 加载数据
        self.data = self._load_data(data_path)

    def _load_data(self, data_path: Path) -> List[str]:
        """加载数据文件"""
        data = []

        if data_path.is_dir():
            # 目录：加载所有文本文件
            for file_path in data_path.glob("*.txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    data.extend([line.strip() for line in f if line.strip()])
        elif data_path.suffix == ".jsonl":
            # JSONL 文件
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    text = item.get(self.text_key, "")
                    if text:
                        data.append(text)
        elif data_path.suffix == ".json":
            # JSON 文件
            with open(data_path, "r", encoding="utf-8") as f:
                items = json.load(f)
                if isinstance(items, list):
                    for item in items:
                        text = (
                            item.get(self.text_key, "")
                            if isinstance(item, dict)
                            else str(item)
                        )
                        if text:
                            data.append(text)
                else:
                    text = items.get(self.text_key, "")
                    if text:
                        data.append(text)
        else:
            # 纯文本文件
            with open(data_path, "r", encoding="utf-8") as f:
                data = [line.strip() for line in f if line.strip()]

        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        text = self.data[idx]

        # Tokenize
        token_ids = self.tokenizer.encode(text, add_special_tokens=True)

        # 截断或填充
        if len(token_ids) > self.max_length:
            token_ids = token_ids[: self.max_length]
        else:
            # 填充到 max_length
            pad_id = self.tokenizer.tokenizer.token_to_id("<pad>")
            if pad_id is None:
                pad_id = self.tokenizer.tokenizer.token_to_id("<unk>") or 0
            token_ids = token_ids + [pad_id] * (
                self.max_length - len(token_ids)
            )

        # 转换为 tensor
        token_ids_tensor = torch.tensor(token_ids, dtype=torch.long)

        return {"input_ids": token_ids_tensor, "labels": token_ids_tensor}
