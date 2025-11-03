import sys
from pathlib import Path

import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.data.tokenizer import BPETokenizer  # noqa: E402
from src.training.model.architecture import DeepSeekForCausalLM  # noqa: E402
from src.training.model.architecture import ModelConfig  # noqa: E402


def load_model(checkpoint_path: Path, device: str = "cpu"):
    # 加载训练配置
    config_path = checkpoint_path / "training_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        import json

        config = json.load(f)

    # 创建模型配置
    model_config_dict = config["model_config"]
    model_config = ModelConfig(**model_config_dict)

    # 创建模型
    model = DeepSeekForCausalLM(model_config)

    # 加载模型权重
    checkpoint = torch.load(
        checkpoint_path / "pytorch_model.bin", map_location=device
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)

    return model, model_config


def generate_text(
    model: DeepSeekForCausalLM,
    tokenizer: BPETokenizer,
    prompt: str,
    max_length: int = 512,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    device: str = "cpu",
):
    """生成文本"""
    # 编码输入
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    # 生成
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_tensor,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
        )

    # 解码生成的文本
    generated_ids = generated_ids[0].cpu().tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return generated_text


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="使用训练好的模型生成文本")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="data/models/checkpoints/final",
        help="检查点目录路径",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="data/models/tokenizer",
        help="Tokenizer 目录路径",
    )
    parser.add_argument(
        "--prompt", type=str, required=True, help="输入提示文本"
    )
    parser.add_argument(
        "--max-length", type=int, default=512, help="最大生成长度"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="采样温度"
    )
    parser.add_argument("--top-k", type=int, default=50, help="Top-k 采样")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p 采样")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="运行设备",
    )

    args = parser.parse_args()

    # 确定设备
    if args.device == "cuda" and torch.cuda.is_available():
        device = "cuda"
    elif args.device == "mps" and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"使用设备: {device}")

    # 加载模型
    checkpoint_path = project_root / args.checkpoint
    print(f"加载模型从: {checkpoint_path}")
    model, model_config = load_model(checkpoint_path, device=device)
    print("模型加载完成")

    # 加载 Tokenizer
    tokenizer_path = project_root / args.tokenizer
    print(f"加载 Tokenizer 从: {tokenizer_path}")
    tokenizer = BPETokenizer.load(tokenizer_path)
    print("Tokenizer 加载完成")

    # 生成文本
    print(f"\n输入提示: {args.prompt}")
    print("生成中...")

    generated = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=device,
    )

    print(f"\n生成结果:\n{generated}")


if __name__ == "__main__":
    main()
