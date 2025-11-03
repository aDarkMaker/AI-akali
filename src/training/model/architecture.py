from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    """模型配置"""

    vocab_size: int = 50000
    hidden_size: int = 2048
    num_layers: int = 24
    num_heads: int = 16
    intermediate_size: int = 5632
    max_position_embeddings: int = 8192
    dropout: float = 0.1
    layer_norm_eps: float = 1e-6
    use_bias: bool = False  # DeepSeek 不使用 bias
    activation: str = "silu"  # SwiGLU activation


class RMSNorm(nn.Module):
    """RMS Normalization (DeepSeek 使用)"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        return self.weight * (x / (norm + self.eps))


class RotaryEmbedding(nn.Module):
    """旋转位置编码 (RoPE)"""

    def __init__(
        self, dim: int, max_seq_len: int = 8192, base: float = 10000.0
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._build_cache(max_seq_len)

    def _build_cache(self, max_seq_len: int):
        """预计算位置编码"""
        t = torch.arange(max_seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None):
        """应用旋转位置编码"""
        if seq_len is None:
            # 根据 x 的维度确定 seq_len
            # 对于 [batch, seq_len, num_heads, head_dim]，seq_len 在维度 1
            # 对于 [batch, seq_len, head_dim]，seq_len 也在维度 1
            if x.dim() >= 2:
                seq_len = x.shape[1]  # 序列长度在维度 1
            else:
                seq_len = x.shape[-2]  # 回退到倒数第二个维度

        cos = self.cos_cached[:seq_len]  # [seq_len, dim]
        sin = self.sin_cached[:seq_len]  # [seq_len, dim]

        x1, x2 = x.chunk(2, dim=-1)
        rotated = torch.cat([-x2, x1], dim=-1)

        # 根据 x 的维度调整 cos 和 sin 的形状以便正确广播
        if x.dim() == 4:  # [batch, seq_len, num_heads, head_dim]
            cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, dim]
            sin = sin.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, dim]
        elif x.dim() == 3:  # [batch, seq_len, head_dim]
            cos = cos.unsqueeze(0)  # [1, seq_len, dim]
            sin = sin.unsqueeze(0)  # [1, seq_len, dim]
        else:
            # 默认处理
            cos = cos.unsqueeze(0)  # [1, seq_len, dim]
            sin = sin.unsqueeze(0)  # [1, seq_len, dim]

        return x * cos + rotated * sin


class SwiGLU(nn.Module):
    """SwiGLU 激活函数"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2


class Attention(nn.Module):
    """多头注意力机制"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.scale = self.head_dim**-0.5

        # QKV 投影
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_heads * self.head_dim,
            bias=config.use_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_heads * self.head_dim,
            bias=config.use_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_heads * self.head_dim,
            bias=config.use_bias,
        )
        self.o_proj = nn.Linear(
            config.num_heads * self.head_dim,
            config.hidden_size,
            bias=config.use_bias,
        )

        # RoPE
        self.rope = RotaryEmbedding(
            self.head_dim, max_seq_len=config.max_position_embeddings
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # QKV 投影
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # 重塑为多头
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # 应用 RoPE
        if position_ids is not None:
            q = self.rope(q, seq_len=seq_len)
            k = self.rope(k, seq_len=seq_len)
        else:
            # 使用相对位置
            q = self.rope(q, seq_len=seq_len)
            k = self.rope(k, seq_len=seq_len)

        # 转置以便计算注意力
        q = q.transpose(1, 2)  # [batch, heads, seq_len, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # 应用注意力掩码
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(
                attention_mask == 0, float("-inf")
            )

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # 应用注意力到值
        attn_output = torch.matmul(attn_probs, v)

        # 重塑并投影
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(
            batch_size, seq_len, self.num_heads * self.head_dim
        )
        attn_output = self.o_proj(attn_output)

        return attn_output


class MLP(nn.Module):
    """前馈神经网络"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.use_bias
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.use_bias
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=config.use_bias
        )
        self.activation = SwiGLU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        x = torch.cat([gate, up], dim=-1)
        x = self.activation(x)
        x = self.down_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer 块"""

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        self.attention = Attention(config)
        self.mlp = MLP(config)

        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-attention norm
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = residual + hidden_states

        # Pre-MLP norm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class DeepSeekModel(nn.Module):
    """DeepSeek-like 模型"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # 词嵌入
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Transformer 层
        self.layers = nn.ModuleList(
            [
                TransformerBlock(config, layer_idx=i)
                for i in range(config.num_layers)
            ]
        )

        # 最终归一化
        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """权重初始化"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 词嵌入
        hidden_states = self.embed_tokens(input_ids)

        # 创建位置 ID（如果没有提供）
        if position_ids is None:
            batch_size, seq_len = input_ids.shape
            position_ids = (
                torch.arange(
                    seq_len, dtype=torch.long, device=input_ids.device
                )
                .unsqueeze(0)
                .expand(batch_size, -1)
            )

        # 创建注意力掩码（如果没有提供）
        if attention_mask is None:
            attention_mask = torch.ones(
                (input_ids.shape[0], input_ids.shape[1]),
                dtype=torch.bool,
                device=input_ids.device,
            )

        # 扩展注意力掩码维度
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # 通过 Transformer 层
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

        # 最终归一化
        hidden_states = self.norm(hidden_states)

        return hidden_states


class DeepSeekForCausalLM(nn.Module):
    """带语言模型头的 DeepSeek 模型"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = DeepSeekModel(config)

        # LM 头（共享权重）
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )

        # 可以共享嵌入权重
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        # 获取隐藏状态
        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        # 语言模型头
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # 计算损失（只对非填充位置计算）
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return (loss, logits) if loss is not None else (logits,)

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 512,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """生成文本"""
        self.eval()
        generated = input_ids.clone()

        with torch.no_grad():
            for _ in range(max_length - input_ids.shape[1]):
                # 前向传播
                outputs = self.forward(generated)
                logits = outputs[0][:, -1, :] / temperature

                if do_sample:
                    # Top-k 采样
                    if top_k > 0:
                        indices_to_remove = (
                            logits
                            < torch.topk(logits, top_k)[0][..., -1, None]
                        )
                        logits[indices_to_remove] = float("-inf")

                    # Top-p 采样
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(
                            logits, descending=True
                        )
                        cumulative_probs = torch.cumsum(
                            F.softmax(sorted_logits, dim=-1), dim=-1
                        )
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = (
                            sorted_indices_to_remove[..., :-1].clone()
                        )
                        sorted_indices_to_remove[..., 0] = 0

                        indices_to_remove = sorted_indices_to_remove.scatter(
                            1, sorted_indices, sorted_indices_to_remove
                        )
                        logits[indices_to_remove] = float("-inf")

                    # 采样
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)

                generated = torch.cat([generated, next_token], dim=1)

        return generated
