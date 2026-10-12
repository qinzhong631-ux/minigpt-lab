"""用基础 PyTorch 层展示字符级 GPT；不依赖预训练模型。"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int = 64
    n_embd: int = 64
    n_head: int = 4
    n_layer: int = 2
    dropout: float = 0.0

    def __post_init__(self):
        for key in ("vocab_size", "block_size", "n_embd", "n_head", "n_layer"):
            if not isinstance(getattr(self, key), int) or getattr(self, key) <= 0:
                raise ValueError(f"{key} 必须为正整数")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd 必须能被 n_head 整除")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout 必须在 [0, 1) 内")


class TokenEmbedding(nn.Embedding):
    """将离散字符编号映射为可学习的稠密向量。"""


class PositionEmbedding(nn.Embedding):
    """学习绝对位置；生成时窗口截断后从位置 0 重新编号。"""


class MultiHeadAttention(nn.Module):
    """固定总宽度拆分为多个头；head 数变化不改变投影参数量。"""
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.output_dropout = nn.Dropout(cfg.dropout)
        self.register_buffer("causal_mask", torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool)), persistent=False)

    def forward(self, x):
        # B: 批量，T: 时间长度，C: 总嵌入维度；H 个头各占 C/H 维。
        b, t, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [a.reshape(b, t, self.n_head, self.head_dim).transpose(1, 2) for a in (q, k, v)]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # 上三角设为负无穷，softmax 后未来位置的权重严格为零。
        scores = scores.masked_fill(~self.causal_mask[:t, :t], float("-inf"))
        weights = self.attn_dropout(F.softmax(scores, dim=-1))
        merged = (weights @ v).transpose(1, 2).contiguous().view(b, t, c)
        return self.output_dropout(self.proj(merged))


class CausalSelfAttention(MultiHeadAttention):
    """单头是多头实现的 H=1 特例，避免比较时混入其他实现差异。"""
    def __init__(self, cfg):
        if cfg.n_head != 1:
            raise ValueError("CausalSelfAttention 单头示例要求 n_head=1")
        super().__init__(cfg)


class FeedForward(nn.Module):
    """逐位置非线性变换，不跨时间交换信息。"""
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cfg.n_embd, 4 * cfg.n_embd), nn.GELU(), nn.Linear(4 * cfg.n_embd, cfg.n_embd), nn.Dropout(cfg.dropout))

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-LayerNorm 与两条残差连接帮助梯度传播。"""
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.attention = CausalSelfAttention(cfg) if cfg.n_head == 1 else MultiHeadAttention(cfg)
        self.ffn = FeedForward(cfg)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class MiniGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.token_embedding = TokenEmbedding(cfg.vocab_size, cfg.n_embd)
        self.position_embedding = PositionEmbedding(cfg.block_size, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg.n_layer)])
        self.final_norm = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        if idx.ndim != 2 or not 0 < idx.shape[1] <= self.config.block_size:
            raise ValueError("输入必须为 [batch, time]，且 1 <= time <= block_size")
        if targets is not None and targets.shape != idx.shape:
            raise ValueError("targets 必须与输入形状相同")
        positions = torch.arange(idx.shape[1], device=idx.device)
        x = self.dropout(self.token_embedding(idx) + self.position_embedding(positions))
        logits = self.lm_head(self.final_norm(self.blocks(x)))
        # targets 是右移一字符的标签；交叉熵同时处理所有位置。
        loss = None if targets is None else F.cross_entropy(logits.reshape(-1, self.config.vocab_size), targets.reshape(-1))
        return logits, loss

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())
