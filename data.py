"""连续切分原始文本，避免训练窗口跨入验证区间。"""
from dataclasses import dataclass
import torch


class CharTokenizer:
    def __init__(self, chars):
        if not chars or chars[0] != "<UNK>" or len(set(chars)) != len(chars):
            raise ValueError("词表需以 <UNK> 开头且不能重复")
        if any(len(c) != 1 for c in chars[1:]):
            raise ValueError("字符词表的元素必须为单字符")
        self.chars = list(chars)
        self.stoi = {c: i for i, c in enumerate(chars)}

    @classmethod
    def from_text(cls, text):
        if not text:
            raise ValueError("不能从空文本构建词表")
        return cls(["<UNK>"] + sorted(set(text)))

    def encode(self, text):
        # 验证集或用户输入的新字符映射到 UNK，不能反向扩充训练词表。
        return [self.stoi.get(c, 0) for c in text]

    def decode(self, ids):
        return "".join(self.chars[i] if i != 0 else "�" for i in ids)


@dataclass
class TextDataset:
    tokenizer: CharTokenizer
    train: torch.Tensor
    val: torch.Tensor
    split_index: int


def make_dataset(text, block_size, train_fraction=0.9):
    if block_size < 1 or not 0 < train_fraction < 1:
        raise ValueError("block_size 和 train_fraction 无效")
    split = int(len(text) * train_fraction)
    if min(split, len(text) - split) <= block_size:
        raise ValueError("训练集和验证集各自至少需要 block_size + 1 个字符")
    tokenizer = CharTokenizer.from_text(text[:split])
    return TextDataset(tokenizer, torch.tensor(tokenizer.encode(text[:split]), dtype=torch.long), torch.tensor(tokenizer.encode(text[split:]), dtype=torch.long), split)


def get_batch(tokens, batch_size, block_size, generator, device="cpu"):
    """局部生成器控制采样顺序；x 与 y 恰好相差一个字符。"""
    if batch_size < 1 or block_size < 1 or len(tokens) <= block_size:
        raise ValueError("批大小/窗口无效，或文本太短")
    starts = torch.randint(len(tokens) - block_size, (batch_size,), generator=generator)
    x = torch.stack([tokens[i:i + block_size] for i in starts.tolist()])
    y = torch.stack([tokens[i + 1:i + block_size + 1] for i in starts.tolist()])
    return x.to(device), y.to(device)
