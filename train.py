"""低资源字符级 GPT 训练；每个输出目录保存独立可复核的运行。"""
import argparse
import csv
import hashlib
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import torch
from data import make_dataset, get_batch
from model import MiniGPT, ModelConfig
from runtime import ROOT, environment, resolve_device, seed_everything, sha256, write_json


@dataclass
class TrainConfig:
    data_path: str = str(ROOT / "data" / "input.txt")
    out: str = str(ROOT / "runs" / "default")
    steps: int = 200
    batch_size: int = 8
    block_size: int = 64
    n_embd: int = 64
    n_head: int = 4
    n_layer: int = 2
    dropout: float = 0.0
    lr: float = 3e-4
    weight_decay: float = 0.01
    eval_interval: int = 50
    eval_batches: int = 10
    seed: int = 42
    device: str = "auto"
    threads: int = 2

    def __post_init__(self):
        for key in ("steps", "batch_size", "eval_interval", "eval_batches", "threads", "block_size", "n_embd", "n_head", "n_layer"):
            if not isinstance(getattr(self, key), int) or getattr(self, key) <= 0:
                raise ValueError(f"{key} 必须是正整数")
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("lr 必须为有限正数")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay 必须为有限非负数")
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("seed 必须为 [0, 2**32) 内的整数")
        ModelConfig(vocab_size=2, block_size=self.block_size, n_embd=self.n_embd, n_head=self.n_head, n_layer=self.n_layer, dropout=self.dropout)


@torch.no_grad()
def estimate_losses(model, dataset, cfg, device):
    """固定抽样评估，无梯度，并恢复进入时的 train/eval 状态。"""
    was_training = model.training
    model.eval()
    losses = {}
    try:
        for offset, split in enumerate(("train", "val")):
            generator = torch.Generator().manual_seed(cfg.seed + 1000 + offset)
            total = 0.0
            for _ in range(cfg.eval_batches):
                x, y = get_batch(getattr(dataset, split), cfg.batch_size, cfg.block_size, generator, device)
                total += model(x, y)[1].item()
            losses[split] = total / cfg.eval_batches
    finally:
        model.train(was_training)
    if not all(math.isfinite(v) for v in losses.values()):
        raise FloatingPointError("评估 loss 非有限数")
    return losses


def train(cfg):
    path = Path(cfg.data_path)
    if not path.is_file():
        raise FileNotFoundError(f"没有数据文件 {path}；请先运行 python prepare_data.py")
    out = Path(cfg.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"输出目录非空，防止覆盖实验：{out}；请指定新的 --out")
    text = path.read_text(encoding="utf-8-sig")
    ds = make_dataset(text, cfg.block_size)
    device = resolve_device(cfg.device)
    seed_everything(cfg.seed, cfg.threads)
    mc = ModelConfig(vocab_size=len(ds.tokenizer.chars), block_size=cfg.block_size, n_embd=cfg.n_embd, n_head=cfg.n_head, n_layer=cfg.n_layer, dropout=cfg.dropout)
    model = MiniGPT(mc).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # 训练数据独立 RNG：增加评估次数不会改变后续训练批次。
    rng = torch.Generator().manual_seed(cfg.seed + 10)
    initial_hash = hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
    metadata = {"train_config": asdict(cfg), "model_config": asdict(mc), "parameter_count": model.parameter_count(), "initial_parameter_sha256": initial_hash, "environment": environment(device), "data": {"path": str(path.resolve()), "sha256": sha256(path), "normalized_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "characters": len(text), "train_characters": len(ds.train), "val_characters": len(ds.val), "split_index": ds.split_index, "train_fraction": 0.9, "vocab_size": len(ds.tokenizer.chars), "val_unknown_count": int((ds.val == 0).sum())}, "source_sha256": {p.name: sha256(p) for p in ROOT.glob("*.py")}, "evaluation": "fixed sampled windows; same windows at every evaluation; no full validation sweep"}
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "metadata.json", metadata)
    write_json(out / "vocab.json", ds.tokenizer.chars)
    start = time.perf_counter()
    best_val, best_step = float("inf"), 0
    last_losses = None

    def checkpoint(step, losses):
        # 仅保存张量和基础类型，可用 weights_only=True 加载。
        return {"format_version": 1, "model_config": asdict(mc), "model_state": model.state_dict(), "vocab": ds.tokenizer.chars, "step": step, "losses": losses, "seed": cfg.seed, "parameter_count": model.parameter_count(), "data_sha256": metadata["data"]["sha256"]}

    with (out / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "train_step_loss", "train_loss", "val_loss", "elapsed_seconds"])
        writer.writeheader()
        for step in range(cfg.steps + 1):
            step_loss = ""
            if step:
                model.train()
                x, y = get_batch(ds.train, cfg.batch_size, cfg.block_size, rng, device)
                optimizer.zero_grad(set_to_none=True)
                loss = model(x, y)[1]
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"step {step}: 训练 loss 非有限数")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                step_loss = loss.item()
            row = {"step": step, "train_step_loss": step_loss, "train_loss": "", "val_loss": ""}
            if step % cfg.eval_interval == 0 or step == cfg.steps:
                last_losses = estimate_losses(model, ds, cfg, device)
                row.update(train_loss=last_losses["train"], val_loss=last_losses["val"])
                print(f"step {step:4d} | train {last_losses['train']:.4f} | val {last_losses['val']:.4f} | {device}", flush=True)
                if last_losses["val"] < best_val:
                    best_val, best_step = last_losses["val"], step
                    torch.save(checkpoint(step, last_losses), out / "best.pt")
            row["elapsed_seconds"] = time.perf_counter() - start
            writer.writerow(row)
            f.flush()
    torch.save(checkpoint(cfg.steps, last_losses), out / "last.pt")
    result = {"steps_completed": cfg.steps, "parameter_count": model.parameter_count(), "best_step": best_step, "best_val_loss": best_val, "final_train_loss": last_losses["train"], "final_val_loss": last_losses["val"], "elapsed_seconds": time.perf_counter() - start, "device": device, "seed": cfg.seed, "n_head": cfg.n_head}
    write_json(out / "summary.json", result)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, field in TrainConfig.__dataclass_fields__.items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(field.default), default=field.default)
    return TrainConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    train(parse_args())
