"""加载本项目本地权重，自回归生成字符。"""
import argparse
import math
from pathlib import Path
import torch
from data import CharTokenizer
from model import MiniGPT, ModelConfig
from runtime import ROOT, resolve_device


def load_checkpoint(path, device="auto"):
    device = resolve_device(device)
    # 先映射到 CPU，再移动模型；只允许张量/基础类型。
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=True)
    if ckpt.get("format_version") != 1:
        raise ValueError("不支持的 checkpoint 格式")
    tokenizer = CharTokenizer(ckpt["vocab"])
    cfg = ModelConfig(**ckpt["model_config"])
    if cfg.vocab_size != len(tokenizer.chars):
        raise ValueError("词表与模型配置不匹配")
    model = MiniGPT(cfg)
    model.load_state_dict(ckpt["model_state"], strict=True)
    return model.to(device).eval(), tokenizer, ckpt


@torch.inference_mode()
def generate_text(model, tokenizer, prompt, max_new_tokens=200, temperature=0.8, top_k=None, seed=42):
    if not isinstance(max_new_tokens, int) or not 0 <= max_new_tokens <= 10000:
        raise ValueError("生成长度必须为 0..10000 的整数")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature 必须为有限正数")
    if top_k is not None and (not isinstance(top_k, int) or top_k < 1):
        raise ValueError("top_k 必须为正整数或 None")
    if not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed 必须为 [0, 2**32) 内的整数")
    if not isinstance(prompt, str):
        raise ValueError("prompt 必须是文本")
    if not prompt:
        prompt = "\n" if "\n" in tokenizer.stoi else tokenizer.chars[1]
    device = next(model.parameters()).device
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    # 局部 CPU RNG 支持 CUDA/MPS 模型采样，不污染其他请求的全局随机状态。
    generator = torch.Generator(device="cpu").manual_seed(seed)
    was_training = model.training
    model.eval()
    suffix = []
    try:
        for _ in range(max_new_tokens):
            logits = model(idx[:, -model.config.block_size:])[0][:, -1, :] / temperature
            if top_k is not None:
                values, indices = torch.topk(logits, min(top_k, logits.shape[-1]))
                logits = torch.full_like(logits, float("-inf")).scatter(1, indices, values)
            probs = torch.softmax(logits, dim=-1).cpu()
            nxt = torch.multinomial(probs, num_samples=1, generator=generator).to(device)
            suffix.append(nxt.item())
            idx = torch.cat((idx, nxt), dim=1)
    finally:
        model.train(was_training)
    # 保留用户输入原文；未知字符在模型内部编码为 UNK。
    return prompt + tokenizer.decode(suffix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/default/best.pt")
    parser.add_argument("--prompt", default="Alice")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    model, tokenizer, _ = load_checkpoint(args.checkpoint, args.device)
    text = generate_text(model, tokenizer, args.prompt, args.max_new_tokens, args.temperature, args.top_k, args.seed)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
