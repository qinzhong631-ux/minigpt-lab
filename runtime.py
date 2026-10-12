"""统一随机种子、设备选择和可审计环境信息。"""
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent


def resolve_device(requested="auto"):
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("当前 PyTorch 无可用 CUDA，请使用 cpu 或安装适合硬件的 CUDA 版 PyTorch")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("当前环境没有可用 MPS")
    if requested not in ("cpu", "cuda", "mps"):
        raise ValueError("device 只能为 auto/cpu/cuda/mps")
    return requested


def seed_everything(seed, threads=2):
    # 严格确定性方便复现；不支持的设备算子报错，不静默降低标准。
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def environment(device):
    packages = {}
    for name in ("torch", "gradio", "pytest", "numpy", "pandas", "gradio_client"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version, "executable": sys.executable, "platform": platform.platform(), "processor": platform.processor(), "cpu_count": os.cpu_count(), "device": device, "cuda_available": torch.cuda.is_available(), "cuda_build": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "threads": torch.get_num_threads(), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(), "packages": packages}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, content):
    Path(path).write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
