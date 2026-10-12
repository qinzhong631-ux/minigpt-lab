"""固定总宽度与训练预算，比较单头和四头；原始运行全部保留。"""
import argparse
import csv
import json
import statistics
from dataclasses import replace
from pathlib import Path
from train import TrainConfig, train
from runtime import ROOT, write_json


def run_comparison(base, seeds=(42, 43, 44)):
    seeds = list(seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("需要非空且不重复的种子列表")
    if base.n_embd % 4:
        raise ValueError("比较 1/4 heads 时 n_embd 必须能被 4 整除")
    out = Path(base.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"比较目录非空，避免覆盖：{out}")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        initial_hashes = []
        for heads in (1, 4):
            cfg = replace(base, seed=seed, n_head=heads, out=str(out / f"heads{heads}_seed{seed}"))
            print(f"=== heads={heads}, seed={seed} ===", flush=True)
            row = train(cfg)
            row["run_directory"] = str(Path(cfg.out).resolve())
            rows.append(row)
            meta = json.loads((Path(cfg.out) / "metadata.json").read_text(encoding="utf-8"))
            initial_hashes.append(meta["initial_parameter_sha256"])
        if len(set(initial_hashes)) != 1:
            raise RuntimeError("配对模型初始化不同，不能把此运行标为受控比较")
    if len({r["parameter_count"] for r in rows}) != 1:
        raise RuntimeError("参数量不一致")
    aggregates = []
    for heads in (1, 4):
        vals = [r["final_val_loss"] for r in rows if r["n_head"] == heads]
        aggregates.append({"n_head": heads, "n_seeds": len(vals), "final_val_loss_mean": statistics.mean(vals), "final_val_loss_sample_std": statistics.stdev(vals) if len(vals) > 1 else None})
    deltas = [{"seed": seed, "delta": next(r["final_val_loss"] for r in rows if r["seed"] == seed and r["n_head"] == 4) - next(r["final_val_loss"] for r in rows if r["seed"] == seed and r["n_head"] == 1)} for seed in seeds]
    result = {"runs": rows, "aggregates": aggregates, "paired_final_val_delta_4_minus_1": deltas, "metric": "Final-step loss on fixed sampled validation windows; nats per character; lower is better.", "limitations": "Tiny model, short budget, one corpus and only a few seeds. Equal parameters do not imply equal activation memory, per-head dimension, wall time or expressive behavior. Validation is also used for best checkpoint selection; it is not a held-out test set. No statistical superiority claim."}
    write_json(out / "comparison.json", result)
    with (out / "comparison.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=str(ROOT / "data/input.txt"))
    parser.add_argument("--out", default=str(ROOT / "runs/comparison"))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42,43,44])
    args = parser.parse_args()
    run_comparison(TrainConfig(data_path=args.data_path, out=args.out, steps=args.steps, device=args.device), args.seeds)
