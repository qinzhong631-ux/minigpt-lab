import json
import pytest


def test_comparison_pairs_controlled_runs(tmp_path):
    from compare import run_comparison
    from train import TrainConfig
    text = tmp_path / "input.txt"
    text.write_text("abc def\n" * 100, encoding="utf-8")
    cfg = TrainConfig(data_path=str(text), out=str(tmp_path / "compare"), steps=2, eval_interval=2, eval_batches=1, batch_size=2, block_size=8, n_embd=16, n_layer=1, device="cpu")
    result = run_comparison(cfg, seeds=[42, 43])
    assert len(result["runs"]) == 4
    assert len({r["parameter_count"] for r in result["runs"]}) == 1
    assert len(result["paired_final_val_delta_4_minus_1"]) == 2
    for seed in (42, 43):
        metas = [json.loads((tmp_path / "compare" / f"heads{h}_seed{seed}" / "metadata.json").read_text(encoding="utf-8")) for h in (1, 4)]
        assert metas[0]["initial_parameter_sha256"] == metas[1]["initial_parameter_sha256"]
        assert metas[0]["data"]["sha256"] == metas[1]["data"]["sha256"]


def test_ui_build_and_handler(tmp_path):
    from app import build_demo, create_handler
    from train import train, TrainConfig
    from generate import load_checkpoint
    text = tmp_path / "input.txt"
    text.write_text("abc def\n" * 100, encoding="utf-8")
    out = tmp_path / "run"
    train(TrainConfig(data_path=str(text), out=str(out), steps=1, eval_batches=1, batch_size=1, block_size=8, n_embd=16, n_layer=1, device="cpu"))
    model,tok,_ = load_checkpoint(out / "best.pt", "cpu")
    handler = create_handler(model,tok)
    assert len(handler("abc", 5, 0.8, 0, 42)) == 8
    demo = build_demo(out / "best.pt", "cpu")
    config = demo.get_config_file()
    assert len(config["components"]) > 5
    assert any(dep.get("api_name") == "generate" for dep in config["dependencies"])
    demo.close()
