import json
import math
from pathlib import Path
import pytest
import torch


def test_gutenberg_extraction():
    from prepare_data import extract_body
    text = "header\n*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\nabc\n*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\nlicense"
    assert extract_body(text) == "abc\n"
    with pytest.raises(ValueError):
        extract_body("no source markers")


def test_train_checkpoint_generate_and_no_overwrite(tmp_path):
    from train import TrainConfig, train
    from generate import load_checkpoint, generate_text
    text = tmp_path / "input.txt"
    text.write_text("abc def\n" * 100, encoding="utf-8")
    cfg = TrainConfig(data_path=str(text), out=str(tmp_path / "run"), steps=3, eval_interval=2, eval_batches=2, batch_size=2, block_size=8, n_embd=16, n_layer=1, n_head=4, device="cpu")
    result = train(cfg)
    assert math.isfinite(result["final_val_loss"])
    assert result["steps_completed"] == 3
    assert (tmp_path / "run" / "metrics.csv").is_file()
    meta = json.loads((tmp_path / "run" / "metadata.json").read_text(encoding="utf-8"))
    assert meta["train_config"]["seed"] == 42
    assert meta["environment"]["device"] == "cpu"
    assert len(meta["data"]["sha256"]) == 64
    model, tok, ckpt = load_checkpoint(tmp_path / "run" / "last.pt", "cpu")
    assert ckpt["step"] == 3
    out1 = generate_text(model, tok, "abc", 12, seed=9)
    assert out1 == generate_text(model, tok, "abc", 12, seed=9)
    assert out1.startswith("abc") and len(out1) == 15
    assert len(generate_text(model, tok, "abc" * 10, 2, top_k=1)) == 32
    assert len(generate_text(model, tok, "", 2)) == 3
    for temp in (0, -1, float("nan")):
        with pytest.raises(ValueError):
            generate_text(model, tok, "a", 2, temperature=temp)
    with pytest.raises(ValueError):
        generate_text(model, tok, "a", -1)
    with pytest.raises(ValueError):
        generate_text(model, tok, "a", 2, top_k=0)
    with pytest.raises(FileExistsError):
        train(cfg)


def test_eval_does_not_mutate_weights_or_training_samples():
    from train import TrainConfig, estimate_losses
    from model import MiniGPT, ModelConfig
    from data import make_dataset, get_batch
    ds = make_dataset("abcd\n" * 100, 8)
    model = MiniGPT(ModelConfig(vocab_size=6, n_embd=16, n_layer=1, block_size=8))
    cfg = TrainConfig(block_size=8, batch_size=2, eval_batches=2)
    weights = {k:v.clone() for k,v in model.state_dict().items()}
    rng = torch.Generator().manual_seed(42)
    state = rng.get_state().clone()
    global_state = torch.get_rng_state().clone()
    loss1 = estimate_losses(model, ds, cfg, "cpu")
    loss2 = estimate_losses(model, ds, cfg, "cpu")
    assert loss1 == loss2
    assert model.training
    assert torch.equal(state, rng.get_state())
    assert torch.equal(global_state, torch.get_rng_state())
    for k,v in model.state_dict().items():
        assert torch.equal(v, weights[k])


def test_invalid_training_config():
    from train import TrainConfig
    for params in ({"steps": 0}, {"batch_size": 0}, {"eval_interval": 0}, {"lr": -0.1}, {"lr": float("nan")}):
        with pytest.raises(ValueError):
            TrainConfig(**params)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="本机 PyTorch 无可用 CUDA；不宣称 GPU 实测通过")
def test_cuda_forward_backward():
    from model import ModelConfig, MiniGPT
    model = MiniGPT(ModelConfig(vocab_size=8, n_embd=16, n_head=4, n_layer=1, block_size=8)).cuda()
    x = torch.randint(0, 8, (2, 8), device="cuda")
    loss = model(x, x)[1]
    loss.backward()
    assert torch.isfinite(loss)


def test_fresh_download_refreshes_source_timestamp(tmp_path, monkeypatch):
    import io
    import prepare_data
    old = "2000-01-01T00:00:00+00:00"
    (tmp_path / "source.json").write_text(json.dumps({"retrieved_utc": old, "raw_sha256": "old"}), encoding="utf-8")
    raw = b"*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\nabc\n*** END OF THE PROJECT GUTENBERG EBOOK TEST ***"
    monkeypatch.setattr(prepare_data, "urlopen", lambda *args, **kwargs: io.BytesIO(raw))
    result = prepare_data.prepare(tmp_path)
    assert result["retrieved_utc"] != old
    # 再次调用使用同一缓存，保留该文件的真实下载时间。
    assert prepare_data.prepare(tmp_path)["retrieved_utc"] == result["retrieved_utc"]
