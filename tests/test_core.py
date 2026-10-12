import math
import pytest
import torch
import torch.nn.functional as F


def components():
    # 动态导入让缺失模块的失败明确定位到所需行为。
    from model import ModelConfig, MiniGPT, CausalSelfAttention, MultiHeadAttention
    return ModelConfig, MiniGPT, CausalSelfAttention, MultiHeadAttention


def test_tokenizer_roundtrip_unknown_and_serialization():
    from data import CharTokenizer
    tok = CharTokenizer.from_text("a中b\n")
    assert tok.decode(tok.encode("中ab\n")) == "中ab\n"
    assert tok.encode("?") == [0]
    assert tok.decode([0]) == "�"
    assert CharTokenizer(tok.chars).chars == tok.chars


def test_split_no_vocabulary_leak_and_shift():
    from data import make_dataset, get_batch
    ds = make_dataset("abcd" * 50 + "Z" * 20, block_size=8)
    assert "Z" not in ds.tokenizer.chars
    assert len(ds.train) + len(ds.val) == 220
    assert len(ds.train) == 198
    x, y = get_batch(ds.train, 4, 8, torch.Generator().manual_seed(42))
    assert x.shape == y.shape == (4, 8)
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_short_data_and_invalid_batch():
    from data import make_dataset, get_batch
    with pytest.raises(ValueError):
        make_dataset("short", 8)
    with pytest.raises(ValueError):
        get_batch(torch.arange(8), 2, 8, torch.Generator())


@pytest.mark.parametrize("heads", [1, 4])
def test_attention_matches_pytorch_reference(heads):
    Config, _, Attention, Multi = components()
    cfg = Config(vocab_size=10, n_embd=16, n_head=heads, block_size=8, dropout=0.0)
    layer = (Attention(cfg) if heads == 1 else Multi(cfg)).eval()
    x = torch.randn(2, 5, 16)
    q, k, v = layer.qkv(x).chunk(3, dim=-1)
    q, k, v = [a.view(2, 5, heads, 16 // heads).transpose(1, 2) for a in (q, k, v)]
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    expected = layer.proj(expected.transpose(1, 2).contiguous().view(2, 5, 16))
    torch.testing.assert_close(layer(x), expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("heads", [1, 4])
def test_future_tokens_do_not_change_prefix(heads):
    Config, GPT, _, _ = components()
    model = GPT(Config(vocab_size=11, n_embd=16, n_head=heads, n_layer=2, block_size=8)).eval()
    a = torch.tensor([[1, 2, 3, 4, 5, 6]])
    b = torch.tensor([[1, 2, 3, 9, 8, 7]])
    torch.testing.assert_close(model(a)[0][:, :3], model(b)[0][:, :3], atol=1e-7, rtol=1e-6)


def test_shape_loss_gradient_and_equal_parameters():
    Config, GPT, _, _ = components()
    models = [GPT(Config(vocab_size=13, n_embd=16, n_head=h, n_layer=1, block_size=8)) for h in (1, 4)]
    assert models[0].parameter_count() == models[1].parameter_count()
    x = torch.randint(0, 13, (2, 8))
    logits, loss = models[1](x, x)
    assert logits.shape == (2, 8, 13)
    assert math.isfinite(loss.item())
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in models[1].parameters())


def test_invalid_config_and_sequence():
    Config, GPT, _, _ = components()
    with pytest.raises(ValueError):
        Config(vocab_size=10, n_embd=15, n_head=4)
    with pytest.raises(ValueError):
        Config(vocab_size=10, dropout=1.0)
    model = GPT(Config(vocab_size=10, n_embd=16, n_head=4, block_size=4))
    for x in (torch.ones(1, 5, dtype=torch.long), torch.ones(1, 0, dtype=torch.long)):
        with pytest.raises(ValueError):
            model(x)


def test_small_batch_can_be_overfit():
    Config, GPT, _, _ = components()
    torch.manual_seed(7)
    torch.set_num_threads(2)
    model = GPT(Config(vocab_size=4, n_embd=16, n_head=4, n_layer=1, block_size=8))
    x = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
    y = (x + 1) % 4
    opt = torch.optim.AdamW(model.parameters(), lr=0.02)
    initial = model(x, y)[1].item()
    for _ in range(30):
        opt.zero_grad(set_to_none=True)
        loss = model(x, y)[1]
        loss.backward()
        opt.step()
    assert model(x, y)[1].item() < initial * 0.3
