"""M12（选做）验收测试：Transformer。不要修改。"""

from __future__ import annotations

import ast
import inspect

import torch

from kws import transformer
from kws.transformer import KWSTransformer, MultiHeadSelfAttention, TransformerBlock


def test_no_builtin_attention():
    tree = ast.parse(inspect.getsource(transformer))
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not names & {
        "MultiheadAttention",
        "TransformerEncoderLayer",
        "scaled_dot_product_attention",
    }


def test_attention_shape():
    attn = MultiHeadSelfAttention(32, 4)
    assert attn(torch.randn(2, 10, 32)).shape == (2, 10, 32)


def test_causal_mask_blocks_future():
    torch.manual_seed(0)
    attn = MultiHeadSelfAttention(16, 2, causal=True).eval()
    x = torch.randn(1, 8, 16)
    y = x.clone()
    y[:, 5:] = torch.randn(1, 3, 16)  # 改未来
    assert torch.allclose(attn(x)[:, :5], attn(y)[:, :5], atol=1e-6)
    assert not torch.allclose(attn(x)[:, 5:], attn(y)[:, 5:])


def test_key_padding_mask_hides_padded_positions():
    torch.manual_seed(0)
    attn = MultiHeadSelfAttention(16, 2).eval()
    x = torch.randn(1, 6, 16)
    y = x.clone()
    y[:, 4:] = 100.0
    pad = torch.tensor([[False, False, False, False, True, True]])
    assert torch.allclose(attn(x, pad)[:, :4], attn(y, pad)[:, :4], atol=1e-5)


def test_block_keeps_shape():
    assert TransformerBlock(32, 4)(torch.randn(2, 7, 32)).shape == (2, 7, 32)


def test_classifier_shape_and_padding_invariance():
    torch.manual_seed(0)
    model = KWSTransformer(n_mels=40, num_classes=12).eval()
    x = torch.randn(2, 40, 50)
    assert model(x).shape == (2, 12)
    mask = torch.ones(2, 50, dtype=torch.bool)
    mask[:, 30:] = False
    y = x.clone()
    y[:, :, 30:] = 55.0
    assert torch.allclose(model(x, mask), model(y, mask), atol=1e-5)


def test_can_overfit_tiny_batch():
    torch.manual_seed(0)
    model = KWSTransformer(n_mels=16, num_classes=4, d_model=32, n_heads=4, n_layers=2)
    x, y = torch.randn(8, 16, 20), torch.arange(8) % 4
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(150):
        loss = torch.nn.functional.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05
