import torch
from attention_lab import TinyAttentionModel

def test_tiny_model_returns_logits_and_attention_shapes():
    model = TinyAttentionModel(vocab_size=128, width=32, heads=4, layers=2, max_length=16)
    ids = torch.tensor([[1, 2, 3, 4]])
    logits, attentions = model(ids)
    assert logits.shape == (1, 4, 128)
    assert len(attentions) == 2
    assert attentions[0].shape == (1, 4, 4, 4)
