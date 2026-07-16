import torch
from src.model import build_model

def test_forward_shape():
    m = build_model(num_labels=14, pretrained=False).eval()
    with torch.no_grad():
        out = m(torch.randn(8, 3, 224, 224))
    assert tuple(out.shape) == (8, 14)
