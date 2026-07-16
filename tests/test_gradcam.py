import torch, numpy as np
from src.model import build_model
from src.gradcam import gradcam

def test_gradcam_shape_and_range():
    model = build_model(14, pretrained=False)
    x = torch.randn(1, 3, 224, 224)
    cam = gradcam(model, x, target_label_idx=1, device=torch.device("cpu"))
    assert cam.shape == (224, 224)
    assert cam.min() >= 0.0 and cam.max() <= 1.0

def test_gradcam_nontrivial_and_label_specific():
    model = build_model(14, pretrained=False)
    x = torch.randn(1, 3, 224, 224)
    cam1 = gradcam(model, x, target_label_idx=1, device=torch.device("cpu"))
    cam5 = gradcam(model, x, target_label_idx=5, device=torch.device("cpu"))
    assert cam1.std() > 0                      # not a constant/all-zero map
    assert abs(cam1 - cam5).mean() > 0         # different targets -> different maps
