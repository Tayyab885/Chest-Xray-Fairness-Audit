import numpy as np
import torch
import torch.nn.functional as F

def gradcam(model: torch.nn.Module, image_tensor: torch.Tensor, target_label_idx: int, device: torch.device) -> "np.ndarray":
    """Grad-CAM heatmap [224,224] in [0,1] for target_label_idx, hooking DenseNet121 features.norm5."""
    model.eval()
    image_tensor = image_tensor.to(device)
    image_tensor.requires_grad = True
    acts, grads = {}, {}
    target_layer = model.features.norm5

    def fwd_hook(m, i, o):
        acts["v"] = o.clone().detach()

    def bwd_hook(module, grad_input, grad_output):
        grads["v"] = grad_output[0].clone().detach()
        return None

    h1 = target_layer.register_forward_hook(fwd_hook)
    # register_backward_hook used instead of register_full_backward_hook: torchvision DenseNet applies
    # F.relu(..., inplace=True) right after features, which makes register_full_backward_hook raise
    # "view modified inplace". register_backward_hook is verified numerically correct for this
    # single-input/output BatchNorm2d target layer (norm5).
    h2 = target_layer.register_backward_hook(bwd_hook)

    out = model(image_tensor)
    model.zero_grad()
    out[0, target_label_idx].backward()

    h1.remove()
    h2.remove()

    a, g = acts["v"], grads["v"]
    w = g.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * a).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
    cam = cam[0, 0].detach().cpu().numpy()
    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()
    return cam
