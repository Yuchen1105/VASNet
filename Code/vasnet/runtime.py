"""Model construction, strict historical checkpoint loading and window inference."""
import itertools
import random
import numpy as np
import torch


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(config, inference_scan="canonical"):
    from models.VASNet import VASNet
    return VASNet(in_chans=1, out_chans=config["num_classes"], inference_scan=inference_scan)


def load_weights(model, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint
    if isinstance(state, dict):
        for key in ("state_dict", "model", "network_weights", "net"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if not isinstance(state, dict) or not state or not all(isinstance(k, str) and torch.is_tensor(v) for k, v in state.items()):
        raise ValueError("Expected a tensor state_dict or a supported checkpoint wrapper")
    if all(k.startswith("module.") for k in state):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return checkpoint


def window_logits(model, image, device, mirror=False, overlap=0.5, sw_batch_size=1):
    from monai.inferers import sliding_window_inference
    image = image.to(device)
    axes = [()] if not mirror else [tuple(i + 2 for i, flag in enumerate(flags) if flag)
                                  for flags in itertools.product((False, True), repeat=3)]
    result = None
    for dims in axes:
        x = torch.flip(image, dims) if dims else image
        logits = sliding_window_inference(x, (128, 128, 128), sw_batch_size, model,
                                         overlap=overlap, mode="gaussian", device="cpu", sw_device=device)
        logits = torch.flip(logits, dims) if dims else logits
        result = logits if result is None else result + logits
    return result / len(axes)
