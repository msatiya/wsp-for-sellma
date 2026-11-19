import torch


def random_edit(model, frac=0.0002, sigma=1.0, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            mask = torch.rand(p.shape, generator=g, device="cpu") < frac
            if mask.any():
                noise = torch.randn_like(p) * (sigma * p.std().item())
                p.add_(noise * mask.to(p.dtype).to(p.device))
    return model