import torch


def random_edit(model, frac=0.0002, sigma=1.0, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            device = p.device
            mask = torch.rand(p.shape, generator=g, device="cpu") < frac
            if mask.any():
                n = int(mask.sum().item())
                noise = torch.randn(n, generator=g, device="cpu")
                std = p.std().item()
                vals = p.detach().cpu()
                flat = vals.view(-1)
                idx = torch.nonzero(mask.view(-1)).squeeze(-1)
                flat[idx] = flat[idx] + noise * (sigma * std)
                p.copy_(vals.to(device))
    return model