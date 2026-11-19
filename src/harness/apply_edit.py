import json

import torch


def apply_patch_to_model(base_id, patch, out_dir, cache_dir=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_id, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        base_id, cache_dir=cache_dir, low_cpu_mem_usage=True
    )
    with torch.no_grad():
        for name, value in patch.items():
            p = dict(model.named_parameters())[name]
            if isinstance(value, dict) and "index" in value:
                idx = value["index"]
                p.view(-1)[idx] = torch.as_tensor(
                    value["value"], device=p.device, dtype=p.dtype
                )
            else:
                p.copy_(torch.as_tensor(value, device=p.device, dtype=p.dtype))
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)


def load_patch(path):
    with open(path) as f:
        return json.load(f)


def save_patch(patch, path):
    with open(path, "w") as f:
        json.dump(patch, f)