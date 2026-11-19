import contextlib
import io

import numpy as np
import torch


class WeightsEnv:
    def __init__(self, model_id, cache_dir=None, dtype=None, device="cpu"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        self.model.to(device)
        self.model.eval()
        self.device = device
        self._original = self._snapshot()

    def _snapshot(self):
        return {k: v.detach().clone() for k, v in self.model.named_parameters()}

    def num_parameters(self):
        return sum(p.numel() for p in self.model.parameters())

    def tensor_summary(self):
        rows = []
        for k, v in self.model.named_parameters():
            w = v.detach().float()
            rows.append(
                {
                    "name": k,
                    "shape": list(v.shape),
                    "dtype": str(v.dtype),
                    "mean": round(float(w.mean()), 6),
                    "std": round(float(w.std()), 6),
                    "min": round(float(w.min()), 6),
                    "max": round(float(w.max()), 6),
                }
            )
        return rows

    def diff(self):
        changed = {}
        total_changed = 0
        total = 0
        for k, v in self.model.named_parameters():
            orig = self._original[k]
            total += orig.numel()
            if orig.shape != v.shape:
                n = orig.numel()
            else:
                n = int((orig != v.detach()).sum().item())
            total_changed += n
            if n > 0:
                changed[k] = n
        changed = dict(sorted(changed.items(), key=lambda kv: -kv[1]))
        return {"changed": changed, "changed_params": total_changed, "total_params": total}

    def exec_code(self, code, extra_ns=None, max_output=20000):
        ns = {
            "model": self.model,
            "torch": torch,
            "np": np,
            "tokenizer": self.tokenizer,
            "diff": self.diff,
        }
        if extra_ns:
            ns.update(extra_ns)
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(code, "<wsp>", "exec"), ns)
            status = "ok"
        except Exception as e:
            print(f"{type(e).__name__}: {e}", file=buf)
            status = "error"
        out = buf.getvalue()
        if len(out) > max_output:
            out = out[:max_output] + f"\n...(truncated, was {len(out)} chars)"
        return {"status": status, "output": out}

    def forward(self, prompts, max_new_tokens=64):
        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        texts = self.tokenizer.batch_decode(out, skip_special_tokens=True)
        return {
            p: {"prompt": p, "completion": t}
            for p, t in zip(prompts, texts)
        }

    def save(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def apply_patch(self, patch):
        changed_before = set(self.diff()["changed"])
        for name, value in patch.items():
            p = dict(self.model.named_parameters())[name]
            with torch.no_grad():
                if isinstance(value, dict) and "index" in value:
                    idx = value["index"]
                    p.view(-1)[idx] = torch.as_tensor(value["value"], device=p.device, dtype=p.dtype)
                else:
                    p.copy_(torch.as_tensor(value, device=p.device, dtype=p.dtype))
        changed_after = set(self.diff()["changed"])
        return changed_after - changed_before