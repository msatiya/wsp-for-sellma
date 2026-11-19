import argparse
import json
import os
from pathlib import Path

import torch


def load_model(model_id, cache_dir, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, cache_dir=cache_dir, low_cpu_mem_usage=True
    )
    model.to(device).eval()
    return model, tokenizer


def eval_metrics(model, tokenizer, device):
    from eval.probes import split_probes, OPENENDED_PROMPTS
    from eval.scoring import evaluate_mc
    from eval.regression import run_regression
    from baselines.generate import generate
    from eval.language_id import detect

    probes = split_probes()["eval"]
    grammar = evaluate_mc(model, tokenizer, probes, device)
    v2 = [r for r in grammar["results"] if r["category"] == "v2"]
    agr = [r for r in grammar["results"] if r["category"] == "agreement"]
    completions = generate(model, tokenizer, OPENENDED_PROMPTS, do_sample=False)
    labels = {}
    for c in completions:
        d = detect(c)
        labels[d["label"]] = labels.get(d["label"], 0) + 1
    regression = run_regression(model, tokenizer, device)
    return {
        "grammar": grammar["accuracy"],
        "v2": sum(r["hit"] for r in v2) / len(v2) if v2 else 0.0,
        "agreement": sum(r["hit"] for r in agr) / len(agr) if agr else 0.0,
        "sv_ratio": labels.get("sv", 0) / max(len(completions), 1),
        "labels": labels,
        "en_regr": regression["accuracy"],
    }


def plant_fault(model, layer, rows, factor):
    t = model.model.layers[layer].mlp.down_proj.weight
    with torch.no_grad():
        t[rows] = t[rows] * factor
    return {"layer": layer, "tensor": f"model.model.layers.{layer}.mlp.down_proj.weight", "rows": rows, "factor": factor}


def blind_repair(model, layer, k=2.0):
    t = model.model.layers[layer].mlp.down_proj.weight
    with torch.no_grad():
        norms = t.norm(dim=1)
        med = norms.median()
        outlier = norms > k * med
        idx = outlier.nonzero().squeeze(-1).tolist()
        for i in idx:
            t[i] = t[i] * (med / norms[i])
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--layer", type=int, default=5)
    ap.add_argument("--n-rows", type=int, default=48)
    ap.add_argument("--factor", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--do-wsp", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model, tok = load_model(args.model, args.cache_dir, args.device)
    out = {}

    out["baseline"] = eval_metrics(model, tok, args.device)

    rows = list(range(args.n_rows))
    spec = plant_fault(model, args.layer, rows, args.factor)
    out["planted"] = spec
    out["after_plant"] = eval_metrics(model, tok, args.device)

    if args.do_wsp:
        from api_client import ChatClient
        from harness.agent import WspAgent
        from harness.weights_env import WeightsEnv

        env = WeightsEnv(args.model, cache_dir=args.cache_dir, device=args.device)
        spec = plant_fault(env.model, args.layer, rows, args.factor)
        client = ChatClient(
            os.environ.get("MODEL_STUDIO_BASE_URL", "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"),
            os.environ.get("MODEL_STUDIO_API_KEY", ""),
            os.environ.get("WSP_EDITOR_MODEL", "deepseek-v4-pro-0813"),
        )
        task = (
            "This language model's weights have been corrupted: some rows of one MLP "
            "matrix were rescaled by a large constant, which broke its Swedish grammar "
            "(verb-second word order and noun agreement). Inspect the weights using "
            "statistics (e.g. per-row norms), locate the anomalous rows and their scale "
            "factor, and restore the rows to plausible values. Edit weights directly, no "
            "gradients. Print a short summary of what you changed."
        )
        agent = WspAgent(client)
        res = agent.run(task, env, max_iters=6, max_tokens=4096, allow_forward=False)
        out["wsp"] = {"diff": res["diff"], "tokens": res["total_tokens"], "final": res["final"]}
        out["after_wsp_repair"] = eval_metrics(env.model, env.tokenizer, args.device)
        Path(args.out).mkdir(parents=True, exist_ok=True)
        with open(Path(args.out) / "wsp_transcript.json", "w") as f:
            json.dump(res["transcript"], f, indent=2, ensure_ascii=False)
        with open(Path(args.out) / "wsp_diff.json", "w") as f:
            json.dump(res["diff"], f, indent=2)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out) / "results.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: (v if not isinstance(v, dict) or "diff" in v else {kk: vv for kk, vv in v.items() if kk in ("grammar", "v2", "agreement", "sv_ratio", "en_regr", "layer", "factor")}) for k, v in out.items()}, ensure_ascii=False, indent=1), flush=True)
    print("DONE plant_and_fix", flush=True)


if __name__ == "__main__":
    main()