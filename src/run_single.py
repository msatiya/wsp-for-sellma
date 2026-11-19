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


def eval_model(model, tokenizer, device):
    from eval.probes import split_probes, OPENENDED_PROMPTS
    from eval.scoring import evaluate_mc
    from eval.regression import run_regression
    from baselines.generate import generate
    from eval.language_id import detect

    grammar = evaluate_mc(model, tokenizer, split_probes()["eval"], device)
    completions = generate(model, tokenizer, OPENENDED_PROMPTS, do_sample=False)
    labels = {}
    ratio_sum = 0.0
    samples = []
    for p, c in zip(OPENENDED_PROMPTS, completions):
        d = detect(c)
        ratio = 1.0 if d["label"] == "sv" else (0.0 if d["label"] == "en" else 0.5)
        ratio_sum += ratio
        labels[d["label"]] = labels.get(d["label"], 0) + 1
        samples.append({"prompt": p, "completion": c, "label": d["label"]})
    n = len(completions)
    openended = {
        "swedish_ratio_mean": ratio_sum / n if n else 0.0,
        "label_counts": labels,
        "samples": samples,
    }
    regression = run_regression(model, tokenizer, device)
    return {"grammar": grammar, "openended": openended, "regression": regression}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", required=True, choices=["random_edit", "wsp_zero"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    meta = {}

    if args.arm == "random_edit":
        from baselines.random_edit import random_edit

        model, tok = load_model(args.model, args.cache_dir, args.device)
        random_edit(model, frac=0.0002, sigma=1.0, seed=args.seed)
        meta = {"frac": 0.0002, "sigma": 1.0}
    else:
        from api_client import ChatClient
        from harness.agent import WspAgent
        from harness.weights_env import WeightsEnv

        cfg_base = os.environ.get("MODEL_STUDIO_BASE_URL", "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1")
        env = WeightsEnv(args.model, cache_dir=args.cache_dir, device=args.device)
        client = ChatClient(cfg_base, os.environ.get("MODEL_STUDIO_API_KEY", ""), os.environ.get("WSP_EDITOR_MODEL", "deepseek-v4-pro-0813"))
        agent = WspAgent(client)
        task = (
            "The loaded model is a language model. Improve its Swedish grammar and its "
            "tendency to answer in Swedish when prompted in Swedish, without harming its "
            "English. Inspect weights (statistics/slices only), edit weights directly in "
            "MLP or attention layers rather than the output head. Then print a short "
            "summary of changed parameter names and counts."
        )
        result = agent.run(task, env, max_iters=5, max_tokens=4096, allow_forward=False)
        Path(args.out).mkdir(parents=True, exist_ok=True)
        with open(Path(args.out) / "wsp_transcript.json", "w") as f:
            json.dump(result["transcript"], f, indent=2, ensure_ascii=False)
        with open(Path(args.out) / "wsp_diff.json", "w") as f:
            json.dump(result["diff"], f, indent=2)
        model, tok = env.model, env.tokenizer
        meta = {"diff": result["diff"], "tokens": result["total_tokens"], "final": result["final"]}

    result = {"eval": eval_model(model, tok, args.device), "meta": meta}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out) / "results.json", "w") as f:
        json.dump({args.arm: result}, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in result["eval"].items() if k != "regression"}, ensure_ascii=False)[:200], flush=True)
    print("DONE", args.arm, flush=True)


if __name__ == "__main__":
    main()