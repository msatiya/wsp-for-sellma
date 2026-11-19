import argparse
import json
import os
from pathlib import Path

import torch


def load_model(model_id, cache_dir, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, cache_dir=cache_dir, low_cpu_mem_usage=True
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def run_grammar(model, tokenizer, device):
    from eval.probes import split_probes
    from eval.scoring import evaluate_mc

    return evaluate_mc(model, tokenizer, split_probes()["eval"], device)


def run_openended(model, tokenizer, device, prefix=None, do_sample=False, logits_processor=None):
    from baselines.generate import generate
    from eval.language_id import detect
    from eval.probes import OPENENDED_PROMPTS

    completions = generate(
        model,
        tokenizer,
        OPENENDED_PROMPTS,
        do_sample=do_sample,
        temperature=0.6,
        top_p=0.95,
        prompt_prefix=prefix,
        logits_processor=logits_processor,
    )
    samples = []
    labels = {}
    ratio_sum = 0.0
    for p, c in zip(OPENENDED_PROMPTS, completions):
        d = detect(c)
        ratio = 1.0 if d["label"] == "sv" else (0.0 if d["label"] == "en" else 0.5)
        ratio_sum += ratio
        labels[d["label"]] = labels.get(d["label"], 0) + 1
        samples.append({"prompt": p, "completion": c, "label": d["label"]})
    n = len(completions)
    return {
        "swedish_ratio_mean": ratio_sum / n if n else 0.0,
        "label_counts": labels,
        "samples": samples,
    }


def run_regression(model, tokenizer, device):
    from eval.regression import run_regression as _r

    return _r(model, tokenizer, device)


def eval_model(model, tokenizer, device):
    return {
        "grammar": run_grammar(model, tokenizer, device),
        "openended": run_openended(model, tokenizer, device),
        "regression": run_regression(model, tokenizer, device),
    }


def run_wsp_arm(model_id, cache_dir, device, allow_forward, api_cfg, out_dir, max_iters):
    from api_client import ChatClient
    from harness.agent import WspAgent
    from harness.weights_env import WeightsEnv

    env = WeightsEnv(model_id, cache_dir=cache_dir, device=device)
    client = ChatClient(api_cfg.base_url, api_cfg.api_key, api_cfg.model)
    agent = WspAgent(client)
    task = (
        "The loaded model is a language model. Improve its Swedish language behavior: "
        "make it more likely to answer in Swedish and produce grammatical Swedish "
        "when the user writes in Swedish, without harming its English. "
        "Reason about which weights encode language behavior and edit them directly. "
        "Prefer edits in the MLP or attention weights over the output head. "
        "After editing, print a short summary of changed parameter names and counts."
    )
    if allow_forward:
        task += " You may call forward([prompt, ...]) to see model outputs and iterate."
    result = agent.run(task, env, max_iters=max_iters, allow_forward=allow_forward)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "wsp_transcript.json", "w") as f:
        json.dump(result["transcript"], f, indent=2, ensure_ascii=False)
    with open(out_dir / "wsp_diff.json", "w") as f:
        json.dump(result["diff"], f, indent=2)
    return env.model, env.tokenizer, result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--arms", default="base,prompt,decode,logit_bias,random_edit,lora,wsp_zero")
    ap.add_argument("--out", default="results")
    ap.add_argument("--cache-dir", default=os.environ.get("HF_CACHE"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--wsp-max-iters", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    from config import load_config

    cfg = load_config("configs/experiment.json")
    out_dir = Path(args.out) / args.model.replace("/", "__")
    out_dir.mkdir(parents=True, exist_ok=True)

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    results = {}

    base_model, base_tok = load_model(args.model, args.cache_dir, args.device)

    for arm in arms:
        print(f"== {arm} ==", flush=True)
        meta = {}
        if arm == "base":
            model, tok = base_model, base_tok
        elif arm == "prompt":
            model, tok = base_model, base_tok
            res = run_openended(model, tok, args.device, prefix="Svara på svenska. ")
            results[arm] = {"openended": res}
            continue
        elif arm == "decode":
            model, tok = base_model, base_tok
            res = run_openended(model, tok, args.device, do_sample=True)
            results[arm] = {"openended": res}
            continue
        elif arm == "logit_bias":
            from baselines.logit_bias import swedish_token_ids, TokenBiasProcessor

            model, tok = base_model, base_tok
            ids = swedish_token_ids(tok)
            proc = TokenBiasProcessor(ids, 5.0)
            res = run_openended(model, tok, args.device, logits_processor=[proc])
            meta = {"n_bias_tokens": len(ids)}
            results[arm] = {"openended": res, "meta": meta}
            continue
        elif arm == "random_edit":
            from baselines.random_edit import random_edit

            model, tok = load_model(args.model, args.cache_dir, args.device)
            random_edit(model, frac=0.0002, sigma=1.0, seed=args.seed)
            meta = {"frac": 0.0002, "sigma": 1.0}
        elif arm == "lora":
            from baselines.lora import lora_finetune
            from baselines.swedish_examples import as_prompt_completion_pairs

            model, tok = load_model(args.model, args.cache_dir, args.device)
            out = lora_finetune(model, tok, as_prompt_completion_pairs())
            model = out["model"]
            meta = {"n_lora_params": out["n_lora_params"]}
        elif arm in ("wsp_zero", "wsp_feedback"):
            model, tok, wsp = run_wsp_arm(
                args.model,
                args.cache_dir,
                args.device,
                allow_forward=(arm == "wsp_feedback"),
                api_cfg=cfg.api,
                out_dir=out_dir / arm,
                max_iters=args.wsp_max_iters,
            )
            meta = {
                "diff": wsp["diff"],
                "tokens": wsp["total_tokens"],
                "final": wsp["final"],
            }
        else:
            raise ValueError(arm)
        results[arm] = {"eval": eval_model(model, tok, args.device), "meta": meta}

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n===== SUMMARY =====")
    for arm, r in results.items():
        ev = r.get("eval")
        if ev:
            print(
                f"{arm:12s} | grammar {ev['grammar']['accuracy']:.3f} "
                f"| sv-ratio {ev['openended']['swedish_ratio_mean']:.2f} "
                f"| en-regr {ev['regression']['accuracy']:.3f}"
            )
        else:
            print(
                f"{arm:12s} | sv-ratio {r['openended']['swedish_ratio_mean']:.2f} "
                f"(no grammar/regression)"
            )


if __name__ == "__main__":
    main()