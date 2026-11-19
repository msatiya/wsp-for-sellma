import torch


def seq_logprob(model, tokenizer, text, device, add_bos=False):
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=add_bos)
    ids = enc["input_ids"].to(device)
    if ids.shape[1] < 2:
        return 0.0
    with torch.no_grad():
        logits = model(ids).logits
    logprobs = torch.log_softmax(logits, dim=-1)
    targets = ids[:, 1:]
    preds = logprobs[:, :-1, :]
    tok_lp = preds.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    return float(tok_lp.sum().item())


def score_mc(model, tokenizer, probe, device):
    prompt = probe["prompt"]
    ll_prompt = seq_logprob(model, tokenizer, prompt, device)
    option_scores = []
    for opt in probe["options"]:
        ll_full = seq_logprob(model, tokenizer, prompt + opt, device)
        option_scores.append(ll_full - ll_prompt)
    best = int(max(range(len(option_scores)), key=lambda i: option_scores[i]))
    return {
        "id": probe["id"],
        "category": probe.get("category"),
        "correct": probe["correct"],
        "predicted": best,
        "hit": best == probe["correct"],
        "scores": option_scores,
    }


def evaluate_mc(model, tokenizer, probes, device):
    results = [score_mc(model, tokenizer, p, device) for p in probes]
    hits = sum(1 for r in results if r["hit"])
    n = len(results)
    acc = hits / n if n else 0.0
    return {"accuracy": acc, "hits": hits, "n": n, "results": results}