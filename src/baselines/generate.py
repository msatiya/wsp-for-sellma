import torch


def generate(
    model,
    tokenizer,
    prompts,
    max_new_tokens=64,
    do_sample=False,
    temperature=1.0,
    top_p=1.0,
    top_k=50,
    prompt_prefix=None,
    logits_processor=None,
):
    if prompt_prefix:
        prompts = [prompt_prefix + p for p in prompts]
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(
        model.device
    )
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs.update({"temperature": temperature, "top_p": top_p, "top_k": top_k})
    with torch.no_grad():
        out = model.generate(
            **enc, logits_processor=logits_processor, **gen_kwargs
        )
    n_in = enc["input_ids"].shape[1]
    return tokenizer.batch_decode(out[:, n_in:], skip_special_tokens=True)