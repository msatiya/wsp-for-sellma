import torch


def lora_finetune(
    model,
    tokenizer,
    examples,
    epochs=5,
    lr=2e-4,
    batch_size=4,
    lora_r=8,
    lora_alpha=16,
    max_len=64,
):
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.train()

    texts = [f"Fråga: {e['prompt']}\nSvar: {e['completion']}" for e in examples]
    batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    input_ids = batch["input_ids"].to(model.device)
    attention_mask = batch["attention_mask"].to(model.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)

    for _ in range(epochs):
        for i in range(0, input_ids.shape[0], batch_size):
            inp = input_ids[i : i + batch_size]
            attn = attention_mask[i : i + batch_size]
            logits = model(input_ids=inp, attention_mask=attn).logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = inp[:, 1:].contiguous()
            shift_attn = attn[:, 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
            )
            loss = (loss * shift_attn.view(-1)).sum() / shift_attn.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    model = model.merge_and_unload()
    return {"model": model, "n_lora_params": n_params}