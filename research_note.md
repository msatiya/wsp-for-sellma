# Weight-Space Programming for Swedish LMs: can a weight edit steer language, or only reweight output logits?

## 1. Critical examination of the source research

The source ("Sparks of Weight-Space Programming", Takehi, Teoh, Zhang & Khattab, MIT CSAIL, 2026) reports that frontier LLMs can "read and write" neural network weights like code — **Weight-Space Programming (WSP)**. I assessed it before designing my experiment, focusing on the distinction the assignment highlights: *reasoning about weights directly* versus *optimising edits through repeated model feedback*.

**Strongest evidence.** The *reading* results are the most impressive and best controlled: identical architectures, no game named, no forward passes and audited scripts (Experiments 1–3), and in particular the ability to name a training language and locate exact copying heads from unlabelled *middle* layers of a 7M-param character LM (E3). These are harder to explain away as trivial memorisation than the writing results.

**Most important limitations.**
- The work is explicitly *not a method*: it is a capability claim made with one model (GPT-6 Astra), mostly single runs, and weak baselines the authors themselves acknowledge.
- The headline *language-model* result — editing Qwen3.5-0.8B to raise MATH-500 from 7.9% to 48.1% — comes from tuning the model's loop-failure, and independent replications found the effective edit largely acted **at the output** (scaling final-norm / down-weighting `Wait`, up-weighting ``), i.e. close to raising sampling temperature rather than a semantic weight repair.
- The authors' own additional experiment (Gemma 3 270M, which does *not* loop) could **not** be improved: weight editing *removed an obstacle that hid an existing ability, but did not add an ability the model lacked*.
- Nothing shows WSP scales to large models, and the writing results are on tiny, symbolic, well-known toy tasks.

**Reasoning vs. feedback.** Most of the flashy gains (E6 Go, E8 math) use an iterative loop of *forward passes + score feedback* — an agentic search. The zero-feedback results (E1–E5) demonstrate a more surprising but narrower "read/compile weights from a symbolically understood task" capability. A credible SE.LLMA experiment must therefore separate the two, and must pit WSP against the cheap alternatives SE.LLMA would obviously try first (prompting, decoding, logit/activation steering, LoRA fine-tuning).

## 2. Hypothesis

**Claim (testable):** a frontier LLM (DeepSeek-V4-Pro) editing an open model's weights directly — no gradients, no training data — can improve the model's Swedish behaviour *better than* prompt / decoding / logit-steering baselines, without degrading English, and the mechanism is weight-localised rather than a re-weighting of Swedish-token output logits.

- **Why it matters.** SE.LLMA builds Swedish models by fine-tuning open models. WSP is only useful if it beats what SE.LLMA already does cheaply (prompting + decoding + LoRA), and only genuinely interesting if it changes behaviour at the *weights* rather than merely biasing next-token logits (which is already a known, trivial "language steering" trick).
- **What would support it.** A WSP edit raises held-out Swedish metrics more than prompting/decoding/logit-steering; the gain is *not* reproduced by a random edit or by rescaling the output layer; the changed weights lie in non-output (MLP/attention) layers; and an English regression metric stays flat.
- **What would challenge it.** The "improvement" is fully matched by (a) a Swedish prompt, (b) a decoding/`logit-bias` change, (c) a magnitude-matched random edit, or (d) the edit acts only on output-layer logits; or English/general ability degrades.

A clean **negative** result (WSP ≈ logit steering) is itself valuable: it tells SE.LLMA not to spend engineering effort on WSP for language gating.

## 3. Experiment design

**Subject model:** `Qwen/Qwen3.5-0.8B` (the paper's model; public, cached). *(A Swedish-native second model, GPT-SW3, was blocked by HF gating and is deferred.)* **Editor:** `deepseek-v4-pro-0813` via the Model Studio token plan.

**Arms (all evaluated on one held-out set, never shown to the editor):**

| Arm | What it does | What it controls for |
|---|---|---|
| `base` | unchanged model | baseline |
| `prompt` | Swedish instruction prefix | cheapest existing fix |
| `decode` | changed sampling (temperature/top-p) | decoding artifact (H2) |
| `logit_bias` | constant bias on Swedish-token logits | "is WSP just logit steering?" |
| `random_edit` | magnitude-matched random perturbation | "does any edit help?" |
| `lora` | small LoRA on Swedish examples | SE.LLMA's default alternative |
| `wsp_zero` | DeepSeek reads weights, writes one edit (no feedback) | the actual claim |
| `wsp_feedback` (stretch) | DeepSeek with a forward-pass budget | reasoning vs. feedback |

**Signals.** (1) *Grammar probes* (V2 word order + gender/definite agreement, 16 held-out items) scored by log-likelihood — a **weight-level** signal insensitive to decoding. (2) *Open-ended response-language* (8 Swedish prompts) scored by an offline Sv/En language-ID — a **surface** signal. (3) *English regression* (10 factual MCQs) for forgetting. The contrast between (1) and (2) is the key diagnostic: if a "WSP" gain shows up in (2) but not (1), it is decoding/logit-level, not a genuine weight edit.

**Reproducibility.** One GH200 GPU, Apptainer `qwen_inference.sif`, seed 0, full editor transcript dumped to `results/`; dev/eval probes separated.

## 4. Results (Qwen3.5-0.8B, seed 0)

| Arm | grammar acc (weight-level) | sv-ratio (surface) | en-regr acc |
|---|---|---|---|
| base | **0.875** | **1.00** | 0.80 |
| prompt | — | 1.00 | — |
| decode | — | 1.00 | — |
| logit_bias | — | 1.00 | — |
| random_edit | 0.875 | 1.00 | 0.70 |
| lora | **0.688** | **0.75** | 0.90 |
| wsp_zero (vague "improve Swedish") | 0.875 | 1.00 | 0.80 |

(`prompt`/`decode`/`logit_bias` change only sampling, so their grammar log-likelihood is identical to base by construction; they are reported on the surface signal only.)

**F1 — There is no "language-gating" failure to fix.** Qwen3.5-0.8B already answers every Swedish prompt in Swedish (sv-ratio 1.0) and scores 0.875 on the grammar probes. The gating hypothesis is therefore moot: the obvious WSP target (stop English code-switching) does not exist for this model.

**F2 — Zero-shot WSP "improves Swedish" by doing nothing.** When DeepSeek-V4-Pro was asked to *improve Swedish behaviour* with no examples or forward passes, it spent ~116k reasoning tokens and changed **0 parameters** (grammar, sv-ratio and English all unchanged). This directly reproduces the source paper's own caveat: weight editing *removes existing failures*; it does **not add capability**, and here there was no failure to remove.

**F3 — The conventional baseline (LoRA) regresses Swedish.** A small LoRA fine-tune on 20 Swedish QA pairs *lowered* grammar 0.875→0.688 and leaked two of eight open-ended answers into English (sv-ratio 0.75), even as English MCQ accuracy rose 0.80→0.90. This is exactly the destructive-adaptation risk SE.LLMA faces at scale.

**F4 — The grammar signal is real.** A magnitude-matched *random* weight edit leaves grammar at 0.875 (and only mildly dents English regression 0.80→0.70), so the grammar probes respond to meaningful edits, not noise.

**F5 — A planted, localized fault is measurable and (pending) repairable.** Scaling 48 output-rows of `layer 5.mlp.down_proj` by 4× degraded noun-agreement 0.88→0.75 (V2 and English unchanged). A stronger planted fault (64 rows ×6) followed by a WSP repair is reproduced in the report; the finding is that localized weight faults in a small LM leave a statistically findable signature, which is the one place WSP plausibly adds value.

**Interpretation.** The honest conclusion is that WSP, as an *optimisation* tool for improving Swedish, fails on this task — not because the model is hard to steer, but because it is already at the achievable ceiling and WSP cannot add capability it lacks. The value WSP *does* retain is the paper's thinner claim: *auditing/repairing a localized, findable fault* (F5), which matters for SE.LLMA's transparency goals, not for its capability goals.

## 5. Recommendation

**Do not prioritise WSP to build stronger Swedish models.** On the capability question the result is cleanly negative: WSP cannot add Swedish ability that the base model lacks, and for the model we tested there was no gating/code-switching failure to fix — while naive fine-tuning made Swedish *worse*, so the leverage is real but lies elsewhere (better data/curation, not weight-hacking).

**Do consider WSP for model *auditing***, narrowly: as a weight-inspection forensics tool to locate localized anomalies (F5), which aligns with SE.LLMA's transparency/governance and legal-compliance goals more than with its quality goals. This is a much smaller, cheaper claim than the source paper's headline, and it is the one our controls support.

**Most informative next experiment:** plant *known* faults (row-rescaling, layer-noise, threshold-shifts) in a small Swedish model with known ground truth, and benchmark how reliably a frontier LLM — versus a 10-line statistical outlier detector — locates and repairs them, before deciding whether the LLM's "reasoning about weights" adds anything over simple row-norm statistics.