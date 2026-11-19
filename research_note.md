# Weight-Space Programming for Swedish LMs: can a weight edit steer language, or only reweight output logits?

**TL;DR.** We tested whether a frontier LLM editing an open model's weights directly (no gradients, no data) can improve Swedish on `Qwen/Qwen3.5-0.8B`. It cannot: the base model already answers Swedish perfectly, a zero-shot "improve Swedish" edit changed **0 parameters**, and a planted localized fault was *located* by the editor but not *repaired*. Meanwhile a small LoRA fine-tune made Swedish *worse* (English leakage, degenerate repeats). **Recommendation: do not prioritise WSP for SE.LLMA's capability goal**; its auditing value is no better than a row-norm statistic.

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

**Subject model:** `Qwen/Qwen3.5-0.8B` (the paper's model; public, cached). We originally planned a Swedish-native second subject (AI Sweden's GPT-SW3), but Hugging-Face access could not be obtained before the deadline, so the experiment uses Qwen alone. **Editor:** `deepseek-v4-pro-0813` via the Model Studio token plan.

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
| `wsp_feedback` *(not run)* | DeepSeek with a forward-pass budget | reasoning vs. feedback |

**Signals.** (1) *Grammar probes* (V2 word order + gender/definite agreement, 16 held-out items) scored by log-likelihood — a **weight-level** signal insensitive to decoding. (2) *Open-ended response-language* (8 Swedish prompts) scored by an offline Sv/En language-ID — a **surface** signal. (3) *English regression* (10 factual MCQs) for forgetting. The contrast between (1) and (2) is the key diagnostic: if a "WSP" gain shows up in (2) but not (1), it is decoding/logit-level, not a genuine weight edit.

**Reproducibility.** CPU-only (no GPU); torch 2.13.0, transformers 5.15.1; model revision `2fc0636…`; seed 0; full editor transcripts in `results/`; dev/eval probes separated (dev was never used).

## 4. Results (Qwen3.5-0.8B, seed 0)

| Arm | grammar acc (weight-level) | sv-ratio (surface) | en-regr acc |
|---|---|---|---|
| base | **0.875 (14/16)** | **1.00 (8/8)** | 0.80 (8/10) |
| prompt | — | 1.00 | — |
| decode | — | 1.00 | — |
| logit_bias | — | 1.00 | — |
| random_edit | 0.875 | 1.00 | 0.70 (7/10) |
| lora | **0.688 (11/16)** | **0.75 (6/8)** | 0.90 (9/10) |
| wsp_zero (vague "improve Swedish") | 0.875 | 1.00 | 0.80 |

(`prompt`/`decode`/`logit_bias` change only sampling, so their grammar log-likelihood is identical to base by construction; they are reported on the surface signal only.)

**F1 — There is no "language-gating" failure to fix.** Qwen3.5-0.8B already answers every Swedish prompt in Swedish (sv-ratio 8/8) and scores 0.875 on the grammar probes. The gating hypothesis is therefore moot: the obvious WSP target (stop English code-switching) does not exist for this model.

**F2 — Zero-shot WSP "improves Swedish" by doing nothing.** When DeepSeek-V4-Pro was asked to *improve Swedish behaviour* with no examples or forward passes, it spent ~116k tokens and changed **0 parameters** (grammar, sv-ratio and English all unchanged). This directly reproduces the source paper's own caveat: weight editing *removes existing failures*; it does **not add capability**, and here there was no failure to remove.

**F3 — The conventional baseline (LoRA) degrades Swedish output.** A LoRA fine-tune on 20 Swedish QA pairs produces qualitatively worse Swedish: two of eight open-ended answers switch to English (sv-ratio 6/8) and several answers degenerate into the model repeating its tiny training set or emitting a multiple-choice list, e.g. *"Vem skrev Pippi Långstrump?"* → *"A. Hans Christian Andersen. B. Oscar Wilde. C. Jane Austen…"*. By contrast the base model answers the same prompt in Swedish. Grammar also drops 14/16→11/16, but (see below) the item count is too small to call this individually significant; the regression is best read as *directional and qualitative*.

**F4 — The grammar signal is real but no edit moves it usefully.** A magnitude-matched *random* edit leaves grammar at 0.875 (14/16) — so the probes respond to the meaningful changes LoRA imposes, not to weight noise in general.

**F5 — A planted, localized fault is found, but WSP cannot repair it.** Scaling 48 output-rows of `layer 5.mlp.down_proj` by 4× degraded noun-agreement 0.88→0.75; scaling 64 rows by 6× degraded V2 0.88→0.75. Given the corrupted weights, the agent identified the right tensor and edited exactly those 64 rows (229,376 params, `down_proj` only) — but its repair left V2 unchanged (still 0.75). It *located* the fault yet could *not restore* the behaviour.

**Interpretation.** As an *optimisation* tool for improving Swedish, WSP fails — not because the model is hard to steer, but because it is already at the achievable ceiling and WSP cannot add capability it lacks. Its *auditing* claim is real but shallow: it points at anomalous weights, but a ten-line row-norm statistic does the same, and the frontier model's "repair" produced no measurable improvement.

**Statistical power & limitations.** These are small-n, single-run results, stated with counts so their strength is visible, not as precise estimates.
- Single seed and single model; the grammar probe set is 16 items, open-ended 8, English 10, so one-item differences dominate. LoRA-vs-base grammar (14 vs 11, 3 discordant, exact McNemar p≈0.25) and the planted-fault effects (7/8→6/8) are *not* individually significant at α=0.05; the conclusions rest on the *qualitative* degeneration (English leakage, degenerate repeats, empty WSP edit, unchanged-after-repair) rather than on any single headline number.
- The Swedish probes are small, author-written and synthetic; the offline Sv/En classifier is a lexicon/count heuristic, adequate for unambiguous full-answer language but not for mixed code-switching.
- One LoRA training example ("Vad heter Sveriges huvudstad?") is semantically close to one eval prompt ("Vad är huvudstaden i Sverige?"); this cannot have inflated the LoRA result since LoRA regressed anyway.
- The `wsp_feedback` (reasoning-vs-feedback) arm was planned but not run within budget; the distinction is addressed analytically (§1). Only zero-feedback edits were tested.
- Everything ran on CPU; Qwen3.5-0.8B is the source paper's exact model and was public/cached (GPT-SW3 was dropped due to HF gating).

## 5. Recommendation

**Do not prioritise WSP to build stronger Swedish models.** On the capability question the result is cleanly negative: WSP cannot add Swedish ability the base model lacks, and for the model we tested there was no gating/code-switching failure to fix — while naive fine-tuning made Swedish *worse*, so the leverage is real but lies elsewhere (data curation and evaluation, not weight-hacking).

**Treat WSP's auditing claim with caution.** It can point at anomalous weights (F5), but a trivial row-norm statistic does the same, and the frontier model's "repair" produced no measurable behaviour change. The auditing angle is worth no more than a cheap heuristic at current reliability, and does not justify engineering investment.

**Most informative next experiment:** plant *known* faults with ground truth and run a head-to-head — frontier LLM vs a ten-line outlier detector — measuring (a) fault *localisation* accuracy and (b) whether any repair actually restores behaviour. That would quantify whether "reasoning about weights" adds anything over statistics before SE.LLMA commits any effort.