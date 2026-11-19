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

## 4. Results

*(pending — job 5516819)*

| Arm | grammar acc | sv-ratio (open) | en-regr acc |
|---|---|---|---|
| base | — | — | — |
| prompt | — | — | — |
| decode | — | — | — |
| logit_bias | — | — | — |
| random_edit | — | — | — |
| lora | — | — | — |
| wsp_zero | — | — | — |

Key questions the results answer: does `wsp_zero` move the grammar signal above `base`/`random_edit` (weight-level evidence), or only the `sv-ratio` (surface), and does it match `logit_bias`/`decode` (artifact)?

## 5. Recommendation

*(pending results)* — provisional framing: if WSP's only reproducible benefit is language gating reducible to logit/decoding steering, SE.LLMA should **not** prioritise WSP for capability, but the *auditing/interpretability* (read) side may still merit a look given their transparency/governance goals.