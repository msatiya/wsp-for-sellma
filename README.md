# Weight-Space Programming for SE.LLMA

Candidate assignment: investigate whether the ideas in *"Sparks of Weight-Space
Programming"* (Takehi et al., MIT CSAIL, 2026) are useful for SE.LLMA's goal of
building strong Swedish language models.

- [`docs/01_assignment-and-research-digest.md`](docs/01_assignment-and-research-digest.md) — assignment, paper digest + critique, SE.LLMA context, environment.
- [`docs/02_experiment-plan.md`](docs/02_experiment-plan.md) — hypothesis, arms, signals, repo layout.
- [`docs/03_decisions-log.md`](docs/03_decisions-log.md) — adaptations & constraints during the run.
- [`research_note.md`](research_note.md) — the 1–2 page research note (hypothesis, results, recommendation).
- [`report/report.pdf`](report/report.pdf) — the longer PDF report.
- [`results/README.md`](results/README.md) — headline numbers and pointers to raw artifacts.
- [`ai_usage.md`](ai_usage.md) — AI tools used and their contributions.

## Hypothesis (narrow, testable)

> A frontier LLM (DeepSeek-V4-Pro) editing an open model's weights directly — no
> gradients, no training data — can improve the model's Swedish behavior better
> than prompt / decoding / steering baselines, without degrading English, and the
> mechanism is a weight-localized change rather than re-weighting Swedish-token
> output logits.

## Experimental arms

`base` · `prompt` · `decode` · `logit_bias` · `random_edit` · `lora` · `wsp_zero`
(· `wsp_feedback` stretch). See the plan doc for what each control tests.

## Signals

- Grammar probes (V2 word order + gender/definite agreement), scored by
  log-likelihood (weight-level signal).
- Open-ended Swedish prompts, scored by offline language identification
  (surface signal / H2 artifact check).
- English regression set (forgetting check).

## Environment

- Any Python env with `torch>=2.6` and `transformers>=5` (the `qwen3_5` architecture
  needs torch>=2.6). We ran CPU-only; no GPU is required.
- Editor API: Alibaba Model Studio "Token Plan", OpenAI-compatible.
- (Cluster-specific notes on how we actually executed it are in `docs/03_decisions-log.md`.)

## Setup

```bash
export MODEL_STUDIO_API_KEY=...            # the token-plan key (never commit it)
export MODEL_STUDIO_BASE_URL="https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
export WSP_EDITOR_MODEL=deepseek-v4-pro-0813

# Build the pure-Python peft + pyyaml wheels (once):
bash scripts/setup_cpu_env.sh
```

## Reproduce

```bash
PYTHONPATH="$PWD/.pip_cpu2/site" python3 -u src/run_all.py \
  --model Qwen/Qwen3.5-0.8B \
  --arms base,prompt,decode,logit_bias,random_edit,lora,wsp_zero \
  --out results --cache-dir <your HF_HOME>/hub --device cpu
```

The `wsp_zero` (vague) arm and the planted-fault repair also have standalone runners:
`src/run_single.py` and `src/plant_and_fix.py`.

Results land under `results*`/; see `results/README.md`.

## Layout

```
docs/        digest + plan
src/         package (harness, eval, baselines, run_all.py)
configs/     experiment.json, probes.json
slurm/       run_experiment.slurm
results/     outputs (gitignored for large artifacts)
```