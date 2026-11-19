# Weight-Space Programming for SE.LLMA

Candidate assignment: investigate whether the ideas in *"Sparks of Weight-Space
Programming"* (Takehi et al., MIT CSAIL, 2026) are useful for SE.LLMA's goal of
building strong Swedish language models.

- [`docs/01_assignment-and-research-digest.md`](docs/01_assignment-and-research-digest.md) — assignment, paper digest + critique, SE.LLMA context, environment.
- [`docs/02_experiment-plan.md`](docs/02_experiment-plan.md) — hypothesis, arms, signals, repo layout.

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

- Cluster `arrhenius` (NAISS). GPU nodes are **ARM64**; use the `qwen_inference.sif`
  Apptainer container on compute nodes (see `slurm/run_experiment.slurm`).
- Shared HF cache (Qwen3.5-0.8B already present):
  `/nobackup/proj/disk/globalpoverty1/personal/satmur/hf_cache`.
- Editor API: Alibaba Model Studio "Token Plan", OpenAI-compatible.

## Setup

```bash
# API credentials (never commit `.env`)
cp .env.example .env   # then fill in MODEL_STUDIO_API_KEY
export $(grep -v '^#' .env | xargs)
```

## Reproduce

Prepare (and, on approval, submit) the SLURM job:

```bash
export MODEL=Qwen/Qwen3.5-0.8B
export ARMS=base,prompt,decode,logit_bias,random_edit,lora,wsp_zero
sbatch slurm/run_experiment.slurm
```

Results land in `results/<model>/results.json`; editor transcripts in
`results/<model>/wsp_zero/wsp_transcript.json`.

## Layout

```
docs/        digest + plan
src/         package (harness, eval, baselines, run_all.py)
configs/     experiment.json, probes.json
slurm/       run_experiment.slurm
results/     outputs (gitignored for large artifacts)
```