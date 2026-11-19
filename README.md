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

- Cluster `arrhenius` (NAISS). The GPU nodes are **ARM64**; the x86 login node can run the
  experiment on CPU via an x86 Apptainer image (`pipeline-cpu.sif`, torch 2.13 + transformers 5.15).
  A GPU path also exists (`slurm/run_experiment.slurm`, `qwen_inference.sif`) but was **not
  validated** (the queue never scheduled it) — the CPU path below is the tested one.
- Shared HF cache (Qwen3.5-0.8B already present):
  `/nobackup/proj/disk/globalpoverty1/personal/satmur/hf_cache`.
- Editor API: Alibaba Model Studio "Token Plan", OpenAI-compatible.

## Setup

```bash
export MODEL_STUDIO_API_KEY=...            # the token-plan key (never commit it)
export MODEL_STUDIO_BASE_URL="https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
export WSP_EDITOR_MODEL=deepseek-v4-pro-0813

# Build the pure-Python peft + pyyaml wheels for the CPU image (once):
bash scripts/setup_cpu_env.sh
```

## Reproduce (CPU, login node)

The heavy lifting uses an x86 Apptainer image that already ships torch>=2.6 + transformers
(the `qwen3_5` architecture needs torch>=2.6, which newer transformers require). `peft` +
`pyyaml` live as unpacked wheels under `.pip_cpu2/site` (see `scripts/setup_cpu_env.sh`).

```bash
PTC=/nobackup/proj/disk/globalpoverty1/personal/audinet/SAE-scrubbing/apptainer/containers/pipeline-cpu.sif
HF_CACHE=/nobackup/proj/disk/globalpoverty1/personal/satmur/hf_cache
apptainer exec --home "$HF_CACHE" --env HF_HOME="$HF_CACHE" --env XDG_CACHE_HOME="$HF_CACHE" \
  --env PYTHONPATH="$PWD/.pip_cpu2/site" --env OMP_NUM_THREADS=16 \
  --bind "$HF_CACHE:/hf_cache" --bind "$PWD:$PWD" "$PTC" \
  python3 -u src/run_all.py --model Qwen/Qwen3.5-0.8B \
  --arms base,prompt,decode,logit_bias,random_edit,lora,wsp_zero \
  --out results --cache-dir /hf_cache/hub --device cpu
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