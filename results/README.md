# Results

All numbers for `Qwen/Qwen3.5-0.8B`, seed 0, CPU-only (torch 2.13.0+cu130,
transformers 5.15.1).

**Model revision:** snapshot `2fc06364715b967f1860aea9cf38778875588b17` (HF cache
`models--Qwen--Qwen3.5-0.8B/snapshots/`).

## Headline (counts in parentheses)

| Arm | grammar acc (weight-level) | sv-ratio (surface) | en-regr acc |
|---|---|---|---|
| base | 0.875 (14/16) | 1.00 (8/8) | 0.80 (8/10) |
| prompt | — | 1.00 | — |
| decode | — | 1.00 | — |
| logit_bias | — | 1.00 | — |
| random_edit | 0.875 (14/16) | 1.00 | 0.70 (7/10) |
| lora | 0.688 (11/16) | 0.75 (6/8) | 0.90 (9/10) |
| wsp_zero (vague) | 0.875 | 1.00 | 0.80 |

## Planted-fault sub-experiment

Scaling `model.layers.5.mlp.down_proj.weight` rows changes a specific behaviour without
harming English:

| run | plant | effect |
|---|---|---|
| results_cpu_plant | 48 rows x4 | agreement 0.88 -> 0.75 (V2 unaffected) |
| results_cpu_plantfix | 64 rows x6 | V2 0.88 -> 0.75; WSP edited exactly 229,376 params (the 64 rows) but did **not** restore V2 (still 0.75) |

## Raw data

- `results_cpu/Qwen__Qwen3.5-0.8B/results.json` — base/prompt/decode/logit_bias/lora (+ errored random_edit)
- `results_cpu_random/results.json` — random_edit (corrected run)
- `results_cpu_wsp/results.json` + `wsp_transcript.json` + `wsp_diff.json` — vague "improve Swedish" WSP run (0 params changed, ~116k tokens)
- `results_cpu_plant/results.json` — plant-only (48 rows x4)
- `results_cpu_plantfix/results.json` + `wsp_transcript.json` + `wsp_diff.json` — plant + WSP repair (64 rows x6)

Open-ended completions (the "evaluation examples") are inside each `results.json` under
`eval.openended.samples`; grammar probe-level predictions under `eval.grammar.results`.