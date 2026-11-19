# AI usage

This work was carried out with an interactive coding agent (opencode, running on
`deepseek-v4-pro-0813`) and a frontier "editor" LLM (`deepseek-v4-pro-0813` via
Alibaba Cloud Model Studio) used **inside** the experiment itself.

- **opencode / deepseek-v4-pro-0813 (assistant)**: digested the assignment PDF and the
  *Sparks of Weight-Space Programming* source; produced this repository's experiment
  code (harness, evaluation, baselines, orchestration), the SLURM scripts, and the
  write-up. All research decisions (hypothesis, experimental arms, controls, probe
  design) were made and reviewed collaboratively with the human author, who supplied
  compute/API access and the final judgment calls.
- **Frontier editor `deepseek-v4-pro-0813` (Model Studio token plan)**: used as the
  Weight-Space Programming "editor" in the `wsp_zero` / `wsp_feedback` arms — i.e. it is
  the *subject* being evaluated, not the author.
- Human author: designed and validated the Swedish grammar probes, chose the hypothesis,
  and will interpret the results and write the recommendation.

No model weights or data were generated that we are not entitled to use; the subject
model (Qwen3.5-0.8B) is open-weight, and the Swedish probes are synthetic/author-written.