# AI usage

This work was carried out with an interactive coding agent (opencode, running on
`deepseek-v4-pro-0813`) and a frontier "editor" LLM (`deepseek-v4-pro-0813` via
Alibaba Cloud Model Studio) used **inside** the experiment itself.

- **opencode / deepseek-v4-pro-0813 (assistant)**: digested the assignment PDF and the
  *Sparks of Weight-Space Programming* source; proposed and implemented this repository's
  experiment (hypothesis, arms, controls, harness, evaluation, baselines, orchestration,
  probes), and drafted the research note, report and docs. The human author set the
  direction at each decision point (hypothesis, model choice, experiment framing),
  supplied compute/API access, and reviewed/approved the Swedish probes and all claims.
- **Frontier editor `deepseek-v4-pro-0813` (Model Studio token plan)**: used as the
  Weight-Space Programming "editor" in the `wsp_zero` and planted-fault arms — i.e. it is
  the *subject* being evaluated inside the experiment, not the author.
- Human author: chose the hypothesis and conscious scope (e.g. dropping GPT-SW3 after HF
  gating), reviewed the Swedish grammar probes, and approved the final conclusions and
  recommendation.

No model weights or data were generated that we are not entitled to use; the subject
model (Qwen3.5-0.8B) is open-weight, and the Swedish probes are synthetic/author-written.