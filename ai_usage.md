# AI usage

Two kinds of tooling were used, with clearly separated roles.

**Coding assistant (opencode, running on `deepseek-v4-pro-0813`):** an execution-level assistant. It read the assignment and the *Sparks of Weight-Space Programming* source, implemented the code (harness, evaluation, baselines, orchestration) and ran the experiment. Every research decision was made by the human author, who directed the assistant throughout.

**Frontier editor `deepseek-v4-pro-0813` (Model Studio token plan):** used *inside* the experiment as the Weight-Space Programming "editor" in the `wsp_zero` and planted-fault arms — i.e. it is the *subject* being evaluated, not an author.

**Human author — all conceptual and directive decisions:** chose the hypothesis and its framing, selected the subject model and the Weight-Space Programming editor, defined the experimental arms and controls, designed and reviewed the Swedish grammar probes, set the scope (e.g. dropping GPT-SW3 after Hugging Face access could not be obtained before the deadline), decided how to handle negative results and small-sample uncertainty, interpreted the findings, and wrote the recommendation. The assistant's outputs were reviewed and revised against these decisions throughout.

No model weights or data were generated that we are not entitled to use; the subject model (Qwen3.5-0.8B) is open-weight, and the Swedish probes are synthetic/author-written.
