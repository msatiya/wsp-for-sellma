import json
import re

from api_client import ChatClient

CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)

SYSTEM_PROMPT = (
    "You are an expert at Weight-Space Programming: reading and editing the raw "
    "weights of a neural network directly, without gradient descent or training. "
    "You are given a language model already loaded in a Python sandbox as the object "
    "`model`, along with `torch`, `np` (numpy), `tokenizer`, and a helper `diff()` "
    "that reports which weight tensors differ from the original snapshot.\n"
    "You MUST work by writing Python inside fenced code blocks (```python ... ```). "
    "Each block is executed in the sandbox and its stdout is returned to you. "
    "You may inspect any weights you like (their `.data`, `.shape`, `.dtype`, "
    "statistics, slices) and modify them in place with normal PyTorch/numpy. "
    "Do NOT call gradient descent, a training loop, or an optimizer. "
    "Do NOT download anything. When you are finished, make your edits, print a short "
    "summary of what you changed (parameter names, indices, counts), and end your "
    "message with no further code blocks so the harness finalizes.\n"
    "Keep outputs small: print statistics and short slices, never entire tensors."
)


def extract_code_blocks(text):
    return [m.group(1) for m in CODE_BLOCK_RE.finditer(text)]


class WspAgent:
    def __init__(self, client: ChatClient):
        self.client = client
        self.total_tokens = 0

    def run(self, task, env, max_iters=10, max_tokens=8192, allow_forward=False):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]
        transcript = []
        extra_ns = {"forward": env.forward} if allow_forward else {}
        final_content = ""
        for _ in range(max_iters):
            resp = self.client.chat(messages, max_tokens=max_tokens)
            content = resp["content"]
            self.total_tokens += resp["usage"].get("total_tokens", 0)
            transcript.append(
                {
                    "role": "assistant",
                    "content": content,
                    "reasoning_len": len(resp["reasoning"]),
                    "finish_reason": resp["finish_reason"],
                }
            )
            blocks = extract_code_blocks(content)
            if not blocks:
                final_content = content
                break
            tool_results = []
            for code in blocks:
                res = env.exec_code(code, extra_ns=extra_ns)
                tool_results.append({"code": code, **res})
                transcript.append({"role": "tool", **res, "code_len": len(code)})
            messages.append({"role": "assistant", "content": content})
            messages.append(
                {
                    "role": "user",
                    "content": "Sandbox output:\n" + json.dumps(tool_results, indent=1),
                }
            )
        return {
            "final": final_content,
            "transcript": transcript,
            "total_tokens": self.total_tokens,
            "diff": env.diff(),
        }