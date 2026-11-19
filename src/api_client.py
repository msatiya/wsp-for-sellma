import json
import urllib.error
import urllib.request


class ApiError(RuntimeError):
    pass


class ChatClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 600):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def _request(self, path: str, payload: dict):
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise ApiError(f"HTTP {e.code}: {body}") from e
        except urllib.error.URLError as e:
            raise ApiError(f"Network error: {e}") from e

    def chat(self, messages, max_tokens=4096, temperature=0.4, top_p=0.95):
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        result = self._request("/chat/completions", payload)
        choice = result["choices"][0]
        msg = choice["message"]
        return {
            "content": msg.get("content") or "",
            "reasoning": msg.get("reasoning_content") or "",
            "finish_reason": choice.get("finish_reason"),
            "usage": result.get("usage", {}),
        }

    def list_models(self):
        req = urllib.request.Request(
            f"{self.base_url}/models",
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))