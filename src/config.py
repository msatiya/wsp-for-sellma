import json
import os
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_BASE_URL = "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
DEFAULT_EDITOR_MODEL = "deepseek-v4-pro-0813"

TARGET_MODELS = {
    "qwen": "Qwen/Qwen3.5-0.8B",
    "gpt-sw3": "AI-Sweden-Models/gpt-sw3-1.3b",
}


@dataclass
class ApiConfig:
    base_url: str
    api_key: str
    model: str


@dataclass
class ExperimentConfig:
    target_models: dict = field(default_factory=lambda: dict(TARGET_MODELS))
    api: ApiConfig = None

    def __post_init__(self):
        if self.api is None:
            self.api = ApiConfig(
                base_url=os.environ.get("MODEL_STUDIO_BASE_URL", DEFAULT_BASE_URL),
                api_key=os.environ.get("MODEL_STUDIO_API_KEY", ""),
                model=os.environ.get("WSP_EDITOR_MODEL", DEFAULT_EDITOR_MODEL),
            )


def load_config(path=None):
    data = {}
    if path:
        p = Path(path)
        if p.exists():
            with p.open() as f:
                data = json.load(f)
    cfg = ExperimentConfig()
    cfg.target_models = data.get("target_models", TARGET_MODELS)
    cfg.api.base_url = data.get("base_url", cfg.api.base_url)
    cfg.api.model = data.get("editor_model", cfg.api.model)
    return cfg