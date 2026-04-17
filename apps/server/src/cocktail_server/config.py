from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    host: str = "0.0.0.0"
    port: int = 8000
    log_level: Literal["debug", "info", "warning", "error"] = "info"

    gpu_concurrency: int = Field(default=1, ge=1)

    hf_home: Path = Path("./data/models")
    images_dir: Path = Path("./data/images")
    weights_dir: Path = Path("./data/weights")

    llm_model_id: str = "google/gemma-4-E4B-it"
    # 画像モデルの指定。次のいずれかを受け付ける:
    #   - Civitai AIR(URN): urn:air:... （デフォルト、wai-anima v10）
    #   - HuggingFace リポ ID: xxx/yyy
    #   - 明示ローカルパス: /path/to/model.safetensors
    image_model_id: str = "urn:air:anima:checkpoint:civitai:2544636@2859702"
    # Civitai の gated モデル用トークン。
    civitai_token: str | None = None

    # VRAM 運用モード: 24GB+ のカードでは coresident に自動切替するのが狙い。
    residency_mode: Literal["auto", "swap", "coresident"] = "auto"
    residency_coresident_threshold_gb: float = Field(default=20.0, ge=0.0)

    # 起動時にモデル取得＋プリロードまで走らせるか（テストでは false にする）。
    startup_preload: bool = True

    default_width: int = Field(default=896, ge=256, le=2048)
    default_height: int = Field(default=1152, ge=256, le=2048)
    default_steps: int = Field(default=32, ge=1, le=100)
    default_cfg: float = Field(default=4.0, ge=0.0, le=20.0)

    def ensure_dirs(self) -> None:
        self.hf_home.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.weights_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
