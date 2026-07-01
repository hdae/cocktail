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

    # ビルド済みの Vite dist ディレクトリ。存在する場合のみ SPA を同一オリジンで配信する。
    # ない時 (dev 起動など) は Vite dev server を別ポートで立てて /api をプロキシする運用。
    client_dist_dir: Path = Path("./apps/client/dist")

    # LLM(日本語指示→画像プロンプト JSON)。llama.cpp(GGUF) で動かす。
    # 形式: "<hf_repo>:<filename.gguf>"（HF から該当ファイルだけ取得）か、ローカル .gguf パス。
    # 既定は無検閲(heretic)12B。検閲版を使うなら公式 QAT 12B に差し替える:
    #   google/gemma-4-12B-it-qat-q4_0-gguf:gemma-4-12b-it-qat-q4_0.gguf
    llm_model_id: str = "igorls/gemma-4-12B-it-heretic-GGUF:gemma-4-12B-it-heretic-Q4_K_M.gguf"
    # Anima のベース(diffusers 形式リポ)。VAE / Qwen3 text encoder / tokenizer /
    # scheduler / modular pipeline 定義をここから読む。派生(WAI-Anima 等)は DiT だけを
    # 差し替えて使うため、ベースは常にこのリポを共有する。
    image_base_model_id: str = "circlestone-labs/Anima-Base-v1.0-Diffusers"
    # 派生モデル(DiT 単体チェックポイント)の指定。次のいずれかを受け付ける:
    #   - Civitai AIR(URN): urn:air:... （デフォルト、wai-anima）
    #   - HuggingFace リポ ID: xxx/yyy
    #   - 明示ローカルパス: /path/to/model.safetensors
    image_model_id: str = "urn:air:anima:checkpoint:civitai:2544636@2983680"
    # Civitai の gated モデル用トークン。
    civitai_token: str | None = None

    # Turbo LoRA(step & CFG 蒸留)。空文字なら無効(base 品質)。値があると有効になり、
    # cold load 時に DiT へ PEFT アダプタとして注入し、生成時は steps/cfg を Turbo 値に切替える。
    # 形式は image_model_id と同じ: Civitai AIR(urn:air:...) / HF リポ / ローカル .safetensors。
    #   例(Anima Turbo LoRA): urn:air:anima:lora:civitai:2560840@2979642
    # 既定は高速化を体感できるよう ON。品質重視に戻すなら .env で空文字にする。
    image_turbo_lora: str = "urn:air:anima:lora:civitai:2560840@2979642"
    # Turbo LoRA の適用強度。1.0 が既定、0.7 程度まで下げると多様性が増す(蒸留は維持)。
    image_turbo_lora_strength: float = Field(default=1.0, ge=0.0, le=2.0)
    # Turbo 有効時の生成パラメータ。この LoRA は「CFG 1 / 8-12 steps」を推奨する蒸留 LoRA。
    image_turbo_steps: int = Field(default=10, ge=1, le=100)
    image_turbo_cfg: float = Field(default=1.0, ge=0.0, le=20.0)

    # VRAM 運用モード: 24GB+ のカードでは coresident に自動切替するのが狙い。
    residency_mode: Literal["auto", "swap", "coresident"] = "auto"
    residency_coresident_threshold_gb: float = Field(default=20.0, ge=0.0)

    # 起動時にモデル取得＋プリロードまで走らせるか（テストでは false にする）。
    startup_preload: bool = True

    default_width: int = Field(default=896, ge=256, le=2048)
    default_height: int = Field(default=1152, ge=256, le=2048)
    # base(Turbo 無効)時の生成パラメータ。CFG は Gemma に選ばせず、モードで一意に決める。
    default_steps: int = Field(default=32, ge=1, le=100)
    default_cfg: float = Field(default=4.0, ge=0.0, le=20.0)

    @property
    def turbo_enabled(self) -> bool:
        """Turbo LoRA が構成されているか。空文字なら base 品質で動く。"""
        return bool(self.image_turbo_lora)

    def image_steps_cfg(self) -> tuple[int, float]:
        """現在のモードから (steps, cfg) を決める。

        CFG と steps は技術的なノブなので Gemma には振らせず、base / Turbo の
        モードだけで一意に定める。Turbo は CFG≈1 の step&CFG 蒸留。
        """
        if self.turbo_enabled:
            return self.image_turbo_steps, self.image_turbo_cfg
        return self.default_steps, self.default_cfg

    def ensure_dirs(self) -> None:
        self.hf_home.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.weights_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
