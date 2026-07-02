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
    # Danbooru タグ検索の索引元 CSV(headerless 4列: tag,category,post_count,"aliases")を置く場所。
    tags_dir: Path = Path("./data/tags")

    # ビルド済みの Vite dist ディレクトリ。存在する場合のみ SPA を同一オリジンで配信する。
    # ない時 (dev 起動など) は Vite dev server を別ポートで立てて /api をプロキシする運用。
    client_dist_dir: Path = Path("./apps/client/dist")

    # LLM(日本語指示→画像プロンプト JSON)。llama.cpp(GGUF) で動かす。
    # 形式: "<hf_repo>:<filename.gguf>"（HF から該当ファイルだけ取得）か、ローカル .gguf パス。
    # 既定は無検閲(heretic)12B。検閲版を使うなら公式 QAT 12B に差し替える:
    #   google/gemma-4-12B-it-qat-q4_0-gguf:gemma-4-12b-it-qat-q4_0.gguf
    llm_model_id: str = "igorls/gemma-4-12B-it-heretic-GGUF:gemma-4-12B-it-heretic-Q4_K_M.gguf"
    # LLM サンプリング。会話品質のため temp>0 を既定にする（temp 0 は長いタグ列で
    # 破滅的なリピートループを誘発することを実機確認済み。repeat_penalty も併用）。
    # 会話・ツール引数生成の双方でこの設定を使う。
    llm_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    llm_top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    llm_top_k: int = Field(default=40, ge=0)
    llm_repeat_penalty: float = Field(default=1.1, ge=0.0, le=2.0)
    # コンテキスト長。長い会話履歴・検索往復・(将来の)画像入力トークンを見込む。16GB VRAM では
    # fp16 KV だと 16384 で溢れるため、KV 量子化を併用して収める（下記 llm_kv_cache_type）。
    llm_n_ctx: int = Field(default=16384, ge=512, le=131072)
    # KV キャッシュの量子化型。f16=無量子化(VRAM 大)、q8_0=品質ほぼ無劣化で半減(既定)、
    # q4_0=更に半減(超長文脈用)。q8_0/q4_0 は flash_attn=True が前提で、K/V は同型必須。
    llm_kv_cache_type: Literal["f16", "q8_0", "q4_0"] = "q8_0"
    # FlashAttention。KV 量子化(特に V)に必須。既定 True。
    llm_flash_attn: bool = True

    # Danbooru タグ検索(search_tags ツール)の索引元 CSV。hdae/danbooru-tagcomplete-extra の
    # prebuilt CSV(日本語エイリアス入り)。`tags_csv` が無いとき `tags_auto_download` が真なら
    # この URL から取得する。
    tags_csv_url: str = "https://github.com/hdae/danbooru-tagcomplete-extra/releases/download/2026-05-23/danbooru.csv"
    # 起動時に CSV 未配置なら自動ダウンロードするか。WSL2 コンテナ+オフライン運用を既定と
    # するため vImagen の True から反転して False にし、事前配置(data/tags/danbooru.csv)を前提と
    # する。未配置なら索引は空のまま起動し search は空を返す（TagService が warning を出す）。
    # DECIDED: 恒久的な取得導線(scripts/ingest_danbooru.py, HF dataset 起点)は ROADMAP M5 に委ね、
    # Phase2 は「事前配置された CSV を読む」までに範囲を絞る（docs/decisions/0002-phase2-agent-loop.md）。
    tags_auto_download: bool = False

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

    @property
    def tags_csv(self) -> Path:
        """Danbooru タグ検索の索引元 CSV パス（`tags_dir` から一意に導出）。"""
        return self.tags_dir / "danbooru.csv"

    def image_steps_cfg(self, *, turbo: bool | None = None) -> tuple[int, float]:
        """モードから (steps, cfg) を決める。

        CFG と steps は技術的なノブなので Gemma には振らせず、base / Turbo の
        モードだけで一意に定める。Turbo は CFG≈1 の step&CFG 蒸留。
        `turbo` を明示すると（`/generate` の base 整合など）その値でモードを固定し、
        既定(None)は `turbo_enabled` に従う。turbo と生成パラメータを構造的に一致させる。
        """
        use_turbo = self.turbo_enabled if turbo is None else turbo
        if use_turbo:
            return self.image_turbo_steps, self.image_turbo_cfg
        return self.default_steps, self.default_cfg

    def ensure_dirs(self) -> None:
        self.hf_home.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.tags_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
