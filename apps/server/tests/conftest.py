from __future__ import annotations

import pytest
from cocktail_server.config import get_settings


@pytest.fixture(autouse=True)
def _skip_startup_preload(monkeypatch: pytest.MonkeyPatch) -> None:
    """全テストで FastAPI lifespan の fetch_models + プリロードを無効化する。

    実ネットワーク / 実 GPU を踏まないための最低限の防御。個別テストは必要に応じて
    `IMAGES_DIR` や `HF_HOME` を上書きする。
    """
    monkeypatch.setenv("STARTUP_PRELOAD", "false")
    # 既定は Turbo ON だが、テストは base モードで決定的にする（出荷既定に依存させない）。
    # Turbo を検証したいテストは個別に IMAGE_TURBO_LORA を設定して cache_clear する。
    monkeypatch.setenv("IMAGE_TURBO_LORA", "")
    get_settings.cache_clear()
