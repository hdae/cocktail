from __future__ import annotations

import logging
import os
import time

from cocktail_server.config import get_settings
from cocktail_server.services.image_gen import ImageGenService
from cocktail_server.services.llm import LlmService

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = get_settings()
    settings.ensure_dirs()
    os.environ.setdefault("HF_HOME", str(settings.hf_home.resolve()))

    print("=== Warmup: Gemma ===")
    t0 = time.perf_counter()
    llm = LlmService(settings.llm_model_id)
    llm.load()
    print(f"Gemma loaded in {time.perf_counter() - t0:.1f}s")
    llm.unload()

    print("=== Warmup: Anima ===")
    t0 = time.perf_counter()
    image_gen = ImageGenService(settings.image_model_id)
    image_gen.load()
    print(f"Anima loaded in {time.perf_counter() - t0:.1f}s")
    image_gen.unload()

    print("Warmup done.")


if __name__ == "__main__":
    main()
