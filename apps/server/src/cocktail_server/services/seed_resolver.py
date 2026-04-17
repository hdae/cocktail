from __future__ import annotations

import logging
import secrets

from cocktail_server.schemas.generate import SeedAction

logger = logging.getLogger(__name__)

_SEED_MAX = 2**63 - 1


def _random_seed() -> int:
    return secrets.randbelow(_SEED_MAX)


def resolve_seed(
    *,
    req_seed: int | None,
    action: SeedAction,
    last_image_seed: int | None,
) -> int:
    """画像生成に使う seed を決定する。

    優先順位:
    1. `req_seed` が明示されていればそれ（開発/デバッグ向け `POST /generate` 経路）
    2. `action="keep"` かつ `last_image_seed` があれば踏襲
    3. それ以外は `secrets.randbelow` で新規採番

    `action="keep"` でも `last_image_seed` が無い場合は WARNING ログを出して
    新規採番に縮退する（初回ターンでの「keep」指定など）。
    """
    if req_seed is not None:
        return req_seed
    if action == "keep":
        if last_image_seed is not None:
            return last_image_seed
        logger.warning("seed_action=keep requested but no previous seed; falling back to new")
    return _random_seed()
