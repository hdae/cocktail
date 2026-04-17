from __future__ import annotations

import logging

import pytest
from cocktail_server.services.seed_resolver import resolve_seed


def test_req_seed_takes_precedence_over_everything() -> None:
    assert resolve_seed(req_seed=42, action="keep", last_image_seed=99) == 42
    assert resolve_seed(req_seed=42, action="new", last_image_seed=99) == 42


def test_keep_reuses_last_image_seed_when_available() -> None:
    assert resolve_seed(req_seed=None, action="keep", last_image_seed=12345) == 12345


def test_keep_without_last_image_seed_falls_back_to_new_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # `cocktail_server` パッケージロガーは他テスト（create_app 経由）で propagate=False
    # に設定される可能性があるため、caplog が拾えるよう一時的に戻す。
    pkg_logger = logging.getLogger("cocktail_server")
    original_propagate = pkg_logger.propagate
    pkg_logger.propagate = True
    try:
        with caplog.at_level(logging.WARNING, logger="cocktail_server.services.seed_resolver"):
            result = resolve_seed(req_seed=None, action="keep", last_image_seed=None)
    finally:
        pkg_logger.propagate = original_propagate
    assert isinstance(result, int)
    assert 0 <= result < 2**63
    assert any("seed_action=keep" in r.message for r in caplog.records)


def test_new_always_generates_fresh_seed() -> None:
    # 乱数なので衝突確率は無視できるほど低い（2**63 空間）。
    seeds = {resolve_seed(req_seed=None, action="new", last_image_seed=777) for _ in range(32)}
    assert len(seeds) == 32
    # last_image_seed は new 時は無視されること
    assert 777 not in seeds


def test_new_range_is_non_negative_and_within_int64() -> None:
    for _ in range(16):
        s = resolve_seed(req_seed=None, action="new", last_image_seed=None)
        assert 0 <= s < 2**63
