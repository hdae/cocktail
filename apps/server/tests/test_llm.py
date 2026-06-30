from __future__ import annotations

from cocktail_server.services.llm import _decode_partial_reasoning


def test_decode_reasoning_plain_japanese() -> None:
    text = '{"reasoning": "やりますね", "tool_calls": []}'
    decoded, ended = _decode_partial_reasoning(text)
    assert ended is True
    assert decoded == "やりますね"


def test_decode_reasoning_combines_surrogate_pair() -> None:
    # 😀(U+1F600) は JSON では 😀 のサロゲートペア。結合して 1 文字にする。
    text = '{"reasoning": "やった\\ud83d\\ude00ね", "tool_calls": []}'
    decoded, ended = _decode_partial_reasoning(text)
    assert ended is True
    assert "\U0001f600" in decoded
    # lone surrogate が混じると SSE の UTF-8 encode で落ちる → 残っていないこと
    assert "\ud83d" not in decoded
    assert "\ude00" not in decoded
    decoded.encode("utf-8")  # raise しない


def test_decode_reasoning_holds_incomplete_high_surrogate() -> None:
    # 下位サロゲートがまだ来ていない場合は、上位サロゲートを emit せず手前まで返して保留する。
    text = '{"reasoning": "x\\ud83d'
    decoded, ended = _decode_partial_reasoning(text)
    assert ended is False
    assert decoded == "x"
    decoded.encode("utf-8")


def test_decode_reasoning_streams_incrementally_until_close() -> None:
    # 閉じクォート未到達なら ended=False で、到達済み分だけ返す。
    partial = '{"reasoning": "途中まで'
    decoded, ended = _decode_partial_reasoning(partial)
    assert ended is False
    assert decoded == "途中まで"
