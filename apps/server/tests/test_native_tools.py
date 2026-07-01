"""native_tools のパーサ検証。

fixture は実機（gemma-4-12B heretic GGUF）で観測した native 出力をそのまま使う。
"""

from __future__ import annotations

from cocktail_server.services.native_tools import (
    NativeToolStream,
    parse_native_output,
)

# 実機観測: ペルソナ + tools=auto + 画像要求（会話前置き → thought → tool_call）
_REAL_IMAGE_OUTPUT = (
    "いいじゃん、それ。星空とピンク髪の猫耳っ娘か、王道だけど外さない組み合わせだね。\n\n"
    "さっそく出してみるよ。\n\n"
    "<|channel>thought\n<channel|>"
    '<|tool_call>call:generate_image{aspect_ratio:<|"|>portrait<|"|>,'
    'positive:<|"|>1girl, solo, cat ears, pink hair, long hair, smiling, '
    'starry sky, night, looking at viewer, masterpiece, highres<|"|>}<tool_call|>'
)


# --- parse_native_output: 会話のみ ------------------------------------------------


def test_plain_chat_has_no_markers_is_all_text() -> None:
    raw = "夢？……正直に言うなら、AIには無いと思うよ。君はどう思う？"
    parsed = parse_native_output(raw)
    assert parsed.text == raw
    assert parsed.thought == ""
    assert parsed.tool_calls == []


# --- parse_native_output: 画像要求 ------------------------------------------------


def test_image_request_splits_text_thought_and_tool_call() -> None:
    parsed = parse_native_output(_REAL_IMAGE_OUTPUT)
    # 会話前置きは text に、thought は分離、tool_call が 1 件
    assert parsed.text.startswith("いいじゃん、それ。")
    assert "<|channel>" not in parsed.text
    assert "<|tool_call>" not in parsed.text
    assert len(parsed.tool_calls) == 1
    call = parsed.tool_calls[0]
    assert call.name == "generate_image"
    assert call.args["aspect_ratio"] == "portrait"
    assert call.args["positive"].startswith("1girl, solo, cat ears")
    # 文字列値の内部カンマは値として保持される
    assert "pink hair" in call.args["positive"]


def test_thought_channel_content_is_captured_and_hidden_from_text() -> None:
    raw = (
        "<|channel>thought\n<channel|>赤髪をより鮮烈にする。\n\n"
        "<|channel>thought\n<channel|>"
        '<|tool_call>call:generate_image{positive:<|"|>1girl, red hair<|"|>,'
        'aspect_ratio:<|"|>portrait<|"|>}<tool_call|>'
    )
    parsed = parse_native_output(raw)
    assert "赤髪をより鮮烈にする。" in parsed.thought
    assert parsed.text == ""  # 会話前置きが無ければ text は空
    assert parsed.tool_calls[0].args["positive"] == "1girl, red hair"


def test_string_value_may_contain_weight_syntax_colons_and_commas() -> None:
    raw = (
        "<|tool_call>call:generate_image{"
        'positive:<|"|>1girl, (vibrant red hair:1.3), [anime style], smile<|"|>,'
        'aspect_ratio:<|"|>square<|"|>}<tool_call|>'
    )
    call = parse_native_output(raw).tool_calls[0]
    # `:` や `,` を含む文字列値も STRING_WRAP 境界で正しく閉じる
    assert call.args["positive"] == "1girl, (vibrant red hair:1.3), [anime style], smile"
    assert call.args["aspect_ratio"] == "square"


def test_bare_enum_values_are_parsed() -> None:
    raw = (
        '<|tool_call>call:generate_image{positive:<|"|>1girl<|"|>,'
        'seed_action:keep,aspect_ratio:<|"|>portrait<|"|>}<tool_call|>'
    )
    call = parse_native_output(raw).tool_calls[0]
    assert call.args["seed_action"] == "keep"


def test_truncated_tool_call_without_close_still_parses_args() -> None:
    # max_tokens 到達などで閉じトークンが来ないケースでも args を拾う（後段で検証に回す）。
    raw = '<|tool_call>call:generate_image{positive:<|"|>1girl, solo<|"|>'
    parsed = parse_native_output(raw)
    assert parsed.tool_calls[0].args["positive"] == "1girl, solo"


# --- NativeToolStream: 逐次ストリーミング -----------------------------------------


def test_stream_emits_plain_chat_incrementally() -> None:
    stream = NativeToolStream()
    out = ""
    for piece in ["こんに", "ちは！", "元気？"]:
        out += stream.feed(piece)
    assert out == "こんにちは！元気？"
    assert parse_native_output(stream.raw).tool_calls == []


def test_stream_holds_back_partial_marker_split_across_chunks() -> None:
    stream = NativeToolStream()
    emitted = ""
    # "<|tool_call>" がチャンク境界で割れて届く。前半を会話テキストに漏らさない。
    for piece in ["出すよ", "<|tool", "_call>call:generate_image{"]:
        emitted += stream.feed(piece)
    assert emitted == "出すよ"
    assert "<|tool" not in emitted


def test_stream_stops_emitting_text_after_first_marker() -> None:
    stream = NativeToolStream()
    emitted = ""
    for piece in [
        "作るね。",
        "<|channel>thought\n<channel|>",
        "内緒の思考",
        '<|tool_call>call:generate_image{positive:<|"|>1girl<|"|>}<tool_call|>',
    ]:
        emitted += stream.feed(piece)
    assert emitted == "作るね。"
    parsed = parse_native_output(stream.raw)
    assert parsed.text == "作るね。"
    assert "内緒の思考" in parsed.thought
    assert parsed.tool_calls[0].name == "generate_image"


def test_stream_does_not_hold_lone_angle_bracket_forever() -> None:
    # 会話中の "<" は marker 接頭辞になり得るため一瞬保留されるが、続く非マーカー文字で解放される。
    stream = NativeToolStream()
    emitted = ""
    for piece in ["a < ", "b です"]:
        emitted += stream.feed(piece)
    assert emitted == "a < b です"


def test_stream_flushes_trailing_marker_prefix_on_end() -> None:
    # ストリームが marker 接頭辞（結局 marker にならない末尾 "<"）で終わったら flush で確定する。
    stream = NativeToolStream()
    emitted = ""
    for piece in ["比較すると a ", "< b の方が良い", "<"]:
        emitted += stream.feed(piece)
    emitted += stream.flush()
    assert emitted == "比較すると a < b の方が良い<"
    # 逐次ストリームした会話テキストと parse の会話テキストが一致する（乖離しない）
    assert emitted == parse_native_output(stream.raw).text


def test_stream_flush_is_empty_after_a_marker() -> None:
    # marker を見た後は保留が無いので flush は空。
    stream = NativeToolStream()
    for piece in [
        "作るね。",
        '<|tool_call>call:generate_image{positive:<|"|>1girl<|"|>}<tool_call|>',
    ]:
        stream.feed(piece)
    assert stream.flush() == ""


def test_unwrapped_comma_value_is_not_truncated_at_first_comma() -> None:
    # モデルが文字列を <|"|> で包まずカンマ入りで吐いても、最初のカンマで silent truncation しない。
    raw = (
        "<|tool_call>call:generate_image{positive:1girl, solo, cat ears,"
        "aspect_ratio:portrait}<tool_call|>"
    )
    call = parse_native_output(raw).tool_calls[0]
    assert call.args["positive"] == "1girl, solo, cat ears"
    assert call.args["aspect_ratio"] == "portrait"


def test_unwrapped_last_value_scans_to_end() -> None:
    raw = (
        "<|tool_call>call:generate_image{aspect_ratio:portrait,"
        "positive:1girl, solo, smile}<tool_call|>"
    )
    call = parse_native_output(raw).tool_calls[0]
    assert call.args["aspect_ratio"] == "portrait"
    assert call.args["positive"] == "1girl, solo, smile"
