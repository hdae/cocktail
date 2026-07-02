"""run_turn のエージェントループ（検索→検索→生成）の振る舞いテスト。

実 llama.cpp / GPU は使わず、スクリプト化した native 出力を返す FakeLlama と、固定候補を
返す FakeTags を DI する（外部 I/O は untestable なので構造でフェイクする）。
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cocktail_server.schemas.messages import Message, TextPart
from cocktail_server.schemas.tags import TagSuggestion
from cocktail_server.services.llm import (
    _MAX_ITERS,
    _SEARCH_TOP_N,
    LlmService,
    LlmTextDelta,
    LlmTurnComplete,
    LlmTurnResult,
)
from cocktail_server.services.native_tools import render_tool_call

# 実機で観測した native 形式のホップ出力（会話前置き + tool_call）。
_SEARCH_CALL = render_tool_call("search_tags", {"query": "arona blue archive", "category": "4"})
_GEN_CALL = render_tool_call(
    "generate_image",
    {
        "aspect_ratio": "portrait",
        "positive": "score_7, 1girl, arona (blue archive), halo",
        "seed_action": "new",
    },
)
_SEARCH_HOP = "探すね。" + _SEARCH_CALL
_GEN_HOP = "できたよ。" + _GEN_CALL
_CHAT_HOP = "こんにちは、元気だよ。"


class FakeLlama:
    """create_chat_completion をスクリプト化する。呼ばれるたび次のホップ出力を返す。"""

    def __init__(self, hop_outputs: list[str]) -> None:
        self._hops = list(hop_outputs)
        self.calls: list[list[dict[str, Any]]] = []  # 呼び出し時点の messages のコピー

    def create_chat_completion(
        self, *, stream: bool = False, messages: list[dict[str, Any]], **_: Any
    ) -> Any:
        self.calls.append([dict(m) for m in messages])
        if not self._hops:
            raise AssertionError("FakeLlama called more times than scripted")
        text = self._hops.pop(0)
        if stream:
            return iter([{"choices": [{"delta": {"content": text}}]}])
        return {"choices": [{"message": {"content": text}}]}


class FakeTags:
    def __init__(
        self, results: list[TagSuggestion], hints: list[TagSuggestion] | None = None
    ) -> None:
        self._results = results
        self._hints = hints or []
        self.queries: list[tuple[str, int, int | None]] = []
        self.match_calls: list[tuple[str, int]] = []

    def search(
        self, query: str, limit: int = 15, category: int | None = None
    ) -> list[TagSuggestion]:
        self.queries.append((query, limit, category))
        return self._results

    def match_in_text(self, text: str, limit: int = 8) -> list[TagSuggestion]:
        self.match_calls.append((text, limit))
        return self._hints


def _service(
    hops: list[str],
    tag_results: list[TagSuggestion],
    *,
    hints: list[TagSuggestion] | None = None,
    tag_hints_enabled: bool = True,
) -> LlmService:
    svc = LlmService(
        "dummy.gguf",
        hf_home=Path("/x"),
        tags=FakeTags(tag_results, hints=hints),  # type: ignore[arg-type]
        tag_hints=tag_hints_enabled,
    )
    svc._llm = FakeLlama(hops)  # type: ignore[assignment]
    return svc


def _fake_llm(svc: LlmService) -> FakeLlama:
    return svc._llm  # type: ignore[return-value]


def _fake_tags(svc: LlmService) -> FakeTags:
    return svc._tags  # type: ignore[return-value]


def _history(text: str = "アロナを描いて") -> list[Message]:
    return [
        Message(
            id="u1",
            conversation_id="c1",
            role="user",
            parts=[TextPart(text=text)],
            created_at=datetime.now(UTC),
        )
    ]


async def _drive(svc: LlmService) -> tuple[list[str], LlmTurnResult]:
    deltas: list[str] = []
    result: LlmTurnResult | None = None
    async for chunk in svc.run_turn(_history()):
        if isinstance(chunk, LlmTextDelta):
            deltas.append(chunk.delta)
        elif isinstance(chunk, LlmTurnComplete):
            result = chunk.result
    assert result is not None, "run_turn ended without LlmTurnComplete"
    return deltas, result


_ARONA = TagSuggestion(
    tag="arona (blue archive)", category=4, post_count=1000, ja="アロナ", matched="アロナ"
)
_HALO = TagSuggestion(tag="halo", category=0, post_count=500, ja="ヘイロー", matched=None)


# --- 基本フロー ------------------------------------------------------------------


async def test_search_then_generate_resolves_to_generate_image() -> None:
    svc = _service([_SEARCH_HOP, _GEN_HOP], [_ARONA])
    _deltas, result = await _drive(svc)
    # 最終確定は generate_image 単一（検索は内部で消費され外に出ない）。
    assert [c.positive for c in result.tool_calls] == ["score_7, 1girl, arona (blue archive), halo"]
    # 検索は上位 N=_SEARCH_TOP_N・category=4 で実行された。
    assert _fake_tags(svc).queries == [("arona blue archive", _SEARCH_TOP_N, 4)]
    assert len(_fake_llm(svc).calls) == 2


async def test_direct_generate_is_single_hop_no_search() -> None:
    # 検索を吐かず直接 generate_image を出す既存挙動の非回帰（1 ホップ・検索なし）。
    svc = _service([_GEN_HOP], [_ARONA])
    _deltas, result = await _drive(svc)
    assert len(result.tool_calls) == 1
    assert _fake_tags(svc).queries == []
    assert len(_fake_llm(svc).calls) == 1


async def test_pure_chat_has_no_tool_calls() -> None:
    # 会話のみのターンはツール無しで確定する非回帰。
    svc = _service([_CHAT_HOP], [])
    deltas, result = await _drive(svc)
    assert result.tool_calls == []
    assert result.text == "こんにちは、元気だよ。"
    assert "".join(deltas) == "こんにちは、元気だよ。"
    assert len(_fake_llm(svc).calls) == 1


# --- 上限・安全弁 ----------------------------------------------------------------


async def test_max_iters_caps_endless_search_loop() -> None:
    # 検索を延々吐くモデルでも _MAX_ITERS で打ち切り、無限ループしない（フォールト注入）。
    svc = _service([_SEARCH_HOP] * (_MAX_ITERS + 3), [_ARONA])
    _deltas, result = await _drive(svc)
    assert len(_fake_llm(svc).calls) == _MAX_ITERS
    # 上限到達時の最終ホップは検索のみ → generate_image は無く会話確定に縮退する。
    assert result.tool_calls == []


async def test_malformed_search_call_finalizes_without_executing_search() -> None:
    # 引数不正（空 query）の検索は from_native で弾かれ、検索を実行せずそのホップで確定する。
    bad = render_tool_call("search_tags", {"query": ""})
    svc = _service([bad], [_ARONA])
    _deltas, result = await _drive(svc)
    assert result.tool_calls == []
    assert _fake_tags(svc).queries == []  # 検索は実行されない
    assert len(_fake_llm(svc).calls) == 1  # 往復せずに確定


# --- 漏洩封じ込め（案A の要）-----------------------------------------------------


async def test_search_summary_normalizes_underscores_to_spaces() -> None:
    # CSV 正規形(texas_(arknights))をそのまま見せるとモデルが positive へ写して
    # 「spaces, no underscores」規約と衝突する(実機で確認)。role:tool 要約はスペース区切り。
    underscored = TagSuggestion(
        tag="texas_(arknights)", category=4, post_count=8125, ja="テキサス", matched="テキサス"
    )
    svc = _service([_SEARCH_HOP, _GEN_HOP], [underscored])
    await _drive(svc)
    tool_msg = next(m for m in _fake_llm(svc).calls[1] if m.get("role") == "tool")
    assert "texas (arknights) [テキサス]" in tool_msg["content"]
    assert "texas_(arknights)" not in tool_msg["content"]


async def test_search_results_reach_next_hop_but_not_final_text() -> None:
    # 中間検索結果は次ホップの messages(role=tool) に入るが、ユーザ向け result.text には漏れない。
    svc = _service([_SEARCH_HOP, _GEN_HOP], [_HALO])
    _deltas, result = await _drive(svc)

    # 最終テキストは会話ナレーションのみ。検索要約(tag/ja)は含まない。
    assert result.text == "探すね。できたよ。"
    assert "halo" not in result.text
    assert "ヘイロー" not in result.text

    # 2 ホップ目の入力 messages には検索結果を載せた role=tool メッセージが入っている。
    second_hop_messages = _fake_llm(svc).calls[1]
    tool_msgs = [m for m in second_hop_messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert "halo" in tool_msgs[0]["content"]


async def test_intermediate_hop_text_streams_and_accumulates() -> None:
    # 各ホップの会話デルタは逐次流れ、確定 text は流したテキストと一致する（単一の真実源）。
    svc = _service([_SEARCH_HOP, _GEN_HOP], [_ARONA])
    deltas, result = await _drive(svc)
    joined = "".join(deltas)
    assert "探すね。" in joined
    assert "できたよ。" in joined
    assert result.text == joined.strip()


# --- タグ候補注入（事前検索: auto tag lookup）-------------------------------------


async def test_tag_hints_injected_into_current_user_message() -> None:
    # ターン開始時に current ユーザ発話が逆引きされ、候補が最初のホップの user content に
    # 注入される。注入は turn-local: ユーザ向け result.text には一切漏れない。
    svc = _service([_GEN_HOP], [], hints=[_ARONA])
    _deltas, result = await _drive(svc)

    tags = _fake_tags(svc)
    assert tags.match_calls == [("アロナを描いて", 8)]
    first_hop = _fake_llm(svc).calls[0]
    user_msg = next(m for m in first_hop if m.get("role") == "user")
    assert "auto tag lookup" in user_msg["content"]
    assert "arona (blue archive) [アロナ]" in user_msg["content"]
    assert "auto tag lookup" not in result.text
    assert "arona" not in result.text


async def test_tag_hints_absent_when_no_match() -> None:
    # ヒットが無い発話（雑談等）はヒントブロック自体が付かない（自然なゲート）。
    svc = _service([_CHAT_HOP], [], hints=[])
    await _drive(svc)
    user_msg = next(m for m in _fake_llm(svc).calls[0] if m.get("role") == "user")
    assert "auto tag lookup" not in user_msg["content"]


async def test_tag_hints_disabled_by_kill_switch() -> None:
    # LLM_TAG_HINTS=false なら逆引き自体を呼ばず、入力は従来と同一。
    svc = _service([_GEN_HOP], [], hints=[_ARONA], tag_hints_enabled=False)
    await _drive(svc)
    assert _fake_tags(svc).match_calls == []
    user_msg = next(m for m in _fake_llm(svc).calls[0] if m.get("role") == "user")
    assert "auto tag lookup" not in user_msg["content"]


# --- フィードバック形式（テンプレ整合）-------------------------------------------


async def test_feedback_uses_structured_tool_calls_and_role_tool() -> None:
    # 検索フィードバックは「構造化 tool_calls を持つ assistant + tool_call_id 一致の role:tool」
    # で積む（Gemma 4 テンプレの OpenAI 互換経路。実機で往復を確認済みの形式）。
    svc = _service([_SEARCH_HOP, _GEN_HOP], [_ARONA])
    await _drive(svc)
    second_hop_messages = _fake_llm(svc).calls[1]

    assistant_calls = [m for m in second_hop_messages if m.get("tool_calls")]
    assert len(assistant_calls) == 1
    # content は空が MUST: テンプレは content をツール応答の後に描画し、非空だと <turn|> で
    # モデルターンを閉じてしまい、継続ホップの生成開始点が消える（cascade 設計の前提）。
    assert assistant_calls[0]["content"] == ""
    fn = assistant_calls[0]["tool_calls"][0]
    assert fn["function"]["name"] == "search_tags"
    # category は既知値のみ int でそのまま replay（未知/None は省く）。
    assert fn["function"]["arguments"] == {"query": "arona blue archive", "category": 4}
    call_id = fn["id"]

    tool_msg = next(m for m in second_hop_messages if m.get("role") == "tool")
    assert tool_msg["tool_call_id"] == call_id
