# 0002. 語彙タグ検索 search_tags とエージェントループ

- Status: Accepted
- Date: 2026-07-02
- Scope: `apps/server` のタグ検索と LLM ターン処理（`services/tags.py`,
  `schemas/tags.py`, `schemas/generate.py`（`SearchTagsCall`）, `services/prompt_builder.py`,
  `services/llm.py`（`run_turn` ループ）, `main.py`, `config.py`）
- 前提: [0001](0001-llm-conversation-harness.md)（会話優先ハーネスと native ツール解析）

## Context

Gemma は Danbooru プロンプトを書くよう訓練されていないため、正規タグ/キャラ表記の綴りを
外しやすい。そこで vImagen の語彙タグ検索を移植した `search_tags` ツールを与え、`run_turn` を
「検索→(必要なだけ)検索→生成」のエージェントループに拡張する。

実装前に実機（gemma-4-12B heretic GGUF, llama-cpp-python 0.3.32）で接地して判明した事実:

- **埋込 chat テンプレはツール応答を OpenAI 互換で描画する**。assistant メッセージが構造化
  `tool_calls`（`id` + `function.name` + mapping/文字列 arguments）を持つと
  `<|tool_call>call:NAME{...}<tool_call|>` に、続く `role:tool`（`tool_call_id` 一致）を
  前方スキャンして `<|tool_response>response:NAME{...}<tool_response|>` に整形する。
  → #2227（native ツール**出力**のパース不良）とは別経路で、ツール結果の**入力**注入は成立する。
- **n_ctx 予算は見積りより広い**。system + search 指針 + 2 tools + 短い user の実 prompt
  トークンは **2036**（tools スキーマ寄与 ≈494）。当初の ~3.4k 見積りは過大で、n_ctx=8192 /
  出力予約 1024 を引いた実質空きは ~5.1k。3 ホップの中間検索要約は十分収まる。
- **モデルは search_tags を自発的には安定して呼ばない**。不確かなキャラ要求でも検索せず
  ナレーションで止まる（ツール呼び忘れと同じ intent-recall の裏返し）。一方、検索結果を
  `role:tool` で戻すと、返ったタグを generate_image の positive に的確に採用する（往復は機能）。
- CSV は headerless 4 列 `tag,category,post_count,"aliases"`（162,840 行 / 6.0MB）。alias 列に
  日英中の読みが混在し、日本語クエリはこの列経由で英語正規タグへ解決する（対訳辞書は不要）。

## Decision

1. **案A: ループを LLM 層（`run_turn` 内）に内包する**。各ホップの native 出力を自前パースし、
   `search_tags` なら `TagService.search` を実行して上位 N 要約を `role:tool` で戻し次ホップへ、
   `generate_image` か会話確定（ツール無し）か上限到達で確定する。
   - **理由**: 検索の中間往復が会話履歴・SSE フレーム契約・永続メッセージに一切出ず run_turn 内で
     完結する。結果 `LlmTurnResult.tool_calls` は `list[GenerateImageCall]` のまま据え置け、
     orchestrator / `api/generate.py` / `_reconstruct_assistant_turn` / `_build_chat_messages` の
     history 経路は**不変**（型リプル無し）。案B（orchestrator にループ）は検索が SSE/永続に必ず
     漏れ二重実装を招くため不採用。案C（2 パス強制）は可変往復要件に構造的に合わず不採用。

2. **フィードバックは構造化 tool_calls + `role:tool`**（テンプレの OpenAI 互換経路）。検索要求を
   構造化 `tool_calls` を持つ assistant メッセージ、結果を `tool_call_id` 一致の `role:tool` として
   一時 `messages` に積む（実機の往復で採用を確認済みの形式）。native 文字列 replay ではなく
   構造化にするのは、テンプレの前方スキャンが構造化 `tool_calls` を要求するため。

3. **n_ctx 予算ガード**: `_MAX_ITERS=3`（検索→検索→生成の最大 3 ホップ）で無限ループと膨張を
   構造的に封じる。中間結果は上位 `_SEARCH_TOP_N=8` 件の `{tag, ja}` 要約のみ戻し（post_count /
   全 alias は落とす）、1 ホップ数百 token に抑える。上限は実測 2036tok の裏取りに基づく。

4. **`search_tags` の引数**は `query`（必須）+ `category`（任意, Danbooru 整数）。返す件数はサーバが
   予算で決めるため **limit はモデルに持たせない**。native は全値文字列なので `SearchTagsCall.from_native`
   が境界でコアーションし、**未知/非整数カテゴリは絞り込み無効（None）**に落とす。引数不正な検索は
   ログに残してスキップし、有効な検索が無ければそのホップで確定する（ターンを落とさない）。

5. **タグ検索は語彙検索主軸**（[0001](0001-llm-conversation-harness.md) の方針を実装）。vImagen
   `services/tags.py` の 3 段検索（prefix→substring→fuzzy + 日本語 alias + post_count ランキング）を
   移植。paste 補完用の `lookup` / REST 面はエージェントループに不要なので移植しない（dead code 回避）。

6. **CSV の扱い**: `parse_csv` は category/post_count の変換失敗を黙って 0 埋めせず **warning + 行スキップ**
   にし、未リリース段階の fail-loud 規約に沿う。`tags_auto_download` は WSL2 オフライン運用を既定に
   するため vImagen の True から **False に反転**（事前配置前提。不在時は起動 warning）。恒久取得導線
   （`scripts/ingest_danbooru.py`, HF dataset 起点）は ROADMAP M5 に委ね、Phase2 は「事前配置 CSV を
   読む」までに範囲を絞る。索引は起動プリロードで CPU ロードする（GPU/ネットワーク非依存）。

## Consequences

- 消費側（orchestrator / api）と履歴再構成は無改修。案A の封じ込めにより、当初見込んだ型リプル
  修正コミットは不要になった。
- 検索フェーズは UI 非表示（[0001](0001-llm-conversation-harness.md) の Phase 3=client 描画と整合）。
  検索進捗を出したくなったら別 hook が要る。
- **search_tags の採用は信頼性の dial**。ループ機構と system 指針で促すが、モデルが実際に検索を選ぶ
  保証はない（自発呼び出しは不安定）。検索が発火すれば結果は的確に使われる。強化候補=システム指針の
  補強 / few-shot / 意図検出→forced tool_choice。ツール呼び忘れと同根で、Phase2 後に挙動を見て判断する。
- 並行性: `TagService` の索引ロードは `asyncio.to_thread`（起動時）で行い、`_index` 参照差し替えは
  GIL 下で原子的。検索は `run_turn` から `asyncio.to_thread` 経由で呼びイベントループを塞がない。

## 既知の制約 / 留保

- native パーサの `<tool_call|>` 閉じ検出は文字列値内を区別しない（[0001](0001-llm-conversation-harness.md)
  の既知制約）。`search_tags.query` は短語へ誘導しているためリテラル `<tool_call|>` は事実上出ない。
- テンプレ描画・往復挙動は実装挙動依存（仕様保証ではない）。#2227 解決で native tool 出力が構造化
  パースされるようになったら、自前パーサと一時措置を [0001](0001-llm-conversation-harness.md) 記載の
  トリガで撤去する。ツール結果注入（構造化 tool_calls + role:tool）はその後も有効。
