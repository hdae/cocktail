# 0001. 会話優先 LLM ハーネスと Gemma 4 native ツール解析

- Status: Accepted
- Date: 2026-07-02
- Scope: `apps/server` の LLM ターン処理（`services/llm.py`, `services/native_tools.py`,
  `services/prompt_builder.py`, `services/orchestrator.py`, `api/generate.py`）

## Context

Gemma を「まず普通に対話できるモデル」として扱い、画像生成は必要時だけツールで呼ぶ
構成へ刷新した。旧実装は毎ターン `{reasoning, tool_calls}` の JSON を強制し、会話を
その一フィールドに押し込んでいたため会話品質が構造的に劣化していた。

実機検証（gemma-4-12B heretic GGUF, llama-cpp-python 0.3.32）で判明した事実:

- 実機は **Gemma 4**（arch `gemma4`）。`tools=` を渡すと埋込テンプレが tool 定義を
  整形し、モデルは native 形式で応答する:
  `<会話テキスト><|channel>thought<channel|>...<|tool_call>call:NAME{key:<|"|>val<|"|>,...}<tool_call|>`。
  会話のみのターンはマーカー無しの素テキスト。
- **llama-cpp-python 0.3.32 はこの native トークンを構造化 `tool_calls` にパースできず**、
  生トークンが `message.content` に漏れ `tool_calls=None` になる
  （[abetlen/llama-cpp-python#2227](https://github.com/abetlen/llama-cpp-python/issues/2227)、
  未マージの修正 [PR #2232](https://github.com/abetlen/llama-cpp-python/pull/2232)）。
- `temperature=0.0` はモデルが巨大な negative を自由生成する文脈で破滅的なリピート
  ループを誘発した。`temperature>0` + `repeat_penalty` で解消。
- system 無しの素の heretic は「Google 製の誠実な Gemma 4」という優等生ペルソナに回帰する。
  abliteration は拒否を外すが既定の声は変えない → 会話ペルソナは system プロンプトで作る。

## Decision

1. **会話優先ハーネス**: 既定は自由文の会話（ペルソナ system プロンプト + サンプリング）。
   `generate_image` は `tools=` で渡し、モデルが必要時だけ native tool を吐く。

2. **[一時措置] native トークンをサーバ側で自前パース** (`services/native_tools.py`)。
   binding のパースが壊れている（#2227）ための繋ぎ。会話テキストのみ逐次ストリームし、
   `<|channel>thought<channel|>`（UI 非表示）と `<|tool_call>...`（引数 DSL）を分離する。
   引数は `GenerateImageCall` で検証し、失敗時はリトライ→チャットのみ縮退。
   **解消トリガ**: #2227 が解決し binding が `message.tool_calls` を返すようになったら、
   `native_tools` を撤去し `tools=` の構造化出力へ載せ替える。

3. **[一時措置] negative はサーバ固定ベース + `negative_extra`**。モデルには追加分だけ
   書かせ、固定ベース（`prompt_builder.NEGATIVE_DEFAULT`）は `compose_negative` が前置する。
   temp 依存のリピートループを構造的に封じるための簡略化。
   **解消トリガ**: サンプリングとパースが十分堅牢だと確認できたら、モデルにフル negative を
   書かせる方式を再検討する。

4. **サンプリング明示**: `temperature=0.7 / top_p=0.95 / top_k=40 / repeat_penalty=1.1`
   を `Settings` に露出（会話・ツール引数生成の双方で使用）。`temperature=0.0` は不可
   （上記リピートループ）。

5. **thought チャネルは UI 非表示**（Phase 1）。内部ログにのみ残す。折りたたみ表示は後日。

6. **タグ検索は語彙検索主軸**（vImagen `services/tags.py` 移植予定、Phase 2）。EmbeddingGemma+
   Faiss は booru 語彙に構造的に不適（人気信号消失・実在検証不可・日本語弱・character 不透明）
   のため当面採用せず、純意味クエリが実運用で頻発すると確認できたら semantic-expand
   フォールバックとして追加検討する。

## Consequences

- 会話品質が大きく改善（実機スモークで率直ペルソナ・無検閲・状況尊重を確認）。
- `native_tools` と negative 簡略化は明示的な一時措置。解消トリガに達したら本 ADR を
  更新して撤去する。
- クライアントは会話テキストを Markdown 描画（`components/Markdown.tsx`、react-markdown）。
- Phase 2 でタグ検索ツール + エージェントループ（検索→検索→生成）を追加予定。パーサは
  複数ツール・ループ対応可能に設計済み（配線は未実施）。

## 既知の制約（native_tools パーサ）

- ツール呼び出しの閉じ検出 `<tool_call|>` は文字列値の内側を区別しない。プロンプト値が
  リテラル `<tool_call|>` を含むと途中で切れる。Danbooru タグ/英語キャプションに当該
  トークンが現れることは事実上無いため、過剰実装を避けて未対応とする（NOTE のみ）。
- 会話テキストは「最初のマーカーより前の領域」と定義する。モデルは実測でマーカー前に
  素テキストで会話を出すためこれで十分だが、仮に `<|channel>final<channel|>` に返答を
  包んだ場合その内容は表示されない（逐次表示との一貫性を優先した割り切り）。

## 多ターンの形式ドリフト対策（実機で判明・修正済み）

**症状**: 5 ターン前後で `generate_image` が壊れる。①モデルが `<|tool_call>` を吐かず、
履歴に見せていた記述的注記 `[generated an image — positive: "..."; ...]` を**そのまま模倣**
して出力（→ ツール未発火・注記が会話に漏れる）。②`<|tool_call>` は出すが enum 引数が
`aspect_ratio: "portrait"` の **ASCII クオート**に崩れ検証失敗。

**根本原因**: 履歴復元 `_reconstruct_assistant_turn` の記述的注記フォーマットが、モデルの
出力形式を汚染していた（自分が過去に「注記」で応答したと誤学習し、次ターンでそれを模倣）。

**修正**（実機多ターン再現で 3/7 劣化 → 2 連続 7/7 に改善）:
1. 過去ツール呼び出しを **native 形式 (`native_tools.render_tool_call`) で replay**。モデルが
   実際に出す `<|tool_call>call:...{k:<|"|>v<|"|>}<tool_call|>` に一致させ、模倣ドリフトを断つ。
2. パーサに **ASCII クオート除去 (`_unquote`)** を追加（残存ドリフトの保険）。

**将来のさらなる堅牢化（未実装・必要時に検討）**: forced `tool_choice` + JSON-schema grammar
の 2 パス（pass1=auto で会話+意図、pass2=強制でクリーンな引数）にすれば引数ドリフトは
**構造的に不可能**になる。コストはツール発火ターンでの追加 prompt-eval 1 回。今回の①②で
実機は安定したため未導入。NOTE: upstream #2227 は未解決・PR #2232 も未マージで、`create_chat_completion`
の native tool パースは 0.3.32 に無い（自前パーサが唯一の API 手段）ことを再確認済み。
- native の特殊トークンを assistant `content` に文字列で入れる方式は「テンプレに structured
  `tool_calls` を渡して native を再生成させる方が理論上正しい」との指摘もあるが、本実装は
  実機多ターンで 7/7 を確認済みのため文字列 replay を採用（tokenize 挙動依存の留保あり）。
