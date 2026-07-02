# 0004. native チャネル文法の是正（span 規則）と検索往復のターン開放

- Status: Accepted
- Date: 2026-07-03
- Scope: `apps/server` の native 出力パース（`services/native_tools.py`）と
  エージェントループの検索往復（`services/llm.py` `_append_search_roundtrip`）
- 前提: [0001](0001-llm-conversation-harness.md)（自前パーサの導入。本 ADR はその形式解釈の
  誤りを訂正する）、[0002](0002-phase2-agent-loop.md)（検索往復）

## Context

実運用（多ターン）で「本文が表示されないターンが多発する」回帰が報告された。生ログ:

```
<|channel>thought\n<channel|>いいね、そのシチュエーション。…仕上げるね。\n\n<|channel>thought\n<channel|><|tool_call>call:generate_image…
```

0001 導入時の観測から、パーサは `<|channel>thought<channel|>` を「チャネルヘッダ」とみなし
「その後ろ〜次マーカーまで」を思考として隠していた。この解釈が誤りだった。

### 正文法（一次ソース 3 系統で確証）

| ソース | 証拠 |
| --- | --- |
| GGUF 埋込チャットテンプレート | 思考の書込形＝`'<|channel>thought\n' + 思考 + '\n<channel|>'`。`strip_thinking` マクロは `<|channel>`〜`<channel|>` の**間**を除去 |
| llama.cpp `peg-gemma4`（PR #21326） | `thought = "<|channel>thought" + reasoning(until "<channel|>")`。content は span の**外側**。裾の空 thought・孤立 `<channel|>` を明示的に許容 |
| ai.google.dev prompt-formatting-gemma4 / thinking | チャネルは `thought` **のみ**（"final" は存在しない）。可視回答は `<channel|>` の**後ろ**のプレーンテキスト |

つまり `<|channel>` が開き・`<channel|>` が閉じの **span** で、思考は span の中、
**本文は span の外側すべて**。

さらに、生成プロンプトは thinking 無効時（本プロジェクトの既定。heretic カードの推奨でも
ある）、空 thought `<|channel>thought\n<channel|>` をプリフィルして「本文から書け」と促す。
モデルは出力中にも **ghost の空 thought** を混ぜる既知の癖があり（Google がプリフィルで
抑止している対象そのもの）、旧文法はこの ghost の直後に来る本文を丸ごと「思考」と誤分類
して隠した。これが本文欠落の直接原因。0001 導入時に観測した
`<text><|channel>thought<channel|><思考>…` も、実際は「text ＋ ghost 空 thought ＋ 可視
テキスト」であり、当時「thought」とログしていた内容の一部は本文だった。

### 併発バグ: 検索往復がモデルターンを閉じる

テンプレートの実レンダリングで判明: assistant（`tool_calls` 付き）の `content` は
**ツール応答の後**に描画され、非空だと `<turn|>` でモデルターンが閉じ、続く生成プロンプト
も抑止される（`prev_message_type == 'tool_response'`）。つまり検索往復の replay に hop
テキストを入れていた従来実装は、**生成開始点が無い**状態で次ホップを生成させていた。
テンプレの設計意図は「継続ホップは content 空 → ターンを開いたまま
`<|tool_response>…` の直後から続けさせる」cascade。

### 悪化のメカニズム（なぜ「多発」し、機械ベンチで出なかったか）

誤分類 → 本文が空で永続化 → `_reconstruct_assistant_turn` が `(no response)` として
replay → モデルが「本文を書かない自分」に条件付けられるスパイラル。0003 の A/B ベンチは
本文入りの綺麗なスクリプト履歴を replay していたため、このスパイラルに構造的に盲目だった
（本文欠落 0/72。実運用でのみ多発する説明がつく）。

## Decision

1. **パーサを span 文法へ書き直す**（`native_tools.py`）。
   - `<|channel>…<channel|>` の中＝思考（チャネル名は問わず非表示。テンプレの
     `strip_thinking` も名前を見ない）。切断 span は末尾まで span 扱い。
   - 孤立 `<channel|>` は制御トークンとして落とし、前後の本文は温存（llama.cpp と同じ寛容則）。
   - `NativeToolStream` は状態機械化し、span が閉じたら可視テキストの emit を**再開**する
     （旧実装は最初のマーカーで打ち切っていた）。可視領域の判定規則は parse と streamer で
     一致させ、逐次表示と確定テキストの乖離を防ぐ。
   - ツール引数 DSL（`<|"|>` ラップ・素値・ASCII クオート吸収）は変更なし。
2. **検索往復の assistant `content` を空にする**（`llm.py`）。ターンを開いたまま継続させる。
   hop テキストは既にストリーム済みで最終 `result.text` にも累積されるため失われない。
3. **旧テストは書き直し**。旧テストは誤文法を仕様化していた（一次ソースで誤り確定）。
   ユーザー実ログ（ghost 空 thought に挟まれた本文）を回帰ケースとして固定。

## Consequences

- 実機スモーク 9/10: 本文欠落ゼロ（ghost チャネル出現 5/5 のシナリオ含む）、stream と確定
  テキストの一致 10/10。唯一の fail は「検索後 generate まで行かず会話で確定」でパース起因
  ではない（後述）。
- 0001 の「多ターン非収束・thought で止まる」観測の一部はパーサ誤分類が見せた幻だった
  可能性が高い。これを受け、収束を強要するプロンプト強化（`0694045`）は本修正の確認後に
  再評価する（別途）。
- llama-cpp-python の修正 PR #2232 がマージされたら、本モジュールごと構造化 `tool_calls`
  へ載せ替える（0001 の撤去トリガは不変。マージ時に取り込みを検討する）。
- 制約（既知・継続）: 文字列値の内側に現れるリテラル `<channel|>` / `<tool_call|>` は
  span 境界と区別しない（0001 の NOTE と同じ割り切り。Danbooru タグ/英語キャプションには
  事実上現れない）。
