# 0006. 事前検索 Stage 2（ベクトル検索）— 保留

- Status: **Deferred（採否ゲート未達につき一旦保留。ユーザ判断）**
- Date: 2026-07-03
- Scope: `apps/server` のタグベクトル索引（`services/tag_vectors.py` — 実装済み・未配線）
- 前提: [0005](0005-tag-hints-presearch.md)（Stage 1 辞書照合と採否ゲートの合意）、
  [0001](0001-llm-conversation-harness.md)（埋め込み検索の保留判断 — 本 ADR で実測を追加）

## Context

Stage 1（辞書完全一致）が拾えない「日本語 alias を持たない英語 general タグ（実測 62%）への
言い換え・情景語」をベクトル検索で橋渡しする計画（0005 の Stage 2）。並列リサーチで
埋め込みモデルを再調査（敵対的検証つき）した上で、日本語情景クエリ 24 本
（Stage 1 非ヒットを機械検証済み・下記付録）の hit@k ゲートで実測した。

事前登録ゲート: A+B 帯 per-primary hit@8 ≥ 0.70 かつ hit@3 ≥ 0.50、C 帯 hit@8 ≥ 0.50。

## 実測結果（採否ゲート）

| 構成 | A+B hit@8 | A+B hit@3 | C hit@8 | any-hit/24 |
| --- | --- | --- | --- | --- |
| **EmbeddingGemma-300m Q8 / 文書=タグ名のみ / β=0.08** | **0.613** | **0.452** | 0.545 ✓ | 17 |
| EmbeddingGemma Q8 / タグ名+alias | 0.548 | 0.355 | 0.455 | 18 |
| EmbeddingGemma BF16 / タグ名+alias | 0.548 | 0.387 | 0.455 | 18 |
| multilingual-e5-base Q8（最良 β） | 0.452 | 0.355 | 0.273 | 10 |
| bge-m3 Q8 dense（最良 β） | 0.387 | 0.226 | 0.455 | 14 |
| DanbotNL-2408-260m（生成型・案Y） | 0.226 | 0.032 | 0.182 | 8 |

スコア = cosine + β·log10(1+post_count)、候補プール 256、8 枠（`_HINT_TOP_N` と一致）。

### 確定した設計知識（再開時に再学習しないこと）

- **モデル序列はリサーチどおり**: EmbeddingGemma（XTREME-UP 47.72 / XOR-ja 81.33 の
  クロスリンガル実測が突出）> bge-m3 > e5-base。Qwen3-Embedding-0.6B は敵対的検証で
  refuted（XTREME-UP 6.64）、ruri v3 は日本語単言語で「62% 問題」により主軸不可。
- **人気事前分布 `+β·log10(post_count)` は必須**（EG: 0.286 → 0.595）。純 cosine は
  `wet_pavement`(post 73) 級の極小タグが `rain`(45k) を押しのけ、Anima の学習が薄い
  タグばかり返す。β スイート・スポットは 0.04–0.08。
- **文書はタグ名のみが最良**。alias 連結（日英中混在）は EG のクロスリンガル照合には
  ノイズ（当初仮説の逆）。Q8 量子化は無罪（BF16 と同等）。読点でのクエリ分割は逆効果
  （文脈喪失で petals 1602→17301 位）。
- **残る構造要因** = 「言い換え → 平易な正準タグ」の橋渡し不足（snow 377 位・ocean 422 位・
  tears 1636 位）。タグ名 1 語の文書では意味の張りが足りない。
- **案Y（DanbotNL 生成型）は不採用**: ①生成順が関連度順でなく汎用フィラー
  （no humans / medium 系）が 8 枠を先に埋める — 枠を絞る根拠スコアを持たない構造欠陥
  ②言い換え・気分・カメラ語で完全脱線（「うれしそうな表情」→鎧・ドラゴン列）。
  NOTE: trust_remote_code はスナップショット `eef951b` の 3 ファイルを全文レビュー
  （純 torch 合成・I/O/exec なし）後、隔離 venv（transformers 4.51/CPU）で実行した。
- **未試行の次善 = 案X（wiki グロス増強）**: Danbooru wiki 本文の先頭 1-2 文を文書へ付与
  して再走。データ源は HF に実在確認済み（`deepghs/danbooru_wikis_full`、
  `isek-ai/danbooru-wiki-2024`、2026 年版複数）。上記の構造要因への正攻法で、再開時は
  ここから。

## Decision

**Stage 2 を一旦保留する**（ユーザ判断。案X 未試行のまま凍結、案Z=既定 OFF 搭載も見送り）。

- `services/tag_vectors.py`（索引ビルダ・fail-loud アーティファクト・cosine 検索、
  テスト付き）は**未配線の資産としてリポジトリに残す**。serving への配線・Settings・
  埋め込みハンドル常駐は一切入っていない。完全撤去する場合は該当コミットを revert する。
- Stage 1（0005）は本番既定 ON のまま変更なし。

## Consequences

- 再開時のコスト: 案X の文書再ビルドは EG で 35k 文書 ~1 分、ゲート再走は付録のクエリ
  集合で機械的に再現できる。
- モデル取得物（HF キャッシュ `data/models/` 配下、計 ~2.6GB: EmbeddingGemma Q8/BF16・
  e5-base Q8・bge-m3 Q8・DanbotNL）は削除可能（再取得容易）。
- 0001 の「埋め込みは booru 語彙に不適」は本実測で部分的に裏付けられた（純 cosine では
  不適。人気事前分布と文書設計を足しても現状ゲート未達）。

## 付録: 採否ゲートのクエリ集合（24 本・Stage 1 非ヒット検証済み）

band A（情景/時間帯/天候）:

| id | クエリ | primary |
| --- | --- | --- |
| A1 | 日が沈みかけの空をバックにした帰り道っぽい場面 | sunset, street |
| A2 | 雨があがった直後、路面の水に景色が映りこんでる | rain, reflection, puddle |
| A3 | 満天の星の下で | starry_sky, night |
| A4 | 葉のすき間から差す光がまだらに落ちる林の中 | dappled_sunlight, forest |
| A5 | 潮風の匂いがして、青い水面がどこまでも広がってる | ocean |
| A6 | しんしんと白いものが降り積もる寒い季節 | snow, snowing, winter |
| A7 | 春先、薄紅色の花が舞い散る並木道 | cherry_blossoms, petals |
| A8 | 夜の繁華街、色とりどりの電飾がまぶしい | neon_lights, city_lights, night |

band B（構図/カメラ）:

| id | クエリ | primary |
| --- | --- | --- |
| B1 | 真下から見上げたような迫力のあるアングル | from_below |
| B2 | 背中を向けたまま、ふと振りむいた一瞬 | looking_back, from_behind |
| B3 | 画面をわざと斜めに傾けた不安定な構図 | dutch_angle |
| B4 | 顔にぐっと寄った画で | close-up, portrait |
| B5 | 腰から上だけが写る距離感 | upper_body |
| B6 | 人物は小さめで、まわりの景色がメインの引きの画 | wide_shot, scenery |
| B7 | 光を背にして、輪郭だけがふちどられて浮かぶ | backlighting, silhouette |
| B8 | 手前の人物にピントを合わせて、うしろはぼんやり | depth_of_field, blurry_background |

band C（気分/抽象 — 緩い基準）:

| id | クエリ | primary |
| --- | --- | --- |
| C1 | どこかもの悲しくて、心が沈むような空気感 | sad |
| C2 | 感情が読めない、静かな顔つき | expressionless |
| C3 | 今にもまぶたが落ちそうな、まどろみの時間 | sleepy |
| C4 | こらえきれずに目もとがうるんでる | tears, crying |
| C5 | 思わずこぼれるうれしそうな表情 | smile, grin |
| C6 | 一瞬の気のゆるみもない、張りつめた目つき | serious |
| C7 | まぶしい光の粒がふわふわ漂う、夢の中みたいな画面 | light_particles, glowing |
| C8 | 色あせた古い写真みたいな雰囲気 | sepia |
