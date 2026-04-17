# Cocktail ロードマップ

日本語指示から Anima
で高品質イラストを生成できるマルチターンチャットアプリを、動く最小から段階的に作る。

本ドキュメントは実装順の指針。詳細設計・背景は
`/home/developer/.claude/plans/elegant-dreaming-dusk.md` を参照。

---

## M0: API だけで画像が出る（完了）

**完了条件**: `curl` で日本語指示を送って、Anima が生成した webp
画像が返ってくる。→ **達成**

- [x] `uv` + `pnpm` モノレポ初期化（`apps/server` のみ実体、`apps/client` /
      `packages/api-types` は M1 で追加予定）
- [x] `.env.example`, `Makefile`, `.python-version`
- [x] `apps/server/pyproject.toml` — `fastapi`, `uvicorn[standard]`, `pydantic`,
      `pydantic-settings`, `pillow`, `torch>=2.8`, `transformers>=4.51`,
      `accelerate`, `bitsandbytes`, `diffusers>=0.36`,
      `diffusers-anima @ git+<pin>`
- [x] Gemma 4 E4B + bnb 4bit ロード（`coder3101/gemma-4-E4B-it-heretic`
      に差し替え済み、拒否挙動回避）
- [x] `build_anima_prompt` を `PromptSpec` pydantic モデルで構造化出力（※
      `outlines` ではなく **正規表現抽出 + pydantic validate + 1 回リトライ**）
- [x] `AnimaPipeline.from_pretrained("hdae/diffusers-anima-preview", torch_dtype=torch.bfloat16)`
      のロード
- [x] 896×1152 / steps=32 / cfg=4.0
      で画像生成（1024²より縦長の方が被写体フィット良）
- [x] `POST /generate` 同期 API、`GET /images/{id}.webp` 配信、`GET /health`
- [x] `uvicorn --host 0.0.0.0 --port 8000` で起動
- [x] コンテナ内 `curl` で日本語指示 → webp
      取得の疎通確認（プロンプトチューニングで多数検証）
- [x] `pytest` / `mypy --strict` / `ruff check` / `ruff format` が通る
- [x] `ModelManager` の `asyncio.Lock` で GPU 役割を直列化（※ `Semaphore(1)` は
      `Lock` と等価のため `Lock` で統一）
- [x] `torch.cuda.empty_cache()` + `gc.collect()` をアンロード時に実行

## M1: フル機能骨格（完了）

UI + SSE + Function Calling で「マルチターンで画像生成」できる状態に。

### M1a: API 契約とサーバ骨格（完了）

- [x] `POST /chat` SSE 対応（`sse-starlette` は未使用、FastAPI
      `StreamingResponse` + 手書きフレームで必要十分）
- [x] `Message` / `ContentPart`（`text` / `image` / `tool_call` /
      `tool_result`）判別 union スキーマ
- [x] SSE イベント: `conversation` / `user_saved` / `assistant_start` /
      `tool_call_start` / `tool_call_end` / `image_ready` / `assistant_end` /
      `error` / `done`
  - ※ 当初案の `token` / `tool_progress` は **M1
    では未実装**。単ターン固定呼び出しのため不要。ストリーミング応答が必要になった時点で追加
- [x] `POST /images` multipart upload（PIL で webp 正規化、10MB
      上限、PNG/JPEG/WebP 受付）
- [x] `ConversationStore`（in-memory、M3 で SQLite に差し替え予定）
- [x] `ChatOrchestrator` — M1 は **`generate_image` の固定呼び出し**（LLM-driven
      tool selection はツールが増える M5 から）
- [x] `ChatRequest` は少なくとも 1 つの text パートを要求（LLM を駆動するため）
- [x] ユーザ添付画像（`ImagePart`）は保存して `tool_call.args.reference_images`
      に記録。Gemma へ食わせるのは M2

### M1b: Web クライアント（完了）

- [x] `packages/api-types` を `openapi-typescript` で自動生成、`pnpm gen-openapi`
      タスク
- [x] `apps/client` を React + Vite + Tailwind で雛形（shadcn は M1c から）
- [x] `ChatView` / `ComposerInput` / `MessageList` / `MessagePart`
- [x] `@microsoft/fetch-event-source` で SSE 受信、`ContentPart`
      型に合わせて部品単位でレンダリング
- [x] `POST /images` のドラッグ&ドロップ・ペースト添付（※ M1c で UI から削除、API
      は残存）
- [x] Zustand で会話状態を管理（TanStack Query/Router は単画面の現時点では不要）
- [x] Zod スキーマで API 境界のランタイム検証（OpenAPI 型と双方向整合）
- [x] ブラウザで「描いて」→ 画像表示、1 会話内で連続生成

### M1c: Gemma 自律 + UI 磨き（完了）

Gemma が喋り、ツール呼び出しの引数も自分で選ぶようにし、UI
を落ち着いたトーンにそろえた。詳細は
`/home/developer/.claude/plans/elegant-dreaming-dusk.md` M1c セクション参照。

- [x] `LlmTurnSpec { reasoning, tool_calls }` 形式の構造化 JSON 出力に移行
      （native tools は M2+ で検討）
- [x] `GenerateImageCall` に `aspect_ratio` / `cfg_preset` / `seed` を持たせ、Gemma
      が選択
  - `aspect_ratio`: portrait 896×1152 / landscape 1152×896 / square 1024×1024
  - `cfg_preset`: soft 3.5 / standard 4.0 / crisp 4.5
  - `seed`: 省略時はターンごとにランダム（`secrets.randbelow`）、指定時はその値を使用
    — **M2 で `seed_action: "new" | "keep"` に置換予定**
- [x] `transformers.TextIteratorStreamer` でトークン逐次受信、正規表現 +
      状態機械で `reasoning` を partial JSON デコードして `TextDeltaEvent` に変換
- [x] `TextDeltaEvent` を SSE union に追加（Python/Zod 両面）
- [x] Gemma が `tool_calls=[]` を返した場合は画像生成せず `TextPart`
      のみで assistant ターンを閉じる
- [x] `diffusers-anima` を `1a59c55` に更新（Comfy 形式
      `waiANIMA_v10.safetensors` 対応）
- [x] `.gitignore` に `*.safetensors` / `*.ckpt`
- [x] `@radix-ui/react-scroll-area` 導入、shadcn 式 `components.json` +
      `ui/scroll-area.tsx`
- [x] 添付ボタン・paperclip・ドラッグ&ドロップ・ペースト画像 UI を
      `ComposerInput` から除去（`POST /images` は残置）
- [x] 水平線（ヘッダ / エラー / コンポーザ）を除去、トーンを落ち着いた無彩色に
- [x] `textarea` を `h-10 min-h-10 max-h-40 field-sizing:content`
      に、送信ボタンと高さ一致
- [x] 例文を単語羅列から「夕暮れの教室…」風の文章 3 例に差し替え
- [x] アシスタント吹き出し: 先頭に Gemma の reasoning、実行中は小スピナー +
      「画像を生成しています…」の 1 行、完了後は `<details>`
      で positive / aspect_ratio / cfg_preset / seed / resolution を展開

### M1d: 起動整備（完了）

「起動＝即使える」を成立させる基盤整備。Makefile を廃し、モデル取得と VRAM
運用切替を起動パイプラインに統合。詳細は同プラン M1d セクション参照。

- [x] ルート `package.json` に pnpm scripts 集約、`concurrently` で
      `pnpm dev` の並列起動、Makefile を削除
- [x] `fetch_models.ensure_all()` で HF snapshot / Civitai AIR(URN) /
      ローカルパスを統一取得、sha256 検証 + atomic rename、起動 lifespan に統合
- [x] `IMAGE_MODEL_ID` が HF リポ ID / `.safetensors` / AIR(URN) いずれも受けられる
      — デフォルト AIR は wai-anima v10
- [x] `RESIDENCY_MODE=auto|swap|coresident` + VRAM 閾値で `ModelManager`
      の evictor を自動選択、coresident 時は LLM/Image 両方を起動時プリロード
- [x] lifespan をノンブロッキング化（ポート bind は即時、モデル準備はバックグラウンド）、
      `/health` に `startup.state` を公開、UI に起動ステータスバナー
- [x] `/health` に `residency_policy` / `vram_total_gb` / `vram_free_gb` を追加
- [x] チャット画像プレビューの最大高を 400px に制限

## M2: 画像メモリ + ギャラリー + シードツール化（次の最優先）

**なぜここか**: Gemma
は現状、前ターンで自分が生成した画像を覚えていない。「もう少し赤を強く」
のような継続指示が成立しない。セッション管理 + 画像入力
経路を先に通し、あわせて UI 側からも生成画像を俯瞰できるギャラリーを出す。
乱数をめぐる Gemma の不安定さもここで潰す。

タグ RAG と参照画像の「再現」は
**このマイルストーンでは扱わない**（タグ辞書は後回し、bit-perfect 再現は Anima
の bf16 + bnb の非決定性で原理的に不可）。

### セッションメタデータと生成画像の来歴

- [ ] `ConversationStore` の既存 UUID を `conversation_id`
      として正式化し、`created_at` / `updated_at` / `generated_image_ids: list[str]`
      をメタデータに追加
- [ ] 生成画像の保存規約を確認: 現状
      `./data/images/{uuid}.webp`（UUID）。**ハッシュ化は見送り**（bnb 量子化 +
      bf16 の非決定性で同一 seed/prompt でも差分が出るため、重複排除は成立しない）
- [ ] `GenerateImageCall` 実行時、画像 ID とパラメータ（positive /
      aspect_ratio / cfg_preset / seed / resolution）をそのセッションの
      `generated_images: list[GeneratedImageRef]` に追記

### ギャラリー（UI）

- [ ] `GET /images` — 生成済み画像のメタデータ一覧（image_id, conversation_id,
      created_at, prompt 抜粋, seed, aspect_ratio）。ページング（`?limit=&cursor=`）
- [ ] サイドバーまたはヘッダから切替できる `Gallery` ビュー — 等間隔グリッド +
      遅延読み込み
- [ ] サムネクリックで元会話へ遷移（M3 で会話 URL ルーティングができたら接続）、
      M2 では詳細パネル（同じパラメータを再試行 / 元 prompt をコンポーザに流し込み）

### Gemma の vision 入力（前回生成画像の再確認）

- [ ] Gemma 4 の画像入力パス整備 — processor 経由で `image + text`
      を Gemma へ投げる。`apps/server/src/cocktail_server/services/llm.py` の
      `generate_turn_spec` に `attached_images: list[Path]` を追加
- [ ] `ChatOrchestrator` が Gemma 呼出時、**直前のアシスタントターンの
      `ImagePart`** を自動で同梱（「もう少し赤を強く」を成立させる最小経路）
- [ ] ユーザが `POST /images` でアップロードした `reference_images`
      も同じ経路で Gemma に食わせる（今は保存されているだけで読まれていない）
- [ ] `ComposerInput` に画像添付 UI を**再導入**（M1c で外したペースト/DnD
      を戻す）。プレビュー付き
- [ ] 画像の渡し方は最大 N 枚（初期 N=2）で切る — VRAM/文脈長の保険

### シードツール化（Gemma は値を持たせない）

- [ ] `GenerateImageCall` の `seed: int | None` を
      `seed_action: Literal["new", "keep"]` に置換
  - `"new"`（既定）: サーバが `secrets.randbelow` で毎回新規採番
  - `"keep"`: 直前ターンで使った seed をセッションから復元して再利用
- [ ] `ConversationStore` に `last_image_seed: int | None`
      を持たせ、ターン完了時に更新
- [ ] `prompt_builder.py` のツール仕様から `seed?: number` 記述を除去し、
      `seed_action: "new" | "keep"` を書く
- [ ] `llm.py` の引数抽出で `seed` の int 受付をやめ、`seed_action` をパース
- [ ] `generate.py` の `POST /generate` は **開発/デバッグ用**として
      `seed: int | None` を残す（API 経由での決定論検証に必要）
- [ ] アシスタント吹き出し `<details>` の seed 表示は「seed: 12345 (new)」
      のように採択経路を併記
- [ ] `tests/test_orchestrator.py` に `"keep"` が last_seed
      を踏襲する/`"new"` が新規採番するケースを追加
- [ ] ドキュメント/README にも反映

## M3: 会話永続化（SQLite）

M1 の `ConversationStore`（in-memory）を SQLite
バックエンドに差し替える。M2 でメタデータ項目が固まった後に入れる順序にする。

- [ ] SQLite + `sqlalchemy[asyncio]` + `aiosqlite`、`Conversation` / `Message` /
      `Attachment` / `GeneratedImage`
- [ ] `ConversationStore` の SQLite 実装（async インタフェースは M1
      で既に整っている）
- [ ] `GET /conversations` 一覧 / `GET /conversations/{id}` 詳細 /
      `DELETE /conversations/{id}`
- [ ] サイドバーで会話一覧・切替・削除、URL ルーティング
      （`/c/:conversationId`、ギャラリーからも飛べる）
- [ ] `PATCH /messages/{id}` でユーザメッセージ編集 → `parent_id`
      で分岐再生成（`Message.parent_id` は M1 で用意済み）
- [ ] `POST /chat/{msg_id}/regenerate`
- [ ] エラーリトライ・キュー表示

## M4: マルチモーダル深堀り

M2 で Gemma への画像入力経路は既に通っている。M4
はその上で自動フィードバックと編集系を積む。

- [ ] `vision_feedback` ツール — 生成画像を Gemma が見て改善案を出し、自動再生成 1 回
- [ ] img2img / inpainting エンドポイント（Anima パイプライン側の対応確認含む）
- [ ] ユーザ提示画像を「参考」ではなく「下絵」として使うモード分け

## M5: Danbooru タグ RAG（オプショナル）

タグ辞書なしでも動く設計を維持。**参照画像の再現は M5 でも対象外**
（bnb 量子化ノイズで bit-perfect は不可、seed 復元は M2 で既にカバー済み）。

- [ ] EmbeddingGemma 300m を常駐に追加（fp16, ~0.6GB）
- [ ] ChromaDB + Matryoshka 256 次元
- [ ] `scripts/ingest_danbooru.py`（HF dataset + Danbooru wiki）
- [ ] `search_tags` ツールを Function Calling に追加
- [ ] Gemma 単体でも Anima 記法を守れる前提は維持

## M6: 仕上げ

- [ ] 仮想スクロール、アクセシビリティ、キーボードショートカット
- [ ] Docker / docker-compose（GPU pass-through）
- [ ] Playwright E2E
- [ ] `/health` にキュー深さ・推定待ち時間
