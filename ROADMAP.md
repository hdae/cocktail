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

## M1: フル機能骨格（進行中）

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
- [x] `ConversationStore`（in-memory、M2 で SQLite に差し替え予定）
- [x] `ChatOrchestrator` — M1 は **`generate_image` の固定呼び出し**（LLM-driven
      tool selection はツールが増える M3 から）
- [x] `ChatRequest` は少なくとも 1 つの text パートを要求（LLM を駆動するため）
- [x] ユーザ添付画像（`ImagePart`）は保存して `tool_call.args.reference_images`
      に記録。Gemma へ食わせるのは M4

### M1b: Web クライアント（未着手）

- [ ] `packages/api-types` を `openapi-typescript` で自動生成、`Makefile` に
      `gen-types` タスク
- [ ] `apps/client` を React + Vite + Tailwind + shadcn-ui で雛形
- [ ] `ChatView` / `ComposerInput` / `MessageList`
- [ ] `@microsoft/fetch-event-source` で SSE 受信、`ContentPart`
      型に合わせて部品単位でレンダリング
- [ ] `POST /images` のドラッグ&ドロップ・ペースト添付
- [ ] Zustand + TanStack Query/Router で状態とルーティング
- [ ] Zod スキーマで API 境界のランタイム検証（OpenAPI 型と双方向整合）
- [ ] 完了条件: ブラウザで「猫の女の子を描いて」→ 画像表示、1 会話内で連続生成

## M2: 会話永続化・編集（目安 1 週間）

M1 の `ConversationStore`（in-memory）を SQLite
バックエンドに差し替える。スキーマは `Message` / `ContentPart` を流用。

- [ ] SQLite + `sqlalchemy[asyncio]` + `aiosqlite`、`Conversation` / `Message` /
      `Attachment`
- [ ] `ConversationStore` の SQLite 実装（async インタフェースは M1
      で既に整っている）
- [ ] `GET /conversations` 一覧 / `GET /conversations/{id}` 詳細 /
      `DELETE /conversations/{id}`
- [ ] サイドバーで会話一覧・切替・削除
- [ ] `PATCH /messages/{id}` でユーザメッセージ編集 → `parent_id`
      で分岐再生成（`Message.parent_id` は M1 で用意済み）
- [ ] `POST /chat/{msg_id}/regenerate`
- [ ] エラーリトライ・キュー表示

## M3: Danbooru タグ RAG（オプショナル・目安 1 週間）

RAG ありでも無しでも動く設計を維持。

- [ ] EmbeddingGemma 300m を**ここで初めて**常駐に追加（fp16, ~0.6GB）
- [ ] ChromaDB + Matryoshka 256 次元
- [ ] `scripts/ingest_danbooru.py`（HF dataset + Danbooru wiki）
- [ ] `search_tags` ツールを Function Calling に追加
- [ ] Gemma 単体でも Anima 記法を守れる前提は維持

## M4: マルチモーダル（目安 1 週間）

アップロード経路（`POST /images` と `ImagePart` / `reference_images` の echo）は
M1 で既に通っている。M4 は **Gemma に画像を読ませる**部分。

- [ ] Gemma 4 の画像入力パス（processor 経由）、`reference_images` を
      orchestrator から Gemma へ受け渡し
- [ ] `vision_feedback` ツール — 生成画像を Gemma が見て改善案 → 自動再生成 1 回
- [ ] img2img / inpainting エンドポイント

## M5: 仕上げ（目安 1 週間）

- [ ] 仮想スクロール、アクセシビリティ、キーボードショートカット
- [ ] Docker / docker-compose（GPU pass-through）
- [ ] Playwright E2E
- [ ] `/health` にキュー深さ・推定待ち時間
