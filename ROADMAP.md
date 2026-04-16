# Cocktail ロードマップ

日本語指示から Anima で高品質イラストを生成できるマルチターンチャットアプリを、動く最小から段階的に作る。

本ドキュメントは実装順の指針。詳細設計・背景は `/home/developer/.claude/plans/elegant-dreaming-dusk.md` を参照。

---

## M0: API だけで画像が出る（最優先・目安 1 週間）

**完了条件**: `curl` で日本語指示を送って、Anima が生成した webp 画像が返ってくる。

- [ ] `uv` + `pnpm` モノレポ初期化（`apps/server`, `apps/client`（空）, `packages/api-types`（空））
- [ ] `.env.example`, `Makefile`, `.python-version`
- [ ] `apps/server/pyproject.toml` — `fastapi`, `uvicorn[standard]`, `pydantic`, `pydantic-settings`, `pillow`, `torch>=2.8`, `transformers>=4.51`, `accelerate`, `bitsandbytes`, `diffusers>=0.36`, `diffusers-anima @ git+<pin>`
- [ ] Gemma 4 E4B + bnb 4bit ロード
- [ ] `build_anima_prompt` を `PromptSpec` pydantic モデルで構造化出力（`outlines` または `transformers` json_mode）
- [ ] `AnimaPipeline.from_pretrained("hdae/diffusers-anima-preview", torch_dtype=torch.bfloat16)` のロード
- [ ] 1024×1024 / steps=32 / cfg=4.0 で画像生成
- [ ] `POST /generate` 同期 API、`GET /images/{id}` webp 配信、`GET /health`
- [ ] `uvicorn --host 0.0.0.0 --port 8000` で起動
- [ ] コンテナ内 `curl` で日本語指示 → webp 取得の疎通確認
- [ ] `pytest` / `mypy --strict` / `ruff check` が通る
- [ ] `ModelManager` の `asyncio.Lock` + GPU `asyncio.Semaphore(1)` で直列化
- [ ] `torch.cuda.empty_cache()` + `gc.collect()` を推論後に実行

## M1: フル機能骨格（目安 2–3 週間）

UI + SSE + Function Calling で「マルチターンで画像生成」できる状態に。

- [ ] `sse-starlette` 追加、`POST /chat` SSE 対応
- [ ] Function Calling ループ（ツールは `generate_image` のみ）、Gemma の `apply_chat_template`
- [ ] SSE イベント: `user_saved` / `token` / `tool_start` / `tool_progress` / `tool_end` / `image_ready` / `error` / `done`
- [ ] React + Vite + Tailwind + shadcn-ui で `ChatView` / `ComposerInput` / `MessageList`
- [ ] `@microsoft/fetch-event-source` で token + `image_ready` 受信
- [ ] `POST /images` multipart upload エンドポイント（箱だけ）
- [ ] `openapi-typescript` で `packages/api-types` 自動生成、`Makefile` の `gen-types` タスク
- [ ] Zustand + TanStack Query/Router で状態とルーティング
- [ ] 完了条件: ブラウザで「猫の女の子を描いて」→ 画像表示、1 会話内で連続生成

## M2: 会話永続化・編集（目安 1 週間）

- [ ] SQLite + `sqlalchemy[asyncio]` + `aiosqlite`、`Conversation` / `Message` / `Attachment`
- [ ] サイドバーで会話一覧・切替・削除
- [ ] `PATCH /messages/{id}` でユーザメッセージ編集 → `parent_id` で分岐再生成
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

- [ ] Gemma 4 の画像入力パス（processor 経由）
- [ ] `vision_feedback` ツール — 生成画像を Gemma が見て改善案 → 自動再生成 1 回
- [ ] img2img / inpainting エンドポイント

## M5: 仕上げ（目安 1 週間）

- [ ] 仮想スクロール、アクセシビリティ、キーボードショートカット
- [ ] Docker / docker-compose（GPU pass-through）
- [ ] Playwright E2E
- [ ] `/health` にキュー深さ・推定待ち時間
