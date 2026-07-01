# Cocktail

日本語指示から Anima（2B DiT、Cosmos-Predict2 派生）で高品質イラストを生成する、マルチターンチャットアプリ。

## スタック

- **Server**: FastAPI + uvicorn（`uv` workspace、Python 3.12）
- **Client**: React + Vite + shadcn-ui（`pnpm` workspace）
- **LLM**: Gemma 4 12B heretic（llama.cpp / GGUF Q4、無検閲・日本語）。Blackwell(sm_120) では CUDA 有効でソースビルド
- **画像生成**: 公式 [diffusers](https://github.com/huggingface/diffusers) の Anima modular pipeline（bfloat16）。派生(WAI-Anima)は DiT 単体をメモリ内変換してベースに差し込む
- **高速化**: 既定で Anima Turbo LoRA（step & CFG 蒸留）を PEFT アダプタとして DiT に注入。steps / CFG を落として高速生成（`IMAGE_TURBO_LORA=` を空にすると base 品質）

## 現在のマイルストーン

**M1d — 起動一本化**。詳細は [ROADMAP.md](./ROADMAP.md)。

## Quick Start

前提: CUDA 対応 GPU（16GB VRAM 目安）、`uv`、`pnpm` がインストールされていること。

```bash
# 初回セットアップ（uv sync + pnpm install）
pnpm bootstrap

# .env を用意（AIR がデフォルトで wai-anima v10 を自動取得する）
cp .env.example .env

# サーバ起動（0.0.0.0:8000 待受）
# 初回のみ: モデル取得（数 GB〜10 GB）+ GPU プリロードで数分かかる。
pnpm dev:api

# UI も同時に立ち上げたいとき
pnpm dev
```

起動時の挙動:

1. `LLM_MODEL_ID`（GGUF）を取得（`repo:file.gguf` なら該当ファイルのみ、ローカル .gguf なら存在確認）
2. `IMAGE_BASE_MODEL_ID`（Anima ベースの diffusers リポ）を `snapshot_download` で取得
3. `IMAGE_MODEL_ID`（派生 DiT 単体）を取得
   - `urn:air:...` なら Civitai API で解決し `${WEIGHTS_DIR}/civitai/{slug}-{sha256[:12]}.{ext}` に配置
   - 明示ローカルパス(.safetensors) なら存在確認のみ
   - 空なら派生なし（ベースだけを使う）
4. `IMAGE_TURBO_LORA`（Turbo LoRA）を取得（空なら skip）。IMAGE_MODEL_ID と同じ解決規則
5. VRAM を検出して `residency_policy` を `swap` / `coresident` に決定
6. LLM をプリロード（coresident なら Image もプリロード）。Image のコールドロード時に Turbo LoRA を DiT へ注入
7. リクエスト受付開始

2 回目以降の起動は sha256 一致で再ダウンロードをスキップする。

### 動作確認

```bash
curl http://localhost:8000/health

curl http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"instruction_ja":"ピンクの髪の猫耳少女が星空の下で微笑んでいる絵"}'
```

## 開発コマンド

```bash
pnpm test        # pytest
pnpm typecheck   # mypy --strict + pnpm -r typecheck
pnpm lint        # ruff check + format --check
pnpm format      # ruff format + check --fix
pnpm gen-openapi # packages/api-types/openapi.json を更新
pnpm clean       # 各種キャッシュを削除
```

## トラブルシューティング

- 起動時に OOM で落ちる: `.env` で `RESIDENCY_MODE=swap` を明示する
- Civitai の gated モデルで 403: `.env` に `CIVITAI_TOKEN=...` を設定する
- sha256 不一致で起動中断: Civitai 側でファイルが差し替わった可能性。`IMAGE_MODEL_ID`(AIR) のバージョン ID を最新に更新する

## ライセンスに関する注意

Anima モデル重みは **Non-Commercial** です（CircleStone Labs + NVIDIA Open Model）。商用利用はできません。
