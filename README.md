# Cocktail

日本語指示から Anima（2B DiT、Cosmos-Predict2 派生）で高品質イラストを生成する、マルチターンチャットアプリ。

## スタック

- **Server**: FastAPI + uvicorn（`uv` workspace、Python 3.12）
- **Client**: React + Vite + shadcn-ui（`pnpm` workspace）
- **LLM**: Gemma 4 E4B（Transformers + bitsandbytes 4bit）
- **画像生成**: [diffusers-anima](https://github.com/hdae/diffusers-anima) の `AnimaPipeline`（bfloat16）

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

1. `LLM_MODEL_ID` の HF リポを `snapshot_download` で取得
2. `IMAGE_MODEL_ID` / `IMAGE_MODEL_AIR` に従って Image モデルを取得
   - HF リポ ID なら `snapshot_download`
   - ローカルパスなら存在確認のみ
   - AIR(URN) なら Civitai API で解決し `${WEIGHTS_DIR}/civitai/{slug}-{sha256[:12]}.{ext}` に配置
3. VRAM を検出して `residency_policy` を `swap` / `coresident` に決定
4. LLM をプリロード（coresident なら Image もプリロード）
5. リクエスト受付開始

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
- sha256 不一致で起動中断: Civitai 側でファイルが差し替わった可能性。`IMAGE_MODEL_AIR` のバージョン ID を最新に更新する

## ライセンスに関する注意

Anima モデル重みは **Non-Commercial** です（CircleStone Labs + NVIDIA Open Model）。商用利用はできません。
