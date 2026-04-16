# Cocktail

日本語指示から Anima（2B DiT、Cosmos-Predict2 派生）で高品質イラストを生成する、マルチターンチャットアプリ。

## スタック

- **Server**: FastAPI + uvicorn（`uv` workspace、Python 3.12）
- **Client**: React + Vite + shadcn-ui（`pnpm` workspace、M1 以降）
- **LLM**: Gemma 4 E4B（Transformers + bitsandbytes 4bit）
- **画像生成**: [diffusers-anima](https://github.com/hdae/diffusers-anima) の `AnimaPipeline`（bfloat16）

## 現在のマイルストーン

**M0 — API だけで画像が出る**。詳細は [ROADMAP.md](./ROADMAP.md)。

## Quick Start（M0）

前提: CUDA 対応 GPU（16GB VRAM 目安）と `uv`、`pnpm` がインストールされていること。

```bash
# 初回セットアップ
make install

# モデル事前ロード（初回のみ時間がかかる）
make warmup

# サーバ起動（0.0.0.0:8000 待受）
make api-only

# 別ターミナルから疎通確認
curl http://localhost:8000/health

# 画像生成
curl http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"instruction_ja":"ピンクの髪の猫耳少女が星空の下で微笑んでいる絵"}'
```

## 開発コマンド

```bash
make test       # pytest
make typecheck  # mypy --strict
make lint       # ruff check + format --check
make format     # ruff format + check --fix
```

## ライセンスに関する注意

Anima モデル重みは **Non-Commercial** です（CircleStone Labs + NVIDIA Open Model）。商用利用はできません。
