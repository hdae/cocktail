# syntax=docker/dockerfile:1.7

# ==============================================================================
# Stage 1 — llama-builder
# llama-cpp-python を CUDA(Blackwell sm_120) 有効でソースビルドし wheel を作る。
# PyPI/事前ビルド wheel の CUDA カーネルは sm_120 で動かない（実機で SIGILL を確認）
# ため、対象アーキ向けに自前コンパイルした wheel をランタイムへ持ち込む。
# 重い CUDA toolkit はこの builder にだけ置き、ランタイムイメージは軽量に保つ。
# ==============================================================================
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS llama-builder

# lock とバージョンを合わせること（apps/server/pyproject.toml / uv.lock）。
ARG LLAMA_CPP_VERSION=0.3.32
# 対象 GPU の compute capability(セミコロン区切り)。CUDA 12.8 で RTX 3000 番代以降を広く
# カバー: 86=Ampere(RTX30xx/A100は80), 89=Ada(RTX40xx), 120=Blackwell(RTX50xx)。
# 載せる世代を絞ればビルドは速く wheel も小さくなる。
ARG CUDA_ARCHITECTURES=86;89;120

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# GGML_CUDA を有効化し、sm_120 のカーネルを焼いた wheel を /wheels に出力する。
# ランタイムの cudart/cublas は torch 同梱のものを使うので、ここでは静的リンクしない。
ENV CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}"
RUN uv pip wheel --python-preference only-managed --python 3.12 \
        --no-binary llama-cpp-python \
        "llama-cpp-python==${LLAMA_CPP_VERSION}" \
        --wheel-dir /wheels

# ==============================================================================
# Stage 2 — runtime
# RunPod 等での実行を想定した軽量イメージ。torch / transformers / diffusers などは
# イメージに焼かず、entrypoint の `uv sync` で /workspace(Network Volume) に入れる。
# llama-cpp-python だけは CUDA ビルドが必要なので builder の wheel を --find-links で使う。
#
# 前提:
#   - `apps/client/dist` を事前ビルド済み（`pnpm --filter @cocktail/client build`）。
#   - ホストに NVIDIA ドライバ + nvidia-container-toolkit。Blackwell は CUDA 12.8 相当。
#     CUDA runtime(cudart/cublas) は PyTorch の wheel 同梱分を llm.py が RTLD で先読みする。
#
# 使い方 (ローカル):
#   pnpm --filter @cocktail/client build
#   docker build -t cocktail:latest .
#   docker run --gpus all -p 8000:8000 -v cocktail-ws:/workspace cocktail:latest
# ==============================================================================
FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # uv の動作パラメータ。
    #   UV_LINK_MODE=copy: volume を跨ぐ hardlink を避ける (RunPod で失敗する)。
    #   UV_COMPILE_BYTECODE=1: .pyc を生成して起動を速くする。
    #   UV_PYTHON_PREFERENCE=only-managed: debian に python を入れないため、uv に
    #     Python 自体もダウンロードさせる。
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_PREFERENCE=only-managed \
    # venv / Python / uv cache をすべて /workspace に寄せる。RunPod では /workspace が
    # Network Volume 扱いで永続化されるので、これで 2 回目以降の起動が速くなる。
    UV_PYTHON_INSTALL_DIR=/workspace/.uv-python \
    UV_PROJECT_ENVIRONMENT=/workspace/.venv \
    UV_CACHE_DIR=/workspace/.uv-cache \
    # アプリの保存先も同じく /workspace 配下に揃える (Settings 側は case-insensitive)。
    HF_HOME=/workspace/models \
    IMAGES_DIR=/workspace/images \
    WEIGHTS_DIR=/workspace/weights

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        libgomp1 \
        # builder で CUDA ビルドした llama-cpp-python(ggml C++) が動的リンクする C++ ランタイム。
        libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# uv / uvx バイナリは astral の公式イメージから持ってくる。apt に無く、公式スクリプトの
# curl | sh より再現性が高い。
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# builder でビルドした llama-cpp-python の CUDA wheel。entrypoint の uv sync が
# UV_FIND_LINKS 経由でこれを使い、sdist からの CPU ビルドを避ける。
COPY --from=llama-builder /wheels /opt/wheels

WORKDIR /app

# 依存解決に必要なメタデータとソース。client は dist のみ。
COPY pyproject.toml uv.lock ./
COPY apps/server/pyproject.toml ./apps/server/pyproject.toml
COPY apps/server/src ./apps/server/src
COPY apps/client/dist ./apps/client/dist

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
