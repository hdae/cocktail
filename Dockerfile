# syntax=docker/dockerfile:1.7

# ------------------------------------------------------------------------------
# Cocktail: RunPod 等での実行を想定した軽量 Docker イメージ。
#
# 狙い:
#   - イメージに torch / transformers などを焼き込まない (RunPod の Docker pull は遅い)。
#   - 代わりに entrypoint で `uv sync --frozen` を走らせ、/workspace (Network Volume) に
#     venv と HuggingFace キャッシュを永続化する。2 回目以降の起動はほぼ瞬時。
#
# 前提:
#   - `apps/client/dist` を事前にビルド済み (`pnpm --filter @cocktail/client build`)。
#     このイメージはクライアントをビルドしない。dist をそのままコピーして同一オリジン配信する。
#   - ホスト側に NVIDIA ドライバ + nvidia-container-toolkit。Blackwell 系なら CUDA 12.8 相当。
#     CUDA runtime 自体は PyTorch / bitsandbytes の wheel に同梱されるのでイメージには入れない。
#
# 使い方 (ローカル):
#   pnpm --filter @cocktail/client build
#   docker build -t cocktail:latest .
#   docker run --gpus all -p 8000:8000 -v cocktail-ws:/workspace cocktail:latest
# ------------------------------------------------------------------------------
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
    && rm -rf /var/lib/apt/lists/*

# uv / uvx バイナリは astral の公式イメージから持ってくる。apt に無く、公式スクリプトの
# curl | sh より再現性が高い。
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

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
