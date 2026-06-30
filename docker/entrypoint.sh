#!/bin/sh
# RunPod / 一般 Docker 環境向けエントリポイント。
#
# 方針:
# - コンテナ起動時に `uv sync --frozen` で仮想環境を作る。イメージに torch などを
#   焼き込まないのは RunPod の Docker pull が遅いため。uv は lock を見るだけなので、
#   volume に venv が残っていれば 2 回目以降は秒で戻る。
# - 重みキャッシュ (HF_HOME など) も /workspace に寄せる前提なので、初回起動以外は
#   モデル取得もスキップされる。
set -eu

cd /app

# volume が未マウントでも落ちないように /workspace を作っておく。
# uv / HF の個別ディレクトリは各ツールが遅延生成するのでここでは触らない。
mkdir -p /workspace

if [ "${SKIP_UV_SYNC:-0}" != "1" ]; then
    echo "[entrypoint] uv sync --frozen (env=${UV_PROJECT_ENVIRONMENT:-.venv})"
    # llama-cpp-python は lock 上 sdist のみ。素の uv sync はそれを CPU ビルドしようとし、
    # runtime にはコンパイラが無いため失敗する。よって llama だけ除外して sync し、builder が
    # 焼いた CUDA(sm_120) ビルド済み wheel を直接インストールする。
    uv sync --frozen --no-install-package llama-cpp-python
    installed=0
    for whl in /opt/wheels/llama_cpp_python-*.whl; do
        [ -e "$whl" ] || continue
        echo "[entrypoint] installing CUDA llama-cpp-python wheel: $whl"
        uv pip install --python "${UV_PROJECT_ENVIRONMENT:-/app/.venv}/bin/python" "$whl"
        installed=1
    done
    if [ "$installed" = "0" ]; then
        echo "[entrypoint] WARNING: /opt/wheels に CUDA llama wheel が無い — LLM は動作しません" >&2
    fi
fi

# uv run でなく venv の python を直接使う。uv run は毎回 sync するので無駄な I/O。
exec "${UV_PROJECT_ENVIRONMENT:-/app/.venv}/bin/python" -m cocktail_server.main "$@"
