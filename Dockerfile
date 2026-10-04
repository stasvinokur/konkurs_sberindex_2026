# syntax=docker/dockerfile:1

# Окружение решения: Python 3.12 и зависимости из uv.lock. Как запускать — в docker-compose.yml.

ARG UV_VERSION=0.12.5
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM python:3.12-slim-bookworm

# libgomp1 — OpenMP для LightGBM; make — цели Makefile.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 make \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /usr/local/bin/

# Окружение лежит вне /app: каталоги проекта монтируются с хоста и не должны его перекрывать.
# UV_NO_SYNC и UV_FROZEN — чтобы `uv run` из Makefile запускал команды в готовом окружении,
# а не пересобирал его. Кэши (веса моделей, numba, matplotlib) — в /cache, это том.
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_SYNC=1 \
    UV_FROZEN=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    XDG_CACHE_HOME=/cache \
    HF_HOME=/cache/huggingface \
    NUMBA_CACHE_DIR=/cache/numba \
    MPLCONFIGDIR=/cache/matplotlib

WORKDIR /app

# Зависимости — отдельным слоем: он пересобирается только при изменении uv.lock.
# Версии и хеши берутся из uv.lock. Исключение одно: на Linux lock тянет torch с CUDA и 2,7 ГБ
# библиотек NVIDIA, а решение считается на CPU. Поэтому пакеты CUDA пропускаются, а torch той
# же версии ставится из CPU-индекса PyTorch.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    set -eu; \
    export UV_CACHE_DIR=/root/.cache/uv; \
    uv venv --python /usr/local/bin/python3 /opt/venv; \
    uv export -q --all-extras --no-emit-project --no-hashes -o /tmp/all.txt; \
    cuda="$(sed -n -E 's/^((torch|triton)|((nvidia|cuda)-[a-z0-9-]+))==.*/--no-emit-package \1/p' /tmp/all.txt)"; \
    uv export -q --all-extras --no-emit-project $cuda -o /tmp/requirements.txt; \
    uv pip install --no-deps --require-hashes -r /tmp/requirements.txt; \
    uv pip install --no-deps --index-url "$TORCH_INDEX_URL" \
        "$(sed -n -E 's/^(torch==[^ ;]+).*/\1/p' /tmp/all.txt)"; \
    rm /tmp/all.txt /tmp/requirements.txt

# Работа не от root. На Linux, чтобы файлы в смонтированных каталогах принадлежали вам:
#   SBX_UID=$(id -u) SBX_GID=$(id -g) docker compose build
ARG SBX_UID=1000
ARG SBX_GID=1000
RUN groupadd --non-unique --gid "$SBX_GID" sbx \
    && useradd --non-unique --uid "$SBX_UID" --gid "$SBX_GID" --create-home sbx \
    && mkdir /cache \
    && chown sbx:sbx /cache /app

# Установка editable: корень проекта вычисляется от расположения пакета (sbx.shell.io.ROOT).
COPY --chown=sbx:sbx . .
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_CACHE_DIR=/root/.cache/uv uv pip install --no-deps --editable .

USER sbx
CMD ["sbx", "--help"]
