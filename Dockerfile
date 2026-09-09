# RetailSupport. Multi-stage, non-root, pinned, and it starts with no credentials
# because the default configuration points at the course gateway.

FROM python:3.12-slim AS builder

WORKDIR /build
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 TIKTOKEN_CACHE_DIR=/opt/tiktoken

COPY requirements.lock .
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir -r requirements.lock

# Warm the tokenizer vocabulary into the image: the container must not need the
# network to start, and tiktoken downloads its vocabulary on first use.
RUN /opt/venv/bin/python -c "import tiktoken; tiktoken.get_encoding('o200k_base').encode('warm'); tiktoken.get_encoding('cl100k_base').encode('warm')"


FROM python:3.12-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUTF8=1 \
    PYTHONPATH=/srv/src \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken \
    RETAIL_SUPPORT_LOG_JSON=1

WORKDIR /srv
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/tiktoken /opt/tiktoken
COPY src ./src
COPY configs ./configs
COPY data ./data
COPY eval ./eval
COPY scripts ./scripts
COPY pyproject.toml requirements.lock ./

RUN useradd --create-home --uid 10001 retail_support \
 && mkdir -p /srv/logs /srv/eval/out \
 && chown -R retail_support:retail_support /srv /opt/tiktoken
USER retail_support

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --retries=5 --start-period=15s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2).status==200 else 1)"

CMD ["uvicorn", "retail_support.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "warning"]
