# The course gateway. Deliberately boring: one stage, one process, no state on disk.
FROM python:3.12-slim

WORKDIR /srv
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TIKTOKEN_CACHE_DIR=/srv/.tiktoken

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Warm the tokenizer vocabulary into the image so the container does not need the
# network at run time — the whole point of this gateway is working offline.
RUN python -c "import tiktoken; tiktoken.get_encoding('o200k_base').encode('warm')"

RUN useradd --create-home --uid 10001 gateway && chown -R gateway:gateway /srv
USER gateway

EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=3s --retries=5 --start-period=10s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=2).status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--log-level", "warning"]
