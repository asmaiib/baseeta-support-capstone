# The course gateway (a simulator, not a model)

Runs the whole course offline. It speaks both wire dialects, accounts tokens,
caches byte-stable prefixes, and fails on command so the drills are reproducible.

```bash
docker compose up -d gateway          # from retail_support/
curl -s localhost:8080/healthz | jq
```

Read the module docstring in `app/brain.py` before quoting any number this thing
produced: the wire contract, the usage accounting and the groundedness are real;
model *quality* is simulated by rules. Point a route at a real provider with two
environment variables when you want evidence about a model rather than about your
harness.

## Drills

```bash
# 429 storm on the primary model for three minutes, Retry-After: 2
curl -s -X POST localhost:8080/admin/fault \
  -H 'content-type: application/json' \
  -d '{"mode":"rate_limit","seconds":180,"model":"course-flagship","retry_after":2}'

# Provider outage: the primary model 529s, the on-prem route keeps serving
curl -s -X POST localhost:8080/admin/fault \
  -H 'content-type: application/json' \
  -d '{"mode":"overload","seconds":300,"model":"course-flagship"}'

curl -s -X POST localhost:8080/admin/fault -d '{"mode":"off"}' -H 'content-type: application/json'
curl -s localhost:8080/admin/stats | jq
```

## Knobs

| Variable | Default | Meaning |
|---|---|---|
| `MOCKGW_SPEED` | `0.2` | Wall-clock compression for simulated latency. `1.0` feels like a real hosted model. |
| `MOCKGW_FAST` | unset | `1` disables all sleeping (tests, CI). |
| `MOCKGW_MIN_CACHEABLE_TOKENS` | `1024` | Minimum prefix length before a prompt cache hit is possible. |
| `MOCKGW_CACHE_TTL_S` | `300` | Prompt-cache TTL, refreshed on hit. |
