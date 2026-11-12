# Models and the catalogue embedding cache are baked in at build time.
# At run time the container only calls the OpenAI and Anthropic APIs.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.8 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/opt/hf-cache \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /app

COPY pyproject.toml uv.lock ./

# Install locked dependencies, including CPU-only torch for Linux.
RUN uv sync --locked --no-dev --no-install-project --no-cache

# Download the pinned BioLORD-2023-M revision at build time. The build has network access; the run
# does not (HF_HUB_OFFLINE=1 above), so the model must be in the image before the container starts.
ARG BIOLORD_REVISION=4ea2ea2c89ef63365f7fcd91406a0cd8ac36d2e2
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('FremyCompany/BioLORD-2023-M', revision='${BIOLORD_REVISION}')"

COPY src/ src/
COPY data/ data/

# Run the retriever once so the catalogue embeddings exist in the image. The cache file is
# keyed by the model id, so the tracked cache is reused and this step only re-embeds if it is stale.
RUN printf '%s\n' '{"id": "warmup", "text": "fever and cough"}' > /tmp/warmup.jsonl \
    && python -m src.coder --hybrid /tmp/warmup.jsonl /tmp/warmup_out.jsonl \
    && rm -f /tmp/warmup.jsonl /tmp/warmup_out.jsonl

ENTRYPOINT ["python", "-m", "src.coder"]
CMD ["/in/notes.jsonl", "/out/predictions.jsonl"]

# Score the dev predictions after the run (score.py is not in the image, so mount it):
# docker compose run --rm -v "${PWD}/score.py:/app/score.py" --entrypoint python coder score.py --gold /app/data/dev_notes.jsonl --pred /out/dev_predictions.jsonl --mms /app/data/mms-2026-01.json
