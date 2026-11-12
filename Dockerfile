# Models and the catalogue embedding cache are baked in at build time.
# At run time the container only calls the OpenAI and Anthropic APIs.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf-cache \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

COPY pyproject.toml uv.lock ./

# Install the locked dependencies with CPU-only torch. The grading box has no GPU,
# so the CUDA wheels would only make the image larger.
RUN pip install --no-cache-dir uv \
    && uv export --frozen --no-hashes --no-emit-project --no-dev -o requirements.txt \
    && grep -v -E '^(torch|nvidia-|triton)' requirements.txt > requirements-cpu.txt \
    && pip install --no-cache-dir torch==2.15.0 --extra-index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-cpu.txt \
    && rm -f requirements.txt requirements-cpu.txt

# Download the embedding model at build time. Run time has no access to Hugging Face.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('FremyCompany/BioLORD-2023-M')"

COPY src/ src/
COPY data/ data/

# Run the retriever once so the catalogue embeddings exist in the image. The cache file is
# keyed by the model id, so the tracked cache is reused and this step only re-embeds if it is stale.
RUN printf '%s\n' '{"id": "warmup", "text": "fever and cough"}' > /tmp/warmup.jsonl \
    && python -m src.coder --hybrid /tmp/warmup.jsonl /tmp/warmup_out.jsonl \
    && rm -f /tmp/warmup.jsonl /tmp/warmup_out.jsonl

# Offline mode stops the model libraries from contacting Hugging Face at run time.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

ENTRYPOINT ["python", "-m", "src.coder"]
CMD ["/in/notes.jsonl", "/out/predictions.jsonl"]
