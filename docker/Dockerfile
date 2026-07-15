FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY frontend ./frontend
COPY scripts ./scripts
COPY sample_data ./sample_data

# Drop root. The chroma volume mounts at /app/chroma_db, so it has to be
# writable by this user before the drop, not after.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/chroma_db \
    && chown -R app:app /app
USER app

EXPOSE 8000
# 0.0.0.0 is the container interface, not the host's. docker-compose maps this
# to 127.0.0.1 on the host, which is what keeps it off the network.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
