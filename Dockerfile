FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y cron \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY moodmetrics ./moodmetrics
COPY scripts ./scripts
RUN pip install --no-cache-dir .

RUN chmod +x scripts/docker-entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["scripts/docker-entrypoint.sh"]
CMD ["api"]
