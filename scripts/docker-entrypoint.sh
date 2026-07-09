#!/bin/sh
set -eu

mode="${1:-api}"

python -m scripts.init_db

if [ -f "${DATASET_PATH:-}" ]; then
    python -m scripts.import_dataset
fi

if [ "$mode" = "api" ]; then
    if [ ! -f "$MODEL_PATH" ]; then
        python -m scripts.train_model
    fi
    exec gunicorn --bind 0.0.0.0:8000 --workers 2 'moodmetrics.app:app'
fi

if [ "$mode" = "trainer" ]; then
    printenv | grep -E '^(DATABASE_URL|DATASET_PATH|EVALUATION_PATH|MODEL_PATH|REPORT_PATH)=' > /etc/environment
    echo '0 3 * * 0 root . /etc/environment; cd /app && { python -m scripts.import_dataset && python -m scripts.train_model && python -m scripts.evaluate_model && python -m scripts.generate_report; } >> /proc/1/fd/1 2>> /proc/1/fd/2' > /etc/cron.d/moodmetrics
    chmod 0644 /etc/cron.d/moodmetrics
    exec cron -f
fi

exec "$@"
