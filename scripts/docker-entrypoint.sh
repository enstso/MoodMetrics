#!/bin/sh
set -eu

mode="${1:-api}"

python -m scripts.init_db

if [ "$mode" = "api" ]; then
    if [ ! -f "$MODEL_PATH" ]; then
        python -m scripts.train_model
    fi
    exec gunicorn --bind 0.0.0.0:8000 --workers 2 'moodmetrics.app:app'
fi

if [ "$mode" = "trainer" ]; then
    printenv | grep -E '^(DATABASE_URL|MODEL_PATH)=' > /etc/environment
    echo '0 3 * * 0 root . /etc/environment; cd /app && python -m scripts.train_model >> /proc/1/fd/1 2>> /proc/1/fd/2' > /etc/cron.d/moodmetrics
    chmod 0644 /etc/cron.d/moodmetrics
    exec cron -f
fi

exec "$@"
