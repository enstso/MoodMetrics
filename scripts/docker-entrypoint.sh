#!/bin/sh
set -eu

mode="${1:-api}"

run_with_retries() {
    attempt=1
    max_attempts="${BOOTSTRAP_MAX_RETRIES:-30}"
    delay_seconds="${BOOTSTRAP_RETRY_SECONDS:-2}"

    until "$@"; do
        if [ "$attempt" -ge "$max_attempts" ]; then
            echo "Échec après $attempt tentatives: $*" >&2
            return 1
        fi
        echo "Commande indisponible, nouvelle tentative $((attempt + 1))/$max_attempts: $*" >&2
        attempt=$((attempt + 1))
        sleep "$delay_seconds"
    done
}

run_with_retries python -m scripts.init_db

import_dataset_if_available() {
    if [ -f "${DATASET_PATH:-}" ]; then
        run_with_retries python -m scripts.import_dataset
    fi
}

generate_evaluation_artifacts_if_missing() {
    if [ ! -f "$EVALUATION_PATH" ] || [ ! -f "$REPORT_PATH" ]; then
        run_with_retries python -m scripts.evaluate_model
        run_with_retries python -m scripts.generate_report
    fi
}

if [ "$mode" = "api" ]; then
    import_dataset_if_available
    if [ ! -f "$MODEL_PATH" ]; then
        run_with_retries python -m scripts.train_model
    fi
    generate_evaluation_artifacts_if_missing
    exec gunicorn --bind 0.0.0.0:8000 --workers 2 'moodmetrics.app:app'
fi

if [ "$mode" = "trainer" ]; then
    printenv | grep -E '^(DATABASE_URL|DATASET_PATH|EVALUATION_PATH|MODEL_PATH|REPORT_PATH)=' > /etc/environment
    echo '0 3 * * 0 root . /etc/environment; cd /app && { python -m scripts.import_dataset && python -m scripts.train_model && python -m scripts.evaluate_model && python -m scripts.generate_report; } >> /proc/1/fd/1 2>> /proc/1/fd/2' > /etc/cron.d/moodmetrics
    chmod 0644 /etc/cron.d/moodmetrics
    exec cron -f
fi

exec "$@"
