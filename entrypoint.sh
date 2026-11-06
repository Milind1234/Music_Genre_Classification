#!/bin/sh
set -e

echo "==> Running Django database migrations..."
python manage.py migrate --noinput

echo "==> Starting production WSGI server (Gunicorn)..."
exec gunicorn MusicClassification.wsgi:application \
    --bind 0.0.0.0:8000 \
    --workers 3 \
    --timeout 120 \
    --access-logfile - \
    --error-logfile -

