#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
cert_file="${TLS_CERT_FILE:-tls/falloutassist-server.crt}"
key_file="${TLS_KEY_FILE:-tls/falloutassist-server.key}"

if [[ ! -r "$cert_file" || ! -r "$key_file" ]]; then
  echo "TLS certificate/key are missing or unreadable: set TLS_CERT_FILE and TLS_KEY_FILE." >&2
  exit 1
fi

exec .venv/bin/python -m uvicorn app:app \
  --host 0.0.0.0 --port 8012 --workers 1 --access-log \
  --ssl-certfile "$cert_file" --ssl-keyfile "$key_file"
