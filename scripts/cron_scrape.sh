#!/usr/bin/env sh

set -eu

REPO_DIR=${REPO_DIR:-/path/to/sdi-helper}
LOG_DIR=${LOG_DIR:-$REPO_DIR/logs}
LOG_FILE=${LOG_FILE:-$LOG_DIR/scrape-cron.log}

mkdir -p "$LOG_DIR"
cd "$REPO_DIR"

make scrape >> "$LOG_FILE" 2>&1