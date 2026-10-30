#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
zip -r ../finsight.zip . -x ".venv/*" "data/cache/*" "data/raw/*.pdf" "data/index/*" ".env"
echo "Created ../finsight.zip"
