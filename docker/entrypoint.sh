#!/bin/sh
# Index the bundled sample repository on first start so the demo has something to search.
set -e
if [ "${CODEFUSION_SKIP_SAMPLE_INDEX:-0}" != "1" ] && [ ! -f "${CODEFUSION_INDEX_DIR:-/app/artifacts/index}/index.db" ]; then
  python scripts/index_repo.py examples/sample_repo --config "$CODEFUSION_CONFIG" ${CODEFUSION_INDEX_DIR:+--index-dir "$CODEFUSION_INDEX_DIR"} || \
    echo "sample indexing failed; the API still starts (POST /index/repository to index)"
fi
exec "$@"
