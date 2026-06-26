#!/bin/bash

if [[ -z "${VIRTUAL_ENV:-}" || "$(basename "$VIRTUAL_ENV")" != "vista_env" ]]; then
  echo "vista_env is not active. Run:"
  echo "source vista_env/bin/activate"
  exit 1
fi

cd object_identifier
python build_product_embeddings.py
cd ..
