#!/usr/bin/env bash
set -euo pipefail

in_dir="${1:-.}"
quality="${2:-95}"
container_image="${CONTAINER_IMAGE:-dpokidov/imagemagick}"

find "$in_dir" -type f \( -iname "*.heic" -o -iname "*.heif" \) -print0 |
while IFS= read -r -d '' f; do
  rel="${f#$in_dir/}"
  out="$in_dir/${rel%.*}.jpg"

  mkdir -p "$(dirname "$out")"

  echo "Converting: $f -> $out"

  docker run --rm \
    --user "$(id -u):$(id -g)" \
    --entrypoint magick \
    -v "$PWD":/work \
    -w /work \
    "$container_image" \
    "$f" -auto-orient -strip -quality "$quality" "$out"
done

