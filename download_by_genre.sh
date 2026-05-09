#!/usr/bin/env bash
# ============================================================
# Genre-sorted yt-dlp downloader
# Reads music_links.txt, detects genre from section comments,
# and downloads each track into its own genre subfolder.
# Usage: bash download_by_genre.sh [--video]
#   Default: audio-only mp3
#   --video : download video (no audio conversion)
# ============================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${0}")" && pwd)"
INPUT="${SCRIPT_DIR}/music_links.txt"
OUTPUT_BASE="${SCRIPT_DIR}/music"
MODE="${1:-}"

if [[ ! -f "$INPUT" ]]; then
  echo "ERROR: music_links.txt not found next to this script."
  exit 1
fi

SUCCESS=0
FAILED=0

current_genre="misc"

while IFS= read -r line || [[ -n "$line" ]]; do
  [[ -z "$line" ]] && continue

  if echo "$line" | grep -qiE '^#\s*={3,}'; then
    continue
  fi

  if echo "$line" | grep -qiE '^#\s*(BLUES|CLASSICAL|COUNTRY|DISCO|HIPHOP|JAZZ|METAL|POP|REGGAE|ROCK)\s*$'; then
    current_genre=$(echo "$line" | sed 's/^#[[:space:]]*//' | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')
    echo ""
    echo ">>> Genre: $current_genre"
    mkdir -p "${OUTPUT_BASE}/${current_genre}"
    continue
  fi

  [[ "$line" == \#* ]] && continue

  url="$line"
  dest="${OUTPUT_BASE}/${current_genre}"
  mkdir -p "$dest"

  echo "  Downloading: $url"

  if [[ "$MODE" == "--video" ]]; then
    if yt-dlp \
      --no-playlist \
      --output "${dest}/%(uploader)s - %(title)s.%(ext)s" \
      "$url"; then
      ((SUCCESS++))
    else
      echo "  SKIP: $url (unavailable or error)"
      ((FAILED++))
    fi
  else
    if yt-dlp \
      --no-playlist \
      -x --audio-format mp3 \
      --output "${dest}/%(uploader)s - %(title)s.%(ext)s" \
      "$url"; then
      ((SUCCESS++))
    else
      echo "  SKIP: $url (unavailable or error)"
      ((FAILED++))
    fi
  fi

done < "$INPUT"

echo ""
echo "============================================================"
echo "Done!"
echo "  Downloaded : $SUCCESS"
echo "  Failed/Skip: $FAILED"
echo "  Files saved: ${OUTPUT_BASE}/"
echo "============================================================"
