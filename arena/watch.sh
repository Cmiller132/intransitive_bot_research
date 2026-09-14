#!/bin/bash
# Export and upload every tenth checkpoint of a training run to the arena:
# watch.sh <arena url> <run> <model> [--once]. Polls runs/<run>/ for ckpt_NNNNNN.pt
# with N a multiple of 10, exports each with `python -m <model>.export` into
# runs/<run>/eval/ and uploads it as <run>_N with upload.sh; a marker beside
# the export records the upload. --once scans once for use by a scheduler.
set -e
if [ "$#" -lt 3 ] || [ "$#" -gt 4 ] || { [ "$#" -eq 4 ] && [ "$4" != "--once" ]; }; then
  echo "usage: watch.sh <arena url> <run> <model> [--once]" >&2
  exit 2
fi
url=$1; run=$2; model=$3
root=$(cd "$(dirname "$0")/.." && pwd)
dir=$root/runs/$run; out=$dir/eval
mkdir -p "$out"
while :; do
  failed=0
  for ckpt in $(find "$dir" -maxdepth 1 -name 'ckpt_[0-9][0-9][0-9][0-9][0-9][0-9].pt' -mmin +1 | sort); do
    name=$(basename "$ckpt" .pt); n=$((10#${name#ckpt_}))
    [ $((n % 10)) -eq 0 ] || continue
    [ -f "$out/$name.uploaded" ] && continue
    if [ ! -f "$out/$name.onnx.json" ]; then
      echo "$(date) exporting $name"
      if ! (cd "$root/models/$model/py" && python -m "$model.export" --ckpt "$ckpt" --out "$out/$name.onnx"); then
        rm -f -- "$out/$name.onnx.json"
        echo "export of $name failed; retrying later" >&2
        failed=1
        continue
      fi
    fi
    echo "$(date) uploading $name as ${run}_$n"
    if "$root/arena/upload.sh" "$url" "${run}_$n" "$model:$name.onnx" "$out/$name.onnx" "$out/$name.onnx.json"; then
      touch "$out/$name.uploaded"
    else
      echo "upload of $name failed; retrying later"
      failed=1
    fi
  done
  [ "${4:-}" != "--once" ] || exit "$failed"
  sleep 120
done
