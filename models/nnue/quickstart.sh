#!/usr/bin/env bash
# One iteration of the NNUE self-play loop on one machine, from a starting network to a stronger one:
#
#   models/nnue/quickstart.sh <start.nnue> [name]
#
#   1. generate   bot selfplay      the network plays itself; every searched root is a labelled position
#                                   (gzip JSONL shards + manifest.json under runs/nnue_selfplay/<name>/)
#   2. import     nnue.importer     the shards become a NumPy dataset under runs/nnue_data/selfplay_<name>/
#   3. encode     nnue.data         the dataset's feature-id cache (the trainer gathers ids instead of boards)
#   4. train      nnue.train        a continuation of the starting network on that dataset -> runs/<name>/best.nnue
#   5. evaluate   bot eval          the new network against the starting one at a fixed node budget
#
# The Rust side and the Python side share nothing but files: records in, a .nnue file out. Run it again with
# NET=runs/<name>/best.nnue to iterate. Every knob is an environment variable (defaults in brackets):
#
#   GAMES [3200]      games to generate           NODES [100000]   search nodes per move (the label budget)
#   THREADS [cores]   games in flight / trainer threads             HASH [64]        transposition table MiB per game
#   EPOCHS [20]       STEPS [1000] steps per epoch   BATCH [8192]  LR [0.0001]      the H512 continuation recipe
#   DEVICE [cpu]      cpu, cuda (or mps to try)   PAIRS [100]      evaluation openings (two games each)
#   SIMS [8]          evaluation budget in units of 2,500 nodes (8 = 20k nodes)     SEED [1]
#
# Timings measured on a Ryzen 7950X (16 one-thread games in flight): about 2,200 games/h at 100k nodes, 800/h at
# 250k; an M3 Max measured 800/h at 250k nodes with 16 in flight, so expect about 2,000/h at 100k. Training the
# full recipe (20 x 1,000 steps at batch 8192, H512) is 11 minutes on an RTX 4070 Ti and hours on a CPU: for a
# first pass on a CPU use EPOCHS=4 STEPS=250 BATCH=2048. Evaluation at SIMS=8 PAIRS=100 is a few minutes.
set -euo pipefail
cd "$(dirname "$0")/../.."

NET="${1:?usage: models/nnue/quickstart.sh <start.nnue> [name]   (a format 8 .nnue file; NET=runs/<name>/best.nnue to iterate)}"
NAME="${2:-qs_$(date +%Y%m%d_%H%M%S)}"
GAMES="${GAMES:-3200}"
NODES="${NODES:-100000}"
if [ -z "${THREADS:-}" ]; then
  THREADS="$( (nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8) | tr -d ' ')"
fi
HASH="${HASH:-64}"
SEED="${SEED:-1}"
EPOCHS="${EPOCHS:-20}"
STEPS="${STEPS:-1000}"
BATCH="${BATCH:-8192}"
LR="${LR:-0.0001}"
WARMUP="${WARMUP:-200}"
DEVICE="${DEVICE:-cpu}"
PAIRS="${PAIRS:-100}"
SIMS="${SIMS:-8}"
PY="${PY:-python3}"
BOT="${BOT:-target/release/bot}"

say() { printf '\n== %s  (%s)\n' "$1" "$(date '+%H:%M:%S')"; }

# ---- preflight: the bot binary, the Python packages ----------------------------------------------------------------
if [ ! -x "$BOT" ] && [ -x "$BOT.exe" ]; then BOT="$BOT.exe"; fi
if [ ! -x "$BOT" ]; then
  say "building the bot binary (cargo build --release -p cli)"
  cargo build --release -p cli
  if [ ! -x "$BOT" ] && [ -x "$BOT.exe" ]; then BOT="$BOT.exe"; fi
fi
if ! "$PY" -c "import engine, nnue, torch, numpy" 2>/dev/null; then
  echo "Python is missing a package. From the workspace root:" >&2
  echo "  $PY -m pip install -e 'models/nnue/py[dev]'     # the nnue package: torch, numpy" >&2
  echo "  $PY -m pip install maturin && cargo xtask wheel  # the engine module (the importer replays games through it)" >&2
  exit 1
fi
if [ ! -f "$NET" ]; then echo "no such network: $NET" >&2; exit 1; fi
HIDDEN="$("$PY" -c "import sys; from pathlib import Path; from nnue import export; print(export.read(Path(sys.argv[1]))['hidden'])" "$NET")"
RECORDS="runs/nnue_selfplay/$NAME"
SET="selfplay_$NAME"
echo "network $NET (H$HIDDEN)  name $NAME  games $GAMES  nodes $NODES  threads $THREADS  device $DEVICE"

# ---- 1. generate ----------------------------------------------------------------------------------------------------
say "1/5 generating $GAMES games at $NODES nodes per move, $THREADS in flight -> $RECORDS"
RESUME=""
if [ -f "$RECORDS/manifest.json" ]; then RESUME="--resume"; fi
"$BOT" selfplay --player "nnue:$NET?hash=$HASH" --nodes "$NODES" --games "$GAMES" --threads "$THREADS" --seed "$SEED" \
  --opening-plies 8 --random-moves 2 --random-from 8 --random-to 40 --records "$RECORDS" $RESUME

# ---- 2. import, 3. encode ---------------------------------------------------------------------------------------------
say "2/5 importing the searched roots -> runs/nnue_data/$SET"
if [ ! -f "runs/nnue_data/$SET/provenance.json" ]; then
  "$PY" -m nnue.importer selfplay --records "$RECORDS" --out "$SET" --min-ply 16
fi
say "3/5 encoding the feature ids"
if [ ! -f "runs/nnue_data/$SET/ids8.npy" ]; then
  "$PY" -m nnue.data encode "$SET"
fi

# ---- 4. train -----------------------------------------------------------------------------------------------------------
say "4/5 training $NAME from $NET on $SET ($EPOCHS x $STEPS steps, batch $BATCH, $DEVICE)"
"$PY" -m nnue.train --run "$NAME" --init "$NET" --data "$SET:1.0" --hidden "$HIDDEN" --version 8 \
  --batch "$BATCH" --steps_per_epoch "$STEPS" --epochs "$EPOCHS" --lr "$LR" --weight_decay 1e-5 \
  --warmup_steps "$WARMUP" --lr_floor 0.15 --qat_start_epoch 1 --raw_weight 0.02 --symmetry_weight 0.2 \
  --grad_clip 5 --patience "$EPOCHS" --val_rows 8192 --threads "$THREADS" --device "$DEVICE" --seed "$SEED"

# ---- 5. evaluate ----------------------------------------------------------------------------------------------------
say "5/5 evaluating runs/$NAME/best.nnue against $NET: $PAIRS openings at $((SIMS * 2500)) nodes"
"$BOT" eval --candidate "nnue:runs/$NAME/best.nnue?hash=$HASH" --reference "nnue:$NET?hash=$HASH" \
  --sims "$SIMS" --pairs "$PAIRS" --threads "$THREADS" --seed "$SEED" > "runs/$NAME/eval.json"
"$PY" - "runs/$NAME/eval.json" <<'PYEOF'
import json, sys
r = json.load(open(sys.argv[1]))
games = r["wins"] + r["draws"] + r["losses"]
score = (r["wins"] + 0.5 * r["draws"]) / games if games else float("nan")
print(f"candidate score {score:.3f} over {games} games (wins {r['wins']}, draws {r['draws']}, losses {r['losses']}); "
      f"margin {r.get('margin')} with bootstrap interval {r.get('interval')}; forfeits {r['forfeits']}")
PYEOF

say "done: runs/$NAME/best.nnue (a score above .5 means it beat the starting network; see runs/$NAME/eval.json)"
echo "next iteration:  NET=runs/$NAME/best.nnue  $0 runs/$NAME/best.nnue"
