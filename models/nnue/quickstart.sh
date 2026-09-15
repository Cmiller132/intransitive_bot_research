#!/usr/bin/env bash
# One iteration of the NNUE self-play loop on one machine (macOS on Apple silicon is the reference platform;
# any Unix with the same tools works). From a starting network to a candidate and its score:
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
# The Rust side and the Python side share nothing but files: records in, a .nnue file out. Run it again with the
# new file as the start to iterate. Every knob is an environment variable (defaults in brackets); QUICKSTART.md
# explains each one, the setup and how to read the result.
#
#   GAMES [3200]        games to generate (about 220 labelled positions each)
#   NODES [100000]      search nodes per move: the label budget (the research loop's value)
#   THREADS [all cores] games in flight, and the trainer's threads; one game per core is the model
#   HASH [64]           transposition table MiB per game in flight
#   EPOCHS [20] BATCH [8192] LR [0.0001]   the H512 continuation recipe's schedule (cosine to a floor, QAT from epoch 1)
#   PASSES [4]          passes over the training rows: STEPS = rows x PASSES / (BATCH x EPOCHS) unless STEPS is set
#   STEPS [] WARMUP []  steps per epoch and warmup steps; sized from the data when unset (WARMUP a tenth of the
#                       steps, at most 200). The research loop's fixed 20 x 1,000 steps is 200 passes over one
#                       3,200-game batch and overfits it (the loop mixes each batch with 13 M older rows; you don't)
#   DEVICE [cpu]        cpu is the tested path; mps may work with a recent PyTorch
#   EXTRA_DATA []       more "<set>:<share>" entries for the training mixture, space separated (earlier sets)
#   PAIRS [100]         evaluation openings, two games each      SIMS [8]   evaluation budget, units of 2,500 nodes
#   SEED [1]            generation, training and evaluation seed (change it for every new batch: the generator is
#                       deterministic, so the same seed from the same network replays the same games)
#   PY [python3]        the interpreter with the nnue and engine packages   BOT [target/release/bot]
#
# Rates: about 130 games per core-hour at 100k nodes (a 16-core machine makes about 2,200 games/h; 250k nodes is
# 2.5 times slower). Training sized to one 3,200-game batch is a few minutes on a GPU and longer on a CPU
# (BATCH=2048 on a small machine). Evaluation at SIMS=8 PAIRS=100 takes a few minutes. A smoke test of the whole
# pipeline: GAMES=8 NODES=5000 EPOCHS=1 STEPS=5 BATCH=256 WARMUP=2 PAIRS=2 SIMS=1 (under a minute).
set -euo pipefail
cd "$(dirname "$0")/../.."

NET="${1:?usage: models/nnue/quickstart.sh <start.nnue> [name]   (a .nnue file, e.g. models/nnue/examples/example.nnue; see models/nnue/QUICKSTART.md)}"
NAME="${2:-qs_$(date +%Y%m%d_%H%M%S)}"
GAMES="${GAMES:-3200}"
NODES="${NODES:-100000}"
if [ -z "${THREADS:-}" ]; then
  THREADS="$( (sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 8) | tr -d ' ')"
fi
HASH="${HASH:-64}"
SEED="${SEED:-1}"
EPOCHS="${EPOCHS:-20}"
STEPS="${STEPS:-}"
BATCH="${BATCH:-8192}"
LR="${LR:-0.0001}"
WARMUP="${WARMUP:-}"
PASSES="${PASSES:-4}"
DEVICE="${DEVICE:-cpu}"
EXTRA_DATA="${EXTRA_DATA:-}"
PAIRS="${PAIRS:-100}"
SIMS="${SIMS:-8}"
PY="${PY:-python3}"
BOT="${BOT:-target/release/bot}"

say() { printf '\n== %s  (%s)\n' "$1" "$(date '+%H:%M:%S')"; }

# ---- preflight: the bot binary, the Python packages, the network --------------------------------------------------
if [ ! -x "$BOT" ] && [ -x "$BOT.exe" ]; then BOT="$BOT.exe"; fi
if [ ! -x "$BOT" ]; then
  say "building the bot binary (cargo build --release -p cli; the first build downloads the ONNX Runtime library)"
  cargo build --release -p cli
  if [ ! -x "$BOT" ] && [ -x "$BOT.exe" ]; then BOT="$BOT.exe"; fi
fi
if ! "$PY" -c "import engine, nnue, torch, numpy" 2>/dev/null; then
  echo "Python is missing a package. From the workspace root:" >&2
  echo "  $PY -m pip install -e 'models/nnue/py[dev]'     # the nnue package: torch, numpy" >&2
  echo "  $PY -m pip install maturin && cargo xtask wheel  # the engine module (the importer replays games through it)" >&2
  echo "  ($PY must be Python 3.11 or newer; PY=<interpreter> selects another one)" >&2
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
rows_of() { "$PY" -c "import json, sys; print(json.load(open('runs/nnue_data/%s/provenance.json' % sys.argv[1]))['rows'])" "$1"; }
DATA_ARGS="--data $SET:1.0"
ROWS="$(rows_of "$SET")"
for entry in $EXTRA_DATA; do DATA_ARGS="$DATA_ARGS --data $entry"; ROWS=$((ROWS + $(rows_of "${entry%%:*}"))); done
if [ -z "$STEPS" ]; then  # PASSES passes over every training row, spread over the epochs
  STEPS=$(( (ROWS * PASSES + BATCH * EPOCHS - 1) / (BATCH * EPOCHS) ))
  [ "$STEPS" -lt 10 ] && STEPS=10
fi
if [ -z "$WARMUP" ]; then  # a tenth of the run, at most 200 steps
  WARMUP=$(( STEPS * EPOCHS / 10 ))
  [ "$WARMUP" -gt 200 ] && WARMUP=200
  [ "$WARMUP" -lt 1 ] && WARMUP=1
fi
say "4/5 training $NAME from $NET on $SET${EXTRA_DATA:+ + $EXTRA_DATA}: $ROWS rows, $EPOCHS x $STEPS steps at batch $BATCH ($((STEPS * EPOCHS * BATCH / (ROWS > 0 ? ROWS : 1))) passes), warmup $WARMUP, $DEVICE"
"$PY" -m nnue.train --run "$NAME" --init "$NET" $DATA_ARGS --hidden "$HIDDEN" --version 8 \
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
echo "next iteration:  $0 runs/$NAME/best.nnue   (EXTRA_DATA=\"$SET:1.0\" keeps this set in the next mixture)"
