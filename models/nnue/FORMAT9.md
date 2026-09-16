# NNUE file format 9

For the DESIGN maintainer: what version 9 is, so it can be folded into sections
2 (evaluator and formats), 3 (accumulator and search), 4.4 (model) and 4.5
(training). Nothing here changes versions 6 and 8; their bytes, their loading,
their evaluation, their training and their export are untouched, and their
tests and fixtures pass unmodified.

Version 9 adds two things to version 8: eight goal-corner feature rows, and
eight output heads selected by the pieces on the board.

## Layout

Little endian, magic `RPSNNUE1`. The 36-byte header is version 9, F = 13,648,
hidden H, QA 255, QB 64, f32 scale 600 and head count 8. H stays a multiple of
32 in 32..1024. The payload is

- bias, H i16;
- the shared feature table, F x H i16, feature-major;
- eight complete heads, one after another, each readout 2H i16 (mover then
  opponent), readout bias i32, dense weights 32 x 2H i8 row-major, dense biases
  32 i32 and residual readout 32 i16.

The dense width u32 of versions 6 and 8 is gone: version 9 fixes the width at
32, and the length check pins it. The total length is exactly

    36 + 2H + 2FH + 8 (68H + 196)  =  1,604 + 2FH + 546H

(H32 892,548; H512 14,256,708; H1024 28,511,812). Any other length, version,
feature count, head count, scale or width is an error, as before.

The feature rows are version 8's, with eight more after the clock rows:

| rows | meaning |
| --- | --- |
| 0 .. 13,121 | 486 piece-square rows in 27 opponent-material contexts |
| 13,122 .. 13,607 | 486 attacked-piece rows |
| 13,608 .. 13,623 | 16 elapsed-clock rows |
| 13,624 .. 13,639 | 16 remaining-clock rows |
| 13,640 .. 13,643 | the occupant of this perspective's own goal (square 80) |
| 13,644 .. 13,647 | the occupant of the opponent's goal (square 0) |
| 13,648 | the padding row, not stored in the file |

Within a goal group the offset is 0 for an empty corner and 1, 2, 3 for rock,
paper, scissors. Exactly one row of each group is active per perspective, so a
perspective has 44 feature slots: 20 pieces, 20 attacked pieces, two clock rows
and two goal rows.

## The goal rows

In a perspective's own frame the mover plays toward square 80 and the opponent
toward square 0. Only an opponent piece can stand on square 80 (one of the
perspective's own pieces there has already won) and only one of the
perspective's own pieces can stand on square 0, so the first group says which
enemy piece type blocks the goal and the second which of its own pieces is
standing in the opponent's corner. The rule is total: anything else on those
squares is an already-decided position, and both the encoder and the evaluator
read it as empty, which keeps a refresh and an incremental update equal on
boards the search never evaluates.

The opponent's perspective is computed in its own frame. The anti-diagonal
reflection maps square 0 to 80 and 80 to 0, so the opponent's half reads the
two corners exchanged and colour-swapped; the Python encoder gets this for free
from the reflected frame, and the Rust evaluator spells it out.

Incremental updates compare the two rows of the side being materialised before
and after the move and exchange them when they differ, exactly as the clock
rows are handled. A move onto a corner, off a corner, and a capture on a corner
are all covered by that comparison, which costs two integer comparisons per
half on every other move. The existing context refresh is unchanged: a half
whose capped context changed refreshes from the bias and every active row
family, goal rows included.

## Output buckets

The head is the piece count of the board, which the accumulator already carries
exactly (`FeatureState.counts`), so no extra state travels through the search:

    bucket = min(7, (total_pieces - 2) * 8 / 19),  total_pieces in 2..20

| pieces | 2-4 | 5-6 | 7-9 | 10-11 | 12-13 | 14-16 | 17-18 | 19-20 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| head | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |

The starting position has 20 pieces and uses head 7; a bare endgame uses head 0.
The head only offsets the readout slice and selects the dense head, so the
scalar, AVX2 and AVX-512/VNNI paths stay bit-identical by construction. The
trainer reads the same count from the ids (one piece slot per occupied square)
and the NumPy oracle from the board.

## Model and trainer

`NNUE(hidden, 9)` adds the `goal` table (8 x H, clamped like the other tables)
and factorises the eight heads the way the context rows are factorised: the
shared `output`, `dense` and `delta` layers plus a zero-started residual per
head, with the range constraint applied to the sum by moving the residual, and
the export writing the sums flattened into eight complete heads. A batch is
evaluated head by head over the rows that select it.

`--version 9` trains it. A format 9 batch also gathers the `board` column: the
dataset id caches are a format 8 encoding (`ids8.npy`, unchanged, so no dataset
has to be re-encoded and the encoder signature is the one the caches carry) and
the goal rows are read from the corners. `--init` walks the conversions in
order, 6 to 8 to 9, so an incumbent of either format starts at its own
evaluation.

## Conversion, the migration path

    python -m nnue.export convert --to 9 <format 8 file> <format 9 file>

copies the bias and the 13,640 shared rows, writes eight zero goal rows and
repeats the one head into all eight with its dense-width field dropped. Zero
rows add nothing to an accumulator and every bucket holds the same head, so the
converted file evaluates every position exactly as its source: this is how a
format 9 run starts from the incumbent's evaluation. `NNUE.widen_buckets()` is
the model-side conversion and exports the same bytes, as `widen_contexts` and
`convert --to 8` already do for version 8.

## Test evidence

- `models/nnue/tests/fixtures/format9_h32.nnue`, a seeded random-weight H32
  network whose eight heads differ in every entry, and
  `format9_positions.jsonl`, 380 positions crossing every piece count from 2 to
  20 with every combination of occupied goal corners. `py/tests/test_fixtures9.py`
  rebuilds both in memory and compares, so they cannot drift.
- `tests/format9_fixtures.rs` runs the diagnostic over those positions: ids,
  contexts, buckets and the raw value against the NumPy oracle over the same
  bytes (exact on the ids and the bucket, 1e-12 on the f64 value), the scalar
  path against the running backend, the AVX2-only backend against both, and an
  incremental update against a refresh on every legal child of every position.
  It asserts that all eight heads and all eight goal rows are reached.
- `tests/format9.rs` builds version 9 bytes in Rust: the header and length
  contract, the goal rows of both frames on hand-built corner positions, the
  bucket table against the documented mapping, a copy with head 0 in every
  bucket that differs exactly where the bucket is not 0, incremental updates
  around both corners (moves onto, off and capturing on them, including the
  terminal ones), and a thousand capture-biased random moves that walk through
  every bucket with every update equal to a refresh.
- `py/tests/test_export.py` checks a trained version 9 network: the file length
  and header, the oracle against the fake-quantised PyTorch forward within
  2e-6, the load/export round trip byte for byte, damaged files refused, and
  that flattening the heads or zeroing the goal rows moves the values.
- The migration: `format9_convert_h32.nnue` is the conversion of the format 8
  fixture; the Python test regenerates it byte for byte and finds identical
  integer values on all 512 shared positions, and the Rust test finds the same
  raw value and the same engine score as the format 8 model on each of them,
  buckets 0 to 7 among them.
- `py/tests/test_train.py` trains one epoch of version 9 from a format 8
  network with the id caches in place: the converted start evaluates the
  validation rows exactly as its source, and after the epoch the eight heads
  have separated and the goal rows have left zero while the export still
  reproduces the fake-quantised forward within 2e-6.

Formats 6 and 8 are proven unchanged by their own tests and fixtures, which
were not modified: `tests/format8.rs`, `tests/migration.rs`,
`src/search/accumulator_tests.rs`, `py/tests/test_fixtures.py`,
`py/tests/test_features.py` and the format 6 and 8 cases of
`py/tests/test_export.py` and `py/tests/test_train.py`.

## Not done

`nnue.widen_check` still accepts only format 6 and 8 arms; a version 9 width
preflight would need its `preserved`/`incoming` lists to cover the goal table
and the head residuals. README and DESIGN still describe versions 6 and 8 only.
