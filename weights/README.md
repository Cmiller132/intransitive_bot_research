# Reference weights

Untracked checkpoints used as training starting points and eval references.

| File | Model | Source |
|---|---|---|
| sq_g128.pt | sq, iteration 39 | previous workspace, run sq_g128f, converted with `python -m sq.convert` |
| sq_g128.onnx, sq_g128.onnx.json | sq | `python -m sq.export` of sq_g128.pt; the default `bot eval --reference` |

Exports for the engine (`<name>.onnx` + `<name>.onnx.json`) are produced by
each model's `export` command and are also untracked.
