# cli

The `bot` binary. It is the only place that knows every model crate, so the
match runner stays model-agnostic.

```
bot eval  --candidate sq:runs/x/export.onnx [--reference sq:weights/sq_g128.onnx] [--pairs 32] [--sims 32] [--threads 4] [--seed 0] [--opening-plies 8] [--records games.jsonl]
bot play  --first sq:a.onnx --second sq:b.onnx [--sims 32] [--seed 0] [--opening-plies 8]
bot rpsi  --player sq:model.onnx [--move-ms 250] [--max-move-ms 250] [--sims 32] [--threads 4] [--name intransitive_bot]
bot --version
```

A player spec is `<model>:<path>`; paths are resolved from the current
directory. `--reference` defaults to the named eval reference,
`sq:weights/sq_g128.onnx`, so `bot eval` is run from the workspace root.
Adding a model means adding one arm to `player_from_spec`. Results are printed
as JSON.

`bot --version` prints the crate version followed by the git short hash and
`-dirty` when the checkout had uncommitted changes at build time (build.rs);
built outside a git history, the version alone.
