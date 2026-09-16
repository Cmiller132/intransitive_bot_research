"""A stand-in for `bot tb-probe` so the importer's tablebase option can be
tested without the prober binary. It answers from the JSON dictionary
`<data>/answers.json`, whose keys are `<the 81 cells, comma separated>:<since
capture>` and whose values are 1, 0, -1 or null; a position the dictionary
does not hold is answered null. Every position it read is copied to
`<data>/seen.jsonl`, so a test can check the contract of `nnue.tablebase`.

    python stub_prober.py [subcommand] --data <dir> --input <jsonl> --output <jsonl>
        [--exit-code N] [--drop-last] [--no-output]

The last three are the failures the importer must catch: a prober that fails,
one that answers fewer positions than it was given and one that writes no
answers at all.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def key(position: dict) -> str:
    return ",".join(str(int(cell)) for cell in position["board"]) + f":{int(position['since_capture'])}"


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subcommand", nargs="?", help="ignored; the real prober is `bot tb-probe`")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exit-code", type=int, default=0, help="fail with this status instead of answering")
    parser.add_argument("--drop-last", action="store_true", help="answer one position fewer than it was given")
    parser.add_argument("--no-output", action="store_true", help="answer nothing at all")
    args = parser.parse_args(argv)
    if args.exit_code:
        print("the stub prober was told to fail", file=sys.stderr)
        raise SystemExit(args.exit_code)
    answers = json.loads((args.data / "answers.json").read_text(encoding="utf-8"))
    lines = [line for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    (args.data / "seen.jsonl").write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    if args.no_output:
        return
    with args.output.open("w", encoding="utf-8") as sink:
        for line in lines[: -1 if args.drop_last else None]:
            sink.write(json.dumps({"value": answers.get(key(json.loads(line)))}) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
