"""The dashboard's web server: the page, the runs, and the controls that run the quickstart's steps.

    python3 -m nnue_dashboard                  from the repository root (or --repo <root>)
    python3 -m nnue_dashboard --read-only      look, never start anything

The server starts the generator, importer, trainer and evaluator itself (runner.py) with the interpreter it runs under
(--python to choose another), from the workspace root, because the trainer writes to <root>/runs whatever the working
directory is. Controls answer only requests that carry the page's token and come from the page's own origin, and only
on the loopback address unless --allow-remote-control is given.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import secrets
import signal
import sys
import threading
import webbrowser
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from urllib.parse import parse_qs

from .analysis import PLAN, load_runs
from .games import GameIndex
from .runner import Runner

STATIC = Path(__file__).parent / "static"


def finite(value):
    """JSON has no NaN or infinity; a diverged run can log them."""
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else None
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    return value


def find_repo(start: Path) -> Path | None:
    for d in [start, *start.parents]:
        if (d / "Cargo.toml").is_file() and (d / "Project.md").is_file():
            return d
    return None


class App:
    def __init__(self, runs_dir: Path, plan: dict, runner: Runner | None, token: str, control_reason: str):
        self.runs_dir, self.plan, self.runner, self.token, self.control_reason = runs_dir, plan, runner, token, control_reason
        self.games = GameIndex(runs_dir)

    def runs_payload(self) -> dict:
        plan = dict(self.plan)
        if self.runner:
            with self.runner.lock:
                plan["reserved_names"] = [j["run"] for j in self.runner.jobs if j["run"]]
        payload = load_runs(self.runs_dir, plan)
        if self.runner:
            self.runner.overlay(payload["runs"])
        payload["sets_on_disk"] = sorted(p.parent.name for p in (self.runs_dir / "nnue_data").glob("*/provenance.json"))
        example = "models/nnue/examples/example.nnue"
        payload["networks"] = ([example] if self.runner and (self.runner.repo / example).is_file() else []) + \
            [f"runs/{r['name']}/best.nnue" for r in payload["runs"] if r["has_net"]] + \
            sorted(f"runs/{p.parent.name}/best.nnue" for p in self.runs_dir.glob("*.replaced_*/best.nnue"))
        return payload

    def control_payload(self) -> dict:
        if not self.runner:
            return {"enabled": False, "reason": self.control_reason}
        return {"enabled": True, **self.runner.snapshot()}


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, app: App, **kwargs):
        self.app = app
        super().__init__(*args, **kwargs)

    def log_message(self, fmt, *args):
        pass

    def send_body(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, status: int, payload) -> None:
        self.send_body(status, json.dumps(finite(payload), allow_nan=False, default=str).encode(), "application/json")

    def do_GET(self):
        path, _, query = self.path.partition("?")
        q = {k: v[-1] for k, v in parse_qs(query).items()}
        games = self.app.games
        try:
            if path == "/api/games":
                run = q.get("run", "")
                return self.send_json(200, {"stats": games.stats(run), "live": games.live(run)})
            if path == "/api/games/list":
                return self.send_json(200, games.listing(q.get("run", ""), q.get("end", ""), q.get("sort", "id"),
                                                         int(q.get("offset", 0)), min(200, int(q.get("limit", 50)))))
            if path == "/api/game":
                game = games.game(q.get("run", ""), int(q.get("id", -1)))
                return self.send_json(200 if game else 404, game or {"error": "no such game"})
            if path == "/api/games/trends":
                names = [n for n in q.get("runs", "").split(",") if n]
                games.request(names)
                return self.send_json(200, {"runs": [games.stats(n) for n in names]})
            if path == "/api/runs":
                return self.send_json(200, self.app.runs_payload())
            if path == "/api/control":
                return self.send_json(200, self.app.control_payload())
        except Exception as error:
            print(f"api error: {error!r}", file=sys.stderr)
            return self.send_json(500, {"error": repr(error)})
        name = "index.html" if path in ("/", "") else path.lstrip("/").removeprefix("static/")
        file = (STATIC / name).resolve()
        if STATIC.resolve() not in file.parents or not file.is_file():
            return self.send_body(404, b"not found", "text/plain")
        body = file.read_bytes()
        if name == "index.html":  # the token reaches only a page served from here
            body = body.replace(b"__TOKEN__", self.app.token.encode())
        kind = mimetypes.guess_type(file.name)[0] or "application/octet-stream"
        if kind.startswith("text/") or kind.endswith("javascript"):
            kind += "; charset=utf-8"
        self.send_body(200, body, kind)

    def do_POST(self):
        runner = self.app.runner
        if not runner:
            return self.send_json(403, {"error": self.app.control_reason})
        origin = self.headers.get("Origin")
        host = self.headers.get("Host", "")
        if self.headers.get("X-Token") != self.app.token or (origin and origin.split("://", 1)[-1] != host):
            return self.send_json(403, {"error": "this request did not come from the dashboard page; reload it"})
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}") if length else {}
        except ValueError:
            return self.send_json(400, {"error": "the request body is not JSON"})
        path = self.path.split("?", 1)[0].rstrip("/")
        parts = path.split("/")
        try:
            if path == "/api/jobs/pipeline":
                job = runner.create_pipeline(body)
            elif path == "/api/jobs/eval":
                job = runner.create_eval(body.get("run", ""), body.get("settings") or {})
            elif path == "/api/jobs/retrain":
                job = runner.create_retrain(body.get("run", ""), body.get("settings") or {})
            elif path == "/api/runs/restore":
                return self.send_json(200, runner.restore_training(body.get("run", ""), body.get("earlier", "")))
            elif path == "/api/jobs/build":
                job = runner.create_build()
            elif len(parts) == 5 and parts[:3] == ["", "api", "jobs"]:
                job = runner.action(parts[3], parts[4], body)
            elif path == "/api/auto":
                return self.send_json(200, {"auto": runner.set_auto(body)})
            elif path == "/api/auto/start":
                return self.send_json(200, {"auto": runner.start_auto_from(body.get("run", ""))})
            else:
                return self.send_json(404, {"error": f"no such action {path}"})
        except (ValueError, KeyError) as error:
            return self.send_json(400, {"error": str(error).strip("'")})
        except Exception as error:
            print(f"control error: {error!r}", file=sys.stderr)
            return self.send_json(500, {"error": repr(error)})
        return self.send_json(200, {"job": {k: v for k, v in job.items() if k != "commands"}})


def targets(text: str) -> list[float]:
    values = [float(v) for v in text.replace(" ", "").split(",") if v]
    if not values or any(not 0.5 < v < 1 for v in values):
        raise argparse.ArgumentTypeError("scores between .5 and 1, e.g. .55,.54,.53,.52")
    return values


def ladder(text: str) -> list[int]:
    values = [int(v) for v in text.replace(" ", "").split(",") if v]
    if not values or any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError("a comma-separated list of positive game counts, e.g. 3200,6400,12800")
    return values


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Dashboard and controller for the NNUE self-play loop")
    parser.add_argument("--repo", help="the repository root (default: found from the working directory)")
    parser.add_argument("--runs", help="look at another runs directory; controls work only on <repo>/runs")
    parser.add_argument("--python", default=sys.executable,
                        help="interpreter with the nnue, engine and torch packages (default: the one running the dashboard)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--open", action="store_true", help="open a browser tab")
    parser.add_argument("--read-only", action="store_true", help="show the runs without the controls")
    parser.add_argument("--allow-remote-control", action="store_true",
                        help="keep the controls when listening beyond this machine (anyone who can load the page can start jobs)")
    plan = parser.add_argument_group("the plan the decisions follow")
    plan.add_argument("--nodes", type=int, default=PLAN["nodes"], help="label nodes per move of new batches (default 50000)")
    plan.add_argument("--ladder", type=ladder, default=PLAN["ladder"],
                      help="games per batch by generation, the last repeating (default 3200,6400,12800,25600,51200)")
    plan.add_argument("--sprt-targets", type=targets, default=PLAN["sprt_targets"],
                      help="the sequential test's target by generation, the last repeating (default .55,.54,.53,.53,.52; "
                           "one value for a fixed target)")
    plan.add_argument("--sims", type=int, default=PLAN["decision_sims"],
                      help="decision budget in units of 2,500 nodes (default 16, the loop's 40k gain test)")
    args = parser.parse_args(argv)

    repo = Path(args.repo).resolve() if args.repo else find_repo(Path.cwd().resolve())
    runs_dir = Path(args.runs).resolve() if args.runs else (repo / "runs" if repo else Path("runs").resolve())
    overrides = {"nodes": args.nodes, "ladder": args.ladder, "decision_sims": args.sims, "sprt_targets": args.sprt_targets}
    reason = ""
    if args.read_only:
        reason = "The dashboard was started with --read-only."
    elif not repo:
        reason = "No repository root (Cargo.toml and Project.md) above the working directory; start it there or pass --repo."
    elif runs_dir != (repo / "runs").resolve():
        reason = f"Controls work on {repo / 'runs'} only: the trainer always writes there."
    elif args.host not in ("127.0.0.1", "localhost", "::1") and not args.allow_remote_control:
        reason = "Controls are off when listening beyond this machine; add --allow-remote-control to keep them."
    runner = None if reason else Runner(repo, args.python, plan=overrides)
    app = App(runs_dir, overrides, runner, secrets.token_urlsafe(24), reason)
    server = ThreadingHTTPServer((args.host, args.port), partial(Handler, app=app))
    url = f"http://{'localhost' if args.host in ('0.0.0.0', '127.0.0.1') else args.host}:{args.port}/"
    print(f"reading {runs_dir}" + (f"; jobs run from {repo} with {args.python}" if runner else f"; controls off: {reason}"))
    print(f"serving {url}  (ctrl-c to stop{'; a running step pauses and continues from the page later' if runner else ''})")
    if args.open:
        webbrowser.open(url)

    signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown, daemon=True).start())
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if runner:
            print("pausing the running step...")
            runner.shutdown()


if __name__ == "__main__":
    main()
