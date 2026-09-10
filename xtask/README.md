# xtask

Workspace tasks, run from any directory of the checkout as
`cargo xtask <command>` (alias in .cargo/config.toml). Dependencies: std and
anyhow. Every step prints its command; the first failing step ends the task
and is named; the passed steps are listed at the end.

## Commands

- `cargo xtask check [--gpu]`: `cargo fmt --check`,
  `cargo clippy --all-targets -- -D warnings`, `cargo test`, the `wheel` task,
  then for each Python package (models/sq/py, models/conv/py, models/nnue/py, arena/backend)
  `ruff check`, `ruff format --check` and `pytest`, the models' with
  `TRITON_INTERPRET=1` (kernels on the CPU; `--gpu` runs them without it),
  then in arena/frontend `npm run typecheck`, `lint`, `format`, `test` and
  `build`.
- `cargo xtask wheel`: `python -m maturin build --release --features python`
  of the engine crate into target/wheels, then
  `python -m pip install --user --force-reinstall` of that wheel.
- `cargo xtask linux`: `nice -n 10 cargo build --release --locked -j 4 -p cli --target-dir
  target/linux` inside the WSL distro `Debian` (Debian 12, x86_64, glibc 2.36:
  the site's platform), on this checkout through its /mnt/<drive>/ path.
  ONNX Runtime is Microsoft's Linux x64 release 1.28.0 (the version the `ort`
  crate targets), downloaded into target/linux on first use and linked
  dynamically; `bot` looks for it beside itself (`$ORIGIN` rpath in
  .cargo/config.toml). dist/linux receives `bot` and `libonnxruntime.so.1`,
  which ship together; the task prints `bot --version` and both sha256 sums.
  Fails, pointing here, when the distro is missing or is not Debian 12.

`python`, `ruff`, `pytest` and `maturin` are those of the interpreter on PATH
(`pip install -e "models/sq/py[dev]"` and the same for models/conv/py, models/nnue/py and
arena/backend); `npm` needs `npm ci` run once in arena/frontend.

## Linux build environment

One-time setup of the WSL distro `Debian`. `wsl --install -d Debian` installs
Debian 13, so a Debian 12 root filesystem is imported instead:

1. Get the `linux/amd64` layer of the `debian:bookworm` image, a gzipped root
   filesystem tar: an anonymous pull token from
   `https://auth.docker.io/token?service=registry.docker.io&scope=repository:library/debian:pull`,
   the manifest list at
   `https://registry-1.docker.io/v2/library/debian/manifests/bookworm`
   (Accept the OCI index and manifest media types), the `linux/amd64`
   manifest by digest, then its single layer from
   `https://registry-1.docker.io/v2/library/debian/blobs/<digest>`.
2. `wsl.exe --import Debian %LOCALAPPDATA%\wsl\Debian <layer.tar.gz> --version 2`
3. `wsl.exe -d Debian -u root -e sh -c 'apt-get update && apt-get install -y --no-install-recommends build-essential pkg-config libssl-dev curl ca-certificates git'`
4. `wsl.exe -d Debian -u root -e sh -c 'curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.95.0'`

`wsl.exe -d Debian -u root -e ldd --version` must report glibc 2.36. The first
`cargo xtask linux` downloads crate sources and the ONNX Runtime release, so
it needs network access.
