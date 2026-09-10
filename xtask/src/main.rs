//! Workspace tasks, run as `cargo xtask <command>`: `check` (formatting,
//! lints, tests, the engine wheel, Python lints and tests, the arena
//! frontend's gates), `wheel` (build and
//! install the engine Python module), `release` (host `bot` build) and `linux` (release build of `bot` for
//! Debian 12 inside the WSL distro `Debian`, into dist/linux).

use std::env;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

use anyhow::{bail, Context, Result};

const USAGE: &str = "\
usage: cargo xtask <command>
  check [--gpu]   cargo fmt --check, clippy, test, wheel, ruff, pytest (CPU unless --gpu), npm gates
  wheel           build the engine wheel and install it into the current python
  release [--portable]  build `bot` for Zen 4, or baseline x86-64 with --portable
  linux [--portable]    release build of `bot` for Debian 12 (WSL distro `Debian`) into dist/linux";

/// Python packages checked after the Rust crates and the wheel.
const PYTHON_PACKAGES: [&str; 4] = [
    "models/sq/py",
    "models/conv/py",
    "models/nnue/py",
    "arena/backend",
];
/// The arena frontend, checked last through its npm scripts.
const FRONTEND: &str = "arena/frontend";
const NPM: &str = if cfg!(windows) { "npm.cmd" } else { "npm" };

const DISTRO: &str = "Debian";
/// ONNX Runtime release linked on Linux: the version ort-sys 2.0.0-rc.13 targets.
const ORT_VERSION: &str = "1.28.0";
const SETUP: &str = "one-time setup: see xtask/README.md, section \"Linux build environment\"";

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask sits under the workspace root")
        .to_path_buf();
    let args: Vec<String> = env::args().skip(1).collect();
    let (command, flags) = match args.split_first() {
        Some((command, flags)) => (command.as_str(), flags),
        None => usage(),
    };
    let mut tasks = Tasks::new(root);
    let result = match (command, flags) {
        ("check", []) => tasks.check(false),
        ("check", [flag]) if flag == "--gpu" => tasks.check(true),
        ("wheel", []) => tasks.wheel(),
        ("release", []) => tasks.release(false),
        ("release", [flag]) if flag == "--portable" => tasks.release(true),
        ("linux", []) => tasks.linux(false),
        ("linux", [flag]) if flag == "--portable" => tasks.linux(true),
        _ => usage(),
    };
    tasks.summary();
    if let Err(error) = result {
        println!("FAILED: {error:#}");
        std::process::exit(1);
    }
}

fn usage() -> ! {
    eprintln!("{USAGE}");
    std::process::exit(2)
}

/// Runs the steps of one command from the workspace root and remembers what passed.
struct Tasks {
    root: PathBuf,
    cargo: String,
    started: Instant,
    passed: Vec<(String, f64)>,
}

impl Tasks {
    fn new(root: PathBuf) -> Tasks {
        env::set_current_dir(&root).expect("workspace root exists");
        Tasks {
            root,
            cargo: env::var("CARGO").unwrap_or_else(|_| "cargo".to_owned()),
            started: Instant::now(),
            passed: Vec::new(),
        }
    }

    /// Run one step with inherited output; a non-zero exit ends the command.
    fn step(&mut self, name: &str, mut command: Command) -> Result<()> {
        println!("\n== {name}");
        let started = Instant::now();
        let status = command
            .status()
            .with_context(|| format!("{name}: cannot start {:?}", command.get_program()))?;
        if !status.success() {
            bail!("step `{name}` failed ({status})");
        }
        self.passed
            .push((name.to_owned(), started.elapsed().as_secs_f64()));
        Ok(())
    }

    fn summary(&self) {
        println!();
        for (name, seconds) in &self.passed {
            println!("ok   {name} ({seconds:.0}s)");
        }
        println!(
            "{} steps passed in {:.0}s",
            self.passed.len(),
            self.started.elapsed().as_secs_f64()
        );
    }

    fn check(&mut self, gpu: bool) -> Result<()> {
        self.step("cargo fmt --check", cmd(&self.cargo, &["fmt", "--check"]))?;
        self.step(
            "cargo clippy",
            cmd(
                &self.cargo,
                &["clippy", "--all-targets", "--", "-D", "warnings"],
            ),
        )?;
        self.step("cargo test", cmd(&self.cargo, &["test"]))?;
        self.wheel()?;
        for package in PYTHON_PACKAGES {
            self.python(package, gpu)?;
        }
        self.frontend()
    }

    /// Lints and tests of one Python package; a model's Triton kernels run
    /// on the CPU unless `gpu`.
    fn python(&mut self, package: &str, gpu: bool) -> Result<()> {
        let dir = self.root.join(package);
        let mut check = cmd("python", &["-m", "ruff", "check"]);
        check.arg(&dir);
        self.step(&format!("ruff check ({package})"), check)?;
        let mut format = cmd("python", &["-m", "ruff", "format", "--check"]);
        format.arg(&dir);
        self.step(&format!("ruff format --check ({package})"), format)?;
        let mut pytest = cmd("python", &["-m", "pytest", "-q", "-p", "no:warnings"]);
        pytest.current_dir(&dir);
        let name = if !package.starts_with("models/") {
            format!("pytest ({package})")
        } else if gpu {
            format!("pytest ({package}, GPU)")
        } else {
            pytest.env("TRITON_INTERPRET", "1");
            format!("pytest ({package}, CPU, TRITON_INTERPRET=1)")
        };
        self.step(&name, pytest)
    }

    /// The arena frontend's gates: type check, lint, format check, tests, build.
    fn frontend(&mut self) -> Result<()> {
        let dir = self.root.join(FRONTEND);
        for script in ["typecheck", "lint", "format", "test", "build"] {
            let mut npm = cmd(NPM, &["run", script]);
            npm.current_dir(&dir);
            self.step(&format!("npm run {script} ({FRONTEND})"), npm)?;
        }
        Ok(())
    }

    /// Build engine's Python module into target/wheels and install it.
    fn wheel(&mut self) -> Result<()> {
        let wheels = self.root.join("target/wheels");
        fs::create_dir_all(&wheels)?;
        for old in engine_wheels(&wheels)? {
            fs::remove_file(old)?;
        }
        let mut build = cmd(
            "python",
            &[
                "-m",
                "maturin",
                "build",
                "--release",
                "--features",
                "python",
                "-o",
            ],
        );
        build.arg(&wheels).current_dir(self.root.join("engine"));
        self.step("maturin build (engine wheel)", build)?;
        let wheel = engine_wheels(&wheels)?
            .into_iter()
            .next()
            .context("maturin built no engine wheel")?;
        let mut install = cmd(
            "python",
            &["-m", "pip", "install", "--user", "--force-reinstall"],
        );
        install.arg(&wheel);
        self.step("pip install (engine wheel)", install)
    }

    /// Build the host CLI for the serving CPU, with a baseline x86-64 option.
    fn release(&mut self, portable: bool) -> Result<()> {
        let mut build = cmd(
            &self.cargo,
            &["build", "--release", "--locked", "-j", "4", "-p", "cli"],
        );
        let flags = env::var("RUSTFLAGS").unwrap_or_default();
        #[cfg(target_os = "linux")]
        let flags = format!("{flags} -C link-arg=-Wl,-rpath,$ORIGIN");
        build.env(
            "RUSTFLAGS",
            format!("{flags} -C target-cpu={}", target_cpu(portable)),
        );
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            build.creation_flags(0x00004000); // BELOW_NORMAL_PRIORITY_CLASS
        }
        self.step("cargo build --release", build)
    }

    /// Release build of `bot` inside the Debian 12 WSL distro, with Microsoft's
    /// ONNX Runtime release linked dynamically; `bot` and the library go to dist/linux.
    fn linux(&mut self, portable: bool) -> Result<()> {
        let release = wsl_output(None, &["sh", "-c", ". /etc/os-release && echo $VERSION_ID"])
            .with_context(|| format!("WSL distro `{DISTRO}` does not start; {SETUP}"))?;
        if release.trim() != "12" {
            bail!(
                "WSL distro `{DISTRO}` is Debian {} but the target is Debian 12; {SETUP}",
                release.trim()
            );
        }
        wsl_output(None, &["sh", "-c", "test -x $HOME/.cargo/bin/cargo"])
            .with_context(|| format!("cargo is missing inside `{DISTRO}`; {SETUP}"))?;
        let windows_root = self.root.to_str().context("workspace path is not UTF-8")?;
        let linux_root = wsl_output(None, &["wslpath", "-a", windows_root])?
            .trim()
            .to_owned();
        let ort = format!("onnxruntime-linux-x64-{ORT_VERSION}");
        let fetch = format!(
            "test -d target/linux/{ort} || (mkdir -p target/linux && curl -sSfL              https://github.com/microsoft/onnxruntime/releases/download/v{ORT_VERSION}/{ort}.tgz              | tar xz -C target/linux)"
        );
        self.step(
            "onnxruntime release (Linux x64)",
            wsl(Some(&linux_root), &["sh", "-c", &fetch]),
        )?;
        let build = format!(
            "export PATH=$HOME/.cargo/bin:$PATH ORT_LIB_LOCATION={linux_root}/target/linux/{ort}/lib              ORT_PREFER_DYNAMIC_LINK=1 RUSTFLAGS='-C target-cpu={} -C link-arg=-Wl,-rpath,$ORIGIN' && nice -n 10 cargo build --release --locked -j 4 -p cli --target-dir target/linux",
            target_cpu(portable)
        );
        self.step(
            "cargo build --release (Debian 12)",
            wsl(Some(&linux_root), &["sh", "-c", &build]),
        )?;
        let ldd = wsl_output(Some(&linux_root), &["ldd", "target/linux/release/bot"])?;
        let soname = ldd
            .lines()
            .filter_map(|line| line.split_whitespace().next())
            .find(|name| name.starts_with("libonnxruntime"))
            .context("bot does not link libonnxruntime")?;
        let collect = format!(
            "mkdir -p dist/linux && cp target/linux/release/bot dist/linux/bot              && cp -L target/linux/{ort}/lib/{soname} dist/linux/{soname}"
        );
        self.step(
            "collect dist/linux",
            wsl(Some(&linux_root), &["sh", "-c", &collect]),
        )?;
        let ldd = wsl_output(Some(&linux_root), &["ldd", "dist/linux/bot"])?;
        if ldd.contains("not found") {
            bail!(
                "dist/linux/bot has unresolved libraries:
{ldd}"
            );
        }
        let version = wsl_output(Some(&linux_root), &["./dist/linux/bot", "--version"])?;
        println!("dist/linux/bot: {}", version.trim());
        let sums = wsl_output(
            Some(&linux_root),
            &[
                "sha256sum",
                "dist/linux/bot",
                &format!("dist/linux/{soname}"),
            ],
        )?;
        print!("{sums}");
        Ok(())
    }
}

fn target_cpu(portable: bool) -> &'static str {
    if portable {
        "x86-64"
    } else {
        "znver4"
    }
}

fn cmd(program: &str, args: &[&str]) -> Command {
    let mut command = Command::new(program);
    command.args(args);
    command
}

/// `engine-*.whl` files in `dir`.
fn engine_wheels(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut wheels = Vec::new();
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        let name = path.file_name().and_then(OsStr::to_str).unwrap_or_default();
        if name.starts_with("engine-") && name.ends_with(".whl") {
            wheels.push(path);
        }
    }
    Ok(wheels)
}

/// A command executed as root inside the Debian distro, optionally in `cwd` (a Linux path).
fn wsl(cwd: Option<&str>, args: &[&str]) -> Command {
    let mut command = cmd("wsl.exe", &["-d", DISTRO, "-u", "root"]);
    if let Some(cwd) = cwd {
        command.args(["--cd", cwd]);
    }
    command.arg("-e").args(args);
    command
}

/// Stdout of a command inside the distro; an error if it fails.
fn wsl_output(cwd: Option<&str>, args: &[&str]) -> Result<String> {
    let output = wsl(cwd, args)
        .stdin(Stdio::null())
        .output()
        .context("cannot start wsl.exe")?;
    if !output.status.success() {
        bail!(
            "`{}` failed inside {DISTRO} ({})",
            args.join(" "),
            output.status
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}
