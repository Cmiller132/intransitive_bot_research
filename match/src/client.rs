//! The seat for an external engine: a process speaking RPSI on its stdin and
//! stdout, driven as a `Player`. Every position goes to the engine as the
//! site's start FEN plus the move list; a wrong, late or missing move forfeits
//! the game.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use engine::{Action, State};

use crate::player::{Clock, History, MoveInfo, Player, Telemetry};
use crate::rpsi::{start_fen, Frame, MODE_ID};

const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(60);
const MOVE_TIMEOUT: Duration = Duration::from_secs(180);
const QUIT_GRACE: Duration = Duration::from_secs(2);
/// Blue's home corner in the positions sent to the engine: i1, as the site serves it.
const BLUE_HOME: u8 = 8;

/// Line transport to an engine.
pub trait Wire {
    fn send(&mut self, line: &str) -> Result<()>;

    /// The next line, or an error when none arrives within `timeout`.
    fn recv(&mut self, timeout: Duration) -> Result<String>;
}

/// A child process with a reader thread on its stdout; its stderr is inherited.
pub struct Process {
    child: Child,
    stdin: ChildStdin,
    lines: Receiver<String>,
}

impl Process {
    pub fn spawn(command: &[String]) -> Result<Process> {
        let (program, args) = command
            .split_first()
            .ok_or_else(|| anyhow!("empty engine command"))?;
        let mut child = Command::new(program)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .with_context(|| format!("starting {program}"))?;
        let stdin = child.stdin.take().expect("piped stdin");
        let stdout = child.stdout.take().expect("piped stdout");
        let (sender, lines) = mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                if sender.send(line).is_err() {
                    break;
                }
            }
        });
        Ok(Process {
            child,
            stdin,
            lines,
        })
    }
}

impl Wire for Process {
    fn send(&mut self, line: &str) -> Result<()> {
        self.stdin.write_all(line.as_bytes())?;
        self.stdin.write_all(b"\n")?;
        self.stdin.flush()?;
        Ok(())
    }

    fn recv(&mut self, timeout: Duration) -> Result<String> {
        match self.lines.recv_timeout(timeout) {
            Ok(line) => Ok(line),
            Err(RecvTimeoutError::Timeout) => bail!("no reply within {} s", timeout.as_secs()),
            Err(RecvTimeoutError::Disconnected) => bail!("the engine closed its output"),
        }
    }
}

impl Drop for Process {
    fn drop(&mut self) {
        let _ = self.send("quit");
        let deadline = Instant::now() + QUIT_GRACE;
        while Instant::now() < deadline {
            if matches!(self.child.try_wait(), Ok(Some(_))) {
                return;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// An RPSI engine as a player.
pub struct RpsiPlayer<W: Wire = Process> {
    wire: W,
    name: String,
    /// Moves of the current game in the engine's spelling.
    moves: Vec<String>,
    frame: Frame,
    forfeited: bool,
    telemetry: Option<Telemetry>,
}

impl RpsiPlayer<Process> {
    /// Start `command` and complete the handshake.
    pub fn spawn(command: &[String]) -> Result<RpsiPlayer> {
        RpsiPlayer::new(Process::spawn(command)?)
    }
}

impl<W: Wire> RpsiPlayer<W> {
    /// Handshake over `wire`: `rpsi` until `rpsiok`, then `isready`.
    pub fn new(mut wire: W) -> Result<RpsiPlayer<W>> {
        let deadline = Instant::now() + HANDSHAKE_TIMEOUT;
        wire.send("rpsi")?;
        let mut name = String::from("rpsi");
        loop {
            let line = wire.recv(remaining(deadline))?;
            if let Some(rest) = line.strip_prefix("id name ") {
                name = rest.trim().to_owned();
            }
            if line.trim() == "rpsiok" {
                break;
            }
        }
        wire.send("isready")?;
        while wire.recv(remaining(deadline))?.trim() != "readyok" {}
        Ok(RpsiPlayer {
            wire,
            name,
            moves: Vec::new(),
            frame: Frame::with_home(BLUE_HOME, false),
            forfeited: false,
            telemetry: None,
        })
    }

    /// The engine's move in `state`, the position after `self.moves`.
    fn ask(&mut self, state: &State, clock: Clock) -> Result<Action> {
        let mut position = format!("position fen {}", start_fen(BLUE_HOME));
        if !self.moves.is_empty() {
            position.push_str(" moves ");
            position.push_str(&self.moves.join(" "));
        }
        self.wire.send(&position)?;
        let legal: Vec<String> = state
            .legal_actions()
            .iter()
            .map(|&a| self.frame.spell(a))
            .collect();
        self.wire.send(&format!("legalmoves {}", legal.join(" ")))?;
        self.wire.send(&match clock {
            Clock::Sims(n) => format!("go sims {n}"),
            Clock::Time(duration) => format!("go movetime {}", duration.as_millis()),
        })?;
        let deadline = Instant::now() + MOVE_TIMEOUT;
        let token = loop {
            let line = self.wire.recv(remaining(deadline))?;
            if let Some(json) = line.strip_prefix("info json ") {
                self.telemetry =
                    Some(serde_json::from_str(json).context("invalid RPSI telemetry")?);
            }
            if let Some(rest) = line.trim().strip_prefix("bestmove") {
                break rest
                    .split_whitespace()
                    .next()
                    .ok_or_else(|| anyhow!("empty bestmove"))?
                    .to_owned();
            }
        };
        let action = self.frame.parse_move(&token)?;
        if !state.legal_mask()[action as usize] {
            bail!("{token} is not legal");
        }
        Ok(action)
    }
}

fn remaining(deadline: Instant) -> Duration {
    deadline.saturating_duration_since(Instant::now())
}

impl<W: Wire> Player for RpsiPlayer<W> {
    fn new_game(&mut self) {
        self.moves.clear();
        self.frame.red_to_move = false;
        self.forfeited = false;
        self.telemetry = None;
        if let Err(error) = self.wire.send(&format!("newgame {MODE_ID}")) {
            eprintln!("{} forfeits: {error}", self.name);
            self.forfeited = true;
        }
    }

    fn observe(&mut self, action: Action) {
        self.moves.push(self.frame.spell(action));
        self.frame.red_to_move = !self.frame.red_to_move;
    }

    fn choose(&mut self, state: &State, _: &History, clock: Clock) -> Action {
        self.telemetry = None;
        if !self.forfeited {
            match self.ask(state, clock) {
                Ok(action) => return action,
                Err(error) => {
                    eprintln!("{} forfeits: {error}", self.name);
                    self.forfeited = true;
                }
            }
        }
        state.legal_actions()[0]
    }

    fn forfeited(&self) -> bool {
        self.forfeited
    }

    fn info(&self) -> Option<MoveInfo> {
        self.telemetry.as_ref().map(|t| t.info.clone())
    }

    fn search_details(&self) -> Option<serde_json::Value> {
        self.telemetry.as_ref().and_then(|t| t.search.clone())
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use engine::Rules;
    use rand::{rngs::StdRng, SeedableRng};

    use super::*;
    use crate::game::{play, End, Opening};
    use crate::rpsi::{FirstMove, Session};

    /// An in-process engine: a `Session` answering each line at once.
    struct Loopback<P: Player = FirstMove> {
        session: Session<P>,
        queue: VecDeque<String>,
        notes: Vec<String>,
    }

    impl Loopback {
        fn new() -> Loopback {
            Loopback {
                session: Session::new(FirstMove, Rules::SITE, "loop", 250, 250, 4),
                queue: VecDeque::new(),
                notes: Vec::new(),
            }
        }
    }

    impl<P: Player> Wire for Loopback<P> {
        fn send(&mut self, line: &str) -> Result<()> {
            let mut out = Vec::new();
            self.session.handle(line, &mut out)?;
            for reply in String::from_utf8(out)?.lines() {
                if let Some(note) = reply.strip_prefix("info string ") {
                    self.notes.push(note.to_owned());
                }
                self.queue.push_back(reply.to_owned());
            }
            Ok(())
        }

        fn recv(&mut self, _: Duration) -> Result<String> {
            self.queue.pop_front().ok_or_else(|| anyhow!("no reply"))
        }
    }

    #[test]
    fn telemetry_round_trips_and_does_not_survive_an_unreported_move() {
        struct Tracked(u32, Action);
        impl Player for Tracked {
            fn new_game(&mut self) {
                self.0 = 0;
            }
            fn choose(&mut self, state: &State, _: &History, _: Clock) -> Action {
                self.0 += 1;
                self.1 = state.legal_actions()[0];
                self.1
            }
            fn name(&self) -> &str {
                "tracked"
            }
            fn info(&self) -> Option<MoveInfo> {
                (self.0 == 1).then(|| MoveInfo {
                    sims: 17,
                    value: 0.25,
                    plies_left: None,
                    q: 0.125,
                    pi: 1.0,
                    exact_win: false,
                    top: vec![(self.1, 17, 0.125)],
                })
            }
            fn search_details(&self) -> Option<serde_json::Value> {
                Some(serde_json::json!({"nodes":12345,"elapsed_ms":2.5,"pv":[self.1]}))
            }
        }
        let wire = Loopback {
            session: Session::new(Tracked(0, 0), Rules::SITE, "tracked", 250, 250, 4),
            queue: VecDeque::new(),
            notes: Vec::new(),
        };
        let mut player = RpsiPlayer::new(wire).unwrap();
        let root = State::initial();
        let action = player.choose(&root, &History::new(), Clock::Sims(17));
        let info = player.info().unwrap();
        assert_eq!((info.sims, info.value, info.top[0].0), (17, 0.25, action));
        assert_eq!(player.search_details().unwrap()["nodes"], 12345);
        assert_eq!(player.search_details().unwrap()["pv"][0], action);
        player.observe(action);
        let child = engine::apply(&Rules::SITE, &root, action).0;
        player.choose(&child, &History::new(), Clock::Sims(17));
        assert!(!player.forfeited());
        assert!(player.info().is_none());
        assert!(player.search_details().is_none());
        player.new_game();
        assert!(player.info().is_none());
    }

    /// Completes the handshake, then answers every `go` with an unplayable move.
    struct Stuck(VecDeque<String>);

    impl Wire for Stuck {
        fn send(&mut self, line: &str) -> Result<()> {
            match line.split_whitespace().next() {
                Some("rpsi") => self.0.push_back("rpsiok".to_owned()),
                Some("isready") => self.0.push_back("readyok".to_owned()),
                Some("go") => self.0.push_back("bestmove a1-a1".to_owned()),
                _ => {}
            }
            Ok(())
        }

        fn recv(&mut self, _: Duration) -> Result<String> {
            self.0.pop_front().ok_or_else(|| anyhow!("no reply"))
        }
    }

    #[test]
    fn a_game_against_the_session_replays_cleanly_in_both_seats() {
        let mut rng = StdRng::seed_from_u64(3);
        for seat in 0..2 {
            let mut engine = RpsiPlayer::new(Loopback::new()).unwrap();
            assert_eq!(engine.name(), "loop");
            let mut other = FirstMove;
            let opening = Opening::random(&Rules::SITE, 8, &mut rng);
            let record = if seat == 0 {
                play(
                    &Rules::SITE,
                    &mut engine,
                    &mut other,
                    &opening,
                    Clock::Sims(2),
                )
            } else {
                play(
                    &Rules::SITE,
                    &mut other,
                    &mut engine,
                    &opening,
                    Clock::Sims(2),
                )
            };
            assert_ne!(record.end, End::Forfeit, "seat {seat}: {record:?}");
            assert!(record.plies > 8);
            assert!(
                engine.wire.notes.is_empty(),
                "seat {seat}: {:?}",
                engine.wire.notes
            );
        }
    }

    #[test]
    fn an_engine_that_cannot_move_forfeits() {
        let mut engine = RpsiPlayer::new(Stuck(VecDeque::new())).unwrap();
        let mut other = FirstMove;
        let opening = Opening::random(&Rules::SITE, 4, &mut StdRng::seed_from_u64(1));
        let record = play(
            &Rules::SITE,
            &mut engine,
            &mut other,
            &opening,
            Clock::Sims(1),
        );
        assert_eq!((record.end, record.winner), (End::Forfeit, Some(1)));
        assert_eq!(record.plies, 4);
        assert!(engine.forfeited());
    }
}
