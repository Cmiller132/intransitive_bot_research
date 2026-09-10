//! The site adapter: RPSI protocol over stdin and stdout, and the site frame.
//!
//! The site sends positions as a FEN with Blue's home in any corner and moves
//! as `a1-b2` tokens in its own frame. `Frame` maps the site's squares to the
//! canonical frame and back; `Session` runs the protocol conversation and asks
//! the player for moves under the site clock.

use std::collections::BTreeSet;
use std::io::{BufRead, Write};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Result};
use engine::notation::{action_between, parse_squares, square_name};
use engine::{apply, flip, from_codes, from_to, mirror_anti, Action, Cell, Piece, Rules, State};

use crate::player::{Clock, History, Player};

pub const PROTOCOL: u32 = 1;
pub const RULES_VERSION: u32 = 2;
pub const MODE_ID: &str = "V6";
pub const MODE_NAME: &str = "Intransitive";

/// Clocks below this play with a one-simulation search.
const PANIC_CLOCK_MS: i64 = 800;
/// Below this the move is planned from the increment alone.
const LOW_CLOCK_MS: i64 = 3000;
const MOVE_OVERHEAD_MS: i64 = 150;
/// The territory field of a site FEN; the engine does not use it.
const TERRITORY: &str = "9/9/9/9/9/9/9/9/9";

/// Wall time for one move under a Fischer clock.
pub fn move_budget_ms(clock_ms: i64, inc_ms: i64, move_ms: i64, cap_ms: i64) -> i64 {
    let clock = clock_ms as f64;
    let increment = inc_ms as f64;
    if clock < LOW_CLOCK_MS as f64 {
        return 50.0f64.max(
            (0.5 * increment)
                .min(clock - PANIC_CLOCK_MS as f64)
                .min(cap_ms as f64),
        ) as i64;
    }
    let safe = clock - 1500.0;
    let target = (0.8 * increment + clock / 30.0 - MOVE_OVERHEAD_MS as f64)
        .min(cap_ms as f64)
        .min(safe);
    target.max((move_ms as f64).min(safe)) as i64
}

const CORNERS: [u8; 4] = [0, 8, 72, 80];

fn opposite(corner: u8) -> u8 {
    80 - corner
}

/// Site square <-> canonical square for one game, fixed by Blue's home corner
/// and by which side is to move.
#[derive(Clone, Debug)]
pub struct Frame {
    /// `transform[site] = absolute`, an involution sending Blue's home to a1.
    transform: [u8; 81],
    pub red_to_move: bool,
}

/// FEN piece field -> 81 cells in the site frame, Blue as own.
pub fn parse_pieces(field: &str) -> Result<[u8; 81]> {
    let rows: Vec<&str> = field.split('/').collect();
    if rows.len() != 9 {
        bail!("FEN has {} rows, expected 9", rows.len());
    }
    let mut board = [0u8; 81];
    for (r, row) in rows.iter().enumerate() {
        let mut file = 0usize;
        for ch in row.chars() {
            if let Some(n) = ch.to_digit(10) {
                file += n as usize;
                continue;
            }
            if ch == '.' {
                file += 1;
                continue;
            }
            let piece = match ch.to_ascii_uppercase() {
                'R' => 1,
                'P' => 2,
                'S' => 3,
                _ => bail!("FEN row {} has an unknown piece {ch:?}", r + 1),
            };
            if file > 8 {
                bail!("FEN row {} overflows the board", r + 1);
            }
            board[r * 9 + file] = if ch.is_uppercase() { piece } else { piece + 3 };
            file += 1;
        }
        if file != 9 {
            bail!("FEN row {} covers {file} files, expected 9", r + 1);
        }
    }
    Ok(board)
}

/// Blue's home corner: the one nearest to Blue's pieces (summed king distance).
fn nearest_corner(squares: &[u8]) -> u8 {
    let mut best = CORNERS[0];
    let mut best_distance = usize::MAX;
    for &corner in &CORNERS {
        let distance: usize = squares
            .iter()
            .map(|&s| ((corner / 9).abs_diff(s / 9)).max((corner % 9).abs_diff(s % 9)) as usize)
            .sum();
        if distance < best_distance {
            best = corner;
            best_distance = distance;
        }
    }
    best
}

impl Frame {
    /// Blue's home corner implied by a starting position in the site frame;
    /// `None` when Blue's and Red's corners are not opposite.
    pub fn detect(site_board: &[u8; 81]) -> Option<Frame> {
        let blue: Vec<u8> = (0..81u8)
            .filter(|&s| (1..=3).contains(&site_board[s as usize]))
            .collect();
        let red: Vec<u8> = (0..81u8).filter(|&s| site_board[s as usize] >= 4).collect();
        let blue_home = if blue.is_empty() {
            0
        } else {
            nearest_corner(&blue)
        };
        let red_home = if red.is_empty() {
            opposite(blue_home)
        } else {
            nearest_corner(&red)
        };
        (red_home == opposite(blue_home)).then(|| Frame::with_home(blue_home, false))
    }

    pub fn with_home(blue_home: u8, red_to_move: bool) -> Frame {
        let flip_file = matches!(blue_home, 8 | 80);
        let flip_rank = matches!(blue_home, 72 | 80);
        let mut transform = [0u8; 81];
        for (i, target) in transform.iter_mut().enumerate() {
            let mut rank = i as u8 / 9;
            let mut file = i as u8 % 9;
            if flip_rank {
                rank = 8 - rank;
            }
            if flip_file {
                file = 8 - file;
            }
            *target = rank * 9 + file;
        }
        Frame {
            transform,
            red_to_move,
        }
    }

    /// Site board -> canonical state for the side to move, at the given ply.
    pub fn canonical(&self, site_board: &[u8; 81], ply: u32) -> Result<State> {
        let mut absolute = [0u8; 81];
        for (site, &code) in site_board.iter().enumerate() {
            absolute[self.transform[site] as usize] = code;
        }
        let mut board = from_codes(&absolute).ok_or_else(|| anyhow!("bad piece code"))?;
        if self.red_to_move {
            board = flip(&board);
        }
        Ok(State {
            board,
            since_capture: 0,
            ply,
        })
    }

    /// Site move token -> canonical action in the current position.
    pub fn parse_move(&self, token: &str) -> Result<Action> {
        let (from, to) =
            parse_squares(token).ok_or_else(|| anyhow!("unparsable move {token:?}"))?;
        let (mut from, mut to) = (self.transform[from as usize], self.transform[to as usize]);
        if self.red_to_move {
            from = mirror_anti(from);
            to = mirror_anti(to);
        }
        action_between(from, to).ok_or_else(|| anyhow!("{token} is not a one-square king move"))
    }

    /// Canonical action -> site move token.
    pub fn spell(&self, action: Action) -> String {
        let (mut from, mut to) = from_to(action);
        if self.red_to_move {
            from = mirror_anti(from);
            to = mirror_anti(to);
        }
        format!(
            "{}-{}",
            square_name(self.transform[from as usize]),
            square_name(self.transform[to as usize])
        )
    }
}

/// `"d7-d6"`, `"Rd7xd6"` -> `"d7-d6"`; unparsable tokens are returned unchanged.
fn normalise(token: &str) -> String {
    match parse_squares(token) {
        Some((from, to)) => format!("{}-{}", square_name(from), square_name(to)),
        None => token.to_owned(),
    }
}

/// One RPSI conversation.
pub struct Session<P: Player> {
    pub player: P,
    pub rules: Rules,
    pub name: String,
    /// Wall time per move without a clock, and the floor with one.
    pub move_ms: i64,
    /// Cap on the per-move thinking time under a clock.
    pub max_move_ms: i64,
    /// Simulations per move when the host gives no time at all.
    pub sims: u32,
    /// Wall time per move in place of the simulation count when the host
    /// gives no time (`go sims N` or a bare `go`): the arena's seats play
    /// under a clock this way. An explicit host movetime or clock still wins.
    pub movetime: Option<i64>,
    state: Option<State>,
    frame: Option<Frame>,
    history: History,
    server_moves: Vec<String>,
    mode: String,
    quit: bool,
}

impl<P: Player> Session<P> {
    pub fn new(
        player: P,
        rules: Rules,
        name: &str,
        move_ms: i64,
        max_move_ms: i64,
        sims: u32,
    ) -> Session<P> {
        Session {
            player,
            rules,
            name: name.to_owned(),
            move_ms: move_ms.max(1),
            max_move_ms: max_move_ms.max(1),
            sims: sims.max(1),
            movetime: None,
            state: None,
            frame: None,
            history: History::new(),
            server_moves: Vec::new(),
            mode: MODE_ID.to_owned(),
            quit: false,
        }
    }

    /// Read lines until `quit`, answering each command on `out`.
    pub fn run(&mut self, input: &mut dyn BufRead, out: &mut dyn Write) -> Result<()> {
        let mut line = String::new();
        while !self.quit {
            line.clear();
            if input.read_line(&mut line)? == 0 {
                break;
            }
            self.handle(&line, out)?;
        }
        Ok(())
    }

    /// Handle one protocol line: `rpsi`, `isready`, `newgame`, `position`,
    /// `legalmoves`, `go`, `stop`, `quit`. Errors become `info string` notes.
    pub fn handle(&mut self, line: &str, out: &mut dyn Write) -> Result<()> {
        let words: Vec<&str> = line.split_whitespace().collect();
        let Some(&command) = words.first() else {
            return Ok(());
        };
        let args = &words[1..];
        let result = match command {
            "rpsi" => self.cmd_rpsi(out),
            "isready" => say(out, "readyok"),
            "setoption" => note(out, "options are fixed on the command line"),
            "newgame" => self.cmd_newgame(args),
            "position" => self.cmd_position(args, out),
            "legalmoves" => self.cmd_legalmoves(args, out),
            "go" => self.cmd_go(args, out),
            "stop" => Ok(()),
            "quit" => {
                self.quit = true;
                Ok(())
            }
            _ => Ok(()),
        };
        if let Err(error) = result {
            note(out, &format!("error in {command}: {error}"))?;
            if command == "go" {
                self.emergency_bestmove(out)?;
            }
        }
        Ok(())
    }

    fn cmd_rpsi(&self, out: &mut dyn Write) -> Result<()> {
        say(out, &format!("id name {}", self.name))?;
        say(out, "id author intransitive_bot_research")?;
        say(out, &format!("protocol {PROTOCOL}"))?;
        say(out, &format!("rules {RULES_VERSION}"))?;
        say(out, &format!("mode {MODE_ID} {MODE_NAME}"))?;
        say(out, "rpsiok")
    }

    fn cmd_newgame(&mut self, args: &[&str]) -> Result<()> {
        self.mode = args.first().copied().unwrap_or(MODE_ID).to_owned();
        self.state = None;
        self.frame = None;
        self.history.clear();
        self.server_moves.clear();
        self.player.new_game();
        Ok(())
    }

    fn cmd_position(&mut self, args: &[&str], out: &mut dyn Write) -> Result<()> {
        if args.first().copied() != Some("fen") {
            return note(out, "position without a fen; ignored");
        }
        let rest = &args[1..];
        let (fields, moves) = match rest.iter().position(|&w| w == "moves") {
            Some(at) => (&rest[..at], &rest[at + 1..]),
            None => (rest, &[][..]),
        };
        if fields.len() < 2 {
            bail!("position fen needs a piece field and a side");
        }
        let site_board = parse_pieces(fields[0])?;
        let red_to_move = match fields[1].to_lowercase().as_str() {
            "r" => true,
            "b" | "-" => false,
            other => {
                note(
                    out,
                    &format!("unknown side to move {other:?}, assuming blue"),
                )?;
                false
            }
        };
        let mut frame = match Frame::detect(&site_board) {
            Some(frame) => frame,
            None => {
                note(out, "home corners disagree; using the identity transform")?;
                Frame::with_home(0, false)
            }
        };
        frame.red_to_move = red_to_move;
        // A book FEN fixes the parity, not the full ply count.
        let mut state = frame.canonical(&site_board, u32::from(red_to_move))?;
        self.history.clear();
        self.history.insert(state.key(), 1);
        for &token in moves {
            let action = frame.parse_move(token)?;
            if !state.legal_mask()[action as usize] {
                note(
                    out,
                    &format!("replayed move {token} is not legal by our rules"),
                )?;
                bail!("cannot replay {token}");
            }
            let (child, outcome) = apply(&self.rules, &state, action);
            if outcome != engine::Outcome::Ongoing {
                note(
                    out,
                    &format!("our rules say the game ended at move {token}"),
                )?;
            }
            state = child;
            frame.red_to_move = !frame.red_to_move;
            *self.history.entry(state.key()).or_insert(0) += 1;
        }
        self.state = Some(state);
        self.frame = Some(frame);
        self.server_moves.clear();
        Ok(())
    }

    fn cmd_legalmoves(&mut self, args: &[&str], out: &mut dyn Write) -> Result<()> {
        self.server_moves = args.iter().map(|t| (*t).to_owned()).collect();
        let (Some(state), Some(frame)) = (&self.state, &self.frame) else {
            return Ok(());
        };
        let ours: BTreeSet<String> = state
            .legal_actions()
            .iter()
            .map(|&a| frame.spell(a))
            .collect();
        let theirs: BTreeSet<String> = args.iter().map(|t| normalise(t)).collect();
        if !theirs.is_empty() && ours != theirs {
            let missing: Vec<&str> = theirs.difference(&ours).map(String::as_str).collect();
            let extra: Vec<&str> = ours.difference(&theirs).map(String::as_str).collect();
            note(
                out,
                &format!(
                    "legalmoves mismatch: {} ours vs {} theirs; we miss [{}] we add [{}]",
                    ours.len(),
                    theirs.len(),
                    missing.join(" "),
                    extra.join(" ")
                ),
            )?;
        }
        Ok(())
    }

    fn cmd_go(&mut self, args: &[&str], out: &mut dyn Write) -> Result<()> {
        let (Some(state), Some(frame)) = (self.state.clone(), self.frame.clone()) else {
            note(out, "go without a position")?;
            return self.emergency_bestmove(out);
        };
        if self.mode != MODE_ID {
            note(
                out,
                &format!(
                    "mode {} is not {MODE_ID}; playing the host's first legal move",
                    self.mode
                ),
            )?;
            return self.emergency_bestmove(out);
        }
        if state.is_stalemated() {
            note(out, "no legal move")?;
            return self.emergency_bestmove(out);
        }
        let params = go_params(args);
        let side = if frame.red_to_move { "r" } else { "b" };
        let clock_ms = params.get(&format!("{side}time")).copied();
        let inc_ms = params.get(&format!("{side}inc")).copied().unwrap_or(0);
        let movetime = params.get("movetime").copied();
        let clock = match (movetime, clock_ms) {
            (Some(ms), _) => Clock::Time(Duration::from_millis(ms.max(1) as u64)),
            (_, Some(ms)) if ms < PANIC_CLOCK_MS => Clock::Sims(1),
            (None, Some(ms)) => Clock::Time(Duration::from_millis(
                move_budget_ms(ms, inc_ms, self.move_ms, self.max_move_ms).max(1) as u64,
            )),
            (None, None) => match self.movetime {
                Some(ms) => Clock::Time(Duration::from_millis(ms.max(1) as u64)),
                None => Clock::Sims(
                    params
                        .get("sims")
                        .map_or(self.sims, |&n| n.clamp(1, u32::MAX as i64) as u32),
                ),
            },
        };
        let started = Instant::now();
        let action = self.player.choose(&state, &self.history, clock);
        let elapsed = started.elapsed().as_millis();
        let token = self.pick_spelling(&frame, action, out)?;
        say(out, &format!("info time {elapsed} pv {token}"))?;
        if let Some(info) = self.player.info() {
            let telemetry = crate::player::Telemetry {
                info,
                search: self.player.search_details(),
            };
            say(
                out,
                &format!("info json {}", serde_json::to_string(&telemetry)?),
            )?;
        }
        say(out, &format!("bestmove {token}"))
    }

    /// Our move in the host's own spelling, or the host's first move if ours
    /// is not in its list.
    fn pick_spelling(&self, frame: &Frame, action: Action, out: &mut dyn Write) -> Result<String> {
        let want = frame.spell(action);
        if self.server_moves.is_empty() {
            return Ok(want);
        }
        if let Some(token) = self.server_moves.iter().find(|t| normalise(t) == want) {
            return Ok(token.clone());
        }
        note(
            out,
            &format!("our move {want} is not in the host's legalmoves; falling back"),
        )?;
        Ok(self.server_moves[0].clone())
    }

    fn emergency_bestmove(&self, out: &mut dyn Write) -> Result<()> {
        if let Some(token) = self.server_moves.first() {
            return say(out, &format!("bestmove {token}"));
        }
        if let (Some(state), Some(frame)) = (&self.state, &self.frame) {
            if let Some(&action) = state.legal_actions().first() {
                return say(out, &format!("bestmove {}", frame.spell(action)));
            }
        }
        note(out, "no legal move is known; answering with a placeholder")?;
        say(out, "bestmove a1-a1")
    }
}

fn say(out: &mut dyn Write, line: &str) -> Result<()> {
    out.write_all(line.as_bytes())?;
    out.write_all(b"\n")?;
    out.flush()?;
    Ok(())
}

fn note(out: &mut dyn Write, message: &str) -> Result<()> {
    say(out, &format!("info string {message}"))
}

fn go_params(args: &[&str]) -> std::collections::HashMap<String, i64> {
    let mut params = std::collections::HashMap::new();
    for pair in args.windows(2) {
        if let Ok(value) = pair[1].parse::<i64>() {
            params.insert(pair[0].to_owned(), value);
        }
    }
    params
}

/// Site board of the start position with Blue's home at `blue_home`, for tests.
pub fn site_start(blue_home: u8) -> [u8; 81] {
    let frame = Frame::with_home(blue_home, false);
    let mut site = [0u8; 81];
    let start = State::initial();
    for (site_square, &absolute) in frame.transform.iter().enumerate() {
        site[site_square] = match start.board[absolute as usize] {
            Cell::Empty => 0,
            Cell::Own(p) => p.code(),
            Cell::Enemy(p) => p.code() + 3,
        };
    }
    site
}

/// The start position as the site's FEN fields (pieces, side to move,
/// territory) with Blue's home at `blue_home`.
pub fn start_fen(blue_home: u8) -> String {
    let site = site_start(blue_home);
    let rows: Vec<String> = site
        .chunks(9)
        .map(|row| {
            row.iter()
                .map(|&code| match code {
                    0 => '.',
                    1 => 'R',
                    2 => 'P',
                    3 => 'S',
                    4 => 'r',
                    5 => 'p',
                    _ => 's',
                })
                .collect()
        })
        .collect();
    format!("{} b {TERRITORY}", rows.join("/"))
}

/// The piece type of a site cell code, for tests.
pub fn piece_of(code: u8) -> Option<Piece> {
    Piece::from_code(if code > 3 { code - 3 } else { code })
}

/// Plays the first legal action; enough to drive the protocol in tests.
#[cfg(test)]
pub(crate) struct FirstMove;

#[cfg(test)]
impl Player for FirstMove {
    fn new_game(&mut self) {}

    fn choose(&mut self, state: &State, _: &History, _: Clock) -> Action {
        state.legal_actions()[0]
    }

    fn name(&self) -> &str {
        "first"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_movetime_reaches_the_player_without_panic_conversion() {
        use std::sync::{Arc, Mutex};
        struct Budgets(Arc<Mutex<Vec<Clock>>>);
        impl Player for Budgets {
            fn new_game(&mut self) {}
            fn choose(&mut self, state: &State, _: &History, clock: Clock) -> Action {
                self.0.lock().unwrap().push(clock);
                state.legal_actions()[0]
            }
            fn name(&self) -> &str {
                "budgets"
            }
        }
        let seen = Arc::new(Mutex::new(Vec::new()));
        let mut s = Session::new(
            Budgets(Arc::clone(&seen)),
            Rules::SITE,
            "budgets",
            250,
            250,
            32,
        );
        let mut out = Vec::new();
        s.handle(
            &format!("position fen {SITE_START_FEN} b {TERRITORY}"),
            &mut out,
        )
        .unwrap();
        for ms in [50, 100, 250, 1000] {
            s.handle(&format!("go movetime {ms} btime 10 rtime 10"), &mut out)
                .unwrap();
        }
        for (clock, ms) in seen.lock().unwrap().iter().zip([50, 100, 250, 1000]) {
            assert!(matches!(clock, Clock::Time(d) if d.as_millis() == ms));
        }
        assert_eq!(seen.lock().unwrap().len(), 4);
    }

    #[test]
    fn a_seat_movetime_replaces_the_simulation_count_but_not_a_host_time() {
        use std::sync::{Arc, Mutex};
        struct Budgets(Arc<Mutex<Vec<Clock>>>);
        impl Player for Budgets {
            fn new_game(&mut self) {}
            fn choose(&mut self, state: &State, _: &History, clock: Clock) -> Action {
                self.0.lock().unwrap().push(clock);
                state.legal_actions()[0]
            }
            fn name(&self) -> &str {
                "budgets"
            }
        }
        let seen = Arc::new(Mutex::new(Vec::new()));
        let mut s = Session::new(
            Budgets(Arc::clone(&seen)),
            Rules::SITE,
            "budgets",
            250,
            250,
            32,
        );
        let mut out = Vec::new();
        s.handle(
            &format!("position fen {SITE_START_FEN} b {TERRITORY}"),
            &mut out,
        )
        .unwrap();
        s.handle("go sims 32", &mut out).unwrap();
        s.movetime = Some(100);
        s.handle("go sims 32", &mut out).unwrap();
        s.handle("go", &mut out).unwrap();
        s.handle("go movetime 50", &mut out).unwrap();
        let seen = seen.lock().unwrap();
        assert!(matches!(seen[0], Clock::Sims(32)));
        assert!(matches!(seen[1], Clock::Time(d) if d.as_millis() == 100));
        assert!(matches!(seen[2], Clock::Time(d) if d.as_millis() == 100));
        assert!(matches!(seen[3], Clock::Time(d) if d.as_millis() == 50));
    }

    #[test]
    fn frames_round_trip_for_every_home_corner() {
        for &home in &CORNERS {
            let site = site_start(home);
            let frame = Frame::detect(&site).expect("consistent corners");
            let state = frame.canonical(&site, 0).unwrap();
            assert_eq!(state, State::initial(), "home {home}");
            for action in state.legal_actions() {
                let token = frame.spell(action);
                assert_eq!(
                    frame.parse_move(&token).unwrap(),
                    action,
                    "home {home} {token}"
                );
            }
        }
    }

    #[test]
    fn red_moves_are_spelled_from_reds_corner() {
        let site = site_start(0);
        let mut frame = Frame::detect(&site).unwrap();
        let blue = frame.canonical(&site, 0).unwrap();
        let (red, _) = apply(&Rules::SITE, &blue, blue.legal_actions()[0]);
        frame.red_to_move = true;
        let token = frame.spell(red.legal_actions()[0]);
        let (from, _) = parse_squares(&token).unwrap();
        assert!(site[from as usize] >= 4, "red moves a red piece: {token}");
        assert_eq!(frame.parse_move(&token).unwrap(), red.legal_actions()[0]);
    }

    #[test]
    fn budget_is_bounded_by_the_cap_and_the_clock() {
        assert_eq!(move_budget_ms(60_000, 1_000, 250, 250), 250);
        assert!(move_budget_ms(2_000, 1_000, 250, 250) <= 250);
        assert!(move_budget_ms(1_000, 0, 250, 250) >= 50);
    }

    /// The opening as the site serves it: rank 1 first, uppercase Blue, Blue's home at i1.
    const SITE_START_FEN: &str =
        "........./....PR.../....SPR../.....SPR./.ps...SP./.rps...../..rps..../...rp..../.........";

    #[test]
    fn start_fen_is_the_site_opening() {
        assert_eq!(start_fen(8), format!("{SITE_START_FEN} b {TERRITORY}"));
    }

    fn session() -> Session<FirstMove> {
        Session::new(FirstMove, Rules::SITE, "test", 250, 250, 4)
    }

    /// Feed `lines` to the session and return every reply line.
    fn send(session: &mut Session<FirstMove>, lines: &[&str]) -> Vec<String> {
        let mut out = Vec::new();
        for line in lines {
            session.handle(line, &mut out).unwrap();
        }
        String::from_utf8(out)
            .unwrap()
            .lines()
            .map(str::to_owned)
            .collect()
    }

    fn bestmoves(lines: &[String]) -> Vec<String> {
        lines
            .iter()
            .filter_map(|l| l.strip_prefix("bestmove "))
            .map(str::to_owned)
            .collect()
    }

    /// Our legal moves in the site's spelling for the position the session holds.
    fn site_legal(session: &Session<FirstMove>) -> Vec<String> {
        let (state, frame) = (
            session.state.as_ref().unwrap(),
            session.frame.as_ref().unwrap(),
        );
        state
            .legal_actions()
            .iter()
            .map(|&a| frame.spell(a))
            .collect()
    }

    #[test]
    fn handshake_identifies_the_engine_and_fixes_options() {
        let lines = send(
            &mut session(),
            &["rpsi", "isready", "setoption name Sims value 8", "unknown"],
        );
        assert_eq!(
            lines,
            [
                "id name test",
                "id author intransitive_bot_research",
                "protocol 1",
                "rules 2",
                "mode V6 Intransitive",
                "rpsiok",
                "readyok",
                "info string options are fixed on the command line",
            ]
        );
    }

    #[test]
    fn site_start_fen_is_the_initial_position_and_a_turn_completes() {
        let mut s = session();
        let position = format!("position fen {SITE_START_FEN} b {TERRITORY}");
        assert!(send(&mut s, &["newgame V6", &position]).is_empty());
        assert_eq!(*s.state.as_ref().unwrap(), State::initial());
        let legal = site_legal(&s);
        let lines = send(
            &mut s,
            &[
                &format!("legalmoves {}", legal.join(" ")),
                "go btime 60000 rtime 60000 binc 1000 rinc 1000",
                "stop",
                "quit",
            ],
        );
        let best = bestmoves(&lines);
        assert_eq!(best.len(), 1, "{lines:?}");
        assert!(legal.contains(&best[0]));
        let pv = format!(" pv {}", best[0]);
        assert!(
            lines
                .iter()
                .any(|l| l.starts_with("info time ") && l.ends_with(&pv)),
            "{lines:?}"
        );
        assert!(!lines.iter().any(|l| l.contains("mismatch")), "{lines:?}");
        assert!(s.quit);
    }

    #[test]
    fn replayed_moves_alternate_sides_and_count_repetitions() {
        let mut s = session();
        let shuttle = "e2-e1 e8-e9 e1-e2 e9-e8";
        send(
            &mut s,
            &[&format!(
                "position fen {SITE_START_FEN} b {TERRITORY} moves {shuttle}"
            )],
        );
        let state = s.state.as_ref().unwrap();
        assert_eq!(state.board, State::initial().board);
        assert_eq!((state.ply, state.since_capture), (4, 4));
        assert_eq!(s.history[&state.key()], 2);
        assert!(!s.frame.as_ref().unwrap().red_to_move);
        send(
            &mut s,
            &[&format!(
                "position fen {SITE_START_FEN} b {TERRITORY} moves e2-e1"
            )],
        );
        assert!(s.frame.as_ref().unwrap().red_to_move);
        let best = bestmoves(&send(&mut s, &["go movetime 1000"]));
        let (from, _) = parse_squares(&best[0]).unwrap();
        assert!(
            parse_pieces(SITE_START_FEN).unwrap()[from as usize] >= 4,
            "red moves a red piece: {}",
            best[0]
        );
    }

    #[test]
    fn host_spellings_and_lists_are_respected() {
        let mut s = session();
        send(
            &mut s,
            &[&format!("position fen {SITE_START_FEN} b {TERRITORY}")],
        );
        let legal = site_legal(&s);
        // The host spells moves without the dash; the reply uses the host's token.
        let hosts: Vec<String> = legal.iter().map(|m| m.replace('-', "")).collect();
        let lines = send(&mut s, &[&format!("legalmoves {}", hosts.join(" ")), "go"]);
        let best = bestmoves(&lines);
        assert!(hosts.contains(&best[0]), "{best:?}");
        assert!(!lines.iter().any(|l| l.contains("mismatch")), "{lines:?}");
        // A list that disagrees with our rules is reported.
        let lines = send(&mut s, &[&format!("legalmoves {}", legal[1..].join(" "))]);
        assert!(
            lines[0].starts_with("info string legalmoves mismatch:"),
            "{lines:?}"
        );
    }

    #[test]
    fn go_without_a_position_or_in_another_mode_still_answers() {
        let mut s = session();
        let lines = send(&mut s, &["go movetime 100"]);
        assert_eq!(lines[0], "info string go without a position");
        assert_eq!(bestmoves(&lines), ["a1-a1"]);
        let lines = send(
            &mut s,
            &[
                "newgame V7",
                &format!("position fen {SITE_START_FEN} b {TERRITORY}"),
                "legalmoves x9-x8 e2-e1",
                "go btime 500 rtime 500",
            ],
        );
        assert!(
            lines
                .iter()
                .any(|l| l.starts_with("info string mode V7 is not V6")),
            "{lines:?}"
        );
        assert_eq!(bestmoves(&lines), ["x9-x8"]);
    }

    #[test]
    fn protocol_errors_become_notes() {
        let lines = send(&mut session(), &["position", "position fen 9/9 b"]);
        assert_eq!(lines[0], "info string position without a fen; ignored");
        assert_eq!(
            lines[1],
            "info string error in position: FEN has 2 rows, expected 9"
        );
    }
}
