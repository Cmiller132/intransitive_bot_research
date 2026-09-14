PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS players (
    name TEXT PRIMARY KEY,
    spec TEXT NOT NULL,
    added TEXT NOT NULL,
    broken INTEGER NOT NULL DEFAULT 0,
    retired INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    a TEXT NOT NULL,
    b TEXT NOT NULL,
    seed INTEGER NOT NULL,
    started TEXT NOT NULL,
    finished TEXT,
    ok INTEGER,
    stderr TEXT
);

CREATE INDEX IF NOT EXISTS jobs_players_idx ON jobs(a, b);

CREATE TABLE IF NOT EXISTS games (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    source_id TEXT,
    source_url TEXT,
    frame TEXT NOT NULL,
    blue TEXT NOT NULL,
    red TEXT NOT NULL,
    blue_kind TEXT NOT NULL,
    red_kind TEXT NOT NULL,
    job INTEGER,
    pair INTEGER,
    game INTEGER,
    result_winner TEXT,
    result_reason TEXT NOT NULL,
    result_reported TEXT,
    ply_count INTEGER NOT NULL,
    started_at TEXT,
    ended_at TEXT,
    imported_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    record_json TEXT NOT NULL,
    live INTEGER NOT NULL DEFAULT 0,
    UNIQUE(source, source_id)
);

CREATE INDEX IF NOT EXISTS games_source_idx ON games(source);
CREATE INDEX IF NOT EXISTS games_players_idx ON games(blue, red);
CREATE INDEX IF NOT EXISTS games_result_idx ON games(result_winner, result_reason);
CREATE INDEX IF NOT EXISTS games_live_idx ON games(live);
CREATE INDEX IF NOT EXISTS games_job_idx ON games(job);

CREATE TABLE IF NOT EXISTS plies (
    game_id TEXT NOT NULL REFERENCES games(id) ON DELETE CASCADE,
    ply INTEGER NOT NULL,
    position_key TEXT NOT NULL,
    symmetry_key TEXT NOT NULL,
    move_from INTEGER,
    move_to INTEGER,
    PRIMARY KEY (game_id, ply)
);

CREATE INDEX IF NOT EXISTS plies_position_idx ON plies(position_key);
CREATE INDEX IF NOT EXISTS plies_symmetry_idx ON plies(symmetry_key);

CREATE TABLE IF NOT EXISTS analyses (
    position_key TEXT NOT NULL,
    engine TEXT NOT NULL,
    sims INTEGER NOT NULL,
    multipv INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    response_json TEXT NOT NULL,
    PRIMARY KEY (position_key, engine, sims)
);

CREATE INDEX IF NOT EXISTS analyses_position_idx ON analyses(position_key, created_at DESC);

CREATE TABLE IF NOT EXISTS reviews (
    game_id TEXT NOT NULL REFERENCES games(id) ON DELETE CASCADE,
    engine TEXT NOT NULL,
    sims INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    review_json TEXT NOT NULL,
    PRIMARY KEY (game_id, engine)
);

CREATE TABLE IF NOT EXISTS review_jobs (
    id TEXT PRIMARY KEY,
    game_id TEXT NOT NULL,
    engine TEXT NOT NULL,
    sims INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'done', 'failed')),
    progress REAL NOT NULL DEFAULT 0.0,
    detail TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS review_jobs_game_idx ON review_jobs(game_id, created_at DESC);

CREATE TABLE IF NOT EXISTS ratings (
    fitted TEXT NOT NULL,
    player TEXT NOT NULL,
    rating REAL NOT NULL,
    se REAL NOT NULL,
    games INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS ratings_player_idx ON ratings(player, fitted);
