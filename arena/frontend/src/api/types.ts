export type Frame = 'meaf' | 'henhen'
export type Side = 'blue' | 'red'
export type PieceType = 'R' | 'P' | 'S'
export type Move = { from: number; to: number; san?: string }
export type ResultReason =
    | 'corner'
    | 'no_pieces'
    | 'no_moves'
    | 'stagnation'
    | 'repetition'
    | 'resign'
    | 'timeout'
    | 'agreed'
    | 'forfeit'
    | 'unknown'
export type Result = {
    winner: Side | null
    reason: ResultReason
    reported?: string
}

export type PositionRef =
    | { game_id: string; ply: number }
    | { fen: string; frame?: Frame }
    /** Board and counters describe the setup; history is played from that setup. */
    | { board: number[]; to_move: Side; psc?: number; ply?: number; history?: Move[] }
export type Budget = 'quick' | 'standard' | 'deep' | { sims: number } | { ms: number }
export type OverlayKind =
    'square_scalar' | 'move_scalar' | 'scalar' | 'histogram' | 'board_forecast' | 'square_vector'
export type Scale = 'sequential' | 'diverging'
export type SquareScalarOverlay = {
    kind: 'square_scalar'
    values: number[]
    range: [number, number]
    scale: Scale
    legend?: string
}
export type MoveScalarOverlay = {
    kind: 'move_scalar'
    moves: Array<{ from: number; to: number; value: number }>
    range: [number, number]
    render: 'arrows' | 'tint'
    scale: Scale
}
export type ScalarOverlay = {
    kind: 'scalar'
    items: Array<{
        label: string
        value: number
        unit?: string
        format?: 'signed' | 'percent' | 'plies' | 'count'
    }>
}
export type HistogramOverlay = {
    kind: 'histogram'
    bins: Array<{ label: string; p: number }>
    expectation?: number
    unit?: string
}
export type BoardForecastOverlay = { kind: 'board_forecast'; cells: Array<{ sq: number; probs: number[] }> }
export type SquareVectorOverlay = {
    kind: 'square_vector'
    source: number
    per_head: number[][]
    labels: string[]
}
export type Overlay =
    | SquareScalarOverlay
    | MoveScalarOverlay
    | ScalarOverlay
    | HistogramOverlay
    | BoardForecastOverlay
    | SquareVectorOverlay
export type ParamSpec = {
    type: 'int' | 'float' | 'square' | 'select' | 'boolean'
    min?: number
    max?: number
    step?: number
    default?: string | number | boolean | null
    options?: Array<string | number>
    optional?: boolean
    label?: string
}
export type HeadSpec = {
    id: string
    label: string
    group: 'policy' | 'value' | 'forecast' | 'input' | 'internals'
    kind: OverlayKind | string
    description: string
    default_on: boolean
    order: number
    params: Record<string, ParamSpec>
}
export type EvalRequest = {
    position: PositionRef
    engine?: string
    budget?: Budget
    multipv?: number
    heads?: string[]
    params?: Record<string, Record<string, unknown>>
}
export type SearchLine = { move: Move; pv: Move[]; q: number; pi: number; visits: number; rank: number }
export type EvalResponse = {
    engine: string
    position_key: string
    symmetry_key: string
    to_move: Side
    legal: Move[]
    value: { blue: number; mover: number; source: 'search' | 'network' }
    search: {
        sims: number
        forwards: number
        ms: number
        lines: SearchLine[]
        played?: { move: Move; q: number; pi: number; visits: number; source: 'search' | 'network' }
    }
    heads: Record<string, Overlay>
    terminal?: Result
}

export type PlayerKind = 'human' | 'bot' | 'engine'
export type PlayerRef = {
    name: string
    id?: string
    rating_before?: number | null
    rating_after?: number | null
    spec?: string
    kind: PlayerKind
}
/** Section 3.1 `info`, re-expressed with absolute moves and values from Blue's point of view. */
export type MoveStats = {
    sims: number
    value: number
    plies_left: number | null
    q: number
    pi: number
    exact_win: boolean
    top: Array<{ move: Move; visits: number; q: number }>
}
export type MoveRecord = {
    from: number
    to: number
    capture: boolean
    san?: string
    emt_ms?: number
    clock_ms?: { blue: number; red: number }
    comment?: string
    nags?: string[]
    stats?: MoveStats | null
}
export type GameSource = 'arena' | 'henhen' | 'meaf' | 'local'
export type GameRecord = {
    id: string | null
    source: GameSource
    source_id: string | null
    source_url: string | null
    frame: Frame
    players: { blue: PlayerRef; red: PlayerRef }
    setup: number[] | null
    moves: MoveRecord[]
    result: { winner: Side | null; reason: string; reported: string }
    time_control: { initial_ms: number; increment_ms: number } | null
    tags: Record<string, string>
    started_at: string | null
    ended_at: string | null
    meta: Record<string, unknown>
    positions?: PositionSummary[]
    review_status?: string
    review_job?: ReviewJob | null
    live?: boolean
    sims?: number
}
export type GameSummary = {
    id: string
    source: GameSource
    source_id: string | null
    frame: Frame
    players: GameRecord['players']
    result: GameRecord['result']
    started_at: string | null
    ply_count: number
    review_status?: string
    engine?: string
    live: boolean
    sims?: number
}
export type PositionSummary = { ply: number; to_move: Side; psc: number; repetition: number; result?: Result }
export type ReviewLabel =
    'best' | 'excellent' | 'good' | 'inaccuracy' | 'mistake' | 'blunder' | 'only' | 'missed_win' | 'book'
export type Review = {
    engine: string
    budget: Budget
    plies: Array<{
        ply: number
        move: Move
        loss: number
        label: ReviewLabel
        best: Move
        lines: SearchLine[]
        value_before: number
        value_after: number
    }>
    accuracy: { blue: number; red: number; by_phase: Record<string, { blue: number; red: number }> }
    key_moments: number[]
    agreement: { blue: number; red: number }
}
export type ReviewJob = { id: string; status: 'queued' | 'running'; progress: number }
export type Job = {
    id: string
    kind: string
    status: 'queued' | 'running' | 'done' | 'failed'
    progress: number
    detail: string
    result_ref: string | null
}
export type EngineKind = 'sq' | 'conv' | 'rpsi'
export type Engine = {
    id: string
    label: string
    run: string
    iter: number
    params: number
    heads: HeadSpec[]
    strongest: boolean
    notes: string
    spec: string
    kind: EngineKind
    rating: number | null
    half_width: number | null
    games: number
    settled: boolean
    broken: boolean
    retired: boolean
    added: string | null
    analysable: boolean
}
export type Meta = {
    name: string
    version: string
    public: boolean
    frames: Frame[]
    default_frame: Frame
    piece_sets: string[]
    target: number
    anchor: string | null
    sims: number
}
export type ImportRequest = {
    source:
        | 'henhen_pgn'
        | 'henhen_id'
        | 'henhen_series'
        | 'meaf_id'
        | 'meaf_line'
        | 'meaf_workshop'
        | 'fen'
        | 'json'
    payload: string
    frame?: Frame
}
export type ImportResponse = { game_ids: string[]; warnings: string[] }
export type ExplorerFilters = {
    line?: string
    symmetry?: boolean
    source?: string
    player?: string
    min_games?: number
}
export type ExplorerResponse = {
    position: PositionRef
    moves: Array<{
        move: Move
        games: number
        wins_blue: number
        draws: number
        wins_red: number
        avg_rating: number | null
        eval?: number
    }>
}
export type InsightsResponse = {
    player: string
    games: number
    wins: number
    draws: number
    losses: number
    score: number
    rating: number | null
    half_width: number | null
    by_color: Record<string, number>
    by_reason: Record<string, number>
    by_source: Record<string, number>
    opponents: Array<{
        name: string
        games: number
        wins: number
        draws: number
        losses: number
        score: number
    }>
    accuracy: number | null
    agreement: number | null
    timeline: Array<{ game_id: string; date: string | null; outcome: string; rating: number | null }>
}
export type ExportFormat = 'henhen_pgn' | 'meaf_line' | 'meaf_workshop_link' | 'henhen_fen' | 'json'

/** Games played against one opponent, from this player's point of view. */
export type HeadToHead = { games: number; wins: number; draws: number; losses: number }

/** One row of the Bayesian Elo table served by /api/ratings and /api/players. */
export type RatedPlayer = {
    name: string
    spec: string
    kind: EngineKind | string
    rating: number | null
    /**
     * The player's own uncertainty: how well its games pin its rating against its
     * opponents' fitted ratings (KataGo-style), not its uncertainty relative to the
     * anchor. The anchor's own row carries the pool's uncertainty about the anchor.
     */
    se: number | null
    half_width: number | null
    games: number
    wins: number
    draws: number
    losses: number
    settled: boolean
    broken: boolean
    retired: boolean
    added: string | null
    /** The previous checkpoint of the same series, or null for an upload without one. */
    parent?: string | null
    /** Head-to-head against the report's anchor, null when the two never met. */
    vs_anchor?: HeadToHead | null
    /** Head-to-head against `parent`, null when there is no parent or no games. */
    vs_parent?: HeadToHead | null
    opponents: Record<string, HeadToHead>
}
export type RatingsResponse = {
    fitted: string | null
    target: number
    anchor: string | null
    players: RatedPlayer[]
}
export type RatingPoint = { fitted: string; rating: number; se: number | null; games: number }
export type RatingsHistory = { players: Record<string, RatingPoint[]> }
export type MatchupCell = {
    a: string
    b: string
    games: number
    wins_a: number
    draws: number
    wins_b: number
    expected_a: number | null
    score_a: number | null
}
export type MatchupsResponse = { players: string[]; cells: MatchupCell[] }
export type MatchupPair = {
    games: GameSummary[]
    wins_a: number
    draws: number
    wins_b: number
    by_colour: Record<string, { games: number; wins_a: number; draws: number; wins_b: number }>
    avg_plies: number | null
    end_reasons: Record<string, number>
}
export type LiveGame = { game_id: string; pair: number; game: number; ply_count: number; live: boolean }
export type LiveJob = {
    id: number | string
    a: string
    b: string
    seed: number
    started: string | null
    games: LiveGame[]
}
export type LiveResponse = { jobs: LiveJob[] }
export type LiveEvent =
    | { type: 'resync' }
    | { type: 'job_start'; job: LiveJob }
    | { type: 'game_start'; job?: number | string; game_id: string; blue?: string; red?: string }
    | { type: 'move'; game_id: string; ply: number; move: Move; stats: MoveStats | null }
    | { type: 'game_end'; game_id: string; winner?: Side | null; reason?: string; ply_count?: number }
    | { type: 'job_end'; job: number | string; ok?: boolean }
    | { type: 'fit'; fitted: string; players?: Array<{ name: string; rating: number; half_width: number }> }
export type SchedulerStatus = {
    jobs_running: number
    slots: number
    target: number
    has_work: boolean
    next_candidate: string | null
}
export type PlayerUpload = { name: string; spec: string; replace: boolean; files: File[] }
