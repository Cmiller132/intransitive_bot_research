import { useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import type {
    Budget,
    Engine,
    EvalResponse,
    ExportFormat,
    Frame,
    GameRecord,
    HeadSpec,
    Move,
    MoveRecord,
    ParamSpec,
    PositionRef,
    Review,
} from '../api/types'
import { api } from '../api/client'
import { Board } from '../board/Board'
import { KNOWN_OVERLAYS, OverlayHost } from '../board/overlays/OverlayHost'
import { EnginePicker } from '../components/EnginePicker'
import { EvalChart } from '../components/EvalChart'
import { PlyControls } from '../components/PlyControls'
import { ReviewPanel } from '../components/ReviewPanel'
import { SideName } from '../components/SideName'
import { visibleMoves } from '../board/overlays/MoveScalarLayer'
import { formatSquareValue } from '../board/overlays/SquareScalarLayer'
import { formatFen, parseFen } from '../rules/fen'
import { applyMove, initialBoard, legalMoves, type PositionState } from '../rules/game'
import { statesFor } from '../rules/replay'
import { parseSquare, squareName } from '../rules/frames'
import { formatMove } from '../rules/notation'
import { useArenaStore } from '../state/store'
import {
    DEFAULT_ANALYSIS_BUDGET,
    defaultAnalysisHeads,
    supportedAnalysisHeads,
    valueSourceLabel,
} from './analysisHeads'
import {
    analysisPosition,
    displayedMoves,
    extendVariation,
    initialPlyFromSearch,
    type AnalysisVariation,
} from './analysisLine'
import { evalSeriesFromMoves, hasMoveStats, signed, topAlternatives } from './gameStats'
import { formatRating } from './runGrouping'

const EMPTY_MOVES: MoveRecord[] = []
const EMPTY_HEADS: HeadSpec[] = []
const NO_PARAMS: Record<string, Record<string, unknown>> = {}

function markClass(label: string): string {
    return label === 'inaccuracy' ? 'inacc' : label
}

function formatPv(
    moves: Move[],
    start: PositionState,
    dialect: 'meaf' | 'henhen',
    frame: 'meaf' | 'henhen',
): string {
    let state = start
    return moves
        .map((move) => {
            const capture = Boolean(state.board[move.to])
            const text = formatMove(move, dialect, state.board, capture, false, frame)
            try {
                state = applyMove(state, move)
            } catch {
                /* A truncated PV is still useful to display. */
            }
            return text
        })
        .join(' ')
}

function ParamControl({
    spec,
    value,
    frame,
    onChange,
}: {
    spec: ParamSpec
    value: unknown
    frame: Frame
    onChange: (value: unknown) => void
}) {
    const shown = value ?? spec.default ?? ''
    if (spec.type === 'boolean')
        return (
            <input
                type="checkbox"
                checked={Boolean(shown)}
                onChange={(event) => onChange(event.target.checked)}
            />
        )
    if (spec.type === 'select') {
        const options = spec.options || []
        return (
            <select
                value={String(shown || options[0] || '')}
                onChange={(event) => {
                    const match = options.find((option) => String(option) === event.target.value)
                    onChange(match ?? event.target.value)
                }}
            >
                {options.map((option) => (
                    <option key={String(option)} value={String(option)}>
                        {String(option)}
                    </option>
                ))}
            </select>
        )
    }
    if (spec.type === 'square')
        return (
            <input
                type="text"
                inputMode="text"
                pattern="[a-iA-I][1-9]"
                defaultValue={typeof shown === 'number' ? squareName(shown, frame) : String(shown)}
                placeholder="e.g. e5"
                onBlur={(event) => {
                    const raw = event.target.value.trim()
                    if (!raw && spec.optional) onChange(undefined)
                    else {
                        try {
                            onChange(parseSquare(raw, frame))
                            event.target.setCustomValidity('')
                        } catch {
                            event.target.setCustomValidity('Use a square from a1 to i9')
                            event.target.reportValidity()
                        }
                    }
                }}
            />
        )
    return (
        <input
            type="number"
            min={spec.min}
            max={spec.max}
            step={spec.step ?? (spec.type === 'int' ? 1 : 'any')}
            value={String(shown)}
            onChange={(event) =>
                onChange(event.target.value === '' && spec.optional ? undefined : Number(event.target.value))
            }
        />
    )
}

const BOARD_OVERLAY_KINDS = new Set(['square_scalar', 'move_scalar', 'board_forecast', 'square_vector'])

function WorkspaceActions({
    gameId,
    position,
    frame,
    ply,
    fen,
}: {
    gameId?: string
    position: PositionRef
    frame: Frame
    ply: number
    fen?: string
}) {
    const [format, setFormat] = useState<ExportFormat>(gameId ? 'meaf_line' : 'henhen_fen')
    const [status, setStatus] = useState('')
    const [pending, setPending] = useState(false)

    const download = async () => {
        setPending(true)
        setStatus('')
        try {
            const result = await api.export(
                gameId ? { game_id: gameId, format, frame } : { position, format, frame },
            )
            const url = URL.createObjectURL(new Blob([result.content], { type: result.mime }))
            const link = document.createElement('a')
            link.href = url
            link.download = result.filename || 'intransitive-export.txt'
            document.body.appendChild(link)
            link.click()
            link.remove()
            URL.revokeObjectURL(url)
            setStatus('Export downloaded.')
        } catch (cause) {
            setStatus(`Export failed: ${cause instanceof Error ? cause.message : String(cause)}`)
        } finally {
            setPending(false)
        }
    }

    const share = async () => {
        const url = new URL(window.location.href)
        if (gameId) url.searchParams.set('ply', String(ply))
        else if (fen) {
            url.pathname = '/analysis'
            url.search = ''
            url.searchParams.set('fen', fen)
            url.searchParams.set('frame', frame)
        }
        try {
            await navigator.clipboard.writeText(url.toString())
            setStatus('Position link copied.')
        } catch {
            setStatus('Could not copy the link. Copy it from the address bar.')
        }
    }

    return (
        <div className="workspace-actions">
            <label>
                Export
                <select value={format} onChange={(event) => setFormat(event.target.value as ExportFormat)}>
                    {gameId ? (
                        <>
                            <option value="meaf_line">MEAF line</option>
                            <option value="henhen_pgn">henhen PGN</option>
                            <option value="meaf_workshop_link">Workshop link</option>
                            <option value="json">Arena JSON</option>
                        </>
                    ) : (
                        <>
                            <option value="henhen_fen">Position FEN</option>
                            <option value="json">Position JSON</option>
                        </>
                    )}
                </select>
            </label>
            <button onClick={() => void download()} disabled={pending}>
                {pending ? 'Exporting…' : 'Download'}
            </button>
            <button onClick={() => void share()}>Copy position link</button>
            {status && <span role="status">{status}</span>}
        </div>
    )
}

type EvaluationState = { key: string; result?: EvalResponse; error?: string }

/** One engine's answer for the current position; two of these can run side by side. */
function useEvaluation(
    enabled: boolean,
    positionRef: PositionRef,
    engine: string,
    budget: Budget,
    activeHeads: string[],
    manifest: HeadSpec[],
    params: Record<string, Record<string, unknown>>,
) {
    const heads = useMemo(() => supportedAnalysisHeads(activeHeads, manifest), [activeHeads, manifest])
    const requestKey = useMemo(
        () => JSON.stringify({ position: positionRef, engine, budget, heads, params }),
        [budget, engine, heads, params, positionRef],
    )
    const [state, setState] = useState<EvaluationState>()

    useEffect(() => {
        if (!enabled) return
        const controller = new AbortController()
        const timer = window.setTimeout(() => {
            api.evaluate({ ...JSON.parse(requestKey), multipv: 4 }, controller.signal)
                .then((result) => {
                    if (!controller.signal.aborted) setState({ key: requestKey, result })
                })
                .catch((error: Error) => {
                    if (!controller.signal.aborted) setState({ key: requestKey, error: error.message })
                })
        }, 150)
        return () => {
            window.clearTimeout(timer)
            controller.abort()
        }
    }, [enabled, requestKey])

    const fresh = state?.key === requestKey ? state : undefined
    return { result: fresh?.result, error: fresh?.error, loading: enabled && state?.key !== requestKey }
}

function ResultPanel({
    title,
    engine,
    evaluation,
    loading,
    error,
    position,
    dialect,
    frame,
    onPlay,
}: {
    title: string
    engine?: Engine
    evaluation?: EvalResponse
    loading: boolean
    error?: string
    position: PositionState
    dialect: 'meaf' | 'henhen'
    frame: Frame
    onPlay: (move: Move) => void
}) {
    const value = evaluation?.value.blue
    const quick = evaluation !== undefined && evaluation.search.sims === 0
    return (
        <div className="panel result-panel">
            <div className="panel-head">
                <b>{title}</b>
                <span aria-live="polite">
                    {loading
                        ? 'Analysing…'
                        : evaluation
                          ? quick
                              ? `network only · ${evaluation.search.ms} ms`
                              : `${evaluation.search.sims} sims · ${evaluation.search.ms} ms`
                          : 'No result'}
                </span>
            </div>
            {engine && (
                <p className="evidence">
                    <b>{engine.label}</b>{' '}
                    <span className="mono">{formatRating(engine.rating, engine.half_width)}</span>
                    {engine.notes ? ` · ${engine.notes}` : ''}
                </p>
            )}
            {error && (
                <p className="form-error" role="alert">
                    Analysis unavailable: {error}
                </p>
            )}
            <div className="value">
                <strong>{value === undefined ? '—' : signed(value)}</strong>
                <span>
                    Blue
                    <br />
                    {!evaluation
                        ? loading
                            ? 'analysis pending'
                            : 'unavailable'
                        : valueSourceLabel(evaluation)}
                </span>
            </div>
            {evaluation?.terminal && (
                <p className="terminal-note">
                    Terminal position · {evaluation.terminal.reason.replace('_', ' ')}
                </p>
            )}
            <div className="pv-lines">
                {evaluation?.search.lines.map((searchLine) => (
                    <button key={searchLine.rank} onClick={() => onPlay(searchLine.move)}>
                        <b>
                            {searchLine.rank}.{' '}
                            {formatMove(
                                searchLine.move,
                                dialect,
                                position.board,
                                Boolean(position.board[searchLine.move.to]),
                                false,
                                frame,
                            )}
                        </b>
                        <span>
                            {searchLine.pv.length > 1
                                ? formatPv(searchLine.pv, position, dialect, frame)
                                : ''}
                        </span>
                        <em>
                            {signed(searchLine.q)} · π {searchLine.pi.toFixed(2)}
                            {quick ? '' : ` · N ${searchLine.visits}`}
                        </em>
                    </button>
                ))}
            </div>
            {!loading && evaluation && !evaluation.search.lines.length && (
                <p className="empty-copy compact">No legal continuation from this position.</p>
            )}
        </div>
    )
}

/** Search statistics recorded while the game was played, for one ply. */
function MoveStatsPanel({
    moves,
    states,
    index,
    dialect,
    frame,
}: {
    moves: MoveRecord[]
    states: PositionState[]
    index: number
    dialect: 'meaf' | 'henhen'
    frame: Frame
}) {
    const move = index >= 0 ? moves[index] : undefined
    const before = index >= 0 ? states[index] : undefined
    const stats = move?.stats
    return (
        <div className="panel stats-panel">
            <div className="panel-head">
                <b>Move statistics</b>
                <span>{move ? `ply ${index + 1}` : 'no move'}</span>
            </div>
            {!(stats && before && move) && (
                <p className="empty-copy compact">
                    {move
                        ? 'No search statistics for this ply (opening ply or external engine).'
                        : 'Hover a ply in the ledger.'}
                </p>
            )}
            {stats && before && move && (
                <>
                    <div className="metric-row">
                        <span>Played</span>
                        <b className="mono">
                            {formatMove(move, dialect, before.board, move.capture, false, frame)}
                            {stats.exact_win ? ' ✔' : ''}
                        </b>
                    </div>
                    <div className="metric-row">
                        <span>Blue value</span>
                        <b className="mono">{signed(stats.value)}</b>
                    </div>
                    <div className="metric-row">
                        <span>q · π</span>
                        <b className="mono">
                            {signed(stats.q)} · {stats.pi.toFixed(2)}
                        </b>
                    </div>
                    <div className="metric-row">
                        <span>Simulations</span>
                        <b className="mono">{stats.sims}</b>
                    </div>
                    <div className="metric-row">
                        <span>Plies left</span>
                        <b className="mono">{stats.plies_left == null ? '—' : stats.plies_left.toFixed(1)}</b>
                    </div>
                    {stats.top.length > 0 && (
                        <div className="alternatives">
                            <span className="group-label">root moves by visits</span>
                            {topAlternatives(stats).map((line) => {
                                const same = line.move.from === move.from && line.move.to === move.to
                                return (
                                    <div
                                        className={`metric-row ${same ? 'played' : ''}`}
                                        key={`${line.move.from}-${line.move.to}`}
                                    >
                                        <span className="mono">
                                            {formatMove(
                                                line.move,
                                                dialect,
                                                before.board,
                                                Boolean(before.board[line.move.to]),
                                                false,
                                                frame,
                                            )}
                                        </span>
                                        <b className="mono">
                                            {signed(line.q)} · {Math.round(line.share * 100)}% · N{' '}
                                            {line.visits}
                                        </b>
                                    </div>
                                )
                            })}
                        </div>
                    )}
                </>
            )}
        </div>
    )
}

type Props = {
    game?: GameRecord
    review?: Review
    reviewLoading?: boolean
    reviewError?: string
    onReviewed?: (review: Review) => void
}

export function AnalysisWorkspace({ game, review, reviewLoading = false, reviewError, onReviewed }: Props) {
    const store = useArenaStore()
    const setFrame = store.setFrame
    const baseMoves = game?.moves || EMPTY_MOVES
    const firstMeta = game?.meta.start_to_move ?? game?.meta.first_side
    const [custom, setCustom] = useState<{ board: number[]; first: 'blue' | 'red' } | null>(() => {
        if (game) return null
        const search = new URLSearchParams(window.location.search)
        const fen = search.get('fen')
        if (!fen) return null
        try {
            const frame = search.get('frame') === 'henhen' ? 'henhen' : 'meaf'
            const parsed = parseFen(fen, frame)
            return { board: parsed.board, first: parsed.toMove }
        } catch {
            return null
        }
    })
    const [fenText, setFenText] = useState('')
    const [fenError, setFenError] = useState('')
    const first = game ? (firstMeta === 'red' ? 'red' : 'blue') : custom?.first || 'blue'
    const setup = game?.setup || custom?.board || null
    const [variation, setVariation] = useState<AnalysisVariation | null>(null)
    const moves = useMemo(() => displayedMoves(baseMoves, variation), [baseMoves, variation])
    const replayed = useMemo(() => statesFor(moves, setup, first), [moves, setup, first])
    const states = replayed.states
    const [ply, setPly] = useState(() =>
        initialPlyFromSearch(window.location.search, baseMoves.length, game?.live ? baseMoves.length : 0),
    )
    const position = states[Math.min(ply, states.length - 1)] || states[0]
    const [engineError, setEngineError] = useState<string>()
    const [engines, setEngines] = useState<Engine[]>([])
    const [engine, setEngine] = useState('strongest')
    const [engine2, setEngine2] = useState('')
    const [budget, setBudget] = useState<Budget>(DEFAULT_ANALYSIS_BUDGET)
    const [activeHeads, setActiveHeads] = useState<string[]>([])
    const [params, setParams] = useState<Record<string, Record<string, unknown>>>(NO_PARAMS)
    const [collapsed, setCollapsed] = useState<string[]>([])
    const [hoveredPly, setHoveredPly] = useState<number>()
    const analysable = useMemo(() => engines.filter((item) => item.analysable), [engines])
    const selectedEngine = engines.find((item) => item.id === engine)
    const secondEngine = engines.find((item) => item.id === engine2)
    const manifest = selectedEngine?.heads || EMPTY_HEADS
    const manifest2 = secondEngine?.heads || EMPTY_HEADS

    const positionRef = useMemo(
        () => analysisPosition(states[0], moves, position.ply, variation ? undefined : game?.id),
        [states, moves, position.ply, variation, game?.id],
    )

    const primary = useEvaluation(
        Boolean(manifest.length),
        positionRef,
        engine,
        budget,
        activeHeads,
        manifest,
        params,
    )
    const secondary = useEvaluation(
        Boolean(engine2 && manifest2.length),
        positionRef,
        engine2,
        budget,
        activeHeads,
        manifest2,
        NO_PARAMS,
    )
    const visibleEvaluation = primary.result
    const evaluationError = engineError || primary.error

    useEffect(() => {
        api.engines()
            .then((items) => {
                const search = new URLSearchParams(window.location.search)
                const requested = search.get('engine')
                const pool = items.filter((item) => item.analysable)
                const selected =
                    pool.find((item) => item.id === requested) ||
                    pool.find((item) => item.strongest) ||
                    pool[0]
                const pinned = search.get('engine2')
                setEngines(items)
                setEngine(selected?.id || 'strongest')
                if (pinned && pool.some((item) => item.id === pinned) && pinned !== selected?.id)
                    setEngine2(pinned)
                setActiveHeads(defaultAnalysisHeads(selected?.heads || []))
                setEngineError(undefined)
            })
            .catch((error: Error) => setEngineError(error.message))
    }, [])

    useEffect(() => {
        const onKey = (event: KeyboardEvent) => {
            if ((event.target as HTMLElement).matches('input,textarea,select')) return
            if (event.key === 'ArrowLeft') setPly((current) => Math.max(0, current - 1))
            if (event.key === 'ArrowRight') setPly((current) => Math.min(states.length - 1, current + 1))
            if (event.key === 'ArrowUp') setPly(0)
            if (event.key === 'ArrowDown') setPly(states.length - 1)
            if (event.key.toLowerCase() === 'f') setFrame(store.frame === 'meaf' ? 'henhen' : 'meaf')
            const shortcut = Number(event.key)
            if (shortcut >= 1 && shortcut <= 9 && manifest[shortcut - 1]) toggle(manifest[shortcut - 1])
        }
        window.addEventListener('keydown', onKey)
        return () => window.removeEventListener('keydown', onKey)
    })

    useEffect(() => {
        const frame = game?.frame || new URLSearchParams(window.location.search).get('frame')
        if (frame === 'meaf' || frame === 'henhen') setFrame(frame)
    }, [game?.frame, setFrame])

    // A live game keeps growing: follow the last ply for anyone already standing there.
    const knownPlies = useRef(baseMoves.length)
    const wasLive = useRef(Boolean(game?.live))
    useEffect(() => {
        const previous = knownPlies.current
        const following = wasLive.current || game?.live
        knownPlies.current = baseMoves.length
        wasLive.current = Boolean(game?.live)
        if (following && !variation && baseMoves.length !== previous)
            setPly((current) => (current === previous ? baseMoves.length : current))
    }, [baseMoves.length, game?.live, variation])

    useEffect(() => {
        if (!game?.id) return
        const url = new URL(window.location.href)
        if (!variation && ply === 0) url.searchParams.delete('ply')
        else url.searchParams.set('ply', String(ply))
        window.history.replaceState(null, '', `${url.pathname}${url.search}${url.hash}`)
    }, [baseMoves.length, game?.id, ply, variation])

    function toggle(head: HeadSpec) {
        if (!KNOWN_OVERLAYS.has(head.kind)) return
        setActiveHeads((ids) => {
            if (ids.includes(head.id)) return ids.filter((id) => id !== head.id)
            if (!BOARD_OVERLAY_KINDS.has(head.kind)) return [...ids, head.id]
            const boardIds = new Set(
                manifest.filter((item) => BOARD_OVERLAY_KINDS.has(item.kind)).map((item) => item.id),
            )
            return [...ids.filter((id) => !boardIds.has(id)), head.id]
        })
    }

    function changeEngine(id: string) {
        const next = engines.find((item) => item.id === id)
        setEngine(id)
        setActiveHeads(defaultAnalysisHeads(next?.heads || []))
        setParams(NO_PARAMS)
        if (id === engine2) setEngine2('')
    }

    function play(move: Move) {
        const next = extendVariation(baseMoves, moves, variation, ply, move, Boolean(position.board[move.to]))
        setVariation(next)
        setPly(ply + 1)
    }

    function resetLine() {
        setVariation(null)
        setPly((current) => (game ? Math.min(current, baseMoves.length) : 0))
    }

    function loadFen() {
        try {
            const parsed = parseFen(fenText, store.frame)
            setCustom({ board: parsed.board, first: parsed.toMove })
            setVariation(null)
            setPly(0)
            setFenError('')
        } catch (cause) {
            setFenError(cause instanceof Error ? cause.message : String(cause))
        }
    }

    function clearPosition() {
        setCustom(null)
        setVariation(null)
        setPly(0)
        setFenText('')
        setFenError('')
    }

    const legal =
        visibleEvaluation?.legal || (position.result ? [] : legalMoves(position.board, position.to_move))
    const activeOverlays = Object.entries(visibleEvaluation?.heads || {})
        .filter(([id]) => activeHeads.includes(id))
        .map(([, overlay]) => overlay)
    const boardOverlays = activeOverlays.filter(
        (overlay) => overlay.kind !== 'scalar' && overlay.kind !== 'histogram',
    )
    const panelOverlays = activeOverlays.filter(
        (overlay) => overlay.kind === 'scalar' || overlay.kind === 'histogram',
    )
    const value = visibleEvaluation?.value.blue
    const valueForBar = value ?? 0
    const groups = [...new Set(manifest.map((head) => head.group))]
    const recordedStats = hasMoveStats(baseMoves)
    const chartValues = review
        ? [review.plies[0]?.value_before ?? 0, ...review.plies.map((item) => item.value_after)]
        : recordedStats
          ? evalSeriesFromMoves(moves)
          : undefined
    const chartLabel = review ? 'from the review' : 'recorded at play time'
    const activeBoardHead = manifest.find(
        (head) => activeHeads.includes(head.id) && BOARD_OVERLAY_KINDS.has(head.kind),
    )
    const quick = visibleEvaluation !== undefined && visibleEvaluation.search.sims === 0
    const topLine = visibleEvaluation?.search.lines[0]
    const hasMoveOverlay = boardOverlays.some((overlay) => overlay.kind === 'move_scalar')
    const bestMove = hasMoveOverlay ? undefined : topLine?.move
    const moveLabel = (move: Move) =>
        formatMove(move, store.dialect, position.board, Boolean(position.board[move.to]), false, store.frame)
    const statsIndex = hoveredPly ?? (ply > 0 ? Math.min(ply, moves.length) - 1 : -1)
    const legend = (() => {
        const overlay = boardOverlays[0]
        if (!overlay || !activeBoardHead) {
            return bestMove ? (
                <>
                    <span className="legend-swatch" /> engine's move <b>{moveLabel(bestMove)}</b>
                    {quick ? ' from the network' : ' from search'}
                </>
            ) : null
        }
        if (overlay.kind === 'move_scalar') {
            const shown = visibleMoves(overlay, store.overlayLabels).length
            return overlay.render === 'tint' ? (
                <>
                    <b>{activeBoardHead.label}</b> <span className="legend-ramp diverging" /> square colour =
                    best Q for the side to move landing there · arrow = top move
                </>
            ) : (
                <>
                    <b>{activeBoardHead.label}</b> <span className="legend-swatch" /> favourite ·{' '}
                    <span className="legend-swatch ink" /> next {Math.max(0, shown - 1)} of{' '}
                    {overlay.moves.length} moves · width = probability
                </>
            )
        }
        if (overlay.kind === 'square_scalar') {
            const peak = Math.max(...overlay.values)
            return (
                <>
                    <b>{activeBoardHead.label}</b> <span className={`legend-ramp ${overlay.scale}`} />{' '}
                    {overlay.legend || activeBoardHead.description} · max {formatSquareValue(peak, overlay)}
                </>
            )
        }
        return (
            <>
                <b>{activeBoardHead.label}</b> {activeBoardHead.description}
            </>
        )
    })()

    return (
        <main className="page analysis-page">
            <div className="page-heading">
                <div>
                    <span className="eyebrow">
                        {game
                            ? `${game.source} · ${game.moves.length} plies${game.sims ? ` · ${game.sims} sims` : ''}`
                            : custom
                              ? 'Custom position · live network'
                              : 'Free analysis · live network'}
                    </span>
                    <h1 className={game ? 'game-title' : undefined}>
                        {game ? (
                            <>
                                <SideName
                                    side="blue"
                                    name={game.players.blue.name}
                                    turn={!position.result && position.to_move === 'blue'}
                                />
                                <span className="versus">vs</span>
                                <SideName
                                    side="red"
                                    name={game.players.red.name}
                                    turn={!position.result && position.to_move === 'red'}
                                />
                            </>
                        ) : (
                            'Analysis board'
                        )}
                        {game?.live && <span className="live-badge heading-badge">live</span>}
                    </h1>
                </div>
                <div className="heading-actions">
                    <WorkspaceActions
                        key={variation ? 'variation' : 'main'}
                        gameId={!variation ? game?.id || undefined : undefined}
                        position={positionRef}
                        frame={store.frame}
                        ply={ply}
                        fen={formatFen(position.board, position.to_move, store.frame)}
                    />
                    {game && (
                        <Link
                            className="secondary-action"
                            to={`/analysis?fen=${encodeURIComponent(
                                formatFen(position.board, position.to_move, store.frame),
                            )}&frame=${store.frame}&engine=${encodeURIComponent(engine)}`}
                        >
                            Analyse this position
                        </Link>
                    )}
                    {review && (
                        <div className="accuracy">
                            <span>
                                <b>{review.accuracy.blue.toFixed(1)}</b>{' '}
                                <span className="side-blue">Blue</span> accuracy
                            </span>
                            <span>
                                <b>{review.accuracy.red.toFixed(1)}</b> <span className="side-red">Red</span>{' '}
                                accuracy
                            </span>
                        </div>
                    )}
                    {reviewLoading && (
                        <span className="status-badge" aria-live="polite">
                            Loading review…
                        </span>
                    )}
                    {game && !review && !reviewLoading && reviewError && (
                        <span className="status-badge muted">{reviewError}</span>
                    )}
                    {variation && (
                        <button className="secondary-action" onClick={resetLine}>
                            {game ? 'Return to main line' : 'Reset position'}
                        </button>
                    )}
                </div>
            </div>
            <div className="double-rule" />

            <div className="analysis-grid">
                <section className="board-column">
                    <div className="board-with-eval">
                        <div
                            className="eval-bar"
                            aria-label={
                                value === undefined
                                    ? 'Evaluation loading'
                                    : `Blue evaluation ${value.toFixed(2)}`
                            }
                        >
                            <i style={{ height: `${(valueForBar + 1) * 50}%` }} />
                            <span>{value === undefined ? '…' : signed(value)}</span>
                        </div>
                        <Board
                            key={`${position.to_move}:${position.board.join('')}`}
                            board={position.board}
                            frame={store.frame}
                            pieceSet={store.pieceSet}
                            lastMove={moves[ply - 1]}
                            legal={legal}
                            overlays={boardOverlays}
                            bestMove={bestMove}
                            labels={store.overlayLabels}
                            onMove={play}
                            orientationLabel={`${store.frame} frame`}
                            homeTint
                        />
                    </div>
                    <div className="board-meta">
                        <span>
                            <span className={`side-${position.to_move}`}>
                                {position.to_move === 'blue' ? 'Blue' : 'Red'}
                            </span>{' '}
                            to move · ply <b>{ply}</b>
                            {variation && <em> · variation from {variation.start}</em>}
                        </span>
                        <span>
                            quiet <b>{position.psc} / 200</b> · repetition <b>{position.repetition}×</b>
                        </span>
                    </div>
                    {replayed.error && (
                        <p className="form-error" role="alert">
                            {replayed.error} Later plies are disabled.
                        </p>
                    )}
                    <PlyControls
                        ply={Math.min(ply, states.length - 1)}
                        max={states.length - 1}
                        onChange={setPly}
                    />
                    {!game && (
                        <form
                            className="position-box"
                            onSubmit={(event) => {
                                event.preventDefault()
                                loadFen()
                            }}
                        >
                            <input
                                value={fenText}
                                placeholder={`Paste a ${store.frame} position FEN, e.g. ${formatFen(initialBoard(), 'blue', store.frame)}`}
                                aria-label="Position FEN"
                                onChange={(event) => setFenText(event.target.value)}
                            />
                            <button type="submit" disabled={!fenText.trim()}>
                                Load position
                            </button>
                            {custom && (
                                <button type="button" onClick={clearPosition}>
                                    Start position
                                </button>
                            )}
                            {fenError && (
                                <p className="form-error" role="alert">
                                    {fenError}
                                </p>
                            )}
                        </form>
                    )}
                    {(legend || activeBoardHead) && (
                        <div className="overlay-legend-row" aria-live="polite">
                            <span>{legend}</span>
                            <span className="grow" />
                            <button
                                type="button"
                                className="labels-toggle"
                                aria-pressed={store.overlayLabels}
                                onClick={() => store.setOverlayLabels(!store.overlayLabels)}
                                title="Print probabilities and values on the board"
                            >
                                {store.overlayLabels ? 'labels on' : 'labels off'}
                            </button>
                        </div>
                    )}
                    <div className="head-groups">
                        {groups.map((group) => {
                            const isCollapsed = collapsed.includes(group)
                            return (
                                <div key={group}>
                                    <button
                                        type="button"
                                        className="group-toggle"
                                        aria-expanded={!isCollapsed}
                                        onClick={() =>
                                            setCollapsed((current) =>
                                                current.includes(group)
                                                    ? current.filter((item) => item !== group)
                                                    : [...current, group],
                                            )
                                        }
                                    >
                                        <span className="group-label">{group}</span>
                                        <i aria-hidden="true">{isCollapsed ? '▸' : '▾'}</i>
                                    </button>
                                    {!isCollapsed && (
                                        <div className="chips">
                                            {manifest
                                                .filter((head) => head.group === group)
                                                .map((head) => {
                                                    const known = KNOWN_OVERLAYS.has(head.kind)
                                                    const shortcut = manifest.indexOf(head) + 1
                                                    return (
                                                        <button
                                                            key={head.id}
                                                            className={`chip ${activeHeads.includes(head.id) ? 'on' : ''}`}
                                                            aria-pressed={activeHeads.includes(head.id)}
                                                            disabled={!known}
                                                            onClick={() => toggle(head)}
                                                            title={
                                                                known
                                                                    ? `${head.description}${shortcut <= 9 ? ` · key ${shortcut}` : ''}`
                                                                    : `Unsupported overlay kind: ${head.kind}`
                                                            }
                                                        >
                                                            {head.label}
                                                            <span className="info" aria-hidden="true">
                                                                i
                                                            </span>
                                                        </button>
                                                    )
                                                })}
                                        </div>
                                    )}
                                </div>
                            )
                        })}
                    </div>
                </section>

                <section className="engine-column">
                    <div className="panel engine-controls">
                        <EnginePicker engines={analysable} value={engine} onChange={changeEngine} />
                        <EnginePicker
                            engines={analysable.filter((item) => item.id !== engine)}
                            value={engine2}
                            onChange={setEngine2}
                            label="Compare with"
                            emptyOption={{ value: '', label: 'No second engine' }}
                        />
                        <label>
                            Analysis
                            <select
                                value={String(budget)}
                                onChange={(event) => setBudget(event.target.value as Budget)}
                            >
                                <option value="quick">No search · network</option>
                                <option value="standard">Standard · 128 simulations</option>
                                <option value="deep">Deep · 800 simulations</option>
                            </select>
                        </label>
                        {evaluationError && (
                            <p className="form-error" role="alert">
                                Analysis unavailable: {evaluationError}
                            </p>
                        )}
                    </div>

                    <div className={`result-pair ${engine2 ? 'paired' : ''}`}>
                        <ResultPanel
                            title="Engine result"
                            engine={selectedEngine}
                            evaluation={primary.result}
                            loading={primary.loading}
                            error={primary.error}
                            position={position}
                            dialect={store.dialect}
                            frame={store.frame}
                            onPlay={play}
                        />
                        {engine2 && (
                            <ResultPanel
                                title="Compared engine"
                                engine={secondEngine}
                                evaluation={secondary.result}
                                loading={secondary.loading}
                                error={secondary.error}
                                position={position}
                                dialect={store.dialect}
                                frame={store.frame}
                                onPlay={play}
                            />
                        )}
                    </div>

                    {panelOverlays.length > 0 && (
                        <div className="panel">
                            <div className="panel-head">
                                <b>Network heads</b>
                                <span>live manifest</span>
                            </div>
                            <OverlayHost overlays={panelOverlays} frame={store.frame} mode="panel" />
                        </div>
                    )}
                    {manifest
                        .filter((head) => activeHeads.includes(head.id) && Object.keys(head.params).length)
                        .map((head) => (
                            <div className="panel param-panel" key={head.id}>
                                <div className="panel-head">
                                    <b>{head.label} parameters</b>
                                </div>
                                {Object.entries(head.params).map(([key, param]) => (
                                    <label key={key}>
                                        {param.label || key}
                                        <ParamControl
                                            spec={param}
                                            value={params[head.id]?.[key]}
                                            frame={store.frame}
                                            onChange={(next) =>
                                                setParams((previous) => ({
                                                    ...previous,
                                                    [head.id]: { ...previous[head.id], [key]: next },
                                                }))
                                            }
                                        />
                                    </label>
                                ))}
                            </div>
                        ))}
                </section>

                <aside className="ledger-column">
                    <div className="ledger">
                        <div className="ledger-head">
                            <span>#</span>
                            <span>Blue</span>
                            <span>Red</span>
                        </div>
                        {Array.from({ length: Math.ceil(moves.length / 2) }, (_, row) => (
                            <div className="ledger-row" key={row}>
                                <span>{row + 1}</span>
                                {[row * 2, row * 2 + 1].map((index) => {
                                    const move = moves[index]
                                    const quality =
                                        !variation || index < variation.start
                                            ? review?.plies[index]?.label
                                            : undefined
                                    const stats = move?.stats
                                    const notation =
                                        move && states[index]
                                            ? formatMove(
                                                  move,
                                                  store.dialect,
                                                  states[index].board,
                                                  move.capture,
                                                  index === moves.length - 1 && Boolean(game?.result.winner),
                                                  store.frame,
                                              )
                                            : ''
                                    return (
                                        <button
                                            key={index}
                                            className={ply === index + 1 ? 'current' : ''}
                                            disabled={!move || index >= states.length - 1}
                                            aria-label={
                                                move
                                                    ? `${notation}${quality ? `, ${quality.replace('_', ' ')}` : ''}`
                                                    : undefined
                                            }
                                            title={
                                                stats
                                                    ? `Blue ${signed(stats.value)} · q ${signed(stats.q)} · π ${stats.pi.toFixed(2)}${
                                                          stats.exact_win ? ' · exact win' : ''
                                                      }`
                                                    : undefined
                                            }
                                            onMouseEnter={() => setHoveredPly(index)}
                                            onFocus={() => setHoveredPly(index)}
                                            onMouseLeave={() => setHoveredPly(undefined)}
                                            onBlur={() => setHoveredPly(undefined)}
                                            onClick={() => setPly(index + 1)}
                                        >
                                            {move && (
                                                <>
                                                    {quality && (
                                                        <i
                                                            aria-hidden="true"
                                                            className={`quality ${markClass(quality)}`}
                                                        />
                                                    )}
                                                    <span>{notation}</span>
                                                    {stats?.exact_win && (
                                                        <em className="exact-win" title="Exact win found">
                                                            ✔
                                                        </em>
                                                    )}
                                                </>
                                            )}
                                        </button>
                                    )
                                })}
                            </div>
                        ))}
                        {!moves.length && (
                            <p className="empty-copy compact">
                                Select a piece, then a highlighted square to begin.
                            </p>
                        )}
                    </div>
                    {review && (
                        <div className="quality-key">
                            <i className="quality best" /> best <i className="quality excellent" /> excellent{' '}
                            <i className="quality inacc" /> inaccuracy <i className="quality mistake" />{' '}
                            mistake <i className="quality blunder" /> blunder
                        </div>
                    )}
                    {recordedStats && moves.length > 0 && (
                        <MoveStatsPanel
                            moves={moves}
                            states={states}
                            index={statsIndex}
                            dialect={store.dialect}
                            frame={store.frame}
                        />
                    )}
                    {chartValues && (
                        <div className="panel chart-panel">
                            <div className="panel-head">
                                <b>Evaluation</b>
                                <span>Blue above · {chartLabel}</span>
                            </div>
                            <EvalChart values={chartValues} cursor={Math.min(ply, chartValues.length - 1)} />
                        </div>
                    )}
                    {game?.id && !review && !reviewLoading && onReviewed && (
                        <ReviewPanel
                            gameId={game.id}
                            engine={engine}
                            job={game.review_job}
                            onReviewed={onReviewed}
                        />
                    )}
                    {review && (
                        <div className="panel key-moments">
                            <div className="panel-head">
                                <b>Key moments</b>
                            </div>
                            {review.key_moments.length ? (
                                review.key_moments.map((moment) => (
                                    <button key={moment} onClick={() => setPly(moment + 1)}>
                                        Ply {moment + 1} · {review.plies[moment]?.label?.replace('_', ' ')}
                                    </button>
                                ))
                            ) : (
                                <p className="empty-copy compact">No major swings in this review.</p>
                            )}
                        </div>
                    )}
                </aside>
            </div>

            {game?.moves.some((move) => move.clock_ms) && (
                <section className="wide-section">
                    <div className="double-rule" />
                    <h2>Clock</h2>
                    <EvalChart
                        label="Blue clock"
                        values={game.moves.map(
                            (move) =>
                                ((move.clock_ms?.blue || 0) - (move.clock_ms?.red || 0)) /
                                Math.max(1, game.time_control?.initial_ms || 60_000),
                        )}
                        cursor={ply}
                    />
                </section>
            )}
        </main>
    )
}
