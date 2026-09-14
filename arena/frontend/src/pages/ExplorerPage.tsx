import { useCallback, useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import { api } from '../api/client'
import type { ExplorerFilters, ExplorerResponse } from '../api/types'
import { Board } from '../board/Board'
import { formatMove } from '../rules/notation'
import { useArenaStore } from '../state/store'

export function ExplorerPage() {
    const { frame, dialect, pieceSet } = useArenaStore()
    const [line, setLine] = useState('')
    const [source, setSource] = useState('')
    const [player, setPlayer] = useState('')
    const [minGames, setMinGames] = useState(1)
    const [symmetry, setSymmetry] = useState(true)
    const [data, setData] = useState<ExplorerResponse>()
    const [error, setError] = useState('')
    const [loading, setLoading] = useState(true)
    const [loadedLine, setLoadedLine] = useState('')
    const requestSequence = useRef(0)

    const load = useCallback((value: string, filters: Omit<ExplorerFilters, 'line'>) => {
        const request = ++requestSequence.current
        setLoading(true)
        setError('')
        return api
            .explorer({ line: value, ...filters })
            .then((response) => {
                if (request !== requestSequence.current) return
                setData(response)
                setLoadedLine(value)
                setLoading(false)
            })
            .catch((cause: Error) => {
                if (request !== requestSequence.current) return
                setData(undefined)
                setError(cause.message)
                setLoading(false)
            })
    }, [])

    // Filters re-query the current line; the line itself is submitted explicitly.
    useEffect(() => {
        void load(loadedLine, { source, player, min_games: minGames, symmetry })
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [source, player, minGames, symmetry])

    const submit = (event: FormEvent) => {
        event.preventDefault()
        void load(line.trim(), { source, player, min_games: minGames, symmetry })
    }
    const reset = () => {
        setLine('')
        void load('', { source, player, min_games: minGames, symmetry })
    }
    const position = data?.position && 'board' in data.position ? data.position : undefined

    const follow = (from: number, to: number) => {
        if (!position) return
        const capture = Boolean(position.board[to])
        const token = formatMove({ from, to }, 'meaf', position.board, capture, false, 'meaf')
        const next = `${loadedLine.trim()} ${token}`.trim()
        setLine(next)
        void load(next, { source, player, min_games: minGames, symmetry })
    }

    return (
        <main className="page">
            <span className="eyebrow">Symmetry-aware corpus</span>
            <h1>Opening explorer</h1>
            <p className="lede">
                Follow a move line to see how the arena and the imported games continued from equivalent
                positions.
            </p>
            <div className="double-rule" />
            <form className="explorer-form" onSubmit={submit}>
                <label>
                    MEAF move line
                    <input
                        value={line}
                        onChange={(event) => setLine(event.target.value)}
                        placeholder="E3-F4 E7-D6"
                    />
                </label>
                <button type="submit" disabled={loading}>
                    Explore
                </button>
                <button type="button" onClick={reset} disabled={loading || !line}>
                    Reset
                </button>
            </form>
            <div className="filters">
                <label>
                    Source
                    <select value={source} onChange={(event) => setSource(event.target.value)}>
                        <option value="">All sources</option>
                        <option value="arena">Arena</option>
                        <option value="henhen">henhen</option>
                        <option value="meaf">meaf</option>
                        <option value="local">local</option>
                    </select>
                </label>
                <label>
                    Player
                    <input
                        value={player}
                        placeholder="Any player"
                        onChange={(event) => setPlayer(event.target.value)}
                    />
                </label>
                <label>
                    Minimum games
                    <input
                        type="number"
                        min={1}
                        max={999}
                        value={minGames}
                        onChange={(event) => setMinGames(Math.max(1, Number(event.target.value) || 1))}
                    />
                </label>
                <label className="checkbox">
                    <input
                        type="checkbox"
                        checked={symmetry}
                        onChange={(event) => setSymmetry(event.target.checked)}
                    />
                    Fold symmetric positions
                </label>
            </div>
            {error && (
                <p className="form-error" role="alert">
                    Could not explore that line: {error}
                </p>
            )}
            {loading && !data && (
                <div className="loading">
                    <span className="cycle-loader">RSP</span> Loading continuations…
                </div>
            )}
            {position && (
                <div className="explorer-layout" aria-busy={loading}>
                    <section>
                        <Board
                            board={position.board}
                            frame={frame}
                            pieceSet={pieceSet}
                            overlays={[]}
                            orientationLabel={`${frame} frame`}
                            homeTint
                        />
                    </section>
                    <section className="panel">
                        <div className="panel-head">
                            <b>Next moves</b>
                            <span>
                                {loading
                                    ? 'Updating…'
                                    : `${data?.moves.reduce((sum, item) => sum + item.games, 0) || 0} continuations`}
                            </span>
                        </div>
                        <div className="explorer-table">
                            <div className="explorer-row explorer-head">
                                <span>Move</span>
                                <span>Games</span>
                                <span>Blue / Draw / Red</span>
                                <span>Blue score</span>
                                <span>Avg rating</span>
                            </div>
                            {data?.moves.map((item) => {
                                const total = Math.max(1, item.games)
                                const score = (item.wins_blue + item.draws * 0.5) / total
                                const capture = Boolean(position.board[item.move.to])
                                return (
                                    <button
                                        className="explorer-row"
                                        key={`${item.move.from}-${item.move.to}`}
                                        onClick={() => follow(item.move.from, item.move.to)}
                                        disabled={loading}
                                    >
                                        <b>
                                            {formatMove(
                                                item.move,
                                                dialect,
                                                position.board,
                                                capture,
                                                false,
                                                frame,
                                            )}
                                        </b>
                                        <span>{item.games}</span>
                                        <span>
                                            {item.wins_blue} / {item.draws} / {item.wins_red}
                                        </span>
                                        <span>{Math.round(score * 100)}%</span>
                                        <span className="mono">
                                            {item.avg_rating == null ? '—' : Math.round(item.avg_rating)}
                                        </span>
                                    </button>
                                )
                            })}
                            {!data?.moves.length && !loading && (
                                <p className="empty-copy">
                                    No recorded game reaches this position with these filters.
                                </p>
                            )}
                        </div>
                    </section>
                </div>
            )}
        </main>
    )
}
