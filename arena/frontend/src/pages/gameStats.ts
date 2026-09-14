import type { MoveRecord, MoveStats } from '../api/types'

/** True when the record carries search statistics recorded at play time. */
export function hasMoveStats(moves: MoveRecord[]): boolean {
    return moves.some((move) => move.stats)
}

/**
 * Blue-view evaluation per position: one point for the start plus one after
 * every move. `stats.value` is already Blue's view (section 6), so a ply
 * without statistics — an opening ply, or an external engine — holds the last
 * known value instead of dropping to zero.
 */
export function evalSeriesFromMoves(moves: MoveRecord[]): number[] {
    const series = [0]
    let last = 0
    for (const move of moves) {
        if (move.stats && Number.isFinite(move.stats.value)) last = move.stats.value
        series.push(last)
    }
    return series
}

export type TopAlternative = { move: { from: number; to: number }; visits: number; q: number; share: number }

/** Root moves by visits with their visit share, played move included. */
export function topAlternatives(stats: MoveStats, limit = 5): TopAlternative[] {
    const total = stats.top.reduce((sum, line) => sum + Math.max(0, line.visits), 0)
    return [...stats.top]
        .sort((a, b) => b.visits - a.visits)
        .slice(0, limit)
        .map((line) => ({ ...line, share: total > 0 ? line.visits / total : 0 }))
}

export const signed = (value: number, digits = 2): string =>
    `${value >= 0 ? '+' : ''}${value.toFixed(digits)}`
