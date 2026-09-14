import type { Move, MoveRecord, PositionRef } from '../api/types'
import type { PositionState } from '../rules/game'

/** Describe a stored ply or a continuation from its original setup. */
export function analysisPosition(
    start: PositionState,
    moves: MoveRecord[],
    ply: number,
    gameId?: string | null,
): PositionRef {
    if (gameId) return { game_id: gameId, ply }
    return {
        board: start.board,
        to_move: start.to_move,
        psc: start.psc,
        ply: start.ply,
        history: moves.slice(0, ply).map(({ from, to }) => ({ from, to })),
    }
}

export type AnalysisVariation = {
    start: number
    moves: MoveRecord[]
}

export function displayedMoves(base: MoveRecord[], variation: AnalysisVariation | null): MoveRecord[] {
    if (!variation) return base
    return [...base.slice(0, variation.start), ...variation.moves]
}

export function extendVariation(
    base: MoveRecord[],
    current: MoveRecord[],
    variation: AnalysisVariation | null,
    ply: number,
    move: Move,
    capture: boolean,
): AnalysisVariation {
    const safePly = Math.max(0, Math.min(ply, current.length))
    const start = variation ? Math.min(variation.start, safePly) : Math.min(safePly, base.length)
    return {
        start,
        moves: [...current.slice(start, safePly), { ...move, capture }],
    }
}

export function initialPlyFromSearch(search: string, moveCount: number, fallback = moveCount): number {
    const raw = new URLSearchParams(search).get('ply')
    const safeFallback = Math.max(0, Math.min(Math.trunc(fallback), moveCount))
    if (raw === null || raw.trim() === '') return safeFallback
    const parsed = Number(raw)
    if (!Number.isFinite(parsed)) return safeFallback
    return Math.max(0, Math.min(Math.trunc(parsed), moveCount))
}
