import type { Move, MoveRecord, MoveStats, Side } from '../api/types'
import { applyMove, detectResult, initialBoard, positionIdentity, type PositionState } from './game'

/**
 * Replay a record into one position per ply. A record that becomes illegal
 * stops there and reports why, so a partial live game still displays.
 */
export function statesFor(
    moves: MoveRecord[],
    setup: number[] | null,
    first: Side,
): { states: PositionState[]; error?: string } {
    const states: PositionState[] = [
        { board: setup ? [...setup] : initialBoard(), to_move: first, ply: 0, psc: 0, repetition: 1 },
    ]
    const repetitions = new Map<string, number>([[positionIdentity(states[0].board, first), 1]])
    for (const [index, move] of moves.entries()) {
        const previous = states.at(-1)!
        try {
            const next = applyMove(previous, move)
            const key = positionIdentity(next.board, next.to_move)
            const repetition = (repetitions.get(key) || 0) + 1
            repetitions.set(key, repetition)
            states.push({
                ...next,
                repetition,
                result: detectResult(next.board, previous.to_move, next.to_move, next.psc, repetition),
            })
        } catch (cause) {
            const detail = cause instanceof Error ? cause.message : String(cause)
            return { states, error: `The record becomes illegal at ply ${index + 1}: ${detail}` }
        }
    }
    return { states }
}

/**
 * Build the record entry for a move that arrived on the live stream. The
 * capture flag comes from the last move touching the destination, or its
 * starting occupancy when no move has touched it.
 */
export function liveMoveRecord(
    moves: MoveRecord[],
    setup: number[] | null,
    move: Move,
    stats: MoveStats | null,
): MoveRecord {
    const last = moves.findLast((played) => played.from === move.to || played.to === move.to)
    const capture = last ? last.to === move.to : Boolean((setup || initialBoard())[move.to])
    return { from: move.from, to: move.to, capture, stats }
}
