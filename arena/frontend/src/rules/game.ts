import type { Move, Result, Side } from '../api/types.ts'
import { parseSquare } from './frames.ts'

export type PositionState = {
    board: number[]
    to_move: Side
    ply: number
    psc: number
    repetition: number
    result?: Result
}
export const sideOf = (piece: number): Side | null =>
    piece >= 1 && piece <= 3 ? 'blue' : piece >= 4 && piece <= 6 ? 'red' : null
export const typeOf = (piece: number): 1 | 2 | 3 | null =>
    piece ? ((((piece - 1) % 3) + 1) as 1 | 2 | 3) : null
export const initialBoard = (): number[] => {
    const board = Array<number>(81).fill(0)
    const setup: Record<number, string[]> = {
        1: ['b4', 'c3', 'd2'],
        2: ['b5', 'c4', 'd3', 'e2'],
        3: ['c5', 'd4', 'e3'],
    }
    Object.entries(setup).forEach(([piece, squares]) =>
        squares.forEach((name) => {
            const sq = parseSquare(name)
            board[sq] = Number(piece)
            const f = sq % 9
            const r = Math.floor(sq / 9)
            board[(8 - f) * 9 + 8 - r] = Number(piece) + 3
        }),
    )
    return board
}
export const canCapture = (attacker: number, defender: number): boolean => {
    const a = typeOf(attacker)
    const d = typeOf(defender)
    return !!a && !!d && ((a === 1 && d === 3) || (a === 3 && d === 2) || (a === 2 && d === 1))
}
export function legalMoves(board: number[], side: Side): Move[] {
    const moves: Move[] = []
    board.forEach((piece, from) => {
        if (sideOf(piece) !== side) return
        const f = from % 9
        const r = Math.floor(from / 9)
        for (let dr = -1; dr <= 1; dr++)
            for (let df = -1; df <= 1; df++) {
                if (!df && !dr) continue
                const nf = f + df
                const nr = r + dr
                if (nf < 0 || nf > 8 || nr < 0 || nr > 8) continue
                const to = nr * 9 + nf
                const target = board[to]
                if (!target || (sideOf(target) !== side && canCapture(piece, target)))
                    moves.push({ from, to })
            }
    })
    return moves
}
export function applyMove(state: PositionState, move: Move): PositionState {
    if (!legalMoves(state.board, state.to_move).some((m) => m.from === move.from && m.to === move.to))
        throw new Error(`Illegal move ${move.from}-${move.to}`)
    const board = [...state.board]
    const capture = board[move.to] !== 0
    board[move.to] = board[move.from]
    board[move.from] = 0
    const to_move: Side = state.to_move === 'blue' ? 'red' : 'blue'
    const psc = capture ? 0 : state.psc + 1
    return {
        board,
        to_move,
        ply: state.ply + 1,
        psc,
        repetition: 1,
        result: detectResult(board, state.to_move, to_move, psc, 1),
    }
}
export function detectResult(
    board: number[],
    mover: Side,
    toMove: Side,
    psc: number,
    repetition: number,
): Result | undefined {
    const goal = mover === 'blue' ? 80 : 0
    if (sideOf(board[goal]) === mover) return { winner: mover, reason: 'corner' }
    if (!board.some((p) => sideOf(p) === toMove)) return { winner: mover, reason: 'no_pieces' }
    if (!legalMoves(board, toMove).length) return { winner: mover, reason: 'no_moves' }
    if (psc >= 200) return { winner: null, reason: 'stagnation' }
    if (repetition >= 3) return { winner: null, reason: 'repetition' }
    return undefined
}
export const initialState = (): PositionState => ({
    board: initialBoard(),
    to_move: 'blue',
    ply: 0,
    psc: 0,
    repetition: 1,
})
export const positionIdentity = (board: number[], toMove: Side): string => `${toMove}:${board.join(',')}`
export function replay(moves: Move[], setup = initialBoard(), first: Side = 'blue'): PositionState[] {
    const states: PositionState[] = [{ board: [...setup], to_move: first, ply: 0, psc: 0, repetition: 1 }]
    const repetitions = new Map<string, number>([[positionIdentity(states[0].board, first), 1]])
    for (const move of moves) {
        const previous = states.at(-1)!
        const next = applyMove(previous, move)
        const key = positionIdentity(next.board, next.to_move)
        const repetition = (repetitions.get(key) || 0) + 1
        repetitions.set(key, repetition)
        states.push({
            ...next,
            repetition,
            result: detectResult(next.board, previous.to_move, next.to_move, next.psc, repetition),
        })
    }
    return states
}
