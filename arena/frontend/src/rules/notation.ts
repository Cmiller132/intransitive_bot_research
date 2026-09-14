import type { Frame, Move } from '../api/types.ts'
import { parseSquare, squareName } from './frames.ts'
import { typeOf } from './game.ts'

export type Dialect = 'meaf' | 'henhen'
export function formatMove(
    move: Move,
    dialect: Dialect,
    board?: number[],
    capture = false,
    terminal = false,
    frame: Frame = dialect,
): string {
    const sep = capture ? 'x' : '-'
    const from = squareName(move.from, frame)
    const to = squareName(move.to, frame)
    if (dialect === 'meaf') return `${from}${sep}${to}`.toUpperCase()
    const piece = board ? ['', 'R', 'P', 'S'][typeOf(board[move.from]) || 0] || '' : ''
    return `${piece}${from}${sep}${to}${terminal ? '#' : ''}`
}
export function parseMove(
    text: string,
    dialect: Dialect = 'meaf',
    frame: Frame = dialect,
): Move & { capture: boolean; terminal: boolean } {
    const m = /^(?:[RPS])?([a-i][1-9])([-x])([a-i][1-9])(#)?$/i.exec(text.trim())
    if (!m) throw new Error(`Invalid ${dialect} move: ${text}`)
    return {
        from: parseSquare(m[1], frame),
        to: parseSquare(m[3], frame),
        capture: m[2].toLowerCase() === 'x',
        terminal: !!m[4],
    }
}
