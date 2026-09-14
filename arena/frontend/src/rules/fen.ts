import type { Frame, Side } from '../api/types.ts'
import { fromFrame } from './frames.ts'

const LETTER_TYPE: Record<string, number> = { r: 1, p: 2, s: 3 }

/**
 * Parse a site FEN (nine rows from rank 1, capitals are Blue, second field is
 * the side to move) written in the given display frame into an absolute board.
 */
export function parseFen(fen: string, frame: Frame): { board: number[]; toMove: Side } {
    const fields = fen
        .trim()
        .replace(/^fen\s+/i, '')
        .split(/\s+/)
    const rows = fields[0].split('/')
    if (rows.length !== 9) throw new Error('A position FEN needs nine rows separated by /')
    const board = Array<number>(81).fill(0)
    rows.forEach((row, rank) => {
        let file = 0
        for (const char of row) {
            if (/[1-9]/.test(char)) file += Number(char)
            else {
                const type = LETTER_TYPE[char.toLowerCase()]
                if (!type || file > 8) throw new Error(`Invalid FEN row "${row}"`)
                board[fromFrame(rank * 9 + file, frame)] = char === char.toUpperCase() ? type : type + 3
                file++
            }
        }
        if (file !== 9) throw new Error(`FEN row "${row}" does not cover nine files`)
    })
    const turn = (fields[1] || 'b').toLowerCase()
    if (!['b', 'blue', 'r', 'red', '-'].includes(turn)) throw new Error(`Unknown side to move "${fields[1]}"`)
    return { board, toMove: turn === 'r' || turn === 'red' ? 'red' : 'blue' }
}

/** Write an absolute board as a two-field FEN in the given display frame. */
export function formatFen(board: number[], toMove: Side, frame: Frame): string {
    const rows: string[] = []
    for (let rank = 0; rank < 9; rank++) {
        let row = ''
        let empty = 0
        for (let file = 0; file < 9; file++) {
            const shown = rank * 9 + file
            const piece = board[fromFrame(shown, frame)]
            if (!piece) {
                empty++
                continue
            }
            if (empty) {
                row += empty
                empty = 0
            }
            const letter = 'RPS'[(piece - 1) % 3]
            row += piece <= 3 ? letter : letter.toLowerCase()
        }
        if (empty) row += empty
        rows.push(row)
    }
    return `${rows.join('/')} ${toMove === 'red' ? 'r' : 'b'}`
}
