import { describe, expect, it } from 'vitest'
import { formatFen, parseFen } from './fen'
import { initialBoard } from './game'
import { parseSquare } from './frames'

describe('position FEN', () => {
    it('round trips the initial position in both frames', () => {
        for (const frame of ['meaf', 'henhen'] as const) {
            const fen = formatFen(initialBoard(), 'blue', frame)
            expect(parseFen(fen, frame)).toEqual({ board: initialBoard(), toMove: 'blue' })
        }
    })
    it('mirrors files for the henhen frame', () => {
        const board = Array<number>(81).fill(0)
        board[parseSquare('a1')] = 1
        expect(formatFen(board, 'red', 'henhen')).toBe('8R/9/9/9/9/9/9/9/9 r')
        expect(formatFen(board, 'red', 'meaf')).toBe('R8/9/9/9/9/9/9/9/9 r')
    })
    it('rejects malformed rows', () => {
        expect(() => parseFen('9/9/9 b', 'meaf')).toThrow()
        expect(() => parseFen('X8/9/9/9/9/9/9/9/9 b', 'meaf')).toThrow()
    })
})
