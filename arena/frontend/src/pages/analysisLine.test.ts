import { describe, expect, it } from 'vitest'
import type { MoveRecord } from '../api/types'
import { analysisPosition, displayedMoves, extendVariation, initialPlyFromSearch } from './analysisLine'
import { initialState, replay } from '../rules/game'

const base: MoveRecord[] = [
    { from: 12, to: 3, capture: false },
    { from: 68, to: 67, capture: false },
    { from: 3, to: 4, capture: false },
]

describe('analysis variations', () => {
    it('sends the original board once when analysing a manually played move', () => {
        const start = initialState()
        const moves = [{ from: 38, to: 48, capture: false }]
        const ref = analysisPosition(start, moves, 1)
        expect(ref).toEqual({
            board: start.board,
            to_move: 'blue',
            psc: 0,
            ply: 0,
            history: [{ from: 38, to: 48 }],
        })
        if (!('board' in ref)) throw new Error('Expected a setup')
        const final = replay(ref.history!, ref.board, ref.to_move).at(-1)!
        expect(final.board[38]).toBe(0)
        expect(final.board[48]).toBe(3)
        expect(final.to_move).toBe('red')
    })

    it('preserves custom Red setups and sends only moves through the selected ply', () => {
        const board = Array<number>(81).fill(0)
        board[40] = 4
        board[20] = 1
        const start = { ...initialState(), board, to_move: 'red' as const, psc: 19 }
        const moves = [
            { from: 40, to: 41, capture: false },
            { from: 20, to: 21, capture: false },
        ]
        expect(analysisPosition(start, moves, 1)).toMatchObject({
            board,
            to_move: 'red',
            psc: 19,
            history: [{ from: 40, to: 41 }],
        })
        expect(analysisPosition(start, moves, 1, 'stored')).toEqual({ game_id: 'stored', ply: 1 })
    })

    it('advances a free-analysis line instead of leaving the board at ply zero', () => {
        const variation = extendVariation([], [], null, 0, { from: 12, to: 4 }, false)
        expect(displayedMoves([], variation)).toEqual([{ from: 12, to: 4, capture: false }])
    })

    it('branches from the selected game ply and retains subsequent variation moves', () => {
        const first = extendVariation(base, base, null, 1, { from: 67, to: 58 }, false)
        const current = displayedMoves(base, first)
        const second = extendVariation(base, current, first, 2, { from: 3, to: 13 }, true)
        expect(second.start).toBe(1)
        expect(displayedMoves(base, second)).toEqual([
            base[0],
            { from: 67, to: 58, capture: false },
            { from: 3, to: 13, capture: true },
        ])
    })
})

describe('shared ply links', () => {
    it('clamps valid values and rejects non-numeric values', () => {
        expect(initialPlyFromSearch('?ply=2', 4)).toBe(2)
        expect(initialPlyFromSearch('?ply=999', 4)).toBe(4)
        expect(initialPlyFromSearch('?ply=-3', 4)).toBe(0)
        expect(initialPlyFromSearch('?ply=nope', 4)).toBe(4)
        expect(initialPlyFromSearch('', 4, 0)).toBe(0)
        expect(initialPlyFromSearch('?ply=nope', 4, 0)).toBe(0)
    })
})
