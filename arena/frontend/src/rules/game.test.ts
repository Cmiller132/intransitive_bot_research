import { describe, expect, it } from 'vitest'
import { canCapture, initialBoard, legalMoves, replay } from './game'
import { fromFrame, parseSquare, squareName, toFrame } from './frames'
import { formatMove, parseMove } from './notation'
import { parsePgn } from './pgn'
import fixture from '../../../backend/tests/golden/altfish_game2.pgn?raw'

describe('rules', () => {
    it('has 36 legal opening moves under the authoritative eight-direction rules', () =>
        expect(legalMoves(initialBoard(), 'blue')).toHaveLength(36))
    it('implements the capture cycle', () => {
        expect(canCapture(1, 6)).toBe(true)
        expect(canCapture(3, 5)).toBe(true)
        expect(canCapture(2, 4)).toBe(true)
        expect(canCapture(1, 5)).toBe(false)
    })
    it('mirrors henhen involutively', () => {
        for (let sq = 0; sq < 81; sq++) expect(fromFrame(toFrame(sq, 'henhen'), 'henhen')).toBe(sq)
    })
    it('round trips both notations', () => {
        const move = { from: parseSquare('e3'), to: parseSquare('f4') }
        expect(parseMove(formatMove(move, 'meaf'), 'meaf')).toMatchObject(move)
        expect(squareName(parseMove('Se3-f4', 'henhen').from, 'henhen')).toBe('e3')
    })
    it('parses all golden plies', () => expect(parsePgn(fixture).moves).toHaveLength(209))
    it('counts repeated positions and detects the third occurrence', () => {
        const board = Array<number>(81).fill(0)
        const e5 = parseSquare('e5')
        const f5 = parseSquare('f5')
        const g7 = parseSquare('g7')
        const h7 = parseSquare('h7')
        board[e5] = 1
        board[g7] = 4
        const cycle = [
            { from: e5, to: f5 },
            { from: g7, to: h7 },
            { from: f5, to: e5 },
            { from: h7, to: g7 },
        ]
        const states = replay([...cycle, ...cycle], board, 'blue')
        expect(states[4].repetition).toBe(2)
        expect(states[8].repetition).toBe(3)
        expect(states[8].result).toEqual({ winner: null, reason: 'repetition' })
    })
})
