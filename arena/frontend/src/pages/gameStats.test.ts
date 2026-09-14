import { describe, expect, it } from 'vitest'
import type { MoveRecord, MoveStats } from '../api/types'
import { evalSeriesFromMoves, hasMoveStats, topAlternatives } from './gameStats'

const stats = (value: number, top: MoveStats['top'] = []): MoveStats => ({
    sims: 32,
    value,
    plies_left: 40,
    q: value,
    pi: 0.5,
    exact_win: false,
    top,
})
const move = (from: number, to: number, s?: MoveStats): MoveRecord => ({
    from,
    to,
    capture: false,
    ...(s ? { stats: s } : {}),
})

describe('per-move statistics to chart values', () => {
    it('starts at the initial position and follows Blue-view values', () => {
        const moves = [move(1, 2, stats(0.2)), move(70, 60, stats(-0.1)), move(2, 3, stats(0.35))]
        expect(evalSeriesFromMoves(moves)).toEqual([0, 0.2, -0.1, 0.35])
    })

    it('carries the last value through opening plies with no statistics', () => {
        const moves = [move(1, 2), move(70, 60), move(2, 3, stats(0.4)), move(60, 50)]
        expect(evalSeriesFromMoves(moves)).toEqual([0, 0, 0, 0.4, 0.4])
        expect(evalSeriesFromMoves([])).toEqual([0])
    })

    it('detects whether a record carries statistics at all', () => {
        expect(hasMoveStats([move(1, 2), move(3, 4)])).toBe(false)
        expect(hasMoveStats([move(1, 2), move(3, 4, stats(0.1))])).toBe(true)
    })
})

describe('top alternatives', () => {
    it('orders root moves by visits and adds the visit share', () => {
        const alternatives = topAlternatives(
            stats(0.1, [
                { move: { from: 1, to: 2 }, visits: 5, q: 0.05 },
                { move: { from: 3, to: 4 }, visits: 15, q: 0.2 },
            ]),
        )
        expect(alternatives.map((line) => line.move.from)).toEqual([3, 1])
        expect(alternatives[0].share).toBeCloseTo(0.75)
        expect(alternatives[1].share).toBeCloseTo(0.25)
    })

    it('survives a record with no visits', () => {
        const alternatives = topAlternatives(stats(0, [{ move: { from: 1, to: 2 }, visits: 0, q: 0 }]))
        expect(alternatives[0].share).toBe(0)
    })
})
