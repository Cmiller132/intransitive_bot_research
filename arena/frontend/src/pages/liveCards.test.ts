import { describe, expect, it } from 'vitest'
import type { LiveJob, MoveStats } from '../api/types'
import {
    FINISHED_CARD_MS,
    endingText,
    lastMover,
    liveCards,
    liveCounts,
    sideToMove,
    valueFor,
} from './liveCards'

const job = (id: number, games: Array<[string, boolean]>): LiveJob => ({
    id,
    a: 'cand',
    b: 'ref',
    seed: 1,
    started: null,
    games: games.map(([game_id, live], index) => ({
        game_id,
        pair: Math.floor(index / 2),
        game: index,
        ply_count: 10,
        live,
    })),
})

const stats = (value: number): MoveStats => ({
    sims: 100,
    value,
    plies_left: null,
    q: value,
    pi: 0.5,
    exact_win: false,
    top: [],
})

describe('live cards', () => {
    it('keeps running games and recently finished ones', () => {
        const jobs = [
            job(1, [
                ['a', true],
                ['b', false],
                ['c', false],
            ]),
        ]
        const endings = {
            b: { winner: 'blue' as const, reason: 'corner', at: 1_000 },
            c: { winner: null, reason: 'stagnation', at: 1_000 - FINISHED_CARD_MS },
        }
        const cards = liveCards(jobs, endings, 1_000)
        expect(cards.map((card) => card.game.game_id)).toEqual(['a', 'b'])
        expect(cards[1].ending?.winner).toBe('blue')
    })

    it('prefers a streamed ending over a stale live flag', () => {
        const jobs = [job(1, [['a', true]])]
        const cards = liveCards(jobs, { a: { winner: 'red', reason: 'resign', at: 5 } }, 10)
        expect(cards).toHaveLength(1)
        expect(cards[0].ending).toBeDefined()
    })

    it('counts live and finished cards across jobs', () => {
        const jobs = [
            job(1, [['a', true]]),
            job(2, [
                ['b', true],
                ['c', false],
            ]),
        ]
        const cards = liveCards(jobs, { c: { winner: 'red', reason: 'corner', at: 0 } }, 0)
        expect(liveCounts(cards)).toEqual({ live: 2, finished: 1, jobs: 2 })
    })
})

describe('live card readings', () => {
    it('reads the side to move and the last mover from the ply count', () => {
        expect(sideToMove(0)).toBe('blue')
        expect(sideToMove(7)).toBe('red')
        expect(lastMover(0)).toBeNull()
        expect(lastMover(7)).toBe('blue')
        expect(lastMover(8)).toBe('red')
    })

    it("turns Blue's recorded value into the asked side's own view", () => {
        expect(valueFor(stats(0.4), 'blue')).toBeCloseTo(0.4)
        expect(valueFor(stats(0.4), 'red')).toBeCloseTo(-0.4)
        expect(valueFor(null, 'blue')).toBeNull()
        expect(valueFor(undefined, 'red')).toBeNull()
    })

    it('names the winner and the reason', () => {
        expect(endingText({ winner: 'blue', reason: 'no_moves', at: 0 }, 'Alice', 'Bob')).toBe(
            'Alice won · no moves',
        )
        expect(endingText({ winner: 'red', reason: 'corner', at: 0 }, 'Alice', 'Bob')).toBe(
            'Bob won · corner',
        )
        expect(endingText({ winner: null, reason: '', at: 0 }, 'Alice', 'Bob')).toBe('Draw · unknown')
    })
})
