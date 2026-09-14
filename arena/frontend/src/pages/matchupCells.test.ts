import { describe, expect, it } from 'vitest'
import { cellIndex, cellKey, cellScoreLabel, matchupTone } from './matchupCells'

describe('matchup cell colouring', () => {
    it('leaves an unplayed pair uncoloured', () => {
        const tone = matchupTone({ games: 0, score_a: null, expected_a: 0.5 })
        expect(tone.level).toBe('none')
        expect(tone.background).toBe('transparent')
        expect(cellScoreLabel({ games: 0, score_a: null })).toBe('–')
    })

    it('calls a pair even when it matches the model', () => {
        expect(matchupTone({ games: 20, score_a: 0.52, expected_a: 0.5 }).level).toBe('even')
    })

    it('greens over-performance and reds under-performance', () => {
        const ahead = matchupTone({ games: 20, score_a: 0.8, expected_a: 0.5 })
        const behind = matchupTone({ games: 20, score_a: 0.2, expected_a: 0.5 })
        expect(ahead.level).toBe('ahead')
        expect(ahead.background).toContain('var(--accent)')
        expect(behind.level).toBe('behind')
        expect(behind.background).toContain('var(--red)')
    })

    it('saturates the tint at a quarter of a point and never beyond', () => {
        expect(matchupTone({ games: 8, score_a: 0.75, expected_a: 0.5 }).intensity).toBeCloseTo(1)
        expect(matchupTone({ games: 8, score_a: 1, expected_a: 0.5 }).intensity).toBe(1)
        expect(matchupTone({ games: 8, score_a: 0.625, expected_a: 0.5 }).intensity).toBeCloseTo(0.5)
    })

    it('compares with an even split when the model has no probability', () => {
        const tone = matchupTone({ games: 4, score_a: 0.75, expected_a: null })
        expect(tone.level).toBe('ahead')
        expect(tone.delta).toBeCloseTo(0.25)
        expect(cellScoreLabel({ games: 4, score_a: 0.75 })).toBe('75%')
    })
})

describe('matchup index', () => {
    it('mirrors a cell so both rows of the matrix can read it', () => {
        const index = cellIndex([
            { a: 'x', b: 'y', games: 10, wins_a: 6, draws: 2, wins_b: 2, expected_a: 0.6, score_a: 0.7 },
        ])
        const mirrored = index.get(cellKey('y', 'x'))!
        expect(mirrored.wins_a).toBe(2)
        expect(mirrored.wins_b).toBe(6)
        expect(mirrored.score_a).toBeCloseTo(0.3)
        expect(mirrored.expected_a).toBeCloseTo(0.4)
        expect(index.get(cellKey('x', 'y'))!.score_a).toBe(0.7)
    })
})
