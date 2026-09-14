import { describe, expect, it } from 'vitest'
import {
    checkpointSeries,
    formatHeadToHead,
    formatRating,
    groupByRun,
    parseCheckpoint,
    sortByRating,
    summariseCounts,
} from './runGrouping'

const name = (item: { name: string }) => item.name

describe('run prefixes', () => {
    it('splits a checkpoint name into run and number', () => {
        expect(parseCheckpoint('conv_g128_240')).toEqual({ run: 'conv_g128', index: 240 })
        expect(parseCheckpoint('sq_g128_8')).toEqual({ run: 'sq_g128', index: 8 })
    })

    it('leaves names that are not checkpoints alone', () => {
        expect(parseCheckpoint('sq_g128')).toBeNull()
        expect(parseCheckpoint('conv_g128_240_b')).toBeNull()
        expect(parseCheckpoint('_5')).toBeNull()
        expect(parseCheckpoint('external engine')).toBeNull()
    })
})

describe('engine sorting and grouping', () => {
    const engines = [
        { name: 'conv_g128_10', rating: 1420 },
        { name: 'conv_g128_20', rating: 1530 },
        { name: 'sq_g128', rating: 1500 },
        { name: 'rpsi_bot', rating: null },
    ]

    it('sorts by rating with unrated engines last', () => {
        expect(sortByRating(engines, name).map(name)).toEqual([
            'conv_g128_20',
            'sq_g128',
            'conv_g128_10',
            'rpsi_bot',
        ])
    })

    it('breaks rating ties alphabetically', () => {
        const tied = [
            { name: 'b', rating: 1500 },
            { name: 'a', rating: 1500 },
        ]
        expect(sortByRating(tied, name).map(name)).toEqual(['a', 'b'])
    })

    it('groups checkpoints under their run, strongest run first', () => {
        const groups = groupByRun(engines, name)
        expect(groups.map((group) => group.run)).toEqual(['conv_g128', 'sq_g128', 'rpsi_bot'])
        expect(groups[0].items.map(name)).toEqual(['conv_g128_20', 'conv_g128_10'])
        expect(groups[0].best).toBe(1530)
        expect(groups[2].best).toBeNull()
    })
})

describe('rating against checkpoint number', () => {
    it('plots one line per run over a shared checkpoint axis', () => {
        const series = checkpointSeries([
            { name: 'conv_g128_10', rating: 1400 },
            { name: 'conv_g128_30', rating: 1560 },
            { name: 'sq_g128_20', rating: 1450 },
            { name: 'sq_g128_30', rating: 1470 },
            { name: 'sq_g128', rating: 1500 },
            { name: 'conv_g64_10', rating: 1300 },
        ])
        expect(series.x).toEqual([10, 20, 30])
        expect(series.runs.map((run) => run.run)).toEqual(['conv_g128', 'sq_g128'])
        expect(series.runs[0].values).toEqual([1400, null, 1560])
        expect(series.runs[1].values).toEqual([null, 1450, 1470])
    })

    it('ignores unrated checkpoints and single-point runs', () => {
        const series = checkpointSeries([
            { name: 'conv_g128_10', rating: 1400 },
            { name: 'conv_g128_20', rating: null },
        ])
        expect(series.runs).toEqual([])
        expect(series.x).toEqual([])
    })
})

describe('rating formatting', () => {
    it('shows the interval only when the fit produced one', () => {
        expect(formatRating(1512.4, 43.2)).toBe('1512 ± 43')
        expect(formatRating(1512.4, null)).toBe('1512')
        expect(formatRating(null, 43)).toBe('—')
    })
})

describe('head-to-head records', () => {
    it('counts a draw as half a point', () => {
        expect(formatHeadToHead({ games: 20, wins: 12, draws: 3, losses: 5 })).toBe('12-3-5 · 67.5%')
        expect(formatHeadToHead({ games: 4, wins: 2, draws: 0, losses: 2 })).toBe('2-0-2 · 50%')
    })

    it('shows an en dash when the pair never met', () => {
        expect(formatHeadToHead(null)).toBe('–')
        expect(formatHeadToHead(undefined)).toBe('–')
        expect(formatHeadToHead({ games: 0, wins: 0, draws: 0, losses: 0 })).toBe('–')
    })
})

describe('the rankings count line', () => {
    it('names the hidden players only when some are hidden', () => {
        expect(summariseCounts(6, 6, 0)).toBe('6 players')
        expect(summariseCounts(5, 6, 1)).toBe('5 shown of 6 players (1 retired)')
        expect(summariseCounts(2, 6, 0)).toBe('2 shown of 6 players')
        expect(summariseCounts(1, 1, 0)).toBe('1 player')
    })
})
