import type { MatchupCell } from '../api/types'

export type CellTone = {
    /** `none` = unplayed, otherwise how A did against the rating model. */
    level: 'none' | 'even' | 'ahead' | 'behind'
    /** score − expected, or null when the pair has no games. */
    delta: number | null
    /** 0…1, saturating at a quarter of a point per game. */
    intensity: number
    background: string
}

/** Below this the pair is called even, so rounding noise does not paint the matrix. */
const EVEN = 0.05
/** A quarter of a point per game is full colour. */
const FULL = 0.25

/**
 * Colour a matchup cell by score against the Bradley-Terry expectation: green
 * when A over-performs, red when it under-performs. With no model probability
 * the plain score is compared with an even split instead.
 */
export function matchupTone(cell: Pick<MatchupCell, 'games' | 'score_a' | 'expected_a'>): CellTone {
    const score = cell.score_a
    if (!cell.games || score === null || score === undefined || !Number.isFinite(score))
        return { level: 'none', delta: null, intensity: 0, background: 'transparent' }
    const expected =
        cell.expected_a === null || cell.expected_a === undefined || !Number.isFinite(cell.expected_a)
            ? 0.5
            : cell.expected_a
    const delta = score - expected
    const intensity = Math.min(1, Math.abs(delta) / FULL)
    if (Math.abs(delta) < EVEN) return { level: 'even', delta, intensity: 0, background: 'transparent' }
    const hue = delta > 0 ? 'var(--accent)' : 'var(--red)'
    const percent = Math.round(12 + intensity * 48)
    return {
        level: delta > 0 ? 'ahead' : 'behind',
        delta,
        intensity,
        background: `color-mix(in srgb, ${hue} ${percent}%, transparent)`,
    }
}

/** `62%` of the available points, or an en dash when the pair never met. */
export function cellScoreLabel(cell: Pick<MatchupCell, 'games' | 'score_a'>): string {
    if (!cell.games || cell.score_a === null || cell.score_a === undefined) return '–'
    return `${Math.round(cell.score_a * 100)}%`
}

/** Player names may contain anything, so the pair key uses a separator they cannot. */
const SEP = '\u0000'
export const cellKey = (a: string, b: string): string => `${a}${SEP}${b}`

/** Index the cell list both ways so the matrix can be read row by row. */
export function cellIndex(cells: MatchupCell[]): Map<string, MatchupCell> {
    const index = new Map<string, MatchupCell>()
    for (const cell of cells) {
        index.set(cellKey(cell.a, cell.b), cell)
        if (!index.has(cellKey(cell.b, cell.a)))
            index.set(cellKey(cell.b, cell.a), {
                ...cell,
                a: cell.b,
                b: cell.a,
                wins_a: cell.wins_b,
                wins_b: cell.wins_a,
                expected_a:
                    cell.expected_a === null || cell.expected_a === undefined ? null : 1 - cell.expected_a,
                score_a: cell.score_a === null || cell.score_a === undefined ? null : 1 - cell.score_a,
            })
    }
    return index
}
