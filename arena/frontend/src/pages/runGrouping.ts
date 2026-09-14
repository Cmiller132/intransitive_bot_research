/** Run/checkpoint naming shared by the rankings chart and every engine picker. */

export type Checkpoint = { run: string; index: number }

/** `conv_g128_240` is checkpoint 240 of run `conv_g128`; anything else stands alone. */
export function parseCheckpoint(name: string): Checkpoint | null {
    const match = /^(.*[^_])_(\d+)$/.exec(name.trim())
    if (!match) return null
    return { run: match[1], index: Number(match[2]) }
}

/** The run a player belongs to: its checkpoint prefix, or the player itself. */
export function runOf(name: string): string {
    return parseCheckpoint(name)?.run ?? name.trim()
}

type Rated = { rating?: number | null }
const ratingOf = (item: Rated): number | null =>
    item.rating === null || item.rating === undefined || !Number.isFinite(item.rating) ? null : item.rating

/** Strongest first; unrated players last, alphabetically. */
export function sortByRating<T extends Rated>(items: T[], nameOf: (item: T) => string): T[] {
    return [...items].sort((a, b) => {
        const ra = ratingOf(a)
        const rb = ratingOf(b)
        if (ra === null && rb === null) return nameOf(a).localeCompare(nameOf(b))
        if (ra === null) return 1
        if (rb === null) return -1
        if (rb !== ra) return rb - ra
        return nameOf(a).localeCompare(nameOf(b))
    })
}

export type RunGroup<T> = { run: string; best: number | null; items: T[] }

/** Group players by run, strongest run first, strongest member first inside each run. */
export function groupByRun<T extends Rated>(items: T[], nameOf: (item: T) => string): Array<RunGroup<T>> {
    const groups = new Map<string, T[]>()
    for (const item of items) {
        const run = runOf(nameOf(item))
        const bucket = groups.get(run)
        if (bucket) bucket.push(item)
        else groups.set(run, [item])
    }
    return [...groups.entries()]
        .map(([run, members]) => {
            const sorted = sortByRating(members, nameOf)
            return { run, best: ratingOf(sorted[0] ?? {}), items: sorted }
        })
        .sort((a, b) => {
            if (a.best === null && b.best === null) return a.run.localeCompare(b.run)
            if (a.best === null) return 1
            if (b.best === null) return -1
            if (b.best !== a.best) return b.best - a.best
            return a.run.localeCompare(b.run)
        })
}

export type CheckpointSeries = {
    x: number[]
    runs: Array<{ run: string; values: Array<number | null> }>
}

/**
 * Rating against checkpoint number, for every run that has at least two rated
 * checkpoints. Runs are ordered by their best rating; gaps are null so uPlot
 * draws one line per run over a shared x axis.
 */
export function checkpointSeries(
    players: Array<{ name: string; rating?: number | null }>,
    minPoints = 2,
): CheckpointSeries {
    const runs = new Map<string, Map<number, number>>()
    for (const player of players) {
        const checkpoint = parseCheckpoint(player.name)
        const rating = ratingOf(player)
        if (!checkpoint || rating === null) continue
        const bucket = runs.get(checkpoint.run) ?? new Map<number, number>()
        // A duplicate index keeps the strongest rating rather than the last row.
        const previous = bucket.get(checkpoint.index)
        if (previous === undefined || rating > previous) bucket.set(checkpoint.index, rating)
        runs.set(checkpoint.run, bucket)
    }
    const kept = [...runs.entries()].filter(([, points]) => points.size >= minPoints)
    const x = [...new Set(kept.flatMap(([, points]) => [...points.keys()]))].sort((a, b) => a - b)
    const series = kept
        .map(([run, points]) => ({
            run,
            values: x.map((index) => points.get(index) ?? null),
            best: Math.max(...points.values()),
        }))
        .sort((a, b) => (b.best !== a.best ? b.best - a.best : a.run.localeCompare(b.run)))
        .map(({ run, values }) => ({ run, values }))
    return { x, runs: series }
}

/** `1512 ± 43`, or an em dash while a player is still unrated. */
export function formatRating(rating: number | null | undefined, halfWidth?: number | null): string {
    if (rating === null || rating === undefined || !Number.isFinite(rating)) return '—'
    const value = Math.round(rating).toString()
    if (halfWidth === null || halfWidth === undefined || !Number.isFinite(halfWidth)) return value
    return `${value} ± ${Math.round(halfWidth)}`
}

type Record4 = { games: number; wins: number; draws: number; losses: number }

/** The en dash standing in for a head-to-head that was never played. */
export const NO_RECORD = '–'

/**
 * `12-3-5 · 61.5%` for a head-to-head, where the score counts a draw as a half
 * point. An en dash when the pair never met.
 */
export function formatHeadToHead(record: Record4 | null | undefined): string {
    if (!record || !record.games) return NO_RECORD
    const score = ((record.wins + record.draws / 2) / record.games) * 100
    const percent = Number.isInteger(score) ? score.toFixed(0) : score.toFixed(1)
    return `${record.wins}-${record.draws}-${record.losses} · ${percent}%`
}

/**
 * `6 players` when every player is on screen, `5 shown of 6 players (1 retired)`
 * when the filters hide some of them.
 */
export function summariseCounts(shown: number, total: number, hiddenRetired = 0): string {
    const players = `${total} player${total === 1 ? '' : 's'}`
    if (shown >= total) return players
    const retired = hiddenRetired > 0 ? ` (${hiddenRetired} retired)` : ''
    return `${shown} shown of ${players}${retired}`
}
