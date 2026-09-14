import type { LiveGame, LiveJob, MoveStats, Side } from '../api/types'

/** How long a game that has just finished keeps its card in the live grid. */
export const FINISHED_CARD_MS = 30_000

/** The end of a game as it arrived on the live stream. */
export type LiveEnding = { winner: Side | null; reason: string; at: number }

/** One card in the live grid: a game, the job playing it, its end if it has one. */
export type LiveCard = { job: LiveJob; game: LiveGame; ending?: LiveEnding }

/**
 * The games worth a card: every running game plus the ones that ended in the
 * last `FINISHED_CARD_MS`, so a finished game holds its slot with its result
 * instead of vanishing while its job plays on. Job and game order is the
 * arena's own, which keeps a card in place while its neighbours come and go.
 */
export function liveCards(jobs: LiveJob[], endings: Record<string, LiveEnding>, now: number): LiveCard[] {
    const cards: LiveCard[] = []
    for (const job of jobs) {
        for (const game of job.games) {
            const ending = endings[game.game_id]
            if (ending) {
                if (now - ending.at < FINISHED_CARD_MS) cards.push({ job, game, ending })
            } else if (game.live) cards.push({ job, game })
        }
    }
    return cards
}

/** Blue opens every game, so an even number of plies leaves Blue to move. */
export function sideToMove(plies: number): Side {
    return plies % 2 === 0 ? 'blue' : 'red'
}

/** The side that played the last ply, or null before the first move. */
export function lastMover(plies: number): Side | null {
    return plies > 0 ? sideToMove(plies - 1) : null
}

/**
 * A root value from one side's own point of view. `stats.value` is recorded
 * from Blue's view (section 6), so Red's own view is its negation.
 */
export function valueFor(stats: MoveStats | null | undefined, side: Side): number | null {
    if (!stats || !Number.isFinite(stats.value)) return null
    return side === 'blue' ? stats.value : -stats.value
}

/** A finished game's one-line result: who won, and how. */
export function endingText(ending: LiveEnding, blue: string, red: string): string {
    const reason = (ending.reason || 'unknown').replace(/_/g, ' ')
    if (!ending.winner) return `Draw · ${reason}`
    return `${ending.winner === 'blue' ? blue : red} won · ${reason}`
}

/** The live/finished split of a set of cards, for the grid's caption. */
export function liveCounts(cards: LiveCard[]): { live: number; finished: number; jobs: number } {
    const jobs = new Set<string>()
    let live = 0
    for (const card of cards) {
        jobs.add(String(card.job.id))
        if (!card.ending) live += 1
    }
    return { live, finished: cards.length - live, jobs: jobs.size }
}
