import { describe, expect, it, vi } from 'vitest'
import { LiveGames } from './liveGames'
import type { GameRecord, LiveEvent } from './types'
import { liveMoveRecord, statesFor } from '../rules/replay'
import { parsePgn } from '../rules/pgn'
import fixture from '../../../backend/tests/golden/altfish_game2.pgn?raw'

const moves = [
    { from: 22, to: 23, capture: false },
    { from: 58, to: 57, capture: false },
    { from: 23, to: 24, capture: false },
]
const record = (plies = 0, live = true): GameRecord => ({
    id: 'game',
    source: 'arena',
    source_id: null,
    source_url: null,
    frame: 'meaf',
    players: { blue: { name: 'Blue', kind: 'engine' }, red: { name: 'Red', kind: 'engine' } },
    setup: null,
    moves: moves.slice(0, plies),
    result: { winner: null, reason: 'unknown', reported: '' },
    time_control: null,
    tags: {},
    started_at: null,
    ended_at: null,
    meta: {},
    live,
})
const event = (ply: number): LiveEvent => ({
    type: 'move',
    game_id: 'game',
    ply,
    move: moves[ply],
    stats: null,
})

function harness() {
    let records: Record<string, GameRecord> = {}
    const requests: Array<{
        resolve: (record: GameRecord) => void
        reject: (error: Error) => void
        signal: AbortSignal
    }> = []
    const load = vi.fn(
        (_id: string, signal: AbortSignal) =>
            new Promise<GameRecord>((resolve, reject) => {
                requests.push({ resolve, reject, signal })
            }),
    )
    const failed = vi.fn()
    const games = new LiveGames(
        load,
        (value) => {
            records = value
        },
        failed,
    )
    games.watch(['game'])
    return { games, requests, load, failed, current: () => records.game }
}
const flush = async () => {
    await new Promise<void>((resolve) => setTimeout(resolve, 0))
}

describe('live record synchronization', () => {
    it('retains capture notation without replaying the game for each streamed move', () => {
        const complete = parsePgn(fixture)
        const first = complete.meta.first_side === 'red' ? 'red' : 'blue'
        const { states, error } = statesFor(complete.moves, complete.setup, first)
        expect(error).toBeUndefined()
        for (const [index, move] of complete.moves.entries()) {
            expect(liveMoveRecord(complete.moves.slice(0, index), complete.setup, move, null).capture).toBe(
                Boolean(states[index].board[move.to]),
            )
        }
    })

    it('recovers a late-game gap and renders every ply of a complete recorded game', async () => {
        const h = harness()
        const complete = { ...parsePgn(fixture), id: 'game', live: true }
        h.requests[0].resolve({ ...complete, moves: complete.moves.slice(0, 171) })
        await flush()
        h.games.receive({ type: 'move', game_id: 'game', ply: 172, move: complete.moves[172], stats: null })
        expect(h.current().moves).toHaveLength(171)
        h.requests[1].resolve({ ...complete, live: false })
        await flush()
        const first = complete.meta.first_side === 'red' ? 'red' : 'blue'
        const replayed = statesFor(h.current().moves, complete.setup, first)
        expect(replayed.error).toBeUndefined()
        expect(replayed.states).toHaveLength(210)
    })

    it('recovers a missing ply without ever appending a move across the gap', async () => {
        const h = harness()
        h.requests[0].resolve(record())
        await flush()
        h.games.receive(event(0))
        h.games.receive(event(2))
        expect(h.current().moves).toHaveLength(1)
        expect(h.load).toHaveBeenCalledTimes(2)
        h.requests[1].resolve(record(3))
        await flush()
        expect(h.current().moves).toHaveLength(3)
        expect(statesFor(h.current().moves, null, 'blue').error).toBeUndefined()
    })

    it('ignores duplicate moves and stale snapshots arriving after newer stream moves', async () => {
        const h = harness()
        h.requests[0].resolve(record())
        await flush()
        h.games.refreshLive()
        h.games.receive(event(0))
        h.games.receive(event(0))
        h.requests[1].resolve(record())
        await flush()
        expect(h.current().moves).toHaveLength(1)
    })

    it('refetches when events race the initial snapshot and coalesces simultaneous requests', async () => {
        const h = harness()
        h.games.receive(event(0))
        h.games.receive(event(1))
        h.games.receive({ type: 'resync' })
        expect(h.load).toHaveBeenCalledTimes(1)
        h.requests[0].resolve(record())
        await flush()
        expect(h.load).toHaveBeenCalledTimes(2)
        h.requests[1].resolve(record(2))
        await flush()
        expect(h.current().moves).toHaveLength(2)
    })

    it('fetches the final position when game_end races an older in-flight snapshot', async () => {
        const h = harness()
        h.requests[0].resolve(record(1))
        await flush()
        h.games.refreshLive()
        h.games.receive({ type: 'game_end', game_id: 'game' })
        h.requests[1].resolve(record(2))
        await flush()
        h.requests[2].resolve(record(3, false))
        await flush()
        h.games.receive(event(2))
        h.games.refreshLive()
        expect(h.current().live).toBe(false)
        expect(h.current().moves).toHaveLength(3)
        expect(h.load).toHaveBeenCalledTimes(3)
    })

    it('retries failed loads on reconciliation and clears their error', async () => {
        const h = harness()
        h.requests[0].reject(new Error('offline'))
        await flush()
        expect(h.failed).toHaveBeenLastCalledWith('game', new Error('offline'))
        h.games.refreshLive()
        h.requests[1].resolve(record(2))
        await flush()
        expect(h.current().moves).toHaveLength(2)
        expect(h.failed).toHaveBeenLastCalledWith('game', null)
    })

    it('discards responses for removed games and aborted subscriptions', async () => {
        const h = harness()
        h.games.watch([])
        expect(h.requests[0].signal.aborted).toBe(true)
        h.requests[0].resolve(record(2))
        await flush()
        expect(h.current()).toBeUndefined()
        h.games.watch(['game'])
        h.games.close()
        expect(h.requests[1].signal.aborted).toBe(true)
        h.requests[1].resolve(record(3))
        await flush()
        expect(h.current()).toBeUndefined()
    })
})
