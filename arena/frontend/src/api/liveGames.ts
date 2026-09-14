import type { GameRecord, LiveEvent } from './types'
import { liveMoveRecord } from '../rules/replay'

/** Keeps watched records contiguous across stream gaps and concurrent snapshot requests. */
export class LiveGames {
    private records: Record<string, GameRecord> = {}
    private watched = new Set<string>()
    private pending = new Map<string, { again: boolean; controller: AbortController }>()
    private load: (id: string, signal: AbortSignal) => Promise<GameRecord>
    private changed: (records: Record<string, GameRecord>) => void
    private failed: (id: string, error: Error | null) => void

    constructor(
        load: (id: string, signal: AbortSignal) => Promise<GameRecord>,
        changed: (records: Record<string, GameRecord>) => void,
        failed: (id: string, error: Error | null) => void,
    ) {
        this.load = load
        this.changed = changed
        this.failed = failed
    }

    watch(ids: string[]): void {
        this.watched = new Set(ids)
        for (const [id, request] of this.pending) {
            if (this.watched.has(id)) continue
            request.controller.abort()
            this.pending.delete(id)
        }
        this.records = Object.fromEntries(Object.entries(this.records).filter(([id]) => this.watched.has(id)))
        this.changed(this.records)
        for (const id of ids) if (!this.records[id]) this.refresh(id)
    }

    refresh(id: string): void {
        if (!this.watched.has(id)) return
        const running = this.pending.get(id)
        if (running) {
            running.again = true
            return
        }
        const request = { again: false, controller: new AbortController() }
        this.pending.set(id, request)
        void this.load(id, request.controller.signal)
            .then((record) => {
                if (this.pending.get(id) !== request) return
                const current = this.records[id]
                if (
                    !current ||
                    (record.moves.length >= current.moves.length && (current.live || !record.live))
                ) {
                    this.records = { ...this.records, [id]: record }
                    this.changed(this.records)
                }
                this.failed(id, null)
            })
            .catch((cause: unknown) => {
                if (this.pending.get(id) === request)
                    this.failed(id, cause instanceof Error ? cause : new Error(String(cause)))
            })
            .finally(() => {
                if (this.pending.get(id) !== request) return
                this.pending.delete(id)
                if (request.again) this.refresh(id)
            })
    }

    refreshLive(): void {
        for (const id of this.watched) if (!this.records[id] || this.records[id].live) this.refresh(id)
    }

    receive(event: LiveEvent): void {
        if (event.type === 'resync') {
            this.refreshLive()
            return
        }
        if (!('game_id' in event) || !this.watched.has(event.game_id)) return
        const id = event.game_id
        const current = this.records[id]
        if (event.type !== 'move' || !current) {
            this.refresh(id)
            return
        }
        if (!current.live || event.ply < current.moves.length) return
        if (event.ply !== current.moves.length) {
            this.refresh(id)
            return
        }
        const move = liveMoveRecord(current.moves, current.setup, event.move, event.stats)
        this.records = { ...this.records, [id]: { ...current, moves: [...current.moves, move] } }
        this.changed(this.records)
    }

    close(): void {
        this.watched.clear()
        for (const request of this.pending.values()) request.controller.abort()
        this.pending.clear()
    }
}
