import { useEffect, useRef, useState } from 'react'
import { api } from './client'
import type { GameRecord, LiveEvent, SchedulerStatus } from './types'
import { LiveGames } from './liveGames'

/** Scheduler status, refreshed on an interval; undefined while the arena is unreachable. */
export function useSchedulerStatus(intervalMs = 10_000): SchedulerStatus | undefined {
    const [status, setStatus] = useState<SchedulerStatus>()
    useEffect(() => {
        let active = true
        const check = () => {
            api.scheduler()
                .then((value) => {
                    if (active) setStatus(value)
                })
                .catch(() => {
                    if (active) setStatus(undefined)
                })
        }
        check()
        const timer = window.setInterval(check, intervalMs)
        return () => {
            active = false
            window.clearInterval(timer)
        }
    }, [intervalMs])
    return status
}

/**
 * Subscribe to the arena's live stream. The handler is kept in a ref so a
 * component may re-render freely without dropping the connection.
 */
export function useLiveEvents(
    onEvent: (event: LiveEvent) => void,
    enabled = true,
    onSync?: () => void,
): void {
    const handler = useRef(onEvent)
    const sync = useRef(onSync)
    useEffect(() => {
        handler.current = onEvent
        sync.current = onSync
    })
    useEffect(() => {
        if (!enabled) return
        return api.liveEvents(
            (event) => handler.current(event),
            () => sync.current?.(),
            () => sync.current?.(),
        )
    }, [enabled])
}

/** Stream watched games and reconcile snapshots on reconnect, foregrounding and every five seconds. */
export function useLiveGames(ids: string[], onEvent?: (event: LiveEvent) => void, onSync?: () => void) {
    const [records, setRecords] = useState<Record<string, GameRecord>>({})
    const [errors, setErrors] = useState<Record<string, string>>({})
    const watched = useRef(ids)
    const [games] = useState(
        () =>
            new LiveGames(api.game, setRecords, (id, error) => {
                setErrors((current) =>
                    Object.fromEntries(
                        Object.entries({ ...current, [id]: error?.message || '' }).filter(
                            ([key, message]) => message && watched.current.includes(key),
                        ),
                    ),
                )
            }),
    )
    useEffect(() => () => games.close(), [games])
    useEffect(() => {
        watched.current = ids
        games.watch(ids)
    }, [games, ids])
    const live = ids.some((id) => !records[id] || records[id].live)
    useLiveEvents(
        (event) => {
            games.receive(event)
            onEvent?.(event)
        },
        live || Boolean(onEvent),
        () => {
            games.refreshLive()
            onSync?.()
        },
    )
    useEffect(() => {
        if (!live) return
        const refresh = () => games.refreshLive()
        const foreground = () => {
            if (document.visibilityState === 'visible') refresh()
        }
        const timer = window.setInterval(refresh, 5_000)
        document.addEventListener('visibilitychange', foreground)
        return () => {
            window.clearInterval(timer)
            document.removeEventListener('visibilitychange', foreground)
        }
    }, [games, live])
    return { records, errors }
}
