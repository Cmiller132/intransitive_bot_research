import type {
    Budget,
    Engine,
    EvalRequest,
    EvalResponse,
    ExplorerFilters,
    ExplorerResponse,
    ExportFormat,
    Frame,
    GameRecord,
    GameSummary,
    HeadSpec,
    ImportRequest,
    ImportResponse,
    InsightsResponse,
    Job,
    LiveEvent,
    LiveResponse,
    MatchupPair,
    MatchupsResponse,
    Meta,
    PlayerUpload,
    PositionRef,
    RatedPlayer,
    RatingsHistory,
    RatingsResponse,
    Review,
    SchedulerStatus,
} from './types'
import { ADMIN_TOKEN_KEY } from '../state/store'

const API_BASE = (import.meta.env.VITE_API_BASE || '/api').replace(/\/$/, '')
type QueryValue = string | number | boolean | undefined

function adminToken(): string {
    try {
        return window.localStorage.getItem(ADMIN_TOKEN_KEY) || ''
    } catch {
        return ''
    }
}

async function fail(response: Response): Promise<never> {
    const body = (await response.json().catch(() => null)) as { error?: { message?: string } } | null
    throw new Error(body?.error?.message || `${response.status} ${response.statusText}`)
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
    const headers: Record<string, string> = { 'Content-Type': 'application/json' }
    // Mutations need the admin token once the site runs in public mode.
    const token = adminToken()
    if (token && init?.method && init.method !== 'GET') headers['X-Admin-Token'] = token
    const response = await fetch(`${API_BASE}${path}`, { ...init, headers: { ...headers, ...init?.headers } })
    if (!response.ok) return fail(response)
    return response.json() as Promise<T>
}

/** Multipart upload: the browser sets its own Content-Type boundary, so we only add the token. */
async function upload<T>(path: string, body: FormData): Promise<T> {
    const headers: Record<string, string> = {}
    const token = adminToken()
    if (token) headers['X-Admin-Token'] = token
    const response = await fetch(`${API_BASE}${path}`, { method: 'POST', body, headers })
    if (!response.ok) return fail(response)
    return response.json() as Promise<T>
}

function query(values: Record<string, QueryValue>) {
    const p = new URLSearchParams()
    Object.entries(values).forEach(([k, v]) => {
        if (v !== undefined && v !== '') p.set(k, String(v))
    })
    const s = p.toString()
    return s ? `?${s}` : ''
}

/** Follow a server-sent event stream, tolerating both default and named events. */
function stream<T>(
    path: string,
    names: string[],
    onEvent: (value: T) => void,
    onError?: () => void,
    onOpen?: () => void,
) {
    const source = new EventSource(`${API_BASE}${path}`)
    const receive = (event: MessageEvent) => {
        let value: T
        try {
            value = JSON.parse(event.data) as T
        } catch {
            onError?.()
            return
        }
        onEvent(value)
    }
    source.onmessage = receive
    names.forEach((name) => source.addEventListener(name, receive as EventListener))
    source.onerror = () => onError?.()
    source.onopen = () => onOpen?.()
    return () => source.close()
}

export const api = {
    meta: () => request<Meta>('/meta'),
    engines: () => request<Engine[]>('/engines'),
    strongest: () => request<{ id: string; evidence: string }>('/engines/strongest'),
    engineHeads: (id: string) => request<HeadSpec[]>(`/engines/${encodeURIComponent(id)}/heads`),
    evaluate: (body: EvalRequest, signal?: AbortSignal) =>
        request<EvalResponse>('/eval', { method: 'POST', body: JSON.stringify(body), signal }),
    importGames: (body: ImportRequest) =>
        request<ImportResponse>('/games/import', { method: 'POST', body: JSON.stringify(body) }),
    games: (filters: Record<string, QueryValue> = {}) =>
        request<{ items: GameSummary[]; total: number }>(`/games${query(filters)}`),
    game: (id: string, signal?: AbortSignal) =>
        request<GameRecord>(`/games/${encodeURIComponent(id)}`, {
            signal: signal
                ? AbortSignal.any([signal, AbortSignal.timeout(15_000)])
                : AbortSignal.timeout(15_000),
        }),
    deleteGame: (id: string) =>
        request<{ ok: boolean }>(`/games/${encodeURIComponent(id)}`, { method: 'DELETE' }),
    review: (id: string) => request<Review>(`/games/${encodeURIComponent(id)}/review`),
    analyse: (id: string, body: { engine?: string; budget?: Budget }) =>
        request<{ job_id: string; reused: boolean }>(`/games/${encodeURIComponent(id)}/analyse`, {
            method: 'POST',
            body: JSON.stringify(body),
        }),
    jobEvents: (id: string, onJob: (job: Job) => void, onError?: () => void) =>
        stream<Job>(`/jobs/${encodeURIComponent(id)}/events`, ['job'], onJob, onError),
    export: (
        body: ({ game_id: string } | { position: PositionRef }) & {
            format: ExportFormat
            frame?: Frame
        },
    ) =>
        request<{ content: string; mime: string; filename?: string }>('/export', {
            method: 'POST',
            body: JSON.stringify(body),
        }),
    explorer: (filters: ExplorerFilters = {}) =>
        request<ExplorerResponse>(
            `/explorer${query({
                line: filters.line,
                symmetry: filters.symmetry === undefined ? undefined : filters.symmetry ? 1 : 0,
                source: filters.source,
                player: filters.player,
                min_games: filters.min_games,
            })}`,
        ),
    insights: (player: string) => request<InsightsResponse>(`/insights/${encodeURIComponent(player)}`),
    ratings: () => request<RatingsResponse>('/ratings'),
    ratingsHistory: (player?: string) => request<RatingsHistory>(`/ratings/history${query({ player })}`),
    matchups: () => request<MatchupsResponse>('/matchups'),
    matchup: (a: string, b: string) =>
        request<MatchupPair>(`/matchups/${encodeURIComponent(a)}/${encodeURIComponent(b)}`),
    live: () => request<LiveResponse>('/live', { signal: AbortSignal.timeout(15_000) }),
    liveEvents: (onEvent: (event: LiveEvent) => void, onError?: () => void, onOpen?: () => void) =>
        stream<LiveEvent>(
            '/live/events',
            ['job_start', 'game_start', 'move', 'game_end', 'job_end', 'fit', 'resync'],
            onEvent,
            onError,
            onOpen,
        ),
    players: () => request<{ players: RatedPlayer[] }>('/players'),
    uploadPlayer: ({ name, spec, replace, files }: PlayerUpload) => {
        const body = new FormData()
        body.set('name', name)
        body.set('spec', spec)
        body.set('replace', replace ? '1' : '0')
        files.forEach((file) => body.append('file', file, file.name))
        return upload<{ name: string; spec: string; replaced?: boolean }>('/players', body)
    },
    retirePlayer: (name: string, retired: boolean) =>
        request<{ name: string; retired: boolean }>(
            `/players/${encodeURIComponent(name)}/${retired ? 'retire' : 'unretire'}`,
            { method: 'POST', body: '{}' },
        ),
    scheduler: () => request<SchedulerStatus>('/scheduler'),
    health: () =>
        fetch('/healthz').then(
            (r) =>
                r.json() as Promise<{
                    ok: boolean
                    engine_loaded: boolean
                    db: boolean | string
                    jobs_running?: number
                    players?: number
                }>,
        ),
}
