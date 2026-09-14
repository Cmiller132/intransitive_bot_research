export type HealthState = 'checking' | 'ready' | 'idle' | 'degraded' | 'offline'

export function classifyHealth(value: {
    ok: boolean
    engine_loaded: boolean
    db: boolean | string
}): HealthState {
    const dbReady = value.db === true || value.db === 'ok' || value.db === 'ready' || value.db === 'true'
    if (!value.ok || !dbReady) return 'degraded'
    return value.engine_loaded ? 'ready' : 'idle'
}
