import { describe, expect, it } from 'vitest'
import { classifyHealth } from './health'

describe('header health status', () => {
    it('treats a healthy cold engine as ready but idle', () => {
        expect(classifyHealth({ ok: true, engine_loaded: false, db: true })).toBe('idle')
    })

    it('distinguishes loaded, degraded, and unavailable states', () => {
        expect(classifyHealth({ ok: true, engine_loaded: true, db: 'ok' })).toBe('ready')
        expect(classifyHealth({ ok: true, engine_loaded: true, db: false })).toBe('degraded')
        expect(classifyHealth({ ok: false, engine_loaded: false, db: true })).toBe('degraded')
    })
})
