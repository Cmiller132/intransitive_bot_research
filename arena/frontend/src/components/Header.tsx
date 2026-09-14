import { useEffect, useState } from 'react'
import { NavLink } from 'react-router-dom'
import { api } from '../api/client'
import { useSchedulerStatus } from '../api/live'
import { useArenaStore } from '../state/store'
import { Logo } from './Logo'
import { classifyHealth, type HealthState } from './health'

export function Header() {
    const { frame, setFrame } = useArenaStore()
    const [engine, setEngine] = useState<string>()
    const [health, setHealth] = useState<HealthState>('checking')
    const scheduler = useSchedulerStatus()
    const running = scheduler?.jobs_running || 0

    useEffect(() => {
        let active = true
        api.strongest()
            .then((value) => {
                if (active) setEngine(value.id)
            })
            .catch(() => {
                if (active) setEngine(undefined)
            })
        return () => {
            active = false
        }
    }, [])

    useEffect(() => {
        let active = true
        const check = () => {
            api.health()
                .then((value) => {
                    if (!active) return
                    setHealth(classifyHealth(value))
                })
                .catch(() => {
                    if (active) setHealth('offline')
                })
        }
        check()
        const timer = window.setInterval(check, 30_000)
        return () => {
            active = false
            window.clearInterval(timer)
        }
    }, [])

    const healthText =
        health === 'checking'
            ? 'Checking analysis service…'
            : health === 'ready'
              ? `${engine || 'Analysis engine'} · online`
              : health === 'idle'
                ? `${engine || 'Analysis engine'} · ready`
                : health === 'degraded'
                  ? 'Analysis service degraded'
                  : 'Analysis service unreachable'
    const problem = health === 'degraded' || health === 'offline'

    return (
        <header className="site-header">
            <NavLink className="wordmark" to="/" aria-label="Intransitive arena — rankings">
                <Logo />
                <span>
                    Intransitive <b>arena</b>
                </span>
            </NavLink>
            <nav aria-label="Primary">
                <NavLink to="/rankings">Rankings</NavLink>
                <NavLink to="/games">Games</NavLink>
                <NavLink to="/matchups">Matchups</NavLink>
                <NavLink to="/analysis">Analysis</NavLink>
                <NavLink to="/explorer">Explorer</NavLink>
                <NavLink to="/engines">Engines</NavLink>
                <NavLink to="/settings">Settings</NavLink>
            </nav>
            <div className="header-tools">
                {running > 0 && (
                    <NavLink className="live-pill" to="/games?live=1" title={`${running} job(s) playing now`}>
                        <i aria-hidden="true" />
                        live <b>{running}</b>
                    </NavLink>
                )}
                <label>
                    Frame
                    <select
                        value={frame}
                        onChange={(event) => setFrame(event.target.value as 'meaf' | 'henhen')}
                    >
                        <option value="meaf">meaf</option>
                        <option value="henhen">henhen</option>
                    </select>
                </label>
                <span
                    className="engine-dot"
                    role="status"
                    aria-live="polite"
                    title={healthText}
                    style={problem ? { color: 'var(--red)', background: 'var(--red-soft)' } : undefined}
                >
                    <i
                        aria-hidden="true"
                        style={{
                            background:
                                health === 'ready' || health === 'idle'
                                    ? 'var(--q-best)'
                                    : health === 'checking'
                                      ? 'var(--line-strong)'
                                      : 'var(--q-blunder)',
                        }}
                    />
                    {healthText}
                </span>
            </div>
        </header>
    )
}
