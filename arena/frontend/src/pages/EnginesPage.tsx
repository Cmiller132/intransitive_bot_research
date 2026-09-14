import { useCallback, useEffect, useMemo, useState } from 'react'
import type { FormEvent } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api/client'
import type { Engine } from '../api/types'
import { useArenaStore } from '../state/store'
import { formatRating, sortByRating } from './runGrouping'

type Strongest = { id: string; evidence: string }

const messageOf = (reason: unknown): string => (reason instanceof Error ? reason.message : String(reason))

export function EnginesPage() {
    const adminToken = useArenaStore((state) => state.adminToken)
    const [engines, setEngines] = useState<Engine[]>([])
    const [strongest, setStrongest] = useState<Strongest>()
    const [loading, setLoading] = useState(true)
    const [registryError, setRegistryError] = useState('')
    const [evidenceError, setEvidenceError] = useState('')
    const [needle, setNeedle] = useState('')
    const [showRetired, setShowRetired] = useState(false)
    const [adminError, setAdminError] = useState('')
    const [adminNote, setAdminNote] = useState('')
    const [pendingName, setPendingName] = useState('')

    const load = useCallback(() => {
        return Promise.allSettled([api.engines(), api.strongest()]).then(([registry, champion]) => {
            if (registry.status === 'fulfilled') {
                setEngines(registry.value)
                setRegistryError('')
            } else setRegistryError(messageOf(registry.reason))
            if (champion.status === 'fulfilled') {
                setStrongest(champion.value)
                setEvidenceError('')
            } else setEvidenceError(messageOf(champion.reason))
            setLoading(false)
        })
    }, [])

    useEffect(() => {
        void load()
    }, [load])

    const pool = useMemo(() => {
        const text = needle.trim().toLowerCase()
        const visible = engines.filter(
            (engine) =>
                (showRetired || !engine.retired) &&
                (!text ||
                    `${engine.id} ${engine.label} ${engine.run} ${engine.spec}`.toLowerCase().includes(text)),
        )
        return sortByRating(visible, (engine) => engine.id)
    }, [engines, needle, showRetired])

    const retire = async (engine: Engine) => {
        setAdminError('')
        setAdminNote('')
        setPendingName(engine.id)
        try {
            await api.retirePlayer(engine.id, !engine.retired)
            setAdminNote(`${engine.id} ${engine.retired ? 'returned to the pool' : 'retired'}.`)
            await load()
        } catch (cause) {
            setAdminError(messageOf(cause))
        } finally {
            setPendingName('')
        }
    }

    return (
        <main className="page">
            <div className="page-heading">
                <div>
                    <span className="eyebrow">Arena pool</span>
                    <h1>Engines</h1>
                </div>
                <span className="record-count mono" aria-live="polite">
                    {loading ? 'Loading…' : `${pool.length} of ${engines.length} players`}
                </span>
            </div>
            <p className="lede">
                Every uploaded model plays rated games and answers analysis requests. Ratings come from the
                same Bradley-Terry fit as the rankings.
            </p>
            <div className="double-rule" />

            {loading && (
                <div className="loading" role="status">
                    <span className="cycle-loader" aria-hidden="true">
                        RSP
                    </span>{' '}
                    Loading the pool…
                </div>
            )}
            {registryError && (
                <p className="form-error" role="alert">
                    Engine registry unavailable: {registryError}
                </p>
            )}
            {!loading && !registryError && engines.length === 0 && (
                <div className="empty-state">
                    <h2>No players yet</h2>
                    <p>Upload a model below to put it in the arena.</p>
                </div>
            )}

            {engines.length > 0 && (
                <>
                    <div className="filters">
                        <label>
                            Search
                            <input
                                value={needle}
                                placeholder="Name, run or spec"
                                onChange={(event) => setNeedle(event.target.value)}
                            />
                        </label>
                        <label className="checkbox">
                            <input
                                type="checkbox"
                                checked={showRetired}
                                onChange={(event) => setShowRetired(event.target.checked)}
                            />
                            Show retired
                        </label>
                    </div>
                    {adminNote && <p className="form-status">{adminNote}</p>}
                    {adminError && (
                        <p className="form-error" role="alert">
                            {adminError}
                            {/admin|token|forbidden|401|403/i.test(adminError) && (
                                <>
                                    {' '}
                                    <Link to="/settings">Add the admin token under Settings.</Link>
                                </>
                            )}
                        </p>
                    )}
                    <div className="engine-table-wrap">
                        <table className="engine-table compact-table">
                            <thead>
                                <tr>
                                    <th>Engine</th>
                                    <th>Run</th>
                                    <th>Rating</th>
                                    <th>Games</th>
                                    <th>Status</th>
                                    <th>Kind</th>
                                    <th>Heads</th>
                                    <th>Actions</th>
                                </tr>
                            </thead>
                            <tbody>
                                {pool.map((engine) => (
                                    <tr key={engine.id} className={engine.retired ? 'muted-row' : ''}>
                                        <td data-label="Engine">
                                            <b>{engine.label}</b>
                                            <small className="mono">{engine.spec}</small>
                                        </td>
                                        <td data-label="Run" className="mono">
                                            {engine.run}
                                            {engine.iter ? ` @${engine.iter}` : ''}
                                        </td>
                                        <td data-label="Rating" className="mono">
                                            {formatRating(engine.rating, engine.half_width)}
                                        </td>
                                        <td data-label="Games" className="mono">
                                            {engine.games}
                                        </td>
                                        <td data-label="Status">
                                            {engine.strongest || strongest?.id === engine.id ? (
                                                <span className="strongest">◆ Strongest</span>
                                            ) : engine.retired ? (
                                                <span className="state-badge muted">retired</span>
                                            ) : engine.broken ? (
                                                <span className="state-badge bad">broken</span>
                                            ) : engine.settled ? (
                                                <span className="state-badge good">settled</span>
                                            ) : (
                                                <span className="state-badge">measuring</span>
                                            )}
                                        </td>
                                        <td data-label="Kind">
                                            <span className="kind-badge">{engine.kind}</span>
                                        </td>
                                        <td data-label="Heads">
                                            {engine.analysable ? (
                                                <>
                                                    {engine.heads.length}
                                                    <small
                                                        title={engine.heads
                                                            .map((head) => head.description)
                                                            .join('\n')}
                                                    >
                                                        {engine.heads.map((head) => head.label).join(' · ') ||
                                                            'No declared heads'}
                                                    </small>
                                                </>
                                            ) : (
                                                <small>External engine · plays only</small>
                                            )}
                                        </td>
                                        <td data-label="Actions">
                                            <div className="row-actions">
                                                {engine.analysable && (
                                                    <Link
                                                        className="button-link"
                                                        to={`/analysis?engine=${encodeURIComponent(engine.id)}`}
                                                    >
                                                        Analyse
                                                    </Link>
                                                )}
                                                <Link
                                                    className="button-link"
                                                    to={`/games?player=${encodeURIComponent(engine.id)}`}
                                                >
                                                    Games
                                                </Link>
                                                <button
                                                    type="button"
                                                    disabled={pendingName === engine.id}
                                                    onClick={() => void retire(engine)}
                                                >
                                                    {engine.retired ? 'Unretire' : 'Retire'}
                                                </button>
                                            </div>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </>
            )}

            {strongest && (
                <section className="evidence-card">
                    <span className="eyebrow">Why this is strongest</span>
                    <h2>{strongest.id}</h2>
                    <p>{strongest.evidence}</p>
                </section>
            )}
            {evidenceError && (
                <section className="evidence-card">
                    <span className="eyebrow">Strongest-engine evidence</span>
                    <p className="form-error" role="alert">
                        Unavailable: {evidenceError}
                    </p>
                </section>
            )}

            <section className="wide-section">
                <div className="double-rule" />
                <h2>Upload a player</h2>
                <p className="section-intro">
                    The files are stored under the player's name and the arena starts scheduling games for it.
                    {adminToken ? '' : ' Needs the admin token, or a request from the LAN.'}
                </p>
                <UploadForm onUploaded={() => void load()} />
            </section>
        </main>
    )
}

function UploadForm({ onUploaded }: { onUploaded: () => void }) {
    const [name, setName] = useState('')
    const [spec, setSpec] = useState('')
    const [replace, setReplace] = useState(false)
    const [files, setFiles] = useState<File[]>([])
    const [pending, setPending] = useState(false)
    const [message, setMessage] = useState('')
    const [error, setError] = useState('')

    const submit = async (event: FormEvent) => {
        event.preventDefault()
        if (pending || !name.trim() || !spec.trim()) return
        setPending(true)
        setMessage('')
        setError('')
        try {
            const result = await api.uploadPlayer({ name: name.trim(), spec: spec.trim(), replace, files })
            setMessage(`${result.name} is in the pool as ${result.spec}.`)
            setName('')
            setSpec('')
            setFiles([])
            onUploaded()
        } catch (cause) {
            setError(messageOf(cause))
        } finally {
            setPending(false)
        }
    }

    return (
        <form className="import-form panel" onSubmit={submit} aria-busy={pending}>
            <div className="form-grid">
                <label>
                    Name
                    <input
                        value={name}
                        disabled={pending}
                        placeholder="conv_g128_240"
                        onChange={(event) => setName(event.target.value)}
                    />
                </label>
                <label>
                    Spec
                    <input
                        value={spec}
                        disabled={pending}
                        placeholder="conv:model.onnx"
                        onChange={(event) => setSpec(event.target.value)}
                    />
                </label>
            </div>
            <label className="file-control">
                Files
                <input
                    type="file"
                    multiple
                    disabled={pending}
                    onChange={(event) => setFiles(Array.from(event.target.files || []))}
                />
            </label>
            <p className="section-intro">
                <span className="mono">sq:&lt;onnx&gt;</span> and{' '}
                <span className="mono">conv:&lt;onnx&gt;</span> need the model file (a conv model also needs
                its <span className="mono">.json</span> beside it);{' '}
                <span className="mono">rpsi:&lt;command&gt;</span> needs no upload.
            </p>
            <label className="checkbox">
                <input
                    type="checkbox"
                    checked={replace}
                    disabled={pending}
                    onChange={(event) => setReplace(event.target.checked)}
                />
                Replace an existing player of this name
            </label>
            {files.length > 0 && (
                <div className="import-preview">
                    <b>{files.length} file(s)</b>
                    <span>{files.map((file) => file.name).join(', ')}</span>
                </div>
            )}
            {error && (
                <p className="form-error" role="alert">
                    Upload failed: {error}
                </p>
            )}
            <div className="form-actions">
                <button className="primary" type="submit" disabled={pending || !name.trim() || !spec.trim()}>
                    {pending ? 'Uploading…' : 'Upload player'}
                </button>
                <span role="status" aria-live="polite">
                    {message}
                </span>
            </div>
        </form>
    )
}
