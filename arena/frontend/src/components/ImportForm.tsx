import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api/client'
import type { Frame, ImportRequest, ImportResponse } from '../api/types'
import { previewPgn } from '../rules/pgn'

const PLACEHOLDERS: Record<ImportRequest['source'], string> = {
    henhen_pgn: 'Paste the complete PGN, including tags and moves…',
    henhen_id: 'Paste a henhen game ID or review URL…',
    henhen_series: 'Paste a henhen player ID or series URL…',
    meaf_id: 'Paste a meaf game ID or game URL…',
    meaf_line: 'Paste a complete MEAF line starting with Blue, e.g. E3-F4 E7-D6…',
    meaf_workshop: 'Paste a workshop ID, URL, or workshop JSON…',
    fen: 'Paste a full 9×9 position FEN…',
    json: 'Paste an arena game JSON object…',
}

export function ImportForm() {
    const [source, setSource] = useState<ImportRequest['source']>('henhen_pgn')
    const [payload, setPayload] = useState('')
    const [frame, setFrame] = useState<Frame>('henhen')
    const [pending, setPending] = useState(false)
    const [message, setMessage] = useState('')
    const [error, setError] = useState('')
    const [result, setResult] = useState<ImportResponse>()

    const preview = useMemo(() => {
        if (source !== 'henhen_pgn' || !payload.trim()) return null
        try {
            const parsed = previewPgn(payload)
            return parsed.plies > 0 ? parsed : null
        } catch {
            return null
        }
    }, [payload, source])
    const previewFailed = source === 'henhen_pgn' && Boolean(payload.trim()) && !preview
    const clearOutcome = () => {
        setMessage('')
        setError('')
        setResult(undefined)
    }

    const submit = async (event: React.FormEvent) => {
        event.preventDefault()
        if (pending || !payload.trim() || previewFailed) return
        setPending(true)
        setMessage('Importing…')
        setError('')
        setResult(undefined)
        try {
            const imported = await api.importGames({ source, payload, frame })
            setResult(imported)
            if (imported.game_ids.length)
                setMessage(
                    `${imported.game_ids.length} game${imported.game_ids.length === 1 ? '' : 's'} imported successfully.`,
                )
            else setMessage('Import completed, but no game was returned.')
        } catch (reason) {
            setMessage('')
            setError(reason instanceof Error ? reason.message : String(reason))
        } finally {
            setPending(false)
        }
    }

    const readFile = async (file?: File) => {
        if (!file) return
        clearOutcome()
        try {
            setPayload(await file.text())
        } catch (reason) {
            setError(reason instanceof Error ? reason.message : 'The selected file could not be read.')
        }
    }

    return (
        <form className="import-form panel" onSubmit={submit} aria-busy={pending}>
            <div className="form-grid">
                <label>
                    Source
                    <select
                        disabled={pending}
                        value={source}
                        onChange={(event) => {
                            setSource(event.target.value as ImportRequest['source'])
                            clearOutcome()
                        }}
                    >
                        <option value="henhen_pgn">henhen PGN</option>
                        <option value="henhen_id">henhen game ID</option>
                        <option value="henhen_series">henhen series</option>
                        <option value="meaf_id">meaf game ID</option>
                        <option value="meaf_line">meaf line</option>
                        <option value="meaf_workshop">meaf workshop</option>
                        <option value="fen">Position FEN</option>
                        <option value="json">Arena JSON</option>
                    </select>
                </label>
                <label>
                    Source frame
                    <select
                        disabled={pending}
                        value={frame}
                        onChange={(event) => {
                            setFrame(event.target.value as Frame)
                            clearOutcome()
                        }}
                    >
                        <option value="meaf">meaf</option>
                        <option value="henhen">henhen</option>
                    </select>
                </label>
            </div>
            <label>
                Paste record
                <textarea
                    disabled={pending}
                    rows={10}
                    value={payload}
                    onChange={(event) => {
                        setPayload(event.target.value)
                        clearOutcome()
                    }}
                    placeholder={PLACEHOLDERS[source]}
                />
            </label>
            <label className="file-control">
                Or choose a file
                <input
                    disabled={pending}
                    type="file"
                    accept=".pgn,.txt,.json"
                    onChange={(event) => {
                        void readFile(event.target.files?.[0])
                    }}
                />
            </label>
            {preview && (
                <div className="import-preview">
                    <b>Ready to import</b>
                    <span>
                        {preview.blue} — {preview.red}
                    </span>
                    <span>
                        {preview.plies} plies · {preview.result}
                    </span>
                </div>
            )}
            {previewFailed && (
                <p className="form-error" role="alert">
                    The preview could not find a move in this PGN. Check the record before importing.
                </p>
            )}
            {result?.warnings.map((warning) => (
                <p className="form-warning" role="status" key={warning}>
                    Import warning: {warning}
                </p>
            ))}
            {error && (
                <p className="form-error" role="alert">
                    Import failed: {error}
                </p>
            )}
            <div className="form-actions">
                <button
                    className="primary"
                    type="submit"
                    disabled={pending || !payload.trim() || previewFailed}
                >
                    {pending ? 'Importing…' : 'Import'}
                </button>
                <span role="status" aria-live="polite">
                    {message}
                </span>
            </div>
            {result && (
                <div className="form-actions">
                    {result.game_ids.slice(0, 3).map((id, index) => (
                        <Link className="button-link" to={`/game/${encodeURIComponent(id)}`} key={id}>
                            {result.game_ids.length === 1 ? 'Open imported game' : `Open game ${index + 1}`}
                        </Link>
                    ))}
                    {result.game_ids.length > 3 && (
                        <Link className="button-link" to="/games">
                            View all {result.game_ids.length} games
                        </Link>
                    )}
                </div>
            )}
        </form>
    )
}
