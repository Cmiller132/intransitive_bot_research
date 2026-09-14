import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api/client'
import type { Budget, Job, Review, ReviewJob } from '../api/types'
import { useArenaStore } from '../state/store'

type Props = {
    gameId: string
    engine: string
    job?: ReviewJob | null
    onReviewed: (review: Review) => void
}

/** Start a game review job, or attach to the one already running, and follow it to the stored review. */
export function ReviewPanel({ gameId, engine, job: initialJob, onReviewed }: Props) {
    const adminToken = useArenaStore((state) => state.adminToken)
    const [budget, setBudget] = useState<Budget>('standard')
    const [job, setJob] = useState<Job | ReviewJob | undefined>(initialJob || undefined)
    const [publicSite, setPublicSite] = useState(false)
    const [error, setError] = useState('')
    const stop = useRef<() => void>(undefined)

    const follow = (jobId: string) => {
        stop.current?.()
        stop.current = api.jobEvents(jobId, (update) => {
            setJob(update)
            if (update.status === 'done') {
                stop.current?.()
                api.review(gameId)
                    .then(onReviewed)
                    .catch((cause: Error) => setError(cause.message))
            }
            if (update.status === 'failed') stop.current?.()
        })
    }

    useEffect(() => {
        api.meta()
            .then((meta) => setPublicSite(meta.public))
            .catch(() => undefined)
        if (initialJob) follow(initialJob.id)
        return () => stop.current?.()
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [])

    const start = async () => {
        setError('')
        try {
            const { job_id } = await api.analyse(gameId, { engine, budget })
            setJob({ id: job_id, status: 'queued', progress: 0 })
            follow(job_id)
        } catch (cause) {
            setError(cause instanceof Error ? cause.message : String(cause))
        }
    }

    const running = job !== undefined && (job.status === 'queued' || job.status === 'running')
    const failed = job !== undefined && job.status === 'failed' ? (job as Job) : undefined
    const deepLocked = publicSite && !adminToken
    const detail =
        job && 'detail' in job ? job.detail : job?.status === 'queued' ? 'waiting for the worker' : ''
    return (
        <div className="panel review-panel">
            <div className="panel-head">
                <b>Game review</b>
                <span>{running ? `${job.status} · ${Math.round(job.progress * 100)}%` : 'not reviewed'}</span>
            </div>
            {running ? (
                <div className="progress" role="progressbar" aria-valuenow={Math.round(job.progress * 100)}>
                    <i style={{ width: `${job.progress * 100}%` }} />
                    <span>{detail}</span>
                </div>
            ) : (
                <>
                    <p>
                        Score every move against {engine} to get accuracy, quality marks and the evaluation
                        graph.
                    </p>
                    <div className="form-actions">
                        <select
                            value={String(budget)}
                            onChange={(event) => setBudget(event.target.value as Budget)}
                        >
                            <option value="quick">Quick · network only</option>
                            <option value="standard">Standard · 128 sims per move</option>
                            <option value="deep" disabled={deepLocked}>
                                Deep · 800 sims per move{deepLocked ? ' (admin token)' : ''}
                            </option>
                        </select>
                        <button className="primary" onClick={() => void start()}>
                            Review this game
                        </button>
                    </div>
                    {failed && <p className="form-error">Review failed: {failed.detail}</p>}
                    {error && (
                        <p className="form-error" role="alert">
                            {error}
                            {/admin/i.test(error) && (
                                <>
                                    {' '}
                                    <Link to="/settings">Add the token under Settings.</Link>
                                </>
                            )}
                        </p>
                    )}
                </>
            )}
        </div>
    )
}
