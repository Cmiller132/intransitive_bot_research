import { useEffect, useMemo, useState } from 'react'
import { useParams } from 'react-router-dom'
import { api } from '../api/client'
import { useLiveGames } from '../api/live'
import type { Review } from '../api/types'
import { AnalysisWorkspace } from './AnalysisWorkspace'

export function GamePage() {
    const { id = '' } = useParams()
    return <GameLoader key={id} id={id} />
}

function GameLoader({ id }: { id: string }) {
    const ids = useMemo(() => [id], [id])
    const { records, errors } = useLiveGames(ids)
    const game = records[id]
    const [review, setReview] = useState<Review>()
    const [reviewError, setReviewError] = useState('')
    const reviewLoading = game?.review_status === 'done' && !review && !reviewError
    const error = errors[id]

    useEffect(() => {
        let cancelled = false
        if (game?.review_status !== 'done') return
        api.review(id)
            .then((result) => {
                if (!cancelled) setReview(result)
            })
            .catch((cause: Error) => {
                if (!cancelled) setReviewError(cause.message)
            })
        return () => {
            cancelled = true
        }
    }, [id, game?.review_status])

    if (error && !game)
        return (
            <main className="page">
                <div className="empty-state">
                    <h1>Game unavailable</h1>
                    <p role="alert">{error}</p>
                </div>
            </main>
        )
    if (!game)
        return (
            <main className="page">
                <div className="loading">
                    <span className="cycle-loader">RSP</span> Loading scorebook…
                </div>
            </main>
        )
    return (
        <>
            {error && (
                <p className="form-error" role="alert">
                    Live updates unavailable: {error}. Retrying…
                </p>
            )}
            <AnalysisWorkspace
                key={game.id || id}
                game={game}
                review={review}
                reviewLoading={reviewLoading}
                reviewError={reviewError}
                onReviewed={setReview}
            />
        </>
    )
}
