import { Link } from 'react-router-dom'
import type { GameSummary } from '../api/types'
import { SideName } from './SideName'

const resultText = (game: GameSummary): string =>
    game.result.reported ||
    (game.result.winner === 'blue' ? '1-0' : game.result.winner === 'red' ? '0-1' : '½–½')

const rating = (value?: number | null) => (value == null ? 'Unrated' : String(Math.round(value)))

/** The shared game row used by the library, the live view and matchup pages. */
export function GameList({
    games,
    loading = false,
    empty = 'No games match these filters.',
}: {
    games: GameSummary[]
    loading?: boolean
    empty?: string
}) {
    return (
        <div className={`game-list ${loading ? 'is-loading' : ''}`} aria-busy={loading}>
            {games.map((game) => (
                <Link to={`/game/${encodeURIComponent(game.id)}`} key={game.id} className="game-row">
                    <div>
                        <SideName side="blue" name={game.players.blue.name} />
                        <span>Blue · {rating(game.players.blue.rating_before)}</span>
                    </div>
                    <strong>{game.live ? '⋯' : resultText(game)}</strong>
                    <div>
                        <SideName side="red" name={game.players.red.name} />
                        <span>Red · {rating(game.players.red.rating_before)}</span>
                    </div>
                    <div className="game-meta">
                        {game.live && <span className="live-badge">live</span>}
                        <span>{game.source}</span>
                        <span>{game.ply_count} plies</span>
                        {game.sims ? <span>{game.sims} sims</span> : null}
                        <span className={`review-state ${game.review_status === 'done' ? 'done' : ''}`}>
                            {game.review_status === 'done' ? 'Reviewed' : 'Not reviewed'}
                        </span>
                    </div>
                </Link>
            ))}
            {!loading && !games.length && <div className="empty-copy">{empty}</div>}
        </div>
    )
}
