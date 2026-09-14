import { Link } from 'react-router-dom'
export function NotFoundPage() {
    return (
        <main className="page">
            <div className="empty-state">
                <span className="eyebrow">404</span>
                <h1>That scorebook is not here.</h1>
                <Link className="primary button-link" to="/">
                    Back to the rankings
                </Link>
            </div>
        </main>
    )
}
