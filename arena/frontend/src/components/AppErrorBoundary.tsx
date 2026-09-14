import { Component, type ErrorInfo, type ReactNode } from 'react'

type Props = { children: ReactNode }
type State = { error?: Error }

export class AppErrorBoundary extends Component<Props, State> {
    state: State = {}

    static getDerivedStateFromError(error: Error): State {
        return { error }
    }

    componentDidCatch(error: Error, info: ErrorInfo) {
        console.error('Route rendering failed', error, info.componentStack)
    }

    render() {
        if (!this.state.error) return this.props.children
        return (
            <main className="page">
                <div className="empty-state route-error">
                    <span className="eyebrow">Page error</span>
                    <h1>This page could not be displayed.</h1>
                    <p role="alert">
                        {this.state.error.message || 'An unexpected rendering error occurred.'}
                    </p>
                    <button className="primary" onClick={() => window.location.reload()}>
                        Reload page
                    </button>
                </div>
            </main>
        )
    }
}
