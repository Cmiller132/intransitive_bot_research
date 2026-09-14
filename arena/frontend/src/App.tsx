import { useEffect } from 'react'
import { Navigate, Route, Routes, useLocation } from 'react-router-dom'
import { AppErrorBoundary } from './components/AppErrorBoundary'
import { Header } from './components/Header'
import { AnalysisPage } from './pages/AnalysisPage'
import { EnginesPage } from './pages/EnginesPage'
import { ExplorerPage } from './pages/ExplorerPage'
import { GamePage } from './pages/GamePage'
import { ImportPage } from './pages/ImportPage'
import { InsightsPage } from './pages/InsightsPage'
import { LibraryPage } from './pages/LibraryPage'
import { MatchupPairPage } from './pages/MatchupPairPage'
import { MatchupsPage } from './pages/MatchupsPage'
import { NotFoundPage } from './pages/NotFoundPage'
import { RankingsPage } from './pages/RankingsPage'
import { SettingsPage } from './pages/SettingsPage'
import { useArenaStore } from './state/store'

export default function App() {
    const theme = useArenaStore((state) => state.theme)
    const location = useLocation()
    useEffect(() => {
        window.scrollTo(0, 0)
    }, [location.pathname])
    useEffect(() => {
        if (theme === 'system') document.documentElement.removeAttribute('data-theme')
        else document.documentElement.dataset.theme = theme
    }, [theme])

    return (
        <>
            <Header />
            <AppErrorBoundary key={location.pathname}>
                <Routes>
                    <Route path="/" element={<RankingsPage />} />
                    <Route path="/rankings" element={<RankingsPage />} />
                    <Route path="/analysis" element={<AnalysisPage />} />
                    <Route path="/game/:id" element={<GamePage />} />
                    <Route path="/review/:id" element={<GamePage />} />
                    <Route path="/games" element={<LibraryPage />} />
                    <Route path="/library" element={<Navigate to="/games" replace />} />
                    <Route path="/matchups" element={<MatchupsPage />} />
                    <Route path="/matchups/:a/:b" element={<MatchupPairPage />} />
                    <Route path="/explorer" element={<ExplorerPage />} />
                    <Route path="/insights" element={<InsightsPage />} />
                    <Route path="/insights/:player" element={<InsightsPage />} />
                    <Route path="/engines" element={<EnginesPage />} />
                    <Route path="/settings" element={<SettingsPage />} />
                    <Route path="/import" element={<ImportPage />} />
                    <Route path="*" element={<NotFoundPage />} />
                </Routes>
            </AppErrorBoundary>
        </>
    )
}
