import { AnalysisWorkspace } from './AnalysisWorkspace'
import { useLocation } from 'react-router-dom'
export function AnalysisPage() {
    const { search } = useLocation()
    return <AnalysisWorkspace key={search} />
}
