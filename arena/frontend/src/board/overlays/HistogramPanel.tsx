import type { HistogramOverlay } from '../../api/types'
import { probabilityHeight } from './semantics'
export function HistogramPanel({ overlay }: { overlay: HistogramOverlay }) {
    return (
        <div className="histogram" aria-label="Probability histogram">
            {overlay.bins.map((bin) => (
                <div className="hist-bin" key={bin.label}>
                    <div className="hist-track">
                        <i style={{ height: probabilityHeight(bin.p) }} />
                    </div>
                    <span>{bin.label}</span>
                    <b>{Math.round(bin.p * 100)}%</b>
                </div>
            ))}
            {overlay.expectation !== undefined && (
                <div className="expectation mono">
                    Expected <b>{overlay.expectation}</b>
                    {overlay.unit ? ` ${overlay.unit}` : ''}
                </div>
            )}
        </div>
    )
}
