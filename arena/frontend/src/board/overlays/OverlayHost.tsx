import type { Frame, Overlay } from '../../api/types'
import { BoardForecastLayer } from './BoardForecastLayer'
import { HistogramPanel } from './HistogramPanel'
import { MoveScalarLayer } from './MoveScalarLayer'
import { ScalarPanel } from './ScalarPanel'
import { SquareScalarLayer } from './SquareScalarLayer'
import { SquareVectorLayer } from './SquareVectorLayer'
export const KNOWN_OVERLAYS = new Set([
    'square_scalar',
    'move_scalar',
    'scalar',
    'histogram',
    'board_forecast',
    'square_vector',
])
export function OverlayHost({
    overlays,
    frame,
    mode = 'board',
    cell = 52,
    margin = 22,
    labels = true,
}: {
    overlays: Overlay[]
    frame: Frame
    mode?: 'board' | 'panel'
    cell?: number
    margin?: number
    labels?: boolean
}) {
    const geometry = { frame, cell, margin, labels }
    if (mode === 'panel')
        return (
            <>
                {overlays.map((overlay, i) =>
                    overlay.kind === 'scalar' ? (
                        <ScalarPanel key={i} overlay={overlay} />
                    ) : overlay.kind === 'histogram' ? (
                        <HistogramPanel key={i} overlay={overlay} />
                    ) : null,
                )}
            </>
        )
    return (
        <>
            {overlays.map((overlay, i) => {
                switch (overlay.kind) {
                    case 'square_scalar':
                        return <SquareScalarLayer key={i} overlay={overlay} geometry={geometry} />
                    case 'move_scalar':
                        return <MoveScalarLayer key={i} overlay={overlay} geometry={geometry} />
                    case 'board_forecast':
                        return <BoardForecastLayer key={i} overlay={overlay} geometry={geometry} />
                    case 'square_vector':
                        return <SquareVectorLayer key={i} overlay={overlay} geometry={geometry} />
                    default:
                        return null
                }
            })}
        </>
    )
}
