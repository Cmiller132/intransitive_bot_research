import type { BoardForecastOverlay } from '../../api/types'
import type { LayerGeometry } from './types'
import { point } from './types'
import { forecastPiece } from './semantics'
const glyphs = ['', 'R', 'P', 'S', 'R', 'P', 'S']
export function BoardForecastLayer({
    overlay,
    geometry,
}: {
    overlay: BoardForecastOverlay
    geometry: LayerGeometry
}) {
    return (
        <g className="overlay-forecast" pointerEvents="none">
            {overlay.cells.map((cell) => {
                const forecast = forecastPiece(cell.probs)
                if (!forecast) return null
                const { piece, probability: max } = forecast
                const p = point(cell.sq, geometry)
                return (
                    <g key={cell.sq} opacity={0.3 + max * 0.7}>
                        <circle
                            cx={p.x + geometry.cell / 2}
                            cy={p.y + geometry.cell / 2}
                            r={geometry.cell * 0.32}
                            fill={piece <= 3 ? 'var(--blue)' : 'var(--red)'}
                            stroke="var(--overlay-outline)"
                            strokeWidth="2"
                            strokeDasharray="3 2"
                        />
                        <text
                            x={p.x + geometry.cell / 2}
                            y={p.y + geometry.cell * 0.61}
                            textAnchor="middle"
                            fill="#fff"
                            fontWeight="700"
                        >
                            {glyphs[piece]}
                        </text>
                        {geometry.labels && (
                            <text
                                x={p.x + geometry.cell / 2}
                                y={p.y + geometry.cell - 4}
                                className="square-value"
                            >
                                {`${Math.round(max * 100)}%`}
                            </text>
                        )}
                        <title>
                            {cell.probs
                                .map((v, i) => `${glyphs[i] || 'empty'} ${Math.round(v * 100)}%`)
                                .join(' · ')}
                        </title>
                    </g>
                )
            })}
        </g>
    )
}
