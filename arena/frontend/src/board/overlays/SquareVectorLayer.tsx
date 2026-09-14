import { useState } from 'react'
import type { SquareVectorOverlay } from '../../api/types'
import type { LayerGeometry } from './types'
import { point } from './types'
export function SquareVectorLayer({
    overlay,
    geometry,
}: {
    overlay: SquareVectorOverlay
    geometry: LayerGeometry
}) {
    const [head, setHead] = useState(0)
    const source = point(overlay.source, geometry)
    const weights = overlay.per_head[head] || []
    const max = Math.max(...weights.map(Math.abs), 0.001)
    return (
        <g>
            {weights.map((value, sq) => {
                if (Math.abs(value) < max * 0.16) return null
                const p = point(sq, geometry)
                return (
                    <g key={sq} pointerEvents="none">
                        <line
                            x1={source.x + geometry.cell / 2}
                            y1={source.y + geometry.cell / 2}
                            x2={p.x + geometry.cell / 2}
                            y2={p.y + geometry.cell / 2}
                            stroke="var(--overlay-outline)"
                            strokeWidth={4 + (5 * Math.abs(value)) / max}
                            opacity=".75"
                        />
                        <line
                            x1={source.x + geometry.cell / 2}
                            y1={source.y + geometry.cell / 2}
                            x2={p.x + geometry.cell / 2}
                            y2={p.y + geometry.cell / 2}
                            stroke="var(--overlay-attention)"
                            strokeWidth={2 + (5 * Math.abs(value)) / max}
                            opacity={0.55 + (0.45 * Math.abs(value)) / max}
                        >
                            <title>{`${overlay.labels[head] || `Head ${head + 1}`} · ${value.toFixed(3)}`}</title>
                        </line>
                    </g>
                )
            })}
            <foreignObject x={geometry.margin + 4} y={geometry.margin + 4} width="145" height="38">
                <select
                    className="overlay-select"
                    value={head}
                    onChange={(e) => setHead(Number(e.target.value))}
                    aria-label="Attention head"
                >
                    {overlay.labels.map((label, i) => (
                        <option key={label} value={i}>
                            {label}
                        </option>
                    ))}
                </select>
            </foreignObject>
        </g>
    )
}
