import type { SquareScalarOverlay } from '../../api/types'
import type { LayerGeometry } from './types'
import { point } from './types'

/** Show a plane value the way a person would count it: pieces, steps or a percentage. */
export function formatSquareValue(value: number, overlay: SquareScalarOverlay): string {
    const [, hi] = overlay.range
    if (hi === 1 && /\/ 8/.test(overlay.legend || '')) return String(Math.round(value * 8))
    if (hi === 1) return `${Math.round(value * 100)}%`
    if (Number.isInteger(hi)) return String(Math.round(value))
    return value.toFixed(2)
}

export function SquareScalarLayer({
    overlay,
    geometry,
}: {
    overlay: SquareScalarOverlay
    geometry: LayerGeometry
}) {
    const [lo, hi] = overlay.range
    const diverging = overlay.scale === 'diverging'
    return (
        <g className="overlay-square-scalar" pointerEvents="none">
            {overlay.values.map((v, sq) => {
                const t = Math.max(0, Math.min(1, (v - lo) / (hi - lo || 1)))
                const strength = diverging ? Math.abs(t - 0.5) * 2 : t
                if (strength < 0.02) return null
                const p = point(sq, geometry)
                const fill = diverging ? (t < 0.5 ? 'var(--q-blunder)' : 'var(--q-best)') : 'var(--accent)'
                return (
                    <g key={sq}>
                        <rect
                            x={p.x}
                            y={p.y}
                            width={geometry.cell}
                            height={geometry.cell}
                            fill={fill}
                            opacity={0.1 + strength * 0.6}
                        />
                        {geometry.labels && strength >= 0.25 && (
                            <text
                                x={p.x + geometry.cell / 2}
                                y={p.y + geometry.cell - 5}
                                className="square-value"
                            >
                                {formatSquareValue(v, overlay)}
                            </text>
                        )}
                        <title>{`${overlay.legend || 'value'}: ${formatSquareValue(v, overlay)}`}</title>
                    </g>
                )
            })}
        </g>
    )
}
