import type { MoveScalarOverlay } from '../../api/types'
import { Arrow } from './Arrow'
import type { LayerGeometry } from './types'
import { centre, point } from './types'

type MoveValue = MoveScalarOverlay['moves'][number]

const MAX_ARROWS = 6
const MIN_ARROWS = 3
const MASS_SHOWN = 0.85
const TINT_ARROWS = 3

/**
 * The moves an overlay draws as arrows. A policy over 36 legal moves is
 * unreadable as 36 arrows, so `arrows` overlays keep the strongest moves until
 * they cover most of the probability mass; `tint` overlays (Q) colour every
 * destination square and only arrow the best few moves.
 */
export function visibleMoves(overlay: MoveScalarOverlay, labels = true): MoveValue[] {
    const sorted = [...overlay.moves].sort((a, b) => b.value - a.value)
    if (overlay.render === 'tint') return sorted.slice(0, labels ? 1 : TINT_ARROWS)
    const total = sorted.reduce((sum, move) => sum + Math.max(0, move.value), 0)
    const shown: MoveValue[] = []
    let mass = 0
    for (const move of sorted) {
        if (shown.length >= MAX_ARROWS) break
        if (shown.length >= MIN_ARROWS && (mass >= MASS_SHOWN * total || move.value <= 0)) break
        shown.push(move)
        mass += Math.max(0, move.value)
    }
    return shown
}

export const signed = (value: number, digits = 2) => {
    const rounded = Number(value.toFixed(digits))
    return rounded === 0 ? (0).toFixed(digits) : `${rounded > 0 ? '+' : ''}${rounded.toFixed(digits)}`
}

export function MoveScalarLayer({
    overlay,
    geometry,
}: {
    overlay: MoveScalarOverlay
    geometry: LayerGeometry
}) {
    const tint = overlay.render === 'tint'
    const max = Math.max(Math.abs(overlay.range[0]), Math.abs(overlay.range[1]), 0.001)
    const arrows = visibleMoves(overlay, geometry.labels)

    // Q tint: the best value among the moves that land on each square.
    const squareValues = new Map<number, number>()
    if (tint) {
        for (const move of overlay.moves) {
            const current = squareValues.get(move.to)
            if (current === undefined || move.value > current) squareValues.set(move.to, move.value)
        }
    }

    return (
        <g className="overlay-move-scalar" pointerEvents="none">
            {[...squareValues].map(([sq, value]) => {
                const p = point(sq, geometry)
                const strength = Math.abs(value) / max
                return (
                    <g key={`tint-${sq}`}>
                        <rect
                            x={p.x}
                            y={p.y}
                            width={geometry.cell}
                            height={geometry.cell}
                            fill={value >= 0 ? 'var(--q-best)' : 'var(--q-blunder)'}
                            opacity={0.08 + strength * 0.62}
                        />
                        {geometry.labels && (
                            <text
                                x={p.x + geometry.cell / 2}
                                y={p.y + geometry.cell - 5}
                                className="square-value"
                            >
                                {signed(value)}
                            </text>
                        )}
                        <title>{`best Q landing here ${signed(value)}`}</title>
                    </g>
                )
            })}
            {arrows.map((move, index) => {
                const from = centre(move.from, geometry)
                const to = centre(move.to, geometry)
                const strength = Math.abs(move.value) / max
                const best = index === 0
                const width = best ? 7 : 3 + strength * 3
                const color = best ? 'var(--accent)' : tint ? 'var(--q-good)' : 'var(--overlay-policy)'
                const label = !geometry.labels
                    ? undefined
                    : tint
                      ? signed(move.value)
                      : `${Math.round(move.value * 100)}%`
                return (
                    <Arrow
                        key={`${move.from}-${move.to}`}
                        x1={from.x}
                        y1={from.y}
                        x2={to.x}
                        y2={to.y}
                        width={width}
                        color={color}
                        opacity={best ? 1 : 0.7 + strength * 0.3}
                        label={label}
                        cell={geometry.cell}
                    />
                )
            })}
        </g>
    )
}
