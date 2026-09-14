/** A board arrow drawn as plain shapes, so no marker ids are needed. */
export function Arrow({
    x1,
    y1,
    x2,
    y2,
    width,
    color,
    opacity = 1,
    label,
    cell,
}: {
    x1: number
    y1: number
    x2: number
    y2: number
    width: number
    color: string
    opacity?: number
    label?: string
    cell: number
}) {
    const dx = x2 - x1
    const dy = y2 - y1
    const length = Math.hypot(dx, dy) || 1
    const ux = dx / length
    const uy = dy / length
    const head = Math.max(10, width * 2.2)
    // Stop the shaft short of the tip so the head is not drawn over it.
    const sx = x2 - ux * head
    const sy = y2 - uy * head
    const px = -uy
    const py = ux
    const tip = `${x2},${y2} ${sx + px * head * 0.55},${sy + py * head * 0.55} ${sx - px * head * 0.55},${sy - py * head * 0.55}`
    const outline = 'var(--overlay-outline)'
    const labelX = x2 - ux * cell * 0.36
    const labelY = y2 - uy * cell * 0.36
    const labelWidth = label ? 8 + label.length * 6.6 : 0
    return (
        <g opacity={opacity} pointerEvents="none">
            <line
                x1={x1}
                y1={y1}
                x2={sx}
                y2={sy}
                stroke={outline}
                strokeWidth={width + 3.5}
                strokeLinecap="round"
            />
            <polygon points={tip} fill={outline} stroke={outline} strokeWidth="3" strokeLinejoin="round" />
            <line x1={x1} y1={y1} x2={sx} y2={sy} stroke={color} strokeWidth={width} strokeLinecap="round" />
            <polygon points={tip} fill={color} />
            {label && (
                <g transform={`translate(${labelX} ${labelY})`} className="arrow-label">
                    <rect
                        x={-labelWidth / 2}
                        y={-9}
                        width={labelWidth}
                        height={18}
                        rx={9}
                        fill="var(--surface)"
                        stroke={color}
                        strokeWidth="1.5"
                    />
                    <text textAnchor="middle" y={4}>
                        {label}
                    </text>
                </g>
            )}
        </g>
    )
}
