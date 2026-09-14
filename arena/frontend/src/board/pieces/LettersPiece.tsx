import type { PieceProps } from './types'
export function LettersPiece({ type, side, size }: PieceProps) {
    return (
        <svg width={size} height={size} viewBox="0 0 64 64" aria-hidden="true">
            <circle
                cx="32"
                cy="32"
                r="28"
                fill={`var(--${side})`}
                stroke="rgba(27,26,36,.5)"
                strokeWidth="2"
            />
            <text
                x="32"
                y="42"
                textAnchor="middle"
                fill="#fff"
                fontFamily="IBM Plex Mono, monospace"
                fontSize="29"
                fontWeight="600"
            >
                {type}
            </text>
        </svg>
    )
}
