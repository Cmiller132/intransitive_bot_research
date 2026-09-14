import type { PieceProps } from './types'
export function GlyphPiece({ type, side, size }: PieceProps) {
    const color = `var(--${side})`
    return (
        <svg width={size} height={size} viewBox="0 0 64 64" aria-hidden="true">
            <circle cx="32" cy="32" r="28" fill={color} stroke="rgba(27,26,36,.45)" strokeWidth="1.5" />
            {type === 'R' && (
                <path
                    d="M18 35c-2-10 6-18 16-17 10-1 16 7 13 17-2 8-10 12-17 11-7 0-11-5-12-11Z"
                    fill="#fff"
                />
            )}
            {type === 'P' && (
                <>
                    <path d="M20 16h18l8 8v25H20Z" fill="#fff" />
                    <path d="M38 16v9h8" fill="none" stroke="rgba(27,26,36,.38)" strokeWidth="2" />
                </>
            )}
            {type === 'S' && (
                <>
                    <path d="m21 18 23 27M43 18 20 44" stroke="#fff" strokeWidth="5" strokeLinecap="round" />
                    <circle cx="19" cy="47" r="6" fill="none" stroke="#fff" strokeWidth="4" />
                    <circle cx="45" cy="47" r="6" fill="none" stroke="#fff" strokeWidth="4" />
                </>
            )}
        </svg>
    )
}
