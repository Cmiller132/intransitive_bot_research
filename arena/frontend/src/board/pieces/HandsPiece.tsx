import type { PieceProps } from './types'
export function HandsPiece({ type, side, size }: PieceProps) {
    const c = `var(--${side})`
    return (
        <svg width={size} height={size} viewBox="0 0 64 64" aria-hidden="true">
            <circle cx="32" cy="32" r="28" fill={c} />
            <g fill="#fff" stroke={c} strokeWidth="2.4" strokeLinejoin="round" strokeLinecap="round">
                {type === 'R' && (
                    <path d="M19 31c0-5 4-8 8-5 0-5 6-7 9-3 3-3 8 0 7 4 5-1 7 5 4 9l-8 11H25c-5-4-7-9-6-16Z" />
                )}
                {type === 'P' && (
                    <path d="M20 37V22c0-5 6-5 6 0v8-12c0-5 6-5 6 0v11-13c0-5 6-5 6 0v14-10c0-5 6-5 6 0v19l-8 10H25Z" />
                )}
                {type === 'S' && (
                    <path d="M21 43c-2-4-1-8 2-11l-2-12c-1-5 6-6 7-1l3 11 5-14c2-5 8-3 6 2l-4 15 5-3c4-2 7 3 4 6l-10 12H27Z" />
                )}
            </g>
        </svg>
    )
}
