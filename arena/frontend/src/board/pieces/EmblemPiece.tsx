import type { PieceProps } from './types'
export function EmblemPiece({ type, side, size }: PieceProps) {
    const c = `var(--${side})`
    const line = 'var(--ink)'
    return (
        <svg width={size} height={size} viewBox="0 0 64 64" aria-hidden="true">
            {type === 'R' && (
                <path
                    d="M11 39c-2-17 10-28 25-25 15 1 23 15 16 29-5 10-17 12-27 8-7-2-11-6-14-12Z"
                    fill={c}
                    stroke={line}
                    strokeWidth="3"
                />
            )}
            {type === 'P' && (
                <>
                    <path d="M14 8h27l10 11v37H14Z" fill={c} stroke={line} strokeWidth="3" />
                    <path d="M41 8v12h10" fill="none" stroke={line} strokeWidth="3" />
                </>
            )}
            {type === 'S' && (
                <>
                    <path d="m18 10 29 37M47 10 17 46" stroke={line} strokeWidth="9" strokeLinecap="round" />
                    <path d="m18 10 29 37M47 10 17 46" stroke={c} strokeWidth="5" strokeLinecap="round" />
                    <circle cx="15" cy="50" r="7" fill={c} stroke={line} strokeWidth="3" />
                    <circle cx="49" cy="50" r="7" fill={c} stroke={line} strokeWidth="3" />
                </>
            )}
        </svg>
    )
}
