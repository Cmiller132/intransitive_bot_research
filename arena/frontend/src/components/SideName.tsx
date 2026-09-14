import type { Side } from '../api/types'

/**
 * A player's name in the colour of the side it is playing, with a filled
 * swatch so the pairing is readable without relying on colour alone. `turn`
 * reserves a dot in front of the name; it is filled for the side to move.
 */
export function SideName({
    side,
    name,
    turn,
    tag,
    className = '',
}: {
    side: Side
    name: string
    turn?: boolean | undefined
    tag?: string
    className?: string
}) {
    const toMove = turn === true
    return (
        <span className={`side-name side-${side}${toMove ? ' to-move' : ''} ${className}`.trim()}>
            {turn === undefined ? null : (
                <i className={`side-turn${toMove ? ' on' : ''}`} aria-hidden="true" />
            )}
            <i className="side-swatch" aria-hidden="true" />
            <span className="side-label">{name}</span>
            {tag ? <em className="side-tag">{tag}</em> : null}
            <span className="sr-only">
                {side === 'blue' ? ' playing Blue' : ' playing Red'}
                {toMove ? ', to move' : ''}
            </span>
        </span>
    )
}
