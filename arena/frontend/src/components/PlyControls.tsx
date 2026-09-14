/** First / previous / slider / next / last controls for stepping through a game. */
export function PlyControls({
    ply,
    max,
    onChange,
}: {
    ply: number
    max: number
    onChange: (ply: number) => void
}) {
    const go = (next: number) => onChange(Math.max(0, Math.min(max, next)))
    return (
        <div className="ply-controls" role="group" aria-label="Move navigation">
            <button type="button" onClick={() => go(0)} disabled={ply === 0} aria-label="Start position">
                «
            </button>
            <button type="button" onClick={() => go(ply - 1)} disabled={ply === 0} aria-label="Previous ply">
                ‹
            </button>
            <input
                type="range"
                min={0}
                max={max}
                value={ply}
                disabled={max === 0}
                aria-label="Ply"
                onChange={(event) => go(Number(event.target.value))}
            />
            <button type="button" onClick={() => go(ply + 1)} disabled={ply >= max} aria-label="Next ply">
                ›
            </button>
            <button type="button" onClick={() => go(max)} disabled={ply >= max} aria-label="Final position">
                »
            </button>
            <span className="mono">
                {ply} / {max}
            </span>
        </div>
    )
}
