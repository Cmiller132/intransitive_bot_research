export function Logo({ size = 34 }: { size?: number }) {
    return (
        <svg className="logo" width={size} height={size} viewBox="0 0 72 72" aria-hidden="true">
            <defs>
                <marker
                    id="logo-arrow"
                    viewBox="0 0 8 8"
                    refX="6"
                    refY="4"
                    markerWidth="6"
                    markerHeight="6"
                    orient="auto"
                >
                    <path d="M0 0L8 4L0 8Z" fill="var(--ink)" />
                </marker>
            </defs>
            <path
                d="M36 12A24 24 0 0 1 56.8 48"
                fill="none"
                stroke="var(--ink)"
                strokeWidth="2.2"
                markerEnd="url(#logo-arrow)"
            />
            <path
                d="M56.8 48A24 24 0 0 1 15.2 48"
                fill="none"
                stroke="var(--ink)"
                strokeWidth="2.2"
                markerEnd="url(#logo-arrow)"
            />
            <path
                d="M15.2 48A24 24 0 0 1 36 12"
                fill="none"
                stroke="var(--ink)"
                strokeWidth="2.2"
                markerEnd="url(#logo-arrow)"
            />
            <circle cx="36" cy="12" r="7" fill="var(--blue)" />
            <circle cx="56.8" cy="48" r="7" fill="var(--red)" />
            <circle cx="15.2" cy="48" r="7" fill="var(--mint)" />
            <text x="36" y="15" textAnchor="middle" fill="#fff" fontSize="8" fontFamily="IBM Plex Mono">
                R
            </text>
            <text x="56.8" y="51" textAnchor="middle" fill="#fff" fontSize="8" fontFamily="IBM Plex Mono">
                S
            </text>
            <text x="15.2" y="51" textAnchor="middle" fill="#1B1A24" fontSize="8" fontFamily="IBM Plex Mono">
                P
            </text>
        </svg>
    )
}
