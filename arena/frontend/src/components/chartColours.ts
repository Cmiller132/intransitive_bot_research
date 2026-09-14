/** uPlot draws on canvas, where `var(--token)` means nothing: resolve it first. */
export function cssColour(token: string, fallback: string): string {
    if (typeof window === 'undefined') return fallback
    const value = getComputedStyle(document.documentElement).getPropertyValue(token).trim()
    return value || fallback
}

export const SERIES_TOKENS = [
    ['--accent', '#287668'],
    ['--blue', '#172fbe'],
    ['--red', '#a92e26'],
    ['--q-inacc', '#9a6711'],
    ['--q-blunder', '#7b2e88'],
    ['--mint', '#a3d3b3'],
] as const
