export function forecastPiece(probs: number[]): { piece: number; probability: number } | null {
    const probability = Math.max(...probs)
    const piece = probs.indexOf(probability)
    return piece <= 0 ? null : { piece, probability }
}

export const probabilityHeight = (probability: number): string => `${Math.max(0, probability * 100)}%`
