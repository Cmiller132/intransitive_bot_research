import type { Frame } from '../api/types.ts'
const FILES = 'abcdefghi'
export const toFrame = (sq: number, frame: Frame): number =>
    frame === 'henhen' ? Math.floor(sq / 9) * 9 + 8 - (sq % 9) : sq
export const fromFrame = toFrame
export const squareName = (sq: number, frame: Frame = 'meaf'): string => {
    const shown = toFrame(sq, frame)
    return `${FILES[shown % 9]}${Math.floor(shown / 9) + 1}`
}
export const parseSquare = (name: string, frame: Frame = 'meaf'): number => {
    const m = /^([a-i])([1-9])$/i.exec(name.trim())
    if (!m) throw new Error(`Invalid square: ${name}`)
    return fromFrame((Number(m[2]) - 1) * 9 + FILES.indexOf(m[1].toLowerCase()), frame)
}
