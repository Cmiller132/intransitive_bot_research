import type { Frame } from '../../api/types'
export type LayerGeometry = { frame: Frame; cell: number; margin: number; labels: boolean }
export const point = (absoluteSq: number, { frame, cell, margin }: LayerGeometry) => {
    const sq = frame === 'henhen' ? Math.floor(absoluteSq / 9) * 9 + 8 - (absoluteSq % 9) : absoluteSq
    return { x: margin + (sq % 9) * cell, y: margin + (8 - Math.floor(sq / 9)) * cell }
}
export const centre = (absoluteSq: number, geometry: LayerGeometry) => {
    const p = point(absoluteSq, geometry)
    return { x: p.x + geometry.cell / 2, y: p.y + geometry.cell / 2 }
}
