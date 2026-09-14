import { describe, expect, it } from 'vitest'
import type { MoveScalarOverlay } from '../../api/types'
import { visibleMoves } from './MoveScalarLayer'

const policy: MoveScalarOverlay = {
    kind: 'move_scalar',
    render: 'arrows',
    scale: 'sequential',
    range: [0, 1],
    moves: [0.4, 0.25, 0.15, 0.08, 0.05, 0.04, 0.02, 0.01].map((value, i) => ({ from: i, to: i + 9, value })),
}

describe('visible overlay moves', () => {
    it('keeps the strongest policy moves until most of the mass is covered', () => {
        const shown = visibleMoves(policy)
        expect(shown.map((m) => m.value)).toEqual([0.4, 0.25, 0.15, 0.08])
    })
    it('arrows only the best Q move when labels print the rest', () => {
        const q: MoveScalarOverlay = { ...policy, render: 'tint', scale: 'diverging', range: [-1, 1] }
        expect(visibleMoves(q, true)).toHaveLength(1)
        expect(visibleMoves(q, false)).toHaveLength(3)
    })
})
