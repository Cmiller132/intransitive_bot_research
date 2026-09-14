import { describe, expect, it } from 'vitest'
import { forecastPiece, probabilityHeight } from './semantics'

describe('overlay semantics', () => {
    it('does not draw a ghost piece when empty is the most likely forecast', () => {
        expect(forecastPiece([0.72, 0.08, 0.07, 0.05, 0.04, 0.02, 0.02])).toBeNull()
        expect(forecastPiece([0.1, 0.6, 0.1, 0.05, 0.05, 0.05, 0.05])).toEqual({ piece: 1, probability: 0.6 })
    })

    it('renders a zero-probability histogram bin at zero height', () => {
        expect(probabilityHeight(0)).toBe('0%')
        expect(probabilityHeight(0.25)).toBe('25%')
    })
})
