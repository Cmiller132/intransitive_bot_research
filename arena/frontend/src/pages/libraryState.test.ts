import { describe, expect, it } from 'vitest'
import { libraryPageCount, libraryRange } from './libraryState'

describe('library pagination', () => {
    it('uses the API total rather than the current page length', () => {
        expect(libraryPageCount(100)).toBe(5)
        expect(libraryRange(1, 20, 100)).toBe('21–40 of 100')
    })

    it('has a readable empty range', () => {
        expect(libraryPageCount(0)).toBe(1)
        expect(libraryRange(0, 0, 0)).toBe('0 games')
    })
})
