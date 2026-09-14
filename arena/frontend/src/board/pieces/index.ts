import { EmblemPiece } from './EmblemPiece'
import { GlyphPiece } from './GlyphPiece'
import { HandsPiece } from './HandsPiece'
import { LettersPiece } from './LettersPiece'
import type { PieceComponent, PieceSetId } from './types'
export const pieceSets: Record<PieceSetId, PieceComponent> = {
    glyph: GlyphPiece,
    hands: HandsPiece,
    emblem: EmblemPiece,
    letters: LettersPiece,
}
export type { PieceProps, PieceSetId } from './types'
