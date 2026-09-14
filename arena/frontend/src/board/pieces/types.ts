import type { PieceType, Side } from '../../api/types'
export type PieceProps = { type: PieceType; side: Side; size: number }
export type PieceSetId = 'glyph' | 'hands' | 'emblem' | 'letters'
export type PieceComponent = (props: PieceProps) => React.ReactNode
