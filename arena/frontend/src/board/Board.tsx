import { useState } from 'react'
import type { Frame, Move, Overlay } from '../api/types'
import { sideOf, typeOf } from '../rules/game'
import { fromFrame, squareName } from '../rules/frames'
import { Arrow } from './overlays/Arrow'
import { OverlayHost } from './overlays/OverlayHost'
import { centre } from './overlays/types'
import { pieceSets, type PieceSetId } from './pieces'

export type BoardProps = {
    board: number[]
    frame: Frame
    pieceSet: PieceSetId
    lastMove?: Move
    legal?: Move[]
    selected?: number
    overlays: Overlay[]
    bestMove?: Move
    labels?: boolean
    onMove?: (move: Move) => void
    orientationLabel: string
    homeTint: boolean
}
const CELL = 52
const M = 22
const SIZE = CELL * 9 + M * 2
const PIECE_NAMES = ['', 'rock', 'paper', 'scissors']
export function Board({
    board,
    frame,
    pieceSet,
    lastMove,
    legal = [],
    selected,
    overlays,
    bestMove,
    labels = true,
    onMove,
    orientationLabel,
    homeTint,
}: BoardProps) {
    const [ownSelected, setOwnSelected] = useState<number>()
    const active = selected ?? ownSelected
    const [dragFrom, setDragFrom] = useState<number>()
    const Piece = pieceSets[pieceSet]
    const targets = active === undefined ? [] : legal.filter((m) => m.from === active).map((m) => m.to)
    const activate = (sq: number) => {
        if (active !== undefined && targets.includes(sq)) {
            onMove?.({ from: active, to: sq })
            setOwnSelected(undefined)
        } else if (legal.some((m) => m.from === sq)) setOwnSelected(sq)
        else setOwnSelected(undefined)
    }
    return (
        <div className="board-shell">
            <svg
                className="game-board"
                viewBox={`0 0 ${SIZE} ${SIZE}`}
                role="grid"
                aria-label={`Intransitive board, ${orientationLabel}`}
            >
                {Array.from({ length: 81 }, (_, shown) => {
                    const file = shown % 9
                    const rank = Math.floor(shown / 9)
                    const x = M + file * CELL
                    const y = M + (8 - rank) * CELL
                    const sq = fromFrame(shown, frame)
                    const isHome = sq === 0 || sq === 80
                    const highlighted = lastMove && (sq === lastMove.from || sq === lastMove.to)
                    return (
                        <rect
                            key={`bg-${shown}`}
                            x={x}
                            y={y}
                            width={CELL}
                            height={CELL}
                            fill={
                                homeTint && isHome
                                    ? sq === 0
                                        ? 'var(--blue-soft)'
                                        : 'var(--red-soft)'
                                    : (file + rank) % 2
                                      ? 'var(--mint-light)'
                                      : 'var(--mint)'
                            }
                            stroke="rgba(27,26,36,.14)"
                            strokeWidth="1"
                            className={highlighted ? 'last-square' : ''}
                        />
                    )
                })}
                <OverlayHost overlays={overlays} frame={frame} cell={CELL} margin={M} labels={labels} />
                {bestMove &&
                    (() => {
                        const geometry = { frame, cell: CELL, margin: M, labels }
                        const from = centre(bestMove.from, geometry)
                        const to = centre(bestMove.to, geometry)
                        return (
                            <Arrow
                                x1={from.x}
                                y1={from.y}
                                x2={to.x}
                                y2={to.y}
                                width={7}
                                color="var(--accent)"
                                label={labels ? 'best' : undefined}
                                cell={CELL}
                            />
                        )
                    })()}
                {lastMove && (
                    <>
                        {[lastMove.from, lastMove.to].map((absolute) => {
                            const shown =
                                frame === 'henhen'
                                    ? Math.floor(absolute / 9) * 9 + 8 - (absolute % 9)
                                    : absolute
                            return (
                                <rect
                                    key={absolute}
                                    x={M + (shown % 9) * CELL + 2}
                                    y={M + (8 - Math.floor(shown / 9)) * CELL + 2}
                                    width={CELL - 4}
                                    height={CELL - 4}
                                    rx="5"
                                    fill="none"
                                    stroke="var(--accent)"
                                    strokeWidth="3"
                                    opacity=".75"
                                />
                            )
                        })}
                    </>
                )}
                {board.map((piece, absolute) => {
                    if (!piece) return null
                    const shown =
                        frame === 'henhen' ? Math.floor(absolute / 9) * 9 + 8 - (absolute % 9) : absolute
                    const x = M + (shown % 9) * CELL + 7
                    const y = M + (8 - Math.floor(shown / 9)) * CELL + 7
                    return (
                        <g
                            key={`piece-${absolute}`}
                            transform={`translate(${x} ${y})`}
                            className="board-piece"
                        >
                            <Piece
                                type={['R', 'P', 'S'][(typeOf(piece) || 1) - 1] as 'R' | 'P' | 'S'}
                                side={sideOf(piece)!}
                                size={38}
                            />
                        </g>
                    )
                })}
                {active !== undefined &&
                    (() => {
                        const shown =
                            frame === 'henhen' ? Math.floor(active / 9) * 9 + 8 - (active % 9) : active
                        return (
                            <rect
                                x={M + (shown % 9) * CELL + 2}
                                y={M + (8 - Math.floor(shown / 9)) * CELL + 2}
                                width={CELL - 4}
                                height={CELL - 4}
                                rx="6"
                                fill="none"
                                stroke="var(--ink)"
                                strokeWidth="3"
                            />
                        )
                    })()}
                {targets.map((absolute) => {
                    const shown =
                        frame === 'henhen' ? Math.floor(absolute / 9) * 9 + 8 - (absolute % 9) : absolute
                    const cx = M + (shown % 9) * CELL + CELL / 2
                    const cy = M + (8 - Math.floor(shown / 9)) * CELL + CELL / 2
                    return board[absolute] ? (
                        <g key={`target-${absolute}`} className="capture-mark">
                            <circle
                                cx={cx}
                                cy={cy}
                                r={CELL * 0.38}
                                fill="none"
                                stroke="var(--ink)"
                                strokeWidth="3"
                                strokeDasharray="4 3"
                            />
                            <path
                                d={`M${cx - 6} ${cy - 6}l12 12m0-12-12 12`}
                                stroke="var(--ink)"
                                strokeWidth="2"
                            />
                        </g>
                    ) : (
                        <circle
                            key={`target-${absolute}`}
                            cx={cx}
                            cy={cy}
                            r="6"
                            fill="var(--ink)"
                            opacity=".58"
                        />
                    )
                })}
                {Array.from({ length: 81 }, (_, shown) => {
                    const file = shown % 9
                    const rank = Math.floor(shown / 9)
                    const sq = fromFrame(shown, frame)
                    const piece = board[sq]
                    const isOrigin = legal.some((move) => move.from === sq)
                    const isTarget = targets.includes(sq)
                    const state =
                        active === sq
                            ? ', selected'
                            : isTarget
                              ? ', legal destination'
                              : isOrigin
                                ? ', movable'
                                : ''
                    const pieceLabel = piece
                        ? `, ${sideOf(piece)} ${PIECE_NAMES[typeOf(piece) || 0]}`
                        : ', empty'
                    return (
                        <rect
                            key={`hit-${shown}`}
                            x={M + file * CELL}
                            y={M + (8 - rank) * CELL}
                            width={CELL}
                            height={CELL}
                            fill="transparent"
                            role="gridcell"
                            tabIndex={onMove && (isOrigin || isTarget) ? 0 : -1}
                            aria-selected={active === sq || undefined}
                            aria-label={`${squareName(sq, frame)}${pieceLabel}${state}`}
                            onClick={() => activate(sq)}
                            onPointerDown={() => setDragFrom(sq)}
                            onPointerUp={() => {
                                if (
                                    dragFrom !== undefined &&
                                    legal.some((m) => m.from === dragFrom && m.to === sq)
                                )
                                    onMove?.({ from: dragFrom, to: sq })
                                setDragFrom(undefined)
                            }}
                            onKeyDown={(e) => {
                                if (e.key === 'Enter' || e.key === ' ') {
                                    e.preventDefault()
                                    activate(sq)
                                }
                            }}
                        />
                    )
                })}
                {Array.from({ length: 9 }, (_, i) => (
                    <text
                        key={`file-${i}`}
                        x={M + i * CELL + CELL / 2}
                        y={SIZE - 6}
                        textAnchor="middle"
                        className="coord"
                    >
                        {'abcdefghi'[i]}
                    </text>
                ))}
                {Array.from({ length: 9 }, (_, i) => (
                    <text key={`rank-${i}`} x="8" y={M + (8 - i) * CELL + CELL / 2 + 4} className="coord">
                        {i + 1}
                    </text>
                ))}
            </svg>
        </div>
    )
}
