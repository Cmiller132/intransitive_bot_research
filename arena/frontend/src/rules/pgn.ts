import type { GameRecord, MoveRecord } from '../api/types.ts'
import { parseMove } from './notation.ts'
import { parseFen } from './fen.ts'

const clockMs = (value: string) => {
    const parts = value.split(':').map(Number)
    return Math.round(((parts[0] * 60 + parts[1]) * 60 + parts[2]) * 1000)
}
export const parseHenhenFen = (fen: string) => parseFen(fen, 'henhen')
export function parsePgn(source: string): GameRecord {
    const tags: Record<string, string> = {}
    for (const m of source.matchAll(/^\[([^\s]+)\s+"([^"]*)"\]$/gm)) tags[m[1]] = m[2]
    const body = source.replace(/^\[[^\n]+\]\s*$/gm, ' ')
    const tokens = [
        ...body.matchAll(/(?:\d+\.{1,3}\s*)?([RPS][a-i][1-9][-x][a-i][1-9]#?)(?:\s*\{([^}]*)\})?/gi),
    ]
    const moves: MoveRecord[] = tokens.map((match) => {
        const parsed = parseMove(match[1], 'henhen', 'henhen')
        const comment = match[2]
        const emt = /%emt\s+([\d.]+)/.exec(comment || '')
        const clk = /%clk\s+([^\s\]]+)\s+([^\s\]]+)/.exec(comment || '')
        return {
            from: parsed.from,
            to: parsed.to,
            capture: parsed.capture,
            san: match[1],
            emt_ms: emt ? Math.round(Number(emt[1]) * 1000) : undefined,
            clock_ms: clk ? { red: clockMs(clk[1]), blue: clockMs(clk[2]) } : undefined,
            comment,
            nags: [],
        }
    })
    return {
        id: tags.GameId || null,
        source: 'henhen',
        source_id: tags.GameId || null,
        source_url: tags.GameId ? `https://rps.henhen1227.com/review?gameId=${tags.GameId}` : null,
        frame: 'henhen',
        players: {
            blue: {
                name: tags.Blue || 'Blue',
                id: tags.BlueId,
                rating_before: Number(tags.BlueElo) || undefined,
                rating_after: Number(tags.BlueEloAfter) || undefined,
                kind: 'bot',
            },
            red: {
                name: tags.Red || 'Red',
                id: tags.RedId,
                rating_before: Number(tags.RedElo) || undefined,
                rating_after: Number(tags.RedEloAfter) || undefined,
                kind: 'bot',
            },
        },
        setup: tags.FEN ? parseHenhenFen(tags.FEN).board : null,
        moves,
        result: {
            winner: tags.Result === '1-0' ? 'red' : tags.Result === '0-1' ? 'blue' : null,
            reason: (tags.EndReason || 'unknown').toLowerCase(),
            reported: tags.Result || '*',
        },
        time_control: tags.TimeControl
            ? {
                  initial_ms: Number(tags.TimeControl.split('+')[0]) * 1000,
                  increment_ms: Number(tags.TimeControl.split('+')[1]) * 1000,
              }
            : null,
        tags,
        started_at: tags.UTCDate
            ? `${tags.UTCDate.replace(/\./g, '-')}T${tags.UTCTime || '00:00:00'}Z`
            : null,
        ended_at: null,
        meta: {
            opening_seed: tags.OpeningSeed,
            book_plies: Number(tags.BookPlies) || 0,
            series: tags.SeriesId,
            first_side: tags.FEN ? parseHenhenFen(tags.FEN).toMove : 'blue',
        },
    }
}
export const previewPgn = (source: string) => {
    const record = parsePgn(source)
    return {
        plies: record.moves.length,
        blue: record.players.blue.name,
        red: record.players.red.name,
        result: record.result.reported,
    }
}
