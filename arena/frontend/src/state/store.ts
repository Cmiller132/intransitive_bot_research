import { create } from 'zustand'
import type { Frame } from '../api/types'
import type { Dialect } from '../rules/notation'
import { pieceSets, type PieceSetId } from '../board/pieces'

type Theme = 'light' | 'dark' | 'system'
type ArenaState = {
    theme: Theme
    frame: Frame
    dialect: Dialect
    pieceSet: PieceSetId
    adminToken: string
    overlayLabels: boolean
    setTheme: (value: Theme) => void
    setFrame: (value: Frame) => void
    setDialect: (value: Dialect) => void
    setPieceSet: (value: PieceSetId) => void
    setAdminToken: (value: string) => void
    setOverlayLabels: (value: boolean) => void
}

export const ADMIN_TOKEN_KEY = 'arena-admin-token'

const readString = (key: string): string => {
    try {
        return window.localStorage.getItem(key) || ''
    } catch {
        return ''
    }
}
const read = <T extends string>(key: string, fallback: T, allowed: readonly T[]): T => {
    const value = readString(key) as T
    return allowed.includes(value) ? value : fallback
}
const persist = <T>(key: string, value: T) => {
    try {
        if (value === '') window.localStorage.removeItem(key)
        else window.localStorage.setItem(key, String(value))
    } catch {
        // Preferences still work for this tab.
    }
    return value
}
const pieceSetIds = Object.keys(pieceSets) as PieceSetId[]

export const useArenaStore = create<ArenaState>((set) => ({
    theme: read('arena-theme', 'system', ['light', 'dark', 'system']),
    frame: read('arena-frame', 'meaf', ['meaf', 'henhen']),
    dialect: read('arena-dialect', 'meaf', ['meaf', 'henhen']),
    pieceSet: read('arena-piece-set', 'glyph', pieceSetIds),
    adminToken: readString(ADMIN_TOKEN_KEY),
    overlayLabels: read('arena-overlay-labels', 'on', ['on', 'off']) === 'on',
    setTheme: (theme) => set({ theme: persist('arena-theme', theme) }),
    setFrame: (frame) => set({ frame: persist('arena-frame', frame) }),
    setDialect: (dialect) => set({ dialect: persist('arena-dialect', dialect) }),
    setPieceSet: (pieceSet) => set({ pieceSet: persist('arena-piece-set', pieceSet) }),
    setAdminToken: (adminToken) => set({ adminToken: persist(ADMIN_TOKEN_KEY, adminToken.trim()) }),
    setOverlayLabels: (overlayLabels) =>
        set({ overlayLabels: persist('arena-overlay-labels', overlayLabels ? 'on' : 'off') === 'on' }),
}))
