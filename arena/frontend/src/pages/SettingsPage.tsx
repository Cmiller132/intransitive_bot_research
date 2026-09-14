import { pieceSets, type PieceSetId } from '../board/pieces'
import { useArenaStore } from '../state/store'
import type { Frame } from '../api/types'
import type { Dialect } from '../rules/notation'
const ids = Object.keys(pieceSets) as PieceSetId[]
export function SettingsPage() {
    const store = useArenaStore()
    return (
        <main className="page">
            <span className="eyebrow">Local preferences</span>
            <h1>Settings</h1>
            <p className="lede">
                Theme, board orientation, notation, pieces and the admin token stay on this device.
            </p>
            <div className="double-rule" />
            <section className="settings-grid">
                <div className="panel">
                    <h2>Display</h2>
                    <label>
                        Theme
                        <select
                            value={store.theme}
                            onChange={(e) => store.setTheme(e.target.value as 'light' | 'dark' | 'system')}
                        >
                            <option value="system">System</option>
                            <option value="light">Light</option>
                            <option value="dark">Dark</option>
                        </select>
                    </label>
                    <label>
                        Default frame
                        <select value={store.frame} onChange={(e) => store.setFrame(e.target.value as Frame)}>
                            <option value="meaf">meaf · Blue home a1</option>
                            <option value="henhen">henhen · Blue home i1</option>
                        </select>
                    </label>
                    <label>
                        Notation
                        <select
                            value={store.dialect}
                            onChange={(e) => store.setDialect(e.target.value as Dialect)}
                        >
                            <option value="meaf">meaf · E3-F3</option>
                            <option value="henhen">henhen · Se3-f4</option>
                        </select>
                    </label>
                </div>
                <div className="panel keyboard-card">
                    <h2>Keyboard</h2>
                    <div>
                        <div>
                            <span>
                                <kbd>←</kbd>
                                <kbd>→</kbd>
                            </span>
                            <span>Step plies</span>
                        </div>
                        <div>
                            <span>
                                <kbd>↑</kbd>
                                <kbd>↓</kbd>
                            </span>
                            <span>Start / end</span>
                        </div>
                        <div>
                            <span>
                                <kbd>F</kbd>
                            </span>
                            <span>Flip frame</span>
                        </div>
                        <div>
                            <span>
                                <kbd>1–9</kbd>
                            </span>
                            <span>Toggle heads</span>
                        </div>
                    </div>
                </div>
                <div className="panel">
                    <h2>Arena admin</h2>
                    <label>
                        Admin token
                        <input
                            type="password"
                            autoComplete="off"
                            value={store.adminToken}
                            placeholder="ARENA_ADMIN_TOKEN"
                            onChange={(e) => store.setAdminToken(e.target.value)}
                        />
                    </label>
                    <p className="section-intro">
                        Needed to upload or retire players and to start deep reviews while the site is public.
                        It stays in this browser and is only sent with requests that change data.
                    </p>
                </div>
            </section>
            <div className="double-rule" />
            <section>
                <span className="eyebrow">Shape-safe at every size</span>
                <h2>Piece set</h2>
                <p className="lede">Each family is shown at 24, 40, and 64 px on mint in both themes.</p>
                <div className="piece-gallery">
                    {ids.map((id) => {
                        const Piece = pieceSets[id]
                        return (
                            <article
                                className={`piece-card ${store.pieceSet === id ? 'chosen' : ''}`}
                                key={id}
                            >
                                <div className="piece-card-head">
                                    <h3>{id}</h3>
                                    <button
                                        className={store.pieceSet === id ? 'selected-control' : ''}
                                        onClick={() => store.setPieceSet(id)}
                                    >
                                        {store.pieceSet === id ? '✓ In use' : 'Use this set'}
                                    </button>
                                </div>
                                {(['light', 'dark'] as const).map((theme) => (
                                    <div className={`theme-preview ${theme}`} key={theme}>
                                        <span>{theme}</span>
                                        {[24, 40, 64].map((size) => (
                                            <div className="piece-size" key={size}>
                                                {(['R', 'P', 'S'] as const).map((type, i) => (
                                                    <Piece
                                                        key={type}
                                                        type={type}
                                                        side={i % 2 ? 'red' : 'blue'}
                                                        size={size}
                                                    />
                                                ))}
                                            </div>
                                        ))}
                                    </div>
                                ))}
                            </article>
                        )
                    })}
                </div>
            </section>
        </main>
    )
}
