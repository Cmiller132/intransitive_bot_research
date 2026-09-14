import { ImportForm } from '../components/ImportForm'
export function ImportPage() {
    return (
        <main className="page narrow-page">
            <span className="eyebrow">Eight sources · one absolute frame</span>
            <h1>Import a game</h1>
            <p className="lede">
                Paste a record or choose a local file. PGN records are previewed on this device before they
                are sent.
            </p>
            <div className="double-rule" />
            <ImportForm />
        </main>
    )
}
