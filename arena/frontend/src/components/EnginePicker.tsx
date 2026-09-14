import { useEffect, useMemo, useRef, useState } from 'react'
import type { Engine } from '../api/types'
import { formatRating, groupByRun } from '../pages/runGrouping'

const matches = (engine: Engine, needle: string) =>
    !needle ||
    [engine.id, engine.label, engine.run, engine.spec, engine.kind]
        .filter(Boolean)
        .some((field) => field.toLowerCase().includes(needle))

/**
 * Searchable engine dropdown: runs collapse a pool of many checkpoints into a
 * handful of groups, each ordered by rating.
 */
export function EnginePicker({
    engines,
    value,
    onChange,
    label = 'Engine',
    placeholder = 'Search engines…',
    emptyOption,
}: {
    engines: Engine[]
    value: string
    onChange: (id: string) => void
    label?: string
    placeholder?: string
    /** Shown as the first choice, e.g. to unpin a comparison engine. */
    emptyOption?: { value: string; label: string }
}) {
    const [open, setOpen] = useState(false)
    const [needle, setNeedle] = useState('')
    const wrapper = useRef<HTMLDivElement>(null)
    const search = useRef<HTMLInputElement>(null)

    const groups = useMemo(() => {
        const filtered = engines.filter((engine) => matches(engine, needle.trim().toLowerCase()))
        return groupByRun(filtered, (engine) => engine.id)
    }, [engines, needle])
    const selected = engines.find((engine) => engine.id === value)

    useEffect(() => {
        if (open) search.current?.focus()
    }, [open])

    return (
        <div
            className={`engine-picker ${open ? 'open' : ''}`}
            ref={wrapper}
            onBlur={(event) => {
                if (!wrapper.current?.contains(event.relatedTarget as Node | null)) setOpen(false)
            }}
            onKeyDown={(event) => {
                if (event.key === 'Escape' && open) {
                    event.stopPropagation()
                    setOpen(false)
                }
            }}
        >
            <span className="picker-label">{label}</span>
            <button
                type="button"
                className="picker-trigger"
                aria-expanded={open}
                aria-haspopup="listbox"
                onClick={() => {
                    setNeedle('')
                    setOpen(!open)
                }}
            >
                <b>
                    {selected?.label ||
                        (value === emptyOption?.value ? emptyOption.label : value || 'Choose…')}
                </b>
                <span className="mono">
                    {selected ? formatRating(selected.rating, selected.half_width) : ''}
                </span>
                <i aria-hidden="true">▾</i>
            </button>
            {open && (
                <div className="picker-popover" role="listbox" aria-label={label}>
                    <input
                        ref={search}
                        className="picker-search"
                        value={needle}
                        placeholder={placeholder}
                        aria-label={placeholder}
                        onChange={(event) => setNeedle(event.target.value)}
                    />
                    <div className="picker-list">
                        {emptyOption && !needle && (
                            <button
                                type="button"
                                role="option"
                                aria-selected={value === emptyOption.value}
                                className={`picker-option ${value === emptyOption.value ? 'chosen' : ''}`}
                                onClick={() => {
                                    onChange(emptyOption.value)
                                    setOpen(false)
                                }}
                            >
                                <b>{emptyOption.label}</b>
                            </button>
                        )}
                        {groups.map((group) => (
                            <div className="picker-group" key={group.run}>
                                <span className="group-label">
                                    {group.run}
                                    <em className="mono">{formatRating(group.best)}</em>
                                </span>
                                {group.items.map((engine) => (
                                    <button
                                        type="button"
                                        role="option"
                                        aria-selected={engine.id === value}
                                        key={engine.id}
                                        className={`picker-option ${engine.id === value ? 'chosen' : ''}`}
                                        onClick={() => {
                                            onChange(engine.id)
                                            setOpen(false)
                                        }}
                                    >
                                        <b>{engine.label}</b>
                                        <span className="mono">
                                            {formatRating(engine.rating, engine.half_width)}
                                        </span>
                                        <em>
                                            {engine.kind}
                                            {engine.strongest ? ' · strongest' : ''}
                                            {engine.retired ? ' · retired' : ''}
                                            {engine.broken ? ' · broken' : ''}
                                        </em>
                                    </button>
                                ))}
                            </div>
                        ))}
                        {!groups.length && (
                            <p className="empty-copy compact">No engine matches “{needle}”.</p>
                        )}
                    </div>
                </div>
            )}
        </div>
    )
}
