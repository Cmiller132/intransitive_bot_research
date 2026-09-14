import type { ScalarOverlay } from '../../api/types'
const display = (value: number, format?: string) =>
    format === 'percent'
        ? `${Math.round(value * 100)}%`
        : format === 'signed'
          ? `${value >= 0 ? '+' : ''}${value.toFixed(2)}`
          : Number.isInteger(value)
            ? String(value)
            : value.toFixed(1)
export function ScalarPanel({ overlay }: { overlay: ScalarOverlay }) {
    return (
        <div className="scalar-list">
            {overlay.items.map((item) => (
                <div key={item.label}>
                    <span>{item.label}</span>
                    <strong className="mono">
                        {display(item.value, item.format)}
                        {item.unit ? ` ${item.unit}` : ''}
                    </strong>
                </div>
            ))}
        </div>
    )
}
