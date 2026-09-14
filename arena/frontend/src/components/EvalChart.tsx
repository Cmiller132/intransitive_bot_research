import { useEffect, useRef } from 'react'
import uPlot from 'uplot'
import 'uplot/dist/uPlot.min.css'
import { cssColour } from './chartColours'
import { useArenaStore } from '../state/store'

export function EvalChart({
    values,
    cursor = values.length - 1,
    label = 'Evaluation',
}: {
    values: number[]
    cursor?: number
    label?: string
}) {
    const ref = useRef<HTMLDivElement>(null)
    const theme = useArenaStore((state) => state.theme)
    useEffect(() => {
        if (!ref.current) return
        const width = Math.max(240, ref.current.clientWidth)
        const ink = cssColour('--ink-3', '#625e6d')
        const line = cssColour('--line', '#e4dad9')
        const blue = cssColour('--blue', '#172fbe')
        const plot = new uPlot(
            {
                width,
                height: 140,
                cursor: { show: true },
                scales: { x: { time: false }, y: { range: [-1, 1] } },
                axes: [
                    { stroke: ink, grid: { stroke: line } },
                    { stroke: ink, grid: { stroke: line } },
                ],
                series: [
                    {},
                    {
                        label,
                        stroke: blue,
                        width: 2,
                        fill: `color-mix(in srgb, ${blue} 13%, transparent)`,
                    },
                ],
            },
            [values.map((_, i) => i), values],
            ref.current,
        )
        plot.setCursor({ left: (cursor / Math.max(1, values.length - 1)) * width, top: 70 })
        return () => plot.destroy()
    }, [values, cursor, label, theme])
    return <div className="plot" ref={ref} />
}
