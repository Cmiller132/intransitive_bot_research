import { useEffect, useRef } from 'react'
import uPlot from 'uplot'
import 'uplot/dist/uPlot.min.css'
import { useArenaStore } from '../state/store'
import { SERIES_TOKENS, cssColour } from './chartColours'

export type ChartSeries = { label: string; values: Array<number | null>; token?: string }

/**
 * One line per series over a shared numeric x axis. Used for rating against
 * checkpoint number and for a player's rating history.
 */
export function LineChart({
    x,
    series,
    height = 220,
    xLabel = '',
    yLabel = '',
    time = false,
}: {
    x: number[]
    series: ChartSeries[]
    height?: number
    xLabel?: string
    yLabel?: string
    time?: boolean
}) {
    const ref = useRef<HTMLDivElement>(null)
    const theme = useArenaStore((state) => state.theme)

    useEffect(() => {
        const host = ref.current
        if (!host || !x.length || !series.length) return
        const ink = cssColour('--ink-3', '#625e6d')
        const line = cssColour('--line', '#e4dad9')
        const plot = new uPlot(
            {
                width: Math.max(260, host.clientWidth),
                height,
                legend: { show: series.length > 1 },
                cursor: { show: true },
                scales: { x: { time } },
                axes: [
                    { stroke: ink, grid: { stroke: line }, ticks: { stroke: line }, label: xLabel },
                    { stroke: ink, grid: { stroke: line }, ticks: { stroke: line }, label: yLabel },
                ],
                series: [
                    {},
                    ...series.map((item, index) => {
                        const [token, fallback] = SERIES_TOKENS[index % SERIES_TOKENS.length]
                        return {
                            label: item.label,
                            stroke: cssColour(item.token || token, fallback),
                            width: 2,
                            points: { show: x.length < 40 },
                            spanGaps: true,
                        }
                    }),
                ],
            },
            [x, ...series.map((item) => item.values)] as unknown as uPlot.AlignedData,
            host,
        )
        const resize = () => {
            plot.setSize({ width: Math.max(260, host.clientWidth), height })
        }
        window.addEventListener('resize', resize)
        return () => {
            window.removeEventListener('resize', resize)
            plot.destroy()
        }
    }, [x, series, height, xLabel, yLabel, time, theme])

    if (!x.length || !series.length) return <p className="empty-copy compact">Not enough data to plot yet.</p>
    return <div className="plot" ref={ref} />
}
