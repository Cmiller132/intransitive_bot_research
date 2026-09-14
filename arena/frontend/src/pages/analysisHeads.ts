import type { Budget, EvalResponse, HeadSpec } from '../api/types'
import { KNOWN_OVERLAYS } from '../board/overlays/OverlayHost'

export const DEFAULT_ANALYSIS_BUDGET: Budget = 'quick'

export function defaultAnalysisHeads(heads: HeadSpec[]): string[] {
    return heads.filter((head) => head.default_on && KNOWN_OVERLAYS.has(head.kind)).map((head) => head.id)
}

export function supportedAnalysisHeads(active: string[], manifest: HeadSpec[]): string[] {
    const supported = new Set(manifest.filter((head) => KNOWN_OVERLAYS.has(head.kind)).map((head) => head.id))
    return active.filter((id) => supported.has(id))
}

/**
 * What the shown value came from. The scalar `value` head labels its state
 * entry "state value head" or "policy-weighted Q" (section 8), so the panel can
 * say which one a model actually has instead of the generic "network value".
 */
export function valueSourceLabel(evaluation: EvalResponse): string {
    const head = evaluation.heads?.value
    const item =
        head && head.kind === 'scalar'
            ? head.items.find((entry) => /state value head|policy-weighted/i.test(entry.label))
            : undefined
    if (evaluation.value.source === 'search')
        return item ? `search root value · ${item.label}` : 'search root value'
    return item ? item.label : 'network value'
}
