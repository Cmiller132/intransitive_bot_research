import { describe, expect, it } from 'vitest'
import type { EvalResponse, HeadSpec } from '../api/types'
import {
    DEFAULT_ANALYSIS_BUDGET,
    defaultAnalysisHeads,
    supportedAnalysisHeads,
    valueSourceLabel,
} from './analysisHeads'

const manifest: HeadSpec[] = [
    {
        id: 'policy',
        label: 'Policy',
        group: 'policy',
        kind: 'move_scalar',
        description: '',
        default_on: true,
        order: 1,
        params: {},
    },
    {
        id: 'value',
        label: 'Value',
        group: 'value',
        kind: 'scalar',
        description: '',
        default_on: true,
        order: 2,
        params: {},
    },
    {
        id: 'attention',
        label: 'Attention',
        group: 'internals',
        kind: 'square_vector',
        description: '',
        default_on: false,
        order: 3,
        params: {},
    },
]

describe('analysis head selection', () => {
    it('defaults to network-only evaluation and the engine manifest defaults', () => {
        expect(DEFAULT_ANALYSIS_BUDGET).toBe('quick')
        expect(defaultAnalysisHeads(manifest)).toEqual(['policy', 'value'])
    })

    it('never sends a stale unsupported head to a newly selected engine', () => {
        expect(supportedAnalysisHeads(['policy', 'atoms', 'attention'], manifest)).toEqual([
            'policy',
            'attention',
        ])
    })
})

const evaluation = (source: 'search' | 'network', label?: string): EvalResponse =>
    ({
        engine: 'e',
        position_key: 'k',
        symmetry_key: 'k',
        to_move: 'blue',
        legal: [],
        value: { blue: 0.1, mover: 0.1, source },
        search: { sims: source === 'search' ? 128 : 0, forwards: 0, ms: 1, lines: [] },
        heads: label ? { value: { kind: 'scalar', items: [{ label, value: 0.1 }] } } : {},
    }) as EvalResponse

describe('value source labelling', () => {
    it('names the head a model actually has', () => {
        expect(valueSourceLabel(evaluation('network', 'policy-weighted Q'))).toBe('policy-weighted Q')
        expect(valueSourceLabel(evaluation('network', 'state value head'))).toBe('state value head')
    })

    it('says when the number came from search instead', () => {
        expect(valueSourceLabel(evaluation('search', 'state value head'))).toBe(
            'search root value · state value head',
        )
        expect(valueSourceLabel(evaluation('search'))).toBe('search root value')
        expect(valueSourceLabel(evaluation('network'))).toBe('network value')
    })
})
