use super::*;

const V8: &[u8] = include_bytes!("../../tests/fixtures/format8_h32.nnue");
const V6: &[u8] = include_bytes!("../../tests/fixtures/format6_h32.nnue");

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Frame {
    game: usize,
    ply: u32,
    board: Vec<u8>,
    since_capture: u32,
    clock: u32,
    outcome: u8,
    legal: usize,
    action: Option<u16>,
    context: [usize; 2],
    ids: [Vec<usize>; 2],
    raw: f64,
    raw6: f64,
}
impl Frame {
    fn state(&self) -> State {
        State {
            board: engine::from_codes(&self.board).unwrap(),
            since_capture: self.since_capture,
            ply: self.ply,
        }
    }
}
fn frames() -> Vec<Frame> {
    include_str!("../../tests/fixtures/format8_trajectories.jsonl")
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}
fn prepared(model: &Model, state: &State, clock: u32) -> Searcher {
    let mut search = Searcher::new(1);
    search.acc = (0..MAX_PLY + 2)
        .map(|_| Accumulator::empty(model.hidden))
        .collect();
    model.refresh_into(&state.board, state.since_capture, clock, &mut search.acc[0]);
    search.hashes[0] = [
        board_hash(&state.board),
        board_hash(&engine::flip(&state.board)),
    ];
    search.metrics = Some(AccumulatorMetrics::default());
    search
}
fn oracle(model: &Model, avx2: &Model, acc: &Accumulator, frame: &Frame) {
    let state = frame.state();
    assert_eq!(
        *acc,
        model.refresh(&state.board, state.since_capture, frame.clock)
    );
    assert_eq!(acc.contexts(), frame.context);
    let expected = if model.features == 1004 {
        frame.raw6
    } else {
        frame.raw
    };
    assert!((model.raw(acc) - expected).abs() <= 1e-12);
    assert_eq!(model.raw(acc), model.raw_scalar(acc));
    assert_eq!(model.raw(acc), avx2.raw(acc));
    for (side, ids) in model
        .feature_ids(&state.board, state.since_capture, frame.clock)
        .iter()
        .enumerate()
    {
        let mut actual: Vec<_> = ids
            .iter()
            .copied()
            .filter(|&id| id != model.features)
            .map(|id| {
                if model.features == 1004 {
                    if id < 486 {
                        id + frame.context[side] * 486
                    } else {
                        id + 12636
                    }
                } else {
                    id
                }
            })
            .collect();
        let mut expected: Vec<_> = frame.ids[side]
            .iter()
            .copied()
            .filter(|&id| id != 13640)
            .collect();
        actual.sort_unstable();
        expected.sort_unstable();
        assert_eq!(actual, expected);
    }
}

#[test]
fn shared_trajectories_replay_lazy_chains_and_every_legal_count_transition() {
    let frames = frames();
    assert_eq!(frames.len(), 2996);
    for (bytes, scalar) in [V6, V8]
        .into_iter()
        .flat_map(|bytes| [(bytes, false), (bytes, true)])
    {
        let mut model = Model::from_bytes(bytes).unwrap();
        if scalar {
            model = model.scalar_clone();
        }
        let mut avx2 = model.clone();
        avx2.disable_vnni();
        let mut transitions = [[[0u32; 5]; 3]; 2];
        let mut search = prepared(&model, &frames[0].state(), frames[0].clock);
        let mut start = 0;
        let mut edges = 0;
        let mut refreshes = 0;
        for (index, frame) in frames.iter().enumerate() {
            let state = frame.state();
            assert_eq!(state.legal_actions().len(), frame.legal);
            let fresh = model.refresh(&state.board, frame.since_capture, frame.clock);
            oracle(&model, &avx2, &fresh, frame);
            if index > 0 && frames[index - 1].game == frame.game {
                let parent = &frames[index - 1];
                let action = parent.action.unwrap();
                let before = parent.state();
                let (child, outcome) = engine::apply(
                    &Rules {
                        capture_clock: Some(frame.clock),
                    },
                    &before,
                    action,
                );
                assert_eq!(child.board, state.board);
                assert_eq!(child.since_capture, state.since_capture);
                assert_eq!(child.ply, state.ply);
                assert_eq!(outcome as u8, frame.outcome);
                let captured = before.board[action_from_to(action).1].code();
                if captured != 0 {
                    let kind = (captured - 4) as usize;
                    let count = FeatureState::new(&before.board, before.since_capture, frame.clock)
                        .counts[1][kind];
                    transitions[(before.ply as usize + 1) % 2][kind][count as usize] += 1;
                }
                search.prepare_child(&before, action, frame.since_capture, index - start - 1);
                let pending = search.pending[index - start].unwrap();
                assert_eq!(pending.after, fresh.state);
                assert_eq!(pending.ready, 0);
                edges += 1;
            }
            let end = frame.action.is_none() || index - start == 31;
            if end {
                let pointers: Vec<_> = search
                    .acc
                    .iter()
                    .map(|a| (a.own.as_ptr(), a.opponent.as_ptr()))
                    .collect();
                search.resolve_acc(&model, index - start);
                for at in (start..=index).rev() {
                    search.resolve_acc(&model, at - start);
                    oracle(&model, &avx2, &search.acc[at - start], &frames[at]);
                }
                assert!(search
                    .acc
                    .iter()
                    .zip(&pointers)
                    .all(|(a, &(own, opp))| a.own.as_ptr() == own && a.opponent.as_ptr() == opp));
                refreshes += search.metrics.unwrap().half_refreshes;
                if let Some(next) = frames.get(index + 1) {
                    if next.game != frame.game {
                        assert!(frame.action.is_none());
                        assert_ne!(frame.outcome, 0);
                        assert_eq!(next.ply, 0);
                        start = index + 1;
                        search = prepared(&model, &next.state(), next.clock);
                    } else {
                        start = index;
                        search = prepared(&model, &state, frame.clock);
                    }
                }
            }
        }
        assert_eq!(edges, 2986);
        for colour in transitions {
            for (kind, counts) in colour.iter().enumerate() {
                for (count, &seen) in counts
                    .iter()
                    .enumerate()
                    .take(if kind == 1 { 5 } else { 4 })
                    .skip(1)
                {
                    assert!(seen > 0, "missing type {kind}, count {count}");
                }
            }
        }
        assert_eq!(refreshes > 0, model.features == 13640);
    }
}

#[test]
fn changed_half_skips_ancestors_and_discard_accounting_conserves_work() {
    let frames = frames();
    let model = Model::from_bytes(V8).unwrap();
    let at = (2..frames.len())
        .find(|&i| {
            let a = &frames[i - 2];
            let b = &frames[i - 1];
            let c = &frames[i];
            a.game == c.game && b.context[0] != c.context[1]
        })
        .unwrap();
    let a = frames[at - 2].state();
    let b = frames[at - 1].state();
    let c = frames[at].state();
    let mut search = prepared(&model, &a, frames[at].clock);
    search.prepare_child(&a, frames[at - 2].action.unwrap(), b.since_capture, 0);
    search.prepare_child(&b, frames[at - 1].action.unwrap(), c.since_capture, 1);
    search.resolve_half(&model, 2, 1);
    assert_eq!(search.pending[1].unwrap().ready, 0);
    assert_eq!(search.pending[2].unwrap().ready, 2);
    assert_eq!(
        search.acc[2].opponent,
        model
            .refresh(&c.board, c.since_capture, frames[at].clock)
            .opponent
    );
    assert_eq!(search.metrics.unwrap().half_refreshes, 1);
    search.discard_pending(1);
    search.discard_pending(2);
    let m = search.metrics.unwrap();
    assert_eq!(
        (m.pending_nodes_discarded, m.pending_halves_discarded),
        (2, 3)
    );
    assert_eq!(
        2 * m.prepared_edges,
        m.materializations + m.pending_halves_discarded
    );
}

#[test]
fn diagnostic_counters_conserve_work_without_changing_search() {
    let model = Model::from_bytes(V8).unwrap();
    let mut pool = SearchPool::new(1, 1);
    let limits = Limits {
        time: Duration::from_secs(60),
        nodes: 4000,
        ..Limits::default()
    };
    let plain = pool.search(&model, &State::initial(), None, limits);
    pool.clear();
    pool.enable_accumulator_metrics();
    let measured = pool.search(&model, &State::initial(), None, limits);
    assert_eq!(
        (
            plain.action,
            plain.score,
            plain.depth,
            plain.nodes,
            plain.qnodes
        ),
        (
            measured.action,
            measured.score,
            measured.depth,
            measured.nodes,
            measured.qnodes
        )
    );
    let m = pool.accumulator_metrics();
    assert_eq!(
        m.materializations,
        2 * m.full_refreshes + m.half_refreshes + m.incremental_updates
    );
    assert_eq!(
        2 * m.prepared_edges + 2,
        m.materializations + m.pending_halves_discarded
    );
    assert!(m.exact_count_transitions <= m.prepared_edges);
    assert_eq!(m.half_refreshes, m.capped_context_changes);
    assert!(m.row_additions > 0 && m.row_subtractions > 0 && m.pending_halves_discarded > 0);
    assert!(m.refresh_ns > 0 && m.update_ns > 0);
}
