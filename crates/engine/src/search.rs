//! Iterative-deepening principal variation search.
//!
//! * Alpha-beta with PVS, aspiration windows and a shared transposition table.
//! * Quiescence is a threat-space search: the side to move may stand pat or
//!   play fours, the defender must answer each four, so victories by
//!   continuous fours (VCF) are found at the leaves.
//! * Forced replies (blocking a five) are extended, so forcing sequences do
//!   not eat into the nominal depth.
//! * Pruning: reverse futility, razoring through the threat search, late
//!   move pruning and late move reductions, all skipped when the opponent
//!   threatens an open four.
//! * Lazy SMP: helper threads search the same tree with their own boards and
//!   histories, sharing only the table and the stop flag.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, OnceLock};

use crate::board::{Board, CELLS, Color, Pos};
use crate::eval::{self, Value};
use crate::movegen::{self, Gen, ScoredMove};
use crate::pattern::Pattern4;
use crate::rule::Rule;
use crate::tt::{Bound, TranspositionTable, TtEntry};

/// Score of a win on the next ply.
pub const MATE: Value = 30_000;
/// Scores at or beyond this are wins or losses.
pub const MATE_BOUND: Value = MATE - 1_000;
/// Larger than any score.
pub const INFINITE: Value = MATE + 1;
/// Deepest ply the search will reach.
pub const MAX_PLY: usize = 160;
/// Deepest nominal depth of iterative deepening.
pub const MAX_DEPTH: i32 = 100;

/// Score of winning `ply` plies from the root.
#[inline]
pub fn mate_in(ply: usize) -> Value {
    MATE - ply as Value
}

/// Score of losing `ply` plies from the root.
#[inline]
pub fn mated_in(ply: usize) -> Value {
    -MATE + ply as Value
}

/// Store mate scores relative to the node, not the root.
fn value_to_tt(v: Value, ply: usize) -> Value {
    if v >= MATE_BOUND {
        v + ply as Value
    } else if v <= -MATE_BOUND {
        v - ply as Value
    } else {
        v
    }
}

/// Whether `pos` is within two cells of `last`, or four along one of its
/// lines (Rapfi's RANGE_SQUARE2_LINE4).
#[inline]
fn near_last(pos: Pos, last: Pos) -> bool {
    let dx = (pos.x() as i32 - last.x() as i32).abs();
    let dy = (pos.y() as i32 - last.y() as i32).abs();
    let d = dx.max(dy);
    d <= 2 || (d <= 4 && (dx == 0 || dy == 0 || dx == dy))
}

/// Chebyshev distance between two cells.
#[inline]
fn chebyshev(a: Pos, b: Pos) -> i32 {
    (a.x() as i32 - b.x() as i32)
        .abs()
        .max((a.y() as i32 - b.y() as i32).abs())
}

fn value_from_tt(v: Value, ply: usize) -> Value {
    if v >= MATE_BOUND {
        v - ply as Value
    } else if v <= -MATE_BOUND {
        v + ply as Value
    } else {
        v
    }
}

/// Milliseconds since an arbitrary epoch.
pub type Clock = fn() -> f64;

/// The default clock: `std::time::Instant`, or a stopped clock on wasm,
/// where the host must supply one with [`Engine::set_clock`].
pub fn default_clock() -> f64 {
    #[cfg(not(target_arch = "wasm32"))]
    {
        static EPOCH: OnceLock<std::time::Instant> = OnceLock::new();
        EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs_f64() * 1000.0
    }
    #[cfg(target_arch = "wasm32")]
    {
        0.0
    }
}

/// When to stop searching. Unset limits do not apply; with none set the
/// search runs until stopped.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Limits {
    /// Maximum nominal depth.
    pub depth: Option<i32>,
    /// Stop after about this many nodes.
    pub nodes: Option<u64>,
    /// Do not start a new iteration after this many milliseconds.
    pub soft_ms: Option<f64>,
    /// Abort after this many milliseconds.
    pub hard_ms: Option<f64>,
    /// Treat `soft_ms` as Rapfi's optimum time: stop earlier when the best
    /// move is stable, later when it keeps changing or the value falls.
    pub dynamic: bool,
}

impl Limits {
    /// Search to a fixed depth.
    pub fn depth(depth: i32) -> Limits {
        Limits {
            depth: Some(depth),
            ..Limits::default()
        }
    }

    /// Use about `ms` milliseconds, never more.
    pub fn movetime(ms: f64) -> Limits {
        Limits {
            soft_ms: Some(ms * 0.5),
            hard_ms: Some(ms),
            ..Limits::default()
        }
    }

    /// Budget a move from a game clock: `time_left_ms` for the rest of the
    /// game and at most `turn_ms` for this move (0 for no per-move cap), as
    /// Rapfi does: at most a seventh of the clock, and when the clock is
    /// short an optimum of a 22nd of it, less as the game goes on.
    pub fn for_clock(time_left_ms: f64, turn_ms: f64, empty_cells: usize, ply: usize) -> Limits {
        let moves_to_go = empty_cells.max(1) as f64;
        let mut hard = time_left_ms / moves_to_go.min(7.0);
        if turn_ms > 0.0 {
            hard = hard.min(turn_ms);
        }
        // Leave room for process and protocol overhead.
        let hard = (hard - 30.0).max(hard * 0.5).max(1.0);
        // When the game clock could pay for a full turn on every remaining
        // move, there is nothing to save time for: keep deepening until
        // most of the turn is used. An unfinished iteration still counts
        // when it finds a better move.
        let ample = turn_ms > 0.0 && turn_ms * moves_to_go.min(64.0) < time_left_ms;
        let mut optimum = hard * 0.9;
        if !ample {
            let by_match = time_left_ms / moves_to_go.min(16.0);
            let by_turn = hard / 1.7 * move_importance(ply);
            optimum = optimum.min(by_match).min(by_turn);
        }
        Limits {
            soft_ms: Some(optimum),
            hard_ms: Some(hard),
            dynamic: !ample,
            ..Limits::default()
        }
    }
}

/// Rapfi's weight of a move by game ply: early moves get a little more.
fn move_importance(ply: usize) -> f64 {
    let ply = ply as f64;
    let basic = (1.0 + ((ply - 50.0) / 8.0).exp()).powf(-0.08);
    let opening = 0.1 * (-0.04 * ply).exp();
    (basic + opening).max(0.01)
}

/// Progress report after each completed iteration.
#[derive(Clone, Debug, PartialEq)]
pub struct Info {
    pub depth: i32,
    pub seldepth: usize,
    pub value: Value,
    pub nodes: u64,
    pub elapsed_ms: f64,
    pub hashfull: u32,
    pub pv: Vec<Pos>,
}

impl Info {
    /// Nodes per second.
    pub fn nps(&self) -> u64 {
        (self.nodes as f64 / (self.elapsed_ms / 1000.0).max(1e-3)) as u64
    }

    /// Moves (not plies) to a forced win (positive) or loss (negative).
    pub fn mate(&self) -> Option<i32> {
        mate_moves(self.value)
    }
}

/// Moves (not plies) to a forced win or loss for a mate score.
pub fn mate_moves(value: Value) -> Option<i32> {
    if value >= MATE_BOUND {
        Some((MATE - value + 1) / 2)
    } else if value <= -MATE_BOUND {
        Some(-(MATE + value) / 2)
    } else {
        None
    }
}

/// Outcome of a search.
#[derive(Clone, Debug, PartialEq)]
pub struct SearchResult {
    /// The move to play; `None` only when the board is full.
    pub best_move: Option<Pos>,
    pub value: Value,
    pub depth: i32,
    pub nodes: u64,
    pub pv: Vec<Pos>,
}

/// A search engine: the shared table plus settings. Positions are passed to
/// each [`Engine::search`] call.
pub struct Engine {
    tt: Arc<TranspositionTable>,
    threads: usize,
    clock: Clock,
    stop: Arc<AtomicBool>,
    /// Rapfi's time reduction from the last move's best-move stability,
    /// and that move's value, carried to the next move of the game.
    time_reduction: f64,
    last_value: Option<Value>,
}

impl Engine {
    /// An engine with a `hash_mb` megabyte table and one thread.
    pub fn new(hash_mb: usize) -> Engine {
        Engine {
            tt: Arc::new(TranspositionTable::new(hash_mb)),
            threads: 1,
            clock: default_clock,
            stop: Arc::new(AtomicBool::new(false)),
            time_reduction: 1.0,
            last_value: None,
        }
    }

    /// Search threads. Native builds need the `threads` feature; WebAssembly
    /// builds with atomics need the host to run [`pool::helper`] in as many
    /// extra threads.
    pub fn set_threads(&mut self, threads: usize) {
        self.threads = threads.clamp(1, 256);
    }

    /// Search threads in use.
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Resize the table, dropping its contents.
    pub fn set_hash(&mut self, megabytes: usize) {
        if megabytes != self.tt.megabytes() {
            self.tt = Arc::new(TranspositionTable::new(megabytes));
        }
    }

    /// Supply the clock used for time limits.
    pub fn set_clock(&mut self, clock: Clock) {
        self.clock = clock;
    }

    /// Forget previous searches (new game).
    pub fn clear(&mut self) {
        self.tt.clear();
        self.time_reduction = 1.0;
        self.last_value = None;
    }

    /// A flag that stops the running search when set.
    pub fn stop_handle(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.stop)
    }

    /// Find the best move for the side to move. `on_info` is called after
    /// every completed iteration.
    pub fn search(&mut self, board: &Board, limits: &Limits, mut on_info: impl FnMut(&Info)) -> SearchResult {
        self.stop.store(false, Relaxed);
        self.tt.new_search();
        let shared = Arc::new(Shared {
            tt: Arc::clone(&self.tt),
            stop: Arc::clone(&self.stop),
            nodes: AtomicU64::new(0),
            clock: self.clock,
            start: (self.clock)(),
            limits: limits.clone(),
            prev_time: (self.time_reduction, self.last_value),
            next_time: AtomicU64::new(1f64.to_bits()),
        });

        #[cfg(feature = "threads")]
        let result = std::thread::scope(|scope| {
            let helpers: Vec<_> = (1..self.threads)
                .map(|id| {
                    let shared = &shared;
                    let board = board.clone();
                    std::thread::Builder::new()
                        .stack_size(16 << 20)
                        .spawn_scoped(scope, move || Worker::new(shared, id, board).think(&mut |_| {}))
                        .expect("spawn search thread")
                })
                .collect();
            let result = Worker::new(&shared, 0, board.clone()).think(&mut on_info);
            shared.stop.store(true, Relaxed);
            for helper in helpers {
                let _ = helper.join();
            }
            result
        });

        #[cfg(all(not(feature = "threads"), target_arch = "wasm32", target_feature = "atomics"))]
        let result = {
            if self.threads > 1 {
                pool::start(&shared, board, self.threads);
            }
            let result = Worker::new(&shared, 0, board.clone()).think(&mut on_info);
            shared.stop.store(true, Relaxed);
            if self.threads > 1 {
                pool::finish();
            }
            result
        };

        #[cfg(all(
            not(feature = "threads"),
            not(all(target_arch = "wasm32", target_feature = "atomics"))
        ))]
        let result = Worker::new(&shared, 0, board.clone()).think(&mut on_info);

        if shared.limits.dynamic {
            self.time_reduction = f64::from_bits(shared.next_time.load(Relaxed));
            self.last_value = Some(result.value);
        }
        SearchResult {
            nodes: shared.nodes.load(Relaxed),
            ..result
        }
    }
}

/// Helper threads started by the host, for WebAssembly with shared memory,
/// where `std::thread::spawn` is unavailable. Each extra thread runs
/// [`pool::helper`] forever; searches hand them work through a shared slot.
#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
pub mod pool {
    use std::sync::{Arc, Condvar, Mutex};

    use super::{Shared, Worker};
    use crate::board::Board;

    struct Job {
        shared: Arc<Shared>,
        board: Board,
        generation: u64,
        threads: usize,
    }

    struct State {
        job: Option<Arc<Job>>,
        generation: u64,
        /// Helpers currently working on the job.
        active: usize,
    }

    static STATE: Mutex<State> = Mutex::new(State {
        job: None,
        generation: 0,
        active: 0,
    });
    static WAKE: Condvar = Condvar::new();

    /// Body of helper thread `id` (1 or more). Never returns.
    pub fn helper(id: usize) -> ! {
        let mut seen = 0;
        loop {
            let job = {
                let mut state = STATE.lock().unwrap_or_else(|e| e.into_inner());
                loop {
                    if let Some(job) = &state.job
                        && job.generation != seen
                    {
                        seen = job.generation;
                        if id < job.threads {
                            let job = Arc::clone(job);
                            state.active += 1;
                            break job;
                        }
                    }
                    state = WAKE.wait(state).unwrap_or_else(|e| e.into_inner());
                }
            };
            Worker::new(&job.shared, id, job.board.clone()).think(&mut |_| {});
            drop(job);
            STATE.lock().unwrap_or_else(|e| e.into_inner()).active -= 1;
            WAKE.notify_all();
        }
    }

    /// Hand a search to the helpers.
    pub(super) fn start(shared: &Arc<Shared>, board: &Board, threads: usize) {
        let mut state = STATE.lock().unwrap_or_else(|e| e.into_inner());
        state.generation += 1;
        let generation = state.generation;
        state.job = Some(Arc::new(Job {
            shared: Arc::clone(shared),
            board: board.clone(),
            generation,
            threads,
        }));
        drop(state);
        WAKE.notify_all();
    }

    /// Wait until every helper has left the (stopped) search.
    pub(super) fn finish() {
        let mut state = STATE.lock().unwrap_or_else(|e| e.into_inner());
        while state.active > 0 {
            state = WAKE.wait(state).unwrap_or_else(|e| e.into_inner());
        }
        state.job = None;
    }
}

/// State shared by every thread of one search.
struct Shared {
    tt: Arc<TranspositionTable>,
    stop: Arc<AtomicBool>,
    nodes: AtomicU64,
    clock: Clock,
    start: f64,
    limits: Limits,
    /// Rapfi's time reduction and best value from the previous move, and
    /// (set by the main thread) the new reduction, as `f64` bits.
    prev_time: (f64, Option<Value>),
    next_time: AtomicU64,
}

impl Shared {
    fn elapsed(&self) -> f64 {
        (self.clock)() - self.start
    }
}

#[derive(Clone, Debug)]
struct RootMove {
    pos: Pos,
    value: Value,
    pv: Vec<Pos>,
}

/// Late move reduction table, indexed by depth and move number.
fn reductions() -> &'static [[u8; 64]; 64] {
    static TABLE: OnceLock<[[u8; 64]; 64]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let p = params();
        let (base, scale) = (p.lmr_base as f64 / 100.0, p.lmr_scale as f64 / 100.0);
        let mut t = [[0; 64]; 64];
        for (d, row) in t.iter_mut().enumerate().skip(1) {
            for (m, r) in row.iter_mut().enumerate().skip(1) {
                *r = (base + scale * (d as f64).ln() * (m as f64).ln()).max(0.0) as u8;
            }
        }
        t
    })
}

/// Search constants that self-play tuning may change (see `tengen-spsa`).
/// The defaults are the values the engine plays with.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    /// Reverse futility pruning: deepest depth, margin per ply, and how
    /// much less margin an improving position needs.
    pub rfp_depth: i32,
    pub rfp_margin: i32,
    pub rfp_improving: i32,
    /// Razoring: deepest depth, base margin and margin per ply.
    pub razor_depth: i32,
    pub razor_base: i32,
    pub razor_margin: i32,
    /// Futility pruning of quiet moves: deepest depth, base margin and
    /// margin per ply.
    pub futility_depth: i32,
    pub futility_base: i32,
    pub futility_margin: i32,
    /// Late move pruning starts after `lmp_base + depth²` moves.
    pub lmp_base: i32,
    /// Late move reductions: `(lmr_base + lmr_scale * ln d * ln m) / 100`,
    /// and history points per ply less.
    pub lmr_base: i32,
    pub lmr_scale: i32,
    pub lmr_history: i32,
    /// Fail-high reduction: static evaluation above beta by this much per
    /// ply.
    pub fail_high_margin: i32,
    /// First aspiration window half-width.
    pub aspiration: i32,
}

impl Default for Params {
    fn default() -> Params {
        Params {
            rfp_depth: 7,
            rfp_margin: 59,
            rfp_improving: 29,
            razor_depth: 3,
            razor_base: 154,
            razor_margin: 113,
            futility_depth: 3,
            futility_base: 117,
            futility_margin: 104,
            lmp_base: 5,
            lmr_base: 62,
            lmr_scale: 104,
            lmr_history: 5873,
            fail_high_margin: 18,
            aspiration: 19,
        }
    }
}

impl Params {
    /// Every parameter with its name, for parsing and printing.
    fn fields(&mut self) -> [(&'static str, &mut i32); 15] {
        [
            ("rfp_depth", &mut self.rfp_depth),
            ("rfp_margin", &mut self.rfp_margin),
            ("rfp_improving", &mut self.rfp_improving),
            ("razor_depth", &mut self.razor_depth),
            ("razor_base", &mut self.razor_base),
            ("razor_margin", &mut self.razor_margin),
            ("futility_depth", &mut self.futility_depth),
            ("futility_base", &mut self.futility_base),
            ("futility_margin", &mut self.futility_margin),
            ("lmp_base", &mut self.lmp_base),
            ("lmr_base", &mut self.lmr_base),
            ("lmr_scale", &mut self.lmr_scale),
            ("lmr_history", &mut self.lmr_history),
            ("fail_high_margin", &mut self.fail_high_margin),
            ("aspiration", &mut self.aspiration),
        ]
    }

    /// The defaults with `name=value` changes, comma-separated.
    pub fn parse(text: &str) -> Result<Params, String> {
        let mut params = Params::default();
        for item in text.split(',').map(str::trim).filter(|i| !i.is_empty()) {
            let (name, value) = item.split_once('=').ok_or(format!("{item}: expected name=value"))?;
            let value: f64 = value.trim().parse().map_err(|e| format!("{item}: {e}"))?;
            let mut fields = params.fields();
            let slot = fields
                .iter_mut()
                .find(|(n, _)| *n == name.trim())
                .ok_or(format!("unknown parameter {name}"))?;
            *slot.1 = value.round() as i32;
        }
        Ok(params)
    }

    /// Names and values, as `parse` reads them.
    pub fn to_text(mut self) -> String {
        let fields = self.fields();
        fields
            .iter()
            .map(|(n, v)| format!("{n}={v}"))
            .collect::<Vec<_>>()
            .join(",")
    }
}

static PARAMS: OnceLock<Params> = OnceLock::new();

/// The search parameters in use.
#[inline]
pub fn params() -> &'static Params {
    PARAMS.get_or_init(Params::default)
}

/// Use `params` for the rest of the process. Must come before the first
/// search; fails afterwards.
pub fn set_params(params: Params) -> Result<(), String> {
    PARAMS
        .set(params)
        .map_err(|_| "search parameters are already in use".to_string())
}

const HISTORY_MAX: i32 = 16_384;
const NODE_CHECK: u64 = 1024;

/// One search thread.
struct Worker<'a> {
    shared: &'a Shared,
    id: usize,
    board: Board,
    nodes: u64,
    flushed: u64,
    seldepth: usize,
    root_depth: i32,
    history: Box<[[i32; CELLS]; 2]>,
    /// History of attacking moves (an open three or more for the mover),
    /// kept apart from quiet moves as in Rapfi.
    attack_history: Box<[[i32; CELLS]; 2]>,
    /// Rapfi's counter moves: by the color and cell of the opponent's last
    /// move, the reply that last refuted it and the combination it made.
    counters: Box<[[(Pos, Pattern4); CELLS]; 2]>,
    killers: Vec<[Pos; 2]>,
    /// Root moves that beat the first one, since the time check last read it.
    best_changes: u32,
    /// The mover's combination at each quiescence move.
    q_p4: Vec<Pattern4>,
    evals: Vec<Value>,
    /// The search parameters, copied for quick access.
    p: Params,
    pv: Vec<Vec<Pos>>,
    lists: Vec<Vec<ScoredMove>>,
    root_moves: Vec<RootMove>,
}

impl<'a> Worker<'a> {
    fn new(shared: &'a Shared, id: usize, board: Board) -> Worker<'a> {
        let plies = MAX_PLY + 4;
        Worker {
            shared,
            id,
            board,
            nodes: 0,
            flushed: 0,
            seldepth: 0,
            root_depth: 0,
            history: Box::new([[0; CELLS]; 2]),
            attack_history: Box::new([[0; CELLS]; 2]),
            counters: Box::new([[(Pos::NONE, Pattern4::None); CELLS]; 2]),
            killers: vec![[Pos::NONE; 2]; plies],
            best_changes: 0,
            q_p4: vec![Pattern4::None; plies],
            p: *params(),
            evals: vec![0; plies],
            pv: (0..plies).map(|_| Vec::with_capacity(32)).collect(),
            lists: (0..plies).map(|_| Vec::with_capacity(128)).collect(),
            root_moves: Vec::new(),
        }
    }

    fn is_main(&self) -> bool {
        self.id == 0
    }

    #[inline]
    fn stopped(&self) -> bool {
        self.shared.stop.load(Relaxed)
    }

    fn total_nodes(&self) -> u64 {
        self.shared.nodes.load(Relaxed) + (self.nodes - self.flushed)
    }

    /// Account nodes and enforce the hard limits.
    fn poll(&mut self) {
        self.shared.nodes.fetch_add(self.nodes - self.flushed, Relaxed);
        self.flushed = self.nodes;
        if !self.is_main() {
            return;
        }
        let limits = &self.shared.limits;
        let over_time = limits.hard_ms.is_some_and(|h| self.shared.elapsed() >= h);
        let over_nodes = limits.nodes.is_some_and(|n| self.shared.nodes.load(Relaxed) >= n);
        if over_time || over_nodes {
            self.shared.stop.store(true, Relaxed);
        }
    }

    /// Iterative deepening at the root.
    fn think(&mut self, on_info: &mut dyn FnMut(&Info)) -> SearchResult {
        let mut list = Vec::new();
        let forced_loss = match movegen::generate(&mut self.board, &mut list) {
            Gen::Win(pos) => {
                return SearchResult {
                    best_move: Some(pos),
                    value: mate_in(1),
                    depth: 1,
                    nodes: 1,
                    pv: vec![pos],
                };
            }
            Gen::Lost(plies) => {
                // Every move loses; pick the most stubborn-looking one.
                let us = self.board.side_to_move();
                let values = self.board.values();
                list = self
                    .board
                    .candidates()
                    .map(|pos| ScoredMove {
                        pos,
                        score: movegen::move_score(&self.board, pos, us, values, None),
                    })
                    .collect();
                list.retain(|m| self.board.is_legal(m.pos));
                Some(plies)
            }
            Gen::Moves { .. } => None,
        };
        if list.is_empty() {
            return SearchResult {
                best_move: None,
                value: 0,
                depth: 0,
                nodes: 0,
                pv: Vec::new(),
            };
        }
        list.sort_by_key(|m| std::cmp::Reverse(m.score));
        if let Some(plies) = forced_loss {
            let pos = list[0].pos;
            return SearchResult {
                best_move: Some(pos),
                value: mated_in(plies as usize),
                depth: 1,
                nodes: 1,
                pv: vec![pos],
            };
        }
        self.root_moves = list
            .iter()
            .map(|m| RootMove {
                pos: m.pos,
                value: -INFINITE,
                pv: vec![m.pos],
            })
            .collect();

        let limits = self.shared.limits.clone();
        let max_depth = limits.depth.unwrap_or(MAX_DEPTH).clamp(1, MAX_DEPTH);
        let mut completed = 0;
        let mut value: Value = 0;
        let mut best = self.root_moves[0].clone();
        // Rapfi's time control: how much the best move has changed lately,
        // and the depth at which it last changed.
        let mut total_changes = 0.0f64;
        let mut last_change_depth = 0;
        let mut last_best = Pos::NONE;
        let mut time_reduction = 1.0f64;

        for depth in 1..=max_depth {
            // Helpers skip ahead on odd ids so threads spread over depths.
            let search_depth = if self.is_main() {
                depth
            } else {
                depth + (self.id as i32 & 1)
            };
            self.root_depth = search_depth;
            self.seldepth = 0;
            total_changes *= 0.5;

            let mut delta = self.p.aspiration;
            let (mut alpha, mut beta) = if depth >= 4 && value.abs() < MATE_BOUND {
                (value - delta, value + delta)
            } else {
                (-INFINITE, INFINITE)
            };
            let v = loop {
                let v = self.search_root(alpha, beta, search_depth);
                self.root_moves.sort_by_key(|m| std::cmp::Reverse(m.value));
                // A full window cannot widen, whatever the value.
                if self.stopped() || (alpha <= -INFINITE && beta >= INFINITE) {
                    break v;
                }
                if v <= alpha {
                    beta = (alpha + beta) / 2;
                    alpha = (v - delta).max(-INFINITE);
                } else if v >= beta {
                    beta = (v + delta).min(INFINITE);
                } else {
                    break v;
                }
                delta += delta;
                if delta > 1000 {
                    (alpha, beta) = (-INFINITE, INFINITE);
                }
            };

            if self.stopped() {
                // A partial iteration only counts if its first move, the
                // previous best, was searched and something beat it.
                if self.root_moves[0].value > -INFINITE
                    && self.root_moves[0].pos != best.pos
                    && self.root_moves[0].value > value
                {
                    best = self.root_moves[0].clone();
                    value = best.value;
                }
                break;
            }
            value = v;
            best = self.root_moves[0].clone();
            completed = search_depth;

            if !self.is_main() {
                continue;
            }
            let elapsed = self.shared.elapsed();
            on_info(&Info {
                depth: search_depth,
                seldepth: self.seldepth.max(search_depth as usize),
                value,
                nodes: self.total_nodes(),
                elapsed_ms: elapsed,
                hashfull: self.shared.tt.hashfull(),
                pv: best.pv.clone(),
            });
            let proven = value.abs() >= MATE_BOUND && depth as Value >= (MATE - value.abs()) + 2;
            if best.pos != last_best {
                last_best = best.pos;
                last_change_depth = search_depth;
            }
            total_changes += std::mem::take(&mut self.best_changes) as f64;
            let timed = match limits.soft_ms {
                Some(optimum) if limits.dynamic => {
                    let (prev_reduction, prev_value) = self.shared.prev_time;
                    // More time when the value is falling, less when it holds.
                    let dropped = prev_value.map_or(f64::INFINITY, |p| (p - value) as f64);
                    let falling = (0.0032 * dropped + 0.544).clamp(0.5, 1.5);
                    // Less time the longer the best move has stayed the same.
                    time_reduction = 1.0 + 0.0125 * (search_depth - last_change_depth) as f64;
                    let reduction = prev_reduction.powf(0.528) / time_reduction;
                    let instability = 1.0 + 2.0 * total_changes;
                    let hard = limits.hard_ms.unwrap_or(f64::INFINITY);
                    let max_turn = hard / (1.25 + 0.02 * (search_depth as f64).powf(1.4));
                    elapsed >= (optimum * instability * reduction * falling).min(max_turn)
                }
                Some(optimum) => elapsed >= optimum,
                None => false,
            };
            let only_move = self.root_moves.len() == 1 && (limits.soft_ms.is_some() || limits.hard_ms.is_some());
            if proven || timed || only_move {
                break;
            }
        }
        if self.is_main() {
            self.shared.stop.store(true, Relaxed);
            self.shared.next_time.store(time_reduction.to_bits(), Relaxed);
        }
        self.poll();
        SearchResult {
            best_move: Some(best.pos),
            value,
            depth: completed,
            nodes: self.nodes,
            pv: best.pv,
        }
    }

    fn search_root(&mut self, mut alpha: Value, beta: Value, depth: i32) -> Value {
        let mut best = -INFINITE;
        for i in 0..self.root_moves.len() {
            let pos = self.root_moves[i].pos;
            let us = self.board.side_to_move();
            let tactical =
                self.board.p4(pos, us) >= Pattern4::HFlex3 || self.board.p4(pos, us.opponent()) >= Pattern4::HFlex3;
            self.nodes += 1;
            self.shared.tt.prefetch(self.board.hash_after(pos));
            self.board.play(pos);
            let v = if i == 0 {
                -self.search::<true>(-beta, -alpha, depth - 1, 1, false)
            } else {
                let r = if depth >= 3 && i >= 3 && !tactical {
                    (reductions()[depth.min(63) as usize][i.min(63)] as i32 - 1).clamp(0, depth - 2)
                } else {
                    0
                };
                let mut v = -self.search::<false>(-alpha - 1, -alpha, depth - 1 - r, 1, true);
                if v > alpha && r > 0 && !self.stopped() {
                    v = -self.search::<false>(-alpha - 1, -alpha, depth - 1, 1, false);
                }
                if v > alpha && !self.stopped() {
                    v = -self.search::<true>(-beta, -alpha, depth - 1, 1, false);
                }
                v
            };
            self.board.undo();
            if self.stopped() {
                break;
            }
            if i > 0 && v > alpha {
                self.best_changes += 1;
            }
            let rm = &mut self.root_moves[i];
            if i == 0 || v > alpha {
                rm.value = v;
                rm.pv.clear();
                rm.pv.push(pos);
                rm.pv.extend_from_slice(&self.pv[1]);
            } else {
                rm.value = -INFINITE;
            }
            if v > best {
                best = v;
            }
            if v > alpha {
                alpha = v;
                if v >= beta {
                    break;
                }
            }
        }
        best
    }

    fn update_pv(&mut self, ply: usize, pos: Pos) {
        let (head, tail) = self.pv.split_at_mut(ply + 1);
        let line = &mut head[ply];
        line.clear();
        line.push(pos);
        line.extend_from_slice(&tail[0]);
    }

    fn history_update(&mut self, color: usize, pos: Pos, bonus: i32) {
        let h = &mut self.history[color][pos.index()];
        *h += bonus - *h * bonus.abs() / HISTORY_MAX;
    }

    fn attack_history_update(&mut self, color: usize, pos: Pos, bonus: i32) {
        let h = &mut self.attack_history[color][pos.index()];
        *h += bonus - *h * bonus.abs() / HISTORY_MAX;
    }

    fn search<const PV: bool>(
        &mut self,
        mut alpha: Value,
        mut beta: Value,
        depth: i32,
        ply: usize,
        cut_node: bool,
    ) -> Value {
        self.pv[ply].clear();
        if PV {
            self.seldepth = self.seldepth.max(ply);
        }
        self.nodes += 1;
        if self.nodes.is_multiple_of(NODE_CHECK) {
            self.poll();
        }
        if self.stopped() {
            return 0;
        }

        let board = &self.board;
        let us = board.side_to_move();
        let them = us.opponent();
        if board.p4_count(us, Pattern4::AFive) > 0 {
            return mate_in(ply + 1);
        }
        let opp_fives = board.p4_count(them, Pattern4::AFive);
        if opp_fives >= 2 {
            return mated_in(ply + 2);
        }
        if board.is_full() {
            return 0;
        }
        if ply >= MAX_PLY - 8 {
            return eval::evaluate(board);
        }
        alpha = alpha.max(mated_in(ply));
        beta = beta.min(mate_in(ply + 1));
        if alpha >= beta {
            return alpha;
        }
        // An open three of the opponent at the horizon: search the defences
        // instead of standing pat in quiescence.
        let depth = if depth <= 0
            && opp_fives == 0
            && board.p4_count(them, Pattern4::BFlex4) > 0
            && ply < 2 * self.root_depth as usize
        {
            1
        } else {
            depth
        };
        if depth <= 0 {
            return self.qsearch(alpha, beta, ply, 0);
        }

        let key = board.hash();
        let tte = self.shared.tt.probe(key);
        let tt_move = tte.and_then(|e| e.best_move).filter(|&m| board.is_empty(m));
        // On the principal variation now or when last stored (Rapfi's and
        // Stockfish's ttPv): such nodes are reduced less.
        let tt_pv = PV || tte.is_some_and(|e| e.is_pv);
        if !PV
            && let Some(e) = tte
            && e.depth >= depth
        {
            let v = value_from_tt(e.value, ply);
            let fits = match e.bound {
                Bound::Exact => true,
                Bound::Lower => v >= beta,
                Bound::Upper => v <= alpha,
                Bound::None => false,
            };
            if fits {
                return v;
            }
        }

        let in_threat = opp_fives > 0 || board.p4_count(them, Pattern4::BFlex4) > 0;
        if opp_fives == 0
            && board.p4_count(us, Pattern4::BFlex4) > 0
            && board.p4_count_at_least(them, Pattern4::EBlock4) == 0
        {
            return mate_in(ply + 3);
        }

        let static_eval = if in_threat {
            -INFINITE
        } else {
            tte.map(|e| e.eval).unwrap_or_else(|| eval::evaluate(board))
        };
        self.evals[ply] = static_eval;
        let improving = !in_threat && ply >= 2 && static_eval > self.evals[ply - 2];

        if !PV && !in_threat && static_eval.abs() < MATE_BOUND {
            // Reverse futility: far above beta with shallow depth left.
            let p = &self.p;
            if depth <= p.rfp_depth && static_eval - (p.rfp_margin * depth - p.rfp_improving * improving as i32) >= beta
            {
                return static_eval;
            }
            // Razoring: far below alpha, and no forcing win changes that.
            if depth <= p.razor_depth && static_eval + p.razor_base + p.razor_margin * depth <= alpha {
                let v = self.qsearch(alpha, alpha + 1, ply, 0);
                if v <= alpha {
                    return v;
                }
            }
        }

        // Internal iterative reduction: no move to try first, search less.
        let mut depth = if (PV || cut_node) && depth >= 4 && tt_move.is_none() {
            depth - 1
        } else {
            depth
        };

        let mut list = std::mem::take(&mut self.lists[ply]);
        let kind = movegen::generate(&mut self.board, &mut list);
        let forced = match kind {
            Gen::Win(_) => {
                self.lists[ply] = list;
                return mate_in(ply + 1);
            }
            Gen::Lost(plies) => {
                self.lists[ply] = list;
                return mated_in(ply + plies as usize);
            }
            Gen::Moves { forced } => forced,
        };
        if list.is_empty() {
            // Only forbidden cells are left for Black. No move can be made,
            // which scores as a draw, like a full board (and like the root).
            self.lists[ply] = list;
            return 0;
        }
        let killers = self.killers[ply];
        let counter = self.board.last_move().map_or((Pos::NONE, Pattern4::None), |last| {
            self.counters[them as usize][last.index()]
        });
        for m in list.iter_mut() {
            if Some(m.pos) == tt_move {
                m.score = i32::MAX;
            } else {
                if killers.contains(&m.pos) {
                    m.score += 300;
                }
                if m.pos == counter.0 && counter.1 <= self.board.p4(m.pos, us) {
                    m.score += 150;
                }
                if self.board.p4(m.pos, us) >= Pattern4::HFlex3 {
                    m.score += self.attack_history[us as usize][m.pos.index()] / 16;
                } else {
                    m.score += self.history[us as usize][m.pos.index()] / 32;
                }
            }
        }
        list.sort_unstable_by_key(|m| std::cmp::Reverse(m.score));

        let original_alpha = alpha;
        let single_reply = list.len() == 1;
        let mut best = -INFINITE;
        let mut best_move = None;
        let mut quiets = [Pos::NONE; 32];
        let mut quiet_count = 0;
        let mut attacks = [Pos::NONE; 16];
        let mut attack_count = 0;
        // Rapfi's move-count pruning: the limit grows as depth^1.3, so it
        // keeps pruning at every depth, not only near the leaves.
        let lmp_limit = (self.p.lmp_base + (depth.max(0) as f64).powf(1.3) as i32) / (2 - improving as i32);
        // Fail-high reduction: a cut node whose static evaluation is far
        // above beta will most likely fail high anyway, so quiet moves are
        // searched a ply shallower.
        let distract_self = if self.board.rule() == Rule::Renju { 5 } else { 4 };
        let likely_fail_high = !PV && cut_node && !in_threat && static_eval >= beta + self.p.fail_high_margin * depth;

        for (index, m) in list.iter().enumerate() {
            let move_count = index as i32 + 1;
            let pos = m.pos;
            let tactical =
                forced || self.board.p4(pos, us) >= Pattern4::HFlex3 || self.board.p4(pos, them) >= Pattern4::HFlex3;
            // Our fours are never reduced, blocks of fours are; threes are, a
            // ply less than quiet moves (Rapfi reduces every move).
            let four = forced || self.board.p4(pos, us) >= Pattern4::EBlock4;

            // Rapfi's move-count pruning also drops late threes; only fours
            // and forced replies are kept.
            // Rapfi prunes by move count at PV nodes too (not the root).
            // Fours too, as in Rapfi.
            if !in_threat && !forced && best > -MATE_BOUND && move_count > lmp_limit {
                continue;
            }
            // Rapfi: against an open four, near the leaves, a move that does
            // not block it and is far from both last moves only delays.
            if in_threat
                && depth <= 4
                && best > -MATE_BOUND
                && self.board.p4(pos, them) < Pattern4::EBlock4
                && self.board.recent_move(0).is_some_and(|m| chebyshev(pos, m) > 4)
                && self
                    .board
                    .recent_move(1)
                    .is_some_and(|m| chebyshev(pos, m) > distract_self)
            {
                continue;
            }
            if !PV && !in_threat && !tactical && best > -MATE_BOUND && move_count > 1 {
                // Futility pruning of quiet moves near the leaves.
                let p = &self.p;
                if depth <= p.futility_depth
                    && static_eval + p.futility_base + p.futility_margin * depth + m.score.min(400) / 4 <= alpha
                {
                    continue;
                }
            }

            let extension = (single_reply && ply < 2 * self.root_depth as usize) as i32;
            let fail_high_reduction = (likely_fail_high && self.board.p4(pos, us) < Pattern4::EBlock4) as i32;
            let new_depth = depth - 1 + extension - fail_high_reduction;

            self.shared.tt.prefetch(self.board.hash_after(pos));
            self.board.play(pos);
            let mut v;
            if move_count == 1 {
                v = if PV {
                    -self.search::<true>(-beta, -alpha, new_depth, ply + 1, false)
                } else {
                    -self.search::<false>(-alpha - 1, -alpha, new_depth, ply + 1, !cut_node)
                };
            } else {
                let mut r = 0;
                if depth >= 3 && !four {
                    r = reductions()[depth.min(63) as usize][(move_count as usize).min(63)] as i32;
                    r += cut_node as i32;
                    r -= tt_pv as i32 + improving as i32;
                    r -= self.history[us as usize][pos.index()] / self.p.lmr_history.max(1);
                    r -= tactical as i32;
                    r = r.clamp(0, new_depth - 1);
                }
                v = -self.search::<false>(-alpha - 1, -alpha, new_depth - r, ply + 1, true);
                if v > alpha && r > 0 {
                    v = -self.search::<false>(-alpha - 1, -alpha, new_depth, ply + 1, !cut_node);
                }
                if PV && v > alpha && v < beta {
                    v = -self.search::<true>(-beta, -alpha, new_depth, ply + 1, false);
                }
            }
            self.board.undo();
            if self.stopped() {
                self.lists[ply] = list;
                return 0;
            }

            if v > best {
                best = v;
                if v > alpha {
                    best_move = Some(pos);
                    if PV {
                        self.update_pv(ply, pos);
                    }
                    if v >= beta {
                        // Not after a four: the reply was forced.
                        if let Some(last) = self.board.last_move()
                            && self.board.p4_count(them, Pattern4::AFive) == 0
                        {
                            self.counters[them as usize][last.index()] = (pos, self.board.p4(pos, us));
                        }
                        let attack = self.board.p4(pos, us) >= Pattern4::HFlex3;
                        if attack {
                            let bonus = (depth * depth * 8).min(3200);
                            self.attack_history_update(us as usize, pos, bonus);
                            for &q in &attacks[..attack_count] {
                                self.attack_history_update(us as usize, q, -bonus);
                            }
                        } else if !tactical {
                            let bonus = (depth * depth * 8).min(3200);
                            self.history_update(us as usize, pos, bonus);
                            for &q in &quiets[..quiet_count] {
                                self.history_update(us as usize, q, -bonus);
                            }
                            if self.killers[ply][0] != pos {
                                self.killers[ply] = [pos, self.killers[ply][0]];
                            }
                        }
                        break;
                    }
                    alpha = v;
                    // Once a move has raised alpha, the rest are searched a
                    // ply shallower, as in Rapfi.
                    if depth > 2 && depth < 12 && v.abs() < MATE_BOUND {
                        depth -= 1;
                    }
                }
            }
            if !tactical && quiet_count < quiets.len() {
                quiets[quiet_count] = pos;
                quiet_count += 1;
            }
            if self.board.p4(pos, us) >= Pattern4::HFlex3 && attack_count < attacks.len() {
                attacks[attack_count] = pos;
                attack_count += 1;
            }
        }
        self.lists[ply] = list;

        let bound = if best >= beta {
            Bound::Lower
        } else if PV && best > original_alpha {
            Bound::Exact
        } else {
            Bound::Upper
        };
        self.shared.tt.store(
            key,
            TtEntry {
                best_move,
                value: value_to_tt(best, ply),
                eval: if in_threat { 0 } else { static_eval },
                depth,
                bound,
                is_pv: tt_pv,
            },
        );
        best
    }

    /// Threat-space quiescence. The side to move blocks the opponent's four
    /// if there is one; otherwise it may stand pat or play a four of its own.
    fn qsearch(&mut self, mut alpha: Value, beta: Value, ply: usize, qply: usize) -> Value {
        self.nodes += 1;
        if self.nodes.is_multiple_of(NODE_CHECK) {
            self.poll();
        }
        if self.stopped() {
            return 0;
        }
        self.seldepth = self.seldepth.max(ply);
        let board = &self.board;
        let us = board.side_to_move();
        let them = us.opponent();
        if board.p4_count(us, Pattern4::AFive) > 0 {
            return mate_in(ply + 1);
        }
        let opp_fives = board.p4_count(them, Pattern4::AFive);
        if opp_fives >= 2 {
            return mated_in(ply + 2);
        }
        if board.is_full() {
            return 0;
        }
        if ply >= MAX_PLY - 4 {
            return eval::evaluate(board);
        }

        if opp_fives == 1 {
            // Answer the opponent's four; they then continue their attack.
            let block = board.five_point(them).expect("counted five point");
            if us == Color::Black && self.board.is_forbidden(block) {
                return mated_in(ply + 2);
            }
            self.shared.tt.prefetch(self.board.hash_after(block));
            self.board.play(block);
            let v = -self.qsearch(-beta, -alpha, ply + 1, qply + 1);
            self.board.undo();
            return v;
        }

        let stand = eval::evaluate(board);
        if stand >= beta {
            return stand;
        }
        if board.p4_count(us, Pattern4::BFlex4) > 0 && board.p4_count_at_least(them, Pattern4::EBlock4) == 0 {
            return mate_in(ply + 3);
        }
        if qply >= 24 {
            return stand;
        }
        let key = board.hash();
        let tte = self.shared.tt.probe(key);
        if let Some(e) = tte
            && e.depth >= 0
        {
            let v = value_from_tt(e.value, ply);
            let fits = match e.bound {
                Bound::Exact => true,
                Bound::Lower => v >= beta,
                Bound::Upper => v <= alpha,
                Bound::None => false,
            };
            if fits {
                return v;
            }
        }
        alpha = alpha.max(stand);
        let original_alpha = alpha;

        let mut list = std::mem::take(&mut self.lists[ply]);
        movegen::generate_fours(&mut self.board, &mut list);
        // Rapfi's quiescent VCF only continues next to the attacker's own
        // last move: within two cells, or four along a line.
        if let Some(last) = self.board.recent_move(1) {
            list.retain(|m| near_last(m.pos, last));
        }
        // After two fours, plain fours only follow two fours that each made
        // more than a four; otherwise a four must come with more.
        if qply >= 4 && (self.q_p4[ply - 2] < Pattern4::DBlock4Plus || self.q_p4[ply - 4] < Pattern4::DBlock4Plus) {
            let board = &self.board;
            list.retain(|m| board.p4(m.pos, us) >= Pattern4::DBlock4Plus);
        }
        if let Some(tt_move) = tte.and_then(|e| e.best_move)
            && let Some(i) = list.iter().position(|m| m.pos == tt_move)
        {
            list[..=i].rotate_right(1);
        }
        let mut best = stand;
        let mut best_move = None;
        for m in list.iter() {
            self.q_p4[ply] = self.board.p4(m.pos, us);
            self.shared.tt.prefetch(self.board.hash_after(m.pos));
            self.board.play(m.pos);
            let v = -self.qsearch(-beta, -alpha, ply + 1, qply + 1);
            self.board.undo();
            if self.stopped() {
                self.lists[ply] = list;
                return 0;
            }
            if v > best {
                best = v;
                if v > alpha {
                    best_move = Some(m.pos);
                    alpha = v;
                    if v >= beta {
                        break;
                    }
                }
            }
        }
        self.lists[ply] = list;
        let bound = if best >= beta {
            Bound::Lower
        } else if best > original_alpha {
            Bound::Exact
        } else {
            Bound::Upper
        };
        self.shared.tt.store(
            key,
            TtEntry {
                best_move,
                value: value_to_tt(best, ply),
                eval: stand,
                depth: 0,
                bound,
                is_pv: false,
            },
        );
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Color;

    fn board_with(moves: &[(usize, usize)]) -> Board {
        let mut board = Board::new(15);
        for &(x, y) in moves {
            board.play(Pos::new(x, y));
        }
        board
    }

    fn best(board: &Board, depth: i32) -> SearchResult {
        Engine::new(16).search(board, &Limits::depth(depth), |_| {})
    }

    #[test]
    fn mate_scores_count_moves() {
        assert_eq!(mate_moves(mate_in(1)), Some(1));
        assert_eq!(mate_moves(mate_in(3)), Some(2));
        assert_eq!(mate_moves(mated_in(2)), Some(-1));
        assert_eq!(mate_moves(mated_in(4)), Some(-2));
        assert_eq!(mate_moves(12), None);
    }

    #[test]
    fn renju_white_wins_by_forcing_a_forbidden_block() {
        use crate::Rule;
        use crate::board::tests::position;
        // Black's (8, 7) is a double three. White's four (9..=12, 6..=3)
        // can only be stopped there.
        let black = [(6, 7), (7, 7), (8, 5), (8, 6), (13, 2)];
        let white = [(10, 5), (11, 4), (12, 3)];
        let board = position(Rule::Renju, &black, &white);
        assert_eq!(board.side_to_move(), Color::White);
        let result = best(&board, 2);
        assert_eq!(result.best_move, Some(Pos::new(9, 6)));
        assert_eq!(result.value, mate_in(3));

        let mut after = board.clone();
        after.play(Pos::new(9, 6));
        let mut list = Vec::new();
        assert_eq!(movegen::generate(&mut after, &mut list), Gen::Lost(2));

        // In freestyle the block is a fine move.
        let free = position(Rule::Freestyle, &black, &white);
        assert!(best(&free, 2).value < MATE_BOUND);
    }

    #[test]
    fn renju_black_never_plays_a_forbidden_move() {
        use crate::Rule;
        use crate::board::tests::position;
        // (8, 7) would be Black's double three, the obvious strong move.
        let mut board = position(
            Rule::Renju,
            &[(6, 7), (7, 7), (8, 5), (8, 6)],
            &[(0, 0), (14, 14), (0, 14)],
        );
        board.play(Pos::new(14, 0));
        assert_eq!(board.side_to_move(), Color::Black);
        assert!(board.is_forbidden(Pos::new(8, 7)));
        let result = best(&board, 4);
        let pos = result.best_move.unwrap();
        assert_ne!(pos, Pos::new(8, 7));
        assert!(board.is_legal(pos));
        let mut list = Vec::new();
        movegen::generate(&mut board, &mut list);
        assert!(list.iter().all(|m| m.pos != Pos::new(8, 7)));
    }

    #[test]
    fn renju_black_without_a_legal_move_is_a_draw() {
        use crate::Rule;
        // A self-play game near its end, Black to move. Three cells are
        // empty: (3, 14), and (7, 14) and (11, 14), where Black would make
        // overlines. After (3, 14) and White's reply, Black has no legal
        // move; the search used to score that -INFINITE and never finish.
        const MOVES: &[&str] = &[
            "10,4 7,4 6,4 8,4 7,6 8,5 9,6 8,6 8,7 8,3 8,2 6,5 9,2 10,5 7,2 10,2 9,3 9,5 7,5 11,5 12,5 7,3 6,2 ",
            "5,2 11,4 10,3 11,7 6,0 6,3 12,4 9,7 10,7 9,9 9,8 10,8 8,10 8,9 7,9 5,6 13,5 9,1 5,4 5,3 6,8 5,7 ",
            "11,3 14,6 5,9 7,7 6,7 4,4 3,5 4,3 4,2 7,1 8,0 9,0 9,4 3,3 2,3 6,6 5,5 10,10 8,8 3,9 4,8 8,1 11,2 ",
            "12,1 6,1 12,10 11,9 13,10 11,10 6,10 4,6 7,8 2,5 4,5 5,10 10,12 10,11 11,11 13,9 9,12 11,12 2,6 ",
            "9,13 5,11 10,9 12,9 12,11 13,12 4,10 13,13 12,12 4,9 12,7 11,6 13,6 11,8 1,9 13,14 13,11 3,11 2,8 ",
            "3,7 3,10 3,8 4,11 2,9 4,7 13,8 13,3 13,4 14,3 12,3 14,8 4,12 1,4 3,6 1,10 2,10 3,2 4,1 14,2 10,6 ",
            "14,4 14,0 2,13 6,14 5,13 3,12 2,2 1,2 1,3 2,4 6,12 3,13 2,14 1,8 7,13 2,7 0,9 1,6 0,5 5,14 7,11 ",
            "4,14 7,12 7,10 8,13 6,13 6,11 9,14 8,11 9,11 8,12 8,14 12,2 13,2 14,10 11,13 1,12 14,9 0,4 1,11 4,0 ",
            "3,1 2,0 0,10 2,12 5,0 0,3 0,2 0,1 0,6 14,13 11,1 1,0 14,5 1,1 3,0 14,11 14,12 0,11 0,0 7,0 10,0 ",
            "11,0 12,0 13,0 2,1 5,1 13,1 10,1 14,1 3,4 1,5 12,6 0,7 1,7 13,7 14,7 0,8 5,8 12,8 6,9 9,10 2,11 ",
            "0,12 5,12 0,13 1,13 4,13 10,13 12,14 12,13 14,14 0,14 10,14 1,14 ",
        ];
        let mut board = Board::with_rule(15, Rule::Renju);
        for (x, y) in MOVES.concat().split_whitespace().map(|m| m.split_once(',').unwrap()) {
            board.play(Pos::new(x.parse().unwrap(), y.parse().unwrap()));
        }
        assert_eq!(board.side_to_move(), Color::Black);
        assert!(board.is_forbidden(Pos::new(7, 14)) && board.is_forbidden(Pos::new(11, 14)));
        let result = best(&board, 6);
        assert_eq!(result.best_move, Some(Pos::new(3, 14)));
        assert!(result.value.abs() < INFINITE);
    }

    #[test]
    fn plays_center_on_empty_board() {
        let result = best(&Board::new(15), 3);
        assert_eq!(result.best_move, Some(Pos::new(7, 7)));
    }

    #[test]
    fn takes_an_immediate_five() {
        let board = board_with(&[(3, 7), (3, 0), (4, 7), (5, 0), (5, 7), (14, 14), (6, 7), (14, 12)]);
        let result = best(&board, 4);
        assert!(matches!(result.best_move, Some(p) if p == Pos::new(2, 7) || p == Pos::new(7, 7)));
        assert_eq!(result.value, mate_in(1));
    }

    #[test]
    fn blocks_the_opponents_four() {
        // White four (3..=6, 0) blocked at (2, 0); Black must take (7, 0).
        let board = board_with(&[(2, 0), (3, 0), (7, 7), (4, 0), (9, 9), (5, 0), (11, 11), (6, 0)]);
        assert_eq!(best(&board, 4).best_move, Some(Pos::new(7, 0)));
    }

    #[test]
    fn finds_a_four_three() {
        // Black to move: (8, 7) makes the open three on row 7 a four and the
        // column (8, 5), (8, 6) an open three, a winning four-three.
        let board = board_with(&[
            (5, 7),
            (0, 0),
            (6, 7),
            (0, 2),
            (7, 7),
            (0, 14),
            (8, 5),
            (14, 0),
            (8, 6),
            (14, 14),
        ]);
        // Many moves win here (Black even has an open three); the search
        // must prove a win quickly.
        let result = best(&board, 6);
        assert!(result.value >= MATE_BOUND, "{result:?}");
        assert_eq!(board.side_to_move(), Color::Black);
    }

    #[test]
    fn finds_vcf_in_quiescence() {
        // Black fours at (8, 7) (White blocks at (9, 7)), then (8, 8) makes a
        // double four. A depth-1 search must see it through the threat search.
        let board = board_with(&[
            (5, 7),
            (4, 7),
            (6, 7),
            (4, 8),
            (7, 7),
            (8, 11),
            (5, 8),
            (0, 0),
            (6, 8),
            (14, 0),
            (7, 8),
            (0, 14),
            (8, 9),
            (14, 14),
            (8, 10),
            (14, 7),
        ]);
        let result = best(&board, 1);
        assert!(result.value >= MATE_BOUND, "{result:?}\n{board}");
    }

    #[test]
    fn deterministic_with_one_thread() {
        let board = board_with(&[(7, 7), (8, 8), (7, 8), (6, 6)]);
        let a = best(&board, 6);
        let b = best(&board, 6);
        assert_eq!(a.best_move, b.best_move);
        assert_eq!(a.value, b.value);
    }

    #[cfg(feature = "threads")]
    #[test]
    fn lazy_smp_finds_the_same_win() {
        let board = board_with(&[
            (5, 7),
            (0, 0),
            (6, 7),
            (0, 2),
            (7, 7),
            (0, 14),
            (8, 5),
            (14, 0),
            (8, 6),
            (14, 14),
        ]);
        let mut engine = Engine::new(16);
        engine.set_threads(2);
        let result = engine.search(&board, &Limits::depth(6), |_| {});
        assert!(result.value >= MATE_BOUND);
    }

    #[test]
    fn respects_the_time_limit() {
        let board = board_with(&[(7, 7), (8, 8), (7, 8), (6, 6), (9, 7)]);
        let mut engine = Engine::new(16);
        let start = default_clock();
        let result = engine.search(&board, &Limits::movetime(200.0), |_| {});
        let elapsed = default_clock() - start;
        assert!(result.best_move.is_some());
        assert!(elapsed < 400.0, "took {elapsed} ms");
    }
}
