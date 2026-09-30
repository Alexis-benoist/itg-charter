//! Step parity: which foot (and which part of the foot) hits each note.
//!
//! This is a Rust port of ITGmania's step parity analysis
//! (`src/StepParityGenerator.cpp`, `StepParityCost.cpp`, `StepParityDatastructs.cpp`,
//! `TechCounts.cpp`, GPL-3.0, original work by tillvit and Michael Votaw). The cost
//! weights are the community-tuned values used by the game for its tech counts; they
//! are not ours to invent. Keep this file close to the upstream code so that future
//! upstream changes can be ported line by line.
//!
//! Known deviations: mines are attached to the next note row, lifts count as taps,
//! and the per-state cache of the C++ implementation is not reproduced.

// Loops index several parallel arrays exactly like the upstream C++; keep them that way.
#![allow(clippy::needless_range_loop)]

use crate::simfile::{Cell, NoteRow, Timing};
use std::collections::HashMap;

pub const MAX_COLUMNS: usize = 8;
pub const INVALID: i8 = -1;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
#[repr(u8)]
pub enum Foot {
    #[default]
    None = 0,
    LeftHeel = 1,
    LeftToe = 2,
    RightHeel = 3,
    RightToe = 4,
}

pub const FEET: [Foot; 4] = [Foot::LeftHeel, Foot::LeftToe, Foot::RightHeel, Foot::RightToe];
const FOOT_MASKS: [u16; 5] = [0, 1, 2, 4, 8];
const NUM_FOOT: usize = 5;

impl Foot {
    fn other_part(self) -> Foot {
        match self {
            Foot::None => Foot::None,
            Foot::LeftHeel => Foot::LeftToe,
            Foot::LeftToe => Foot::LeftHeel,
            Foot::RightHeel => Foot::RightToe,
            Foot::RightToe => Foot::RightHeel,
        }
    }
    fn mask(self) -> u16 {
        FOOT_MASKS[self as usize]
    }
    pub fn is_left(self) -> bool {
        matches!(self, Foot::LeftHeel | Foot::LeftToe)
    }
    pub fn is_right(self) -> bool {
        matches!(self, Foot::RightHeel | Foot::RightToe)
    }
}

// Cost weights, from StepParityCost.h.
const DOUBLESTEP: f32 = 850.0;
const BRACKETJACK: f32 = 20.0;
const JACK: f32 = 30.0;
const SLOW_BRACKET: f32 = 300.0;
const TWISTED_FOOT: f32 = 100000.0;
const BRACKETTAP: f32 = 400.0;
const HOLDSWITCH: f32 = 55.0;
const MINE: f32 = 10000.0;
const FOOTSWITCH: f32 = 325.0;
const MISSED_FOOTSWITCH: f32 = 500.0;
const FACING: f32 = 2.0;
const DISTANCE: f32 = 6.0;
const SPIN: f32 = 1000.0;
const SIDESWITCH: f32 = 130.0;
const JACK_THRESHOLD: f32 = 0.1;
const SLOW_BRACKET_THRESHOLD: f32 = 0.15;
const SLOW_FOOTSWITCH_THRESHOLD: f32 = 0.2;
const SLOW_FOOTSWITCH_IGNORE: f32 = 0.4;

// Tech count cutoffs, from TechCounts.cpp.
const JACK_CUTOFF: f32 = 0.176;
const FOOTSWITCH_CUTOFF: f32 = 0.3;
const DOUBLESTEP_CUTOFF: f32 = 0.235;

#[derive(Clone, Copy, Debug, Default)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

pub type Placement = [Foot; MAX_COLUMNS];

/// Relative position of each panel, plus precomputed lookup tables.
pub struct Layout {
    pub column_count: usize,
    pub columns: [Point; MAX_COLUMNS],
    pub up: Vec<usize>,
    pub down: Vec<usize>,
    pub side: Vec<usize>,
    avg_points: Vec<Point>,
    distances: Vec<f32>,
    facing_x: Vec<f32>,
    facing_y: Vec<f32>,
    permute_cache: Vec<Option<Vec<Placement>>>,
}

fn facing_penalty(v: f32) -> f32 {
    let base = -v.min(0.0);
    ((base as f64).powf(1.8) * 100.0) as f32
}

impl Layout {
    /// The 4-panel layout of `Steps.cpp` (Left, Down, Up, Right).
    pub fn dance_single() -> Layout {
        Layout::new(
            &[
                Point { x: 0.0, y: 1.0 },
                Point { x: 1.0, y: 0.0 },
                Point { x: 1.0, y: 2.0 },
                Point { x: 2.0, y: 1.0 },
            ],
            vec![2],
            vec![1],
            vec![0, 3],
        )
    }

    fn new(cols: &[Point], up: Vec<usize>, down: Vec<usize>, side: Vec<usize>) -> Layout {
        let n = cols.len();
        let mut columns = [Point::default(); MAX_COLUMNS];
        columns[..n].copy_from_slice(cols);
        let mut l = Layout {
            column_count: n,
            columns,
            up,
            down,
            side,
            avg_points: vec![Point::default(); n * n],
            distances: vec![0.0; n * n],
            facing_x: vec![0.0; n * n],
            facing_y: vec![0.0; n * n],
            permute_cache: Vec::new(),
        };
        for left in 0..n {
            for right in 0..n {
                let idx = left * n + right;
                let (a, b) = (l.columns[left], l.columns[right]);
                l.avg_points[idx] = Point {
                    x: (a.x + b.x) / 2.0,
                    y: (a.y + b.y) / 2.0,
                };
                let (dx, dy) = (a.x - b.x, a.y - b.y);
                let dist = (dx * dx + dy * dy).sqrt();
                l.distances[idx] = dist;
                if dist != 0.0 {
                    l.facing_x[idx] = facing_penalty(l.x_difference(left, right)).max(0.0);
                    l.facing_y[idx] = facing_penalty(l.y_difference(left, right)).max(0.0);
                }
            }
        }
        l.permute_cache = (0..(1usize << n))
            .map(|mask| {
                if mask == 0 {
                    return Some(Vec::new());
                }
                if mask.count_ones() > 4 {
                    return None;
                }
                let p = l.permute(mask, [Foot::None; MAX_COLUMNS], 0);
                if p.is_empty() { None } else { Some(p) }
            })
            .collect();
        l
    }

    fn permute(&self, mask: usize, cols: Placement, column: usize) -> Vec<Placement> {
        if column >= self.column_count {
            let find = |f: Foot| cols[..self.column_count].iter().position(|c| *c == f);
            let (lh, lt) = (find(Foot::LeftHeel), find(Foot::LeftToe));
            let (rh, rt) = (find(Foot::RightHeel), find(Foot::RightToe));
            if (lh.is_none() && lt.is_some()) || (rh.is_none() && rt.is_some()) {
                return Vec::new();
            }
            if let (Some(h), Some(t)) = (lh, lt)
                && !self.bracket_check(h, t)
            {
                return Vec::new();
            }
            if let (Some(h), Some(t)) = (rh, rt)
                && !self.bracket_check(h, t)
            {
                return Vec::new();
            }
            return vec![cols];
        }
        if mask & (1 << column) != 0 {
            let mut out = Vec::new();
            for foot in FEET {
                if cols.contains(&foot) {
                    continue;
                }
                let mut next = cols;
                next[column] = foot;
                out.extend(self.permute(mask, next, column + 1));
            }
            return out;
        }
        self.permute(mask, cols, column + 1)
    }

    fn bracket_check(&self, c1: usize, c2: usize) -> bool {
        let d = self.distance(c1 as i8, c2 as i8);
        d * d <= 2.0
    }

    fn distance_sq(&self, c1: usize, c2: usize) -> f32 {
        let (a, b) = (self.columns[c1], self.columns[c2]);
        (a.y - b.y) * (a.y - b.y) + (a.x - b.x) * (a.x - b.x)
    }

    fn distance(&self, l: i8, r: i8) -> f32 {
        if l == INVALID || r == INVALID {
            return 0.0;
        }
        self.distances[l as usize * self.column_count + r as usize]
    }

    fn x_facing(&self, l: i8, r: i8) -> f32 {
        if l == INVALID || r == INVALID {
            return 0.0;
        }
        self.facing_x[l as usize * self.column_count + r as usize]
    }

    fn y_facing(&self, l: i8, r: i8) -> f32 {
        if l == INVALID || r == INVALID {
            return 0.0;
        }
        self.facing_y[l as usize * self.column_count + r as usize]
    }

    fn x_difference(&self, l: usize, r: usize) -> f32 {
        if l == r {
            return 0.0;
        }
        let dx = self.columns[r].x - self.columns[l].x;
        let dy = self.columns[r].y - self.columns[l].y;
        let d = dx / (dx * dx + dy * dy).sqrt();
        let v = d.powi(4);
        if d <= 0.0 { -v } else { v }
    }

    fn y_difference(&self, l: usize, r: usize) -> f32 {
        if l == r {
            return 0.0;
        }
        let dx = self.columns[r].x - self.columns[l].x;
        let dy = self.columns[r].y - self.columns[l].y;
        let d = dy / (dx * dx + dy * dy).sqrt();
        let v = d.powi(4);
        if d <= 0.0 { -v } else { v }
    }

    pub fn average_point(&self, l: i8, r: i8) -> Point {
        match (l == INVALID, r == INVALID) {
            (true, true) => Point::default(),
            (true, false) => self.columns[r as usize],
            (false, true) => self.columns[l as usize],
            (false, false) => self.avg_points[l as usize * self.column_count + r as usize],
        }
    }

    /// Valid foot placements for a row, following `getFootPlacementPermutations`.
    pub fn placements(&self, note_mask: u16, hold_mask: u16) -> &[Placement] {
        let get = |m: u16| self.permute_cache.get(m as usize).and_then(|p| p.as_deref());
        get(note_mask | hold_mask)
            .or_else(|| get(note_mask))
            .unwrap_or(&[])
    }
}

/// A hold that is active on a row (started on an earlier row).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hold {
    pub beat: f32,
    pub length: f32,
}

/// A non-empty row of a chart, as seen by the parity analysis.
#[derive(Clone, Debug)]
pub struct Row {
    pub notes: [bool; MAX_COLUMNS],
    pub holds: [Option<Hold>; MAX_COLUMNS],
    pub mines: [f32; MAX_COLUMNS],
    pub second: f32,
    pub beat: f32,
    pub column_count: usize,
    pub note_mask: u16,
    pub hold_mask: u16,
    pub mine_mask: u16,
    // Filled in by the analysis.
    pub columns: [Foot; MAX_COLUMNS],
    pub where_feet: [i8; NUM_FOOT],
    pub note_count: usize,
}

impl Row {
    pub fn new(column_count: usize, beat: f32, second: f32) -> Row {
        Row {
            notes: [false; MAX_COLUMNS],
            holds: [None; MAX_COLUMNS],
            mines: [0.0; MAX_COLUMNS],
            second,
            beat,
            column_count,
            note_mask: 0,
            hold_mask: 0,
            mine_mask: 0,
            columns: [Foot::None; MAX_COLUMNS],
            where_feet: [INVALID; NUM_FOOT],
            note_count: 0,
        }
    }

    pub fn add_note(&mut self, c: usize) {
        self.notes[c] = true;
        self.note_mask |= 1 << c;
    }

    pub fn add_hold(&mut self, c: usize, hold: Hold) {
        self.holds[c] = Some(hold);
        self.hold_mask |= 1 << c;
    }

    fn set_foot_placement(&mut self, state: &State) {
        for c in 0..self.column_count {
            if self.notes[c] {
                self.columns[c] = state.combined[c];
                self.where_feet[state.combined[c] as usize] = c as i8;
                self.note_count += 1;
            }
        }
    }
}

/// Builds parity rows from parsed simfile rows (`CreateRows`).
pub fn rows_from_chart(note_rows: &[NoteRow], timing: &Timing, column_count: usize) -> Vec<Row> {
    // Hold lengths: find the tail of each hold head.
    let mut hold_len = vec![vec![None; column_count]; note_rows.len()];
    for c in 0..column_count {
        let mut open: Option<usize> = None;
        for (i, r) in note_rows.iter().enumerate() {
            match r.cells.get(c).copied().unwrap_or(Cell::Empty) {
                Cell::HoldHead | Cell::RollHead => open = Some(i),
                Cell::Tail => {
                    if let Some(h) = open.take() {
                        hold_len[h][c] = Some((r.beat - note_rows[h].beat) as f32);
                    }
                }
                _ => {}
            }
        }
    }
    let mut rows: Vec<Row> = Vec::new();
    let mut active: Vec<Option<(Hold, f32)>> = vec![None; column_count];
    let mut pending_mines = [0f32; MAX_COLUMNS];
    for (i, nr) in note_rows.iter().enumerate() {
        let beat = nr.beat as f32;
        let second = timing.seconds(nr.beat) as f32;
        let mut row = Row::new(column_count, beat, second);
        for (c, a) in active.iter_mut().enumerate() {
            if let Some((h, start_second)) = *a {
                if beat > h.beat + h.length {
                    *a = None;
                } else if start_second < second {
                    row.add_hold(c, h);
                }
            }
        }
        let mut row_mines = [0f32; MAX_COLUMNS];
        for c in 0..column_count {
            match nr.cells.get(c).copied().unwrap_or(Cell::Empty) {
                Cell::Tap | Cell::Lift => row.add_note(c),
                Cell::HoldHead | Cell::RollHead => {
                    row.add_note(c);
                    if let Some(len) = hold_len[i][c] {
                        active[c] = Some((Hold { beat, length: len }, second));
                    }
                }
                Cell::Mine => row_mines[c] = second,
                _ => {}
            }
        }
        if row.note_mask == 0 {
            // A row of mines only: they apply to the next note row.
            for c in 0..column_count {
                if row_mines[c] != 0.0 {
                    pending_mines[c] = row_mines[c];
                }
            }
            continue;
        }
        for c in 0..column_count {
            if pending_mines[c] != 0.0 {
                row.mines[c] = pending_mines[c];
                row.mine_mask |= 1 << c;
            }
        }
        pending_mines = row_mines;
        rows.push(row);
    }
    rows
}

/// A possible position of the player after a row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct State {
    pub combined: [Foot; MAX_COLUMNS],
    pub moved_mask: u16,
    pub holding_mask: u16,
    pub combined_mask: u32,
    pub where_feet: [i8; NUM_FOOT],
    pub what_note: [i8; NUM_FOOT],
    pub did_move: [bool; NUM_FOOT],
    pub holding: [bool; NUM_FOOT],
}

impl State {
    /// The state before the first note. Mirrors the zero-initialised C++ struct,
    /// where every foot "is" on column 0.
    pub fn beginning() -> State {
        State {
            combined: [Foot::None; MAX_COLUMNS],
            moved_mask: 0,
            holding_mask: 0,
            combined_mask: 0,
            where_feet: [0; NUM_FOOT],
            what_note: [0; NUM_FOOT],
            did_move: [false; NUM_FOOT],
            holding: [false; NUM_FOOT],
        }
    }

    pub fn key(&self) -> u64 {
        self.combined_mask as u64 | ((self.moved_mask as u64) << 30) | ((self.holding_mask as u64) << 46)
    }

    /// `initResultState` + `mergeInitialAndResultPosition`.
    pub fn result(initial: &State, row: &Row, cols: &Placement) -> State {
        let n = row.column_count;
        let mut r = State {
            combined: [Foot::None; MAX_COLUMNS],
            moved_mask: 0,
            holding_mask: 0,
            combined_mask: 0,
            where_feet: [INVALID; NUM_FOOT],
            what_note: [INVALID; NUM_FOOT],
            did_move: [false; NUM_FOOT],
            holding: [false; NUM_FOOT],
        };
        for i in 0..n {
            let f = cols[i];
            if f == Foot::None {
                continue;
            }
            r.what_note[f as usize] = i as i8;
            if row.holds[i].is_none() {
                r.did_move[f as usize] = true;
                continue;
            }
            if initial.combined[i] != f {
                r.did_move[f as usize] = true;
            }
        }
        for i in 0..n {
            let f = cols[i];
            if f == Foot::None {
                continue;
            }
            if row.holds[i].is_some() {
                r.holding[f as usize] = true;
            }
            let bit = 1u16 << i;
            if row.hold_mask & bit != 0 {
                r.holding_mask |= f.mask();
            }
            if row.hold_mask & bit == 0 || initial.combined[i] != f {
                r.moved_mask |= f.mask();
            }
        }
        for i in 0..n {
            if cols[i] != Foot::None {
                r.combined[i] = cols[i];
                continue;
            }
            let prev = initial.combined[i];
            match prev {
                Foot::LeftHeel | Foot::RightHeel => {
                    if !r.did_move[prev as usize] {
                        r.combined[i] = prev;
                    }
                }
                Foot::LeftToe => {
                    if !r.did_move[Foot::LeftToe as usize] && !r.did_move[Foot::LeftHeel as usize] {
                        r.combined[i] = prev;
                    }
                }
                Foot::RightToe => {
                    if !r.did_move[Foot::RightToe as usize] && !r.did_move[Foot::RightHeel as usize] {
                        r.combined[i] = prev;
                    }
                }
                Foot::None => {}
            }
        }
        for i in 0..n {
            if r.combined[i] != Foot::None {
                r.where_feet[r.combined[i] as usize] = i as i8;
            }
            r.combined_mask |= (r.combined[i] as u32) << (i * 3);
        }
        r
    }

    fn moved_foot_not_holding(&self, f: Foot) -> bool {
        self.did_move[f as usize] && !self.holding[f as usize]
    }
}

/// `StepParityCost::getActionCost`.
pub fn action_cost(
    layout: &Layout,
    initial: &State,
    result: &State,
    row: &Row,
    prev_row: Option<&Row>,
    cols: &Placement,
    elapsed: f32,
) -> f32 {
    use Foot::*;
    let lh = result.what_note[LeftHeel as usize];
    let lt = result.what_note[LeftToe as usize];
    let rh = result.what_note[RightHeel as usize];
    let rt = result.what_note[RightToe as usize];
    let moved_left = result.did_move[LeftHeel as usize] || result.did_move[LeftToe as usize];
    let moved_right = result.did_move[RightHeel as usize] || result.did_move[RightToe as usize];
    // Whether the *previous* state was a jump.
    let did_jump = (initial.moved_foot_not_holding(LeftHeel) || initial.moved_foot_not_holding(LeftToe))
        && (initial.moved_foot_not_holding(RightHeel) || initial.moved_foot_not_holding(RightToe));
    let prev_moved_left = initial.moved_foot_not_holding(LeftHeel) || initial.moved_foot_not_holding(LeftToe);
    let prev_moved_right =
        initial.moved_foot_not_holding(RightHeel) || initial.moved_foot_not_holding(RightToe);
    let jacked = |heel: i8, toe: i8, fh: Foot, ft: Foot, moved: bool, prev_moved: bool| {
        if did_jump || !moved {
            return false;
        }
        (heel > INVALID
            && initial.combined[heel as usize] == fh
            && !result.holding[fh as usize]
            && prev_moved)
            || (toe > INVALID
                && initial.combined[toe as usize] == ft
                && !result.holding[ft as usize]
                && prev_moved)
    };
    let jacked_left = jacked(lh, lt, LeftHeel, LeftToe, moved_left, prev_moved_left);
    let jacked_right = jacked(rh, rt, RightHeel, RightToe, moved_right, prev_moved_right);
    let n = row.column_count;
    let mut cost = 0.0;

    // Mines
    if row.mine_mask != 0 && (0..n).any(|i| result.combined[i] != None && row.mines[i] != 0.0) {
        cost += MINE;
    }

    // Hold switch
    if row.hold_mask != 0 {
        for c in 0..n {
            if row.holds[c].is_none() {
                continue;
            }
            let rf = result.combined[c];
            let inf = initial.combined[c];
            if (rf.is_left() && !inf.is_left()) || (rf.is_right() && !inf.is_right()) {
                let prev_foot = initial.where_feet[rf as usize];
                cost += HOLDSWITCH
                    * if prev_foot == INVALID {
                        1.0
                    } else {
                        layout.distance_sq(c, prev_foot as usize).sqrt()
                    };
            }
        }
    }

    // Bracket tap
    if row.hold_mask != 0 {
        let mut tap = |heel: i8, toe: i8, fh: Foot, ft: Foot| {
            if heel == INVALID || toe == INVALID {
                return;
            }
            let mut jack_penalty = 1.0;
            if initial.did_move[fh as usize] || initial.did_move[ft as usize] {
                jack_penalty = 1.0 / elapsed;
            }
            let (hh, th) = (
                row.holds[heel as usize].is_some(),
                row.holds[toe as usize].is_some(),
            );
            if hh && !th {
                cost += BRACKETTAP * jack_penalty;
            }
            if th && !hh {
                cost += BRACKETTAP * jack_penalty;
            }
        };
        tap(lh, lt, LeftHeel, LeftToe);
        tap(rh, rt, RightHeel, RightToe);
    }

    // Bracket jack
    if moved_left != moved_right && result.holding_mask == 0 && !did_jump {
        if jacked_left && result.did_move[LeftHeel as usize] && result.did_move[LeftToe as usize] {
            cost += BRACKETJACK;
        }
        if jacked_right && result.did_move[RightHeel as usize] && result.did_move[RightToe as usize] {
            cost += BRACKETJACK;
        }
    }

    // Doublestep
    if moved_left != moved_right && result.holding_mask == 0 && !did_jump {
        let mut doublestepped = (moved_left && !jacked_left && prev_moved_left)
            || (moved_right && !jacked_right && prev_moved_right);
        if let Some(last) = prev_row {
            for h in last.holds[..n].iter().flatten() {
                let end = h.beat + h.length;
                if (end > last.beat && end < row.beat) || end >= row.beat {
                    doublestepped = false;
                }
            }
        }
        if doublestepped {
            cost += DOUBLESTEP;
        }
    }

    // Slow bracket
    if elapsed > SLOW_BRACKET_THRESHOLD
        && moved_left != moved_right
        && row.notes[..n].iter().filter(|x| **x).count() >= 2
    {
        cost += (elapsed - SLOW_BRACKET_THRESHOLD) * SLOW_BRACKET;
    }

    // Twisted foot
    let left_pos = layout.average_point(lh, lt);
    let right_pos = layout.average_point(rh, rt);
    let crossed_over = right_pos.x < left_pos.x;
    let right_backwards =
        rh != INVALID && rt != INVALID && layout.columns[rt as usize].y < layout.columns[rh as usize].y;
    let left_backwards =
        lh != INVALID && lt != INVALID && layout.columns[lt as usize].y < layout.columns[lh as usize].y;
    if !crossed_over && (right_backwards || left_backwards) {
        cost += TWISTED_FOOT;
    }

    // Facing
    let end_lh = result.where_feet[LeftHeel as usize];
    let mut end_lt = result.where_feet[LeftToe as usize];
    let end_rh = result.where_feet[RightHeel as usize];
    let mut end_rt = result.where_feet[RightToe as usize];
    if end_lt == INVALID {
        end_lt = end_lh;
    }
    if end_rt == INVALID {
        end_rt = end_rh;
    }
    cost += layout.x_facing(end_lh, end_rh) * FACING
        + layout.x_facing(end_lt, end_rt) * FACING
        + layout.y_facing(end_lh, end_lt) * FACING
        + layout.y_facing(end_rh, end_rt) * FACING;

    // Spin
    let prev_left = layout.average_point(
        initial.where_feet[LeftHeel as usize],
        initial.where_feet[LeftToe as usize],
    );
    let prev_right = layout.average_point(
        initial.where_feet[RightHeel as usize],
        initial.where_feet[RightToe as usize],
    );
    let lp = layout.average_point(end_lh, end_lt);
    let rp = layout.average_point(end_rh, end_rt);
    if rp.x < lp.x && prev_right.x < prev_left.x && rp.y < lp.y && prev_right.y > prev_left.y {
        cost += SPIN;
    }
    if rp.x < lp.x && prev_right.x < prev_left.x && rp.y > lp.y && prev_right.y < prev_left.y {
        cost += SPIN;
    }

    // Footswitch (too slow)
    if (SLOW_FOOTSWITCH_THRESHOLD..SLOW_FOOTSWITCH_IGNORE).contains(&elapsed) && row.mine_mask == 0 {
        let time_scaled = elapsed - SLOW_FOOTSWITCH_THRESHOLD;
        for i in 0..n {
            if initial.combined[i] == None || cols[i] == None {
                continue;
            }
            if initial.combined[i] != cols[i] && initial.combined[i] != cols[i].other_part() {
                cost += (time_scaled / (SLOW_FOOTSWITCH_THRESHOLD + time_scaled)) * FOOTSWITCH;
                break;
            }
        }
    }

    // Sideswitch
    for &c in &layout.side {
        let prev = initial.combined[c];
        if prev != cols[c] && cols[c] != None && prev != None && !result.did_move[prev as usize] {
            cost += SIDESWITCH;
        }
    }

    // Missed footswitch
    if (jacked_left || jacked_right) && row.mine_mask != 0 {
        cost += MISSED_FOOTSWITCH;
    }

    // Jack (too fast)
    if elapsed < JACK_THRESHOLD && moved_left != moved_right && (jacked_left || jacked_right) {
        let time_scaled = JACK_THRESHOLD - elapsed;
        cost += (1.0 / time_scaled - 1.0 / JACK_THRESHOLD) * JACK;
    }

    // Big movements quickly
    for foot in FEET {
        if result.moved_mask & foot.mask() == 0 {
            continue;
        }
        let initial_pos = initial.where_feet[foot as usize];
        if initial_pos == INVALID {
            continue;
        }
        let result_pos = result.what_note[foot as usize];
        let other = result.what_note[foot.other_part() as usize];
        let bracketing = other != INVALID;
        if bracketing && other == initial_pos {
            continue;
        }
        let mut dist = layout.distance(initial_pos, result_pos) * DISTANCE / elapsed;
        if bracketing {
            dist *= 0.2;
        }
        cost += dist;
    }

    cost
}

/// Finds the cheapest foot placement for every row (`buildStateGraph` +
/// `computeCheapestPath`), annotates the rows and returns the total cost.
/// Returns `None` when some row has no valid placement.
pub fn analyze(layout: &Layout, rows: &mut [Row]) -> Option<f32> {
    struct Node {
        state: State,
        second: f32,
        total: f32,
        prev: usize,
    }
    if rows.is_empty() {
        return Some(0.0);
    }
    let mut nodes = vec![Node {
        state: State::beginning(),
        second: rows[0].second - 1.0,
        total: 0.0,
        prev: usize::MAX,
    }];
    let mut previous = vec![0usize];
    let mut by_key: HashMap<u64, usize> = HashMap::new();
    for i in 0..rows.len() {
        by_key.clear();
        let mut result_nodes = Vec::new();
        let row = &rows[i];
        let prev_row = if i > 0 { Some(&rows[i - 1]) } else { None };
        let placements = layout.placements(row.note_mask, row.hold_mask);
        for &init in &previous {
            let elapsed = row.second - nodes[init].second;
            for p in placements {
                let state = State::result(&nodes[init].state, row, p);
                let cost = action_cost(layout, &nodes[init].state, &state, row, prev_row, p, elapsed);
                let total = nodes[init].total + cost;
                match by_key.get(&state.key()) {
                    Some(&existing) => {
                        if total < nodes[existing].total {
                            nodes[existing].total = total;
                            nodes[existing].prev = init;
                        }
                    }
                    None => {
                        nodes.push(Node {
                            state,
                            second: row.second,
                            total,
                            prev: init,
                        });
                        by_key.insert(state.key(), nodes.len() - 1);
                        result_nodes.push(nodes.len() - 1);
                    }
                }
            }
        }
        if result_nodes.is_empty() {
            return None;
        }
        previous = result_nodes;
    }
    let mut best = previous[0];
    for &n in &previous {
        if nodes[n].total < nodes[best].total {
            best = n;
        }
    }
    let total = nodes[best].total;
    let mut path = Vec::with_capacity(rows.len());
    let mut cur = best;
    while cur != 0 {
        path.push(cur);
        cur = nodes[cur].prev;
    }
    path.reverse();
    for (row, &n) in rows.iter_mut().zip(&path) {
        row.set_foot_placement(&nodes[n].state);
    }
    Some(total)
}

/// Whether the step from `initial` to `cols` on `row` is a footswitch as counted by
/// [`TechCounts::from_rows`]: an up or down panel of the previous row (`prev_notes`)
/// hit again by the other foot less than `FOOTSWITCH_CUTOFF` seconds later. Not
/// upstream: lets the generator see footswitches while it builds a chart.
pub fn is_footswitch(
    layout: &Layout,
    initial: &State,
    prev_notes: u16,
    row: &Row,
    cols: &Placement,
    elapsed: f32,
) -> bool {
    elapsed < FOOTSWITCH_CUTOFF
        && layout.up.iter().chain(&layout.down).any(|&c| {
            let (prev, cur) = (initial.combined[c], cols[c]);
            row.notes[c]
                && prev_notes & (1 << c) != 0
                && cur != Foot::None
                && prev != Foot::None
                && prev != cur
                && prev.other_part() != cur
        })
}

/// Tech counts, as displayed by the game (`TechCounts::CalculateTechCountsFromRows`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TechCounts {
    pub crossovers: u32,
    pub half_crossovers: u32,
    pub full_crossovers: u32,
    pub footswitches: u32,
    pub up_footswitches: u32,
    pub down_footswitches: u32,
    pub sideswitches: u32,
    pub jacks: u32,
    pub doublesteps: u32,
    pub brackets: u32,
}

impl TechCounts {
    pub fn from_rows(layout: &Layout, rows: &[Row]) -> TechCounts {
        use Foot::*;
        let mut out = TechCounts::default();
        let footswitch = |c: usize, cur: &Row, prev: &Row, elapsed: f32| {
            cur.columns[c] != None
                && prev.columns[c] != None
                && prev.columns[c] != cur.columns[c]
                && prev.columns[c].other_part() != cur.columns[c]
                && elapsed < FOOTSWITCH_CUTOFF
        };
        for i in 0..rows.len() {
            let cur = &rows[i];
            let wf = |r: &Row, f: Foot| r.where_feet[f as usize];
            if cur.note_count >= 2 {
                if wf(cur, LeftHeel) != INVALID && wf(cur, LeftToe) != INVALID {
                    out.brackets += 1;
                }
                if wf(cur, RightHeel) != INVALID && wf(cur, RightToe) != INVALID {
                    out.brackets += 1;
                }
            }
            if i == 0 {
                continue;
            }
            let prev = &rows[i - 1];
            let elapsed = cur.second - prev.second;
            if cur.note_count == 1 && prev.note_count == 1 {
                for foot in FEET {
                    let (c, p) = (wf(cur, foot), wf(prev, foot));
                    if c == INVALID || p == INVALID {
                        continue;
                    }
                    if c == p {
                        if elapsed < JACK_CUTOFF {
                            out.jacks += 1;
                        }
                    } else if elapsed < DOUBLESTEP_CUTOFF {
                        out.doublesteps += 1;
                    }
                }
            }
            for &c in &layout.up {
                if footswitch(c, cur, prev, elapsed) {
                    out.up_footswitches += 1;
                    out.footswitches += 1;
                }
            }
            for &c in &layout.down {
                if footswitch(c, cur, prev, elapsed) {
                    out.down_footswitches += 1;
                    out.footswitches += 1;
                }
            }
            for &c in &layout.side {
                if footswitch(c, cur, prev, elapsed) {
                    out.sideswitches += 1;
                }
            }
            let (lh, lt, rh, rt) = (
                wf(cur, LeftHeel),
                wf(cur, LeftToe),
                wf(cur, RightHeel),
                wf(cur, RightToe),
            );
            let (plh, plt, prh, prt) = (
                wf(prev, LeftHeel),
                wf(prev, LeftToe),
                wf(prev, RightHeel),
                wf(prev, RightToe),
            );
            let mut crossover = |full: bool| {
                if full {
                    out.full_crossovers += 1;
                } else {
                    out.half_crossovers += 1;
                }
                out.crossovers += 1;
            };
            if rh != INVALID && plh != INVALID && prh == INVALID {
                let left_pos = layout.average_point(plh, plt);
                let right_pos = layout.average_point(rh, rt);
                if right_pos.x < left_pos.x {
                    if i > 1 {
                        let pprh = wf(&rows[i - 2], RightHeel);
                        if pprh != INVALID && pprh != rh {
                            crossover(layout.columns[pprh as usize].x > left_pos.x);
                        }
                    } else {
                        crossover(false);
                    }
                }
            } else if lh != INVALID && prh != INVALID && plh == INVALID {
                let left_pos = layout.average_point(lh, lt);
                let right_pos = layout.average_point(prh, prt);
                if right_pos.x < left_pos.x {
                    if i > 1 {
                        let pplh = wf(&rows[i - 2], LeftHeel);
                        if pplh != INVALID && pplh != lh {
                            crossover(right_pos.x > layout.columns[pplh as usize].x);
                        }
                    } else {
                        crossover(false);
                    }
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds rows from a compact pattern like "L D U R" at a fixed interval.
    fn pattern(p: &str, interval: f32) -> Vec<Row> {
        p.split_whitespace()
            .enumerate()
            .map(|(i, tok)| {
                let mut r = Row::new(4, i as f32 * 0.5, i as f32 * interval);
                for ch in tok.chars() {
                    r.add_note(match ch {
                        'L' => 0,
                        'D' => 1,
                        'U' => 2,
                        'R' => 3,
                        _ => panic!("bad token"),
                    });
                }
                r
            })
            .collect()
    }

    fn counts(p: &str, interval: f32) -> (TechCounts, Vec<Row>) {
        let layout = Layout::dance_single();
        let mut rows = pattern(p, interval);
        analyze(&layout, &mut rows).expect("valid path");
        (TechCounts::from_rows(&layout, &rows), rows)
    }

    #[test]
    fn simple_stream_alternates_feet_without_tech() {
        let (tc, rows) = counts("L D U R L U D R L D U R", 0.15);
        assert_eq!(tc.crossovers, 0);
        assert_eq!(tc.footswitches, 0);
        assert_eq!(tc.doublesteps, 0);
        assert!(rows[0].columns[0].is_left());
        assert!(rows[3].columns[3].is_right());
        for w in rows.windows(2) {
            let a = w[0].columns.iter().find(|f| **f != Foot::None).unwrap();
            let b = w[1].columns.iter().find(|f| **f != Foot::None).unwrap();
            assert_ne!(a.is_left(), b.is_left(), "feet must alternate");
        }
    }

    #[test]
    fn sideways_candle_is_a_crossover() {
        // L D R U L: the right foot steps up then the left foot crosses... (classic LDR-LUR)
        let (tc, _) = counts("L D R U L D R U R D L", 0.15);
        assert!(tc.crossovers > 0, "{tc:?}");
    }

    #[test]
    fn fast_same_arrow_twice_is_footswitch_or_jack() {
        let (tc, _) = counts("L D U U R", 0.12);
        assert_eq!(tc.footswitches + tc.jacks, 1, "{tc:?}");
    }

    #[test]
    fn is_footswitch_agrees_with_tech_counts() {
        let layout = Layout::dance_single();
        for (p, interval) in [
            ("L D U U R", 0.12),
            ("L D D R U U L D D", 0.12),
            ("L U U D D R", 0.25),
            ("L U U D D R", 0.35),
            ("L D U R L U D R", 0.15),
        ] {
            let (tc, rows) = counts(p, interval);
            let found = rows
                .windows(2)
                .filter(|w| {
                    let mut initial = State::beginning();
                    initial.combined = w[0].columns;
                    let elapsed = w[1].second - w[0].second;
                    is_footswitch(&layout, &initial, w[0].note_mask, &w[1], &w[1].columns, elapsed)
                })
                .count();
            assert_eq!(found as u32, tc.footswitches, "{p} {tc:?}");
        }
    }

    #[test]
    fn jump_uses_both_feet() {
        let (_, rows) = counts("LR D U LR", 0.3);
        assert!(rows[0].columns[0].is_left() && rows[0].columns[3].is_right());
    }

    #[test]
    fn placements_exclude_impossible_brackets() {
        let l = Layout::dance_single();
        // Up+Down cannot be bracketed by one foot; L+R neither.
        for p in l.placements(0b0110, 0) {
            let f1 = p[1];
            let f2 = p[2];
            assert!(!(f1.is_left() && f2.is_left()) && !(f1.is_right() && f2.is_right()));
        }
        assert!(!l.placements(0b1111, 0).is_empty());
    }
}
