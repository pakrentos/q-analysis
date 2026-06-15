use hashbrown::HashSet;

use crate::bitset::Bitset;
use crate::max_cliques::{BatchDedup, CliqueId, CliqueStore, ScanScratch, VertexId};

/// Persistence intervals in CSR form: entry i owns members[offsets[i]..offsets[i+1]].
pub struct PersistenceDiagram {
    pub q: Vec<i32>,
    pub birth: Vec<f64>,
    pub death: Vec<f64>,
    pub offsets: Vec<u64>,
    pub members: Vec<u32>,
}

impl PersistenceDiagram {
    fn new() -> Self {
        PersistenceDiagram {
            q: Vec::new(),
            birth: Vec::new(),
            death: Vec::new(),
            offsets: vec![0],
            members: Vec::new(),
        }
    }

    fn push(&mut self, q: i32, birth: f64, death: f64, members: impl Iterator<Item = u32>) {
        self.q.push(q);
        self.birth.push(birth);
        self.death.push(death);
        self.members.extend(members);
        self.offsets.push(self.members.len() as u64);
    }
}

/// Union-find over the objects of one q level (cliques for q >= 1, vertices for q = 0),
/// tracking component vertex sets and births.
///
/// `snapshot` keeps the pre-threshold composition of roots modified within the
/// current threshold batch, so deaths report the last composition that was
/// observable across thresholds rather than a transient intra-batch state.
struct LevelDsu {
    parent: Vec<u32>,
    /// Last known root per slot; valid while it is still a root, since components
    /// only ever grow — this skips find() entirely for already-merged pairs.
    hint: Vec<u32>,
    birth: Vec<f64>,
    members: Vec<Option<Bitset>>,
    popcount: Vec<u32>,
    snapshot: Vec<Option<Bitset>>,
    snap_popcount: Vec<u32>,
    touched: Vec<u32>,
    /// Live-object count per root; a component with none can never gain members.
    root_live: Vec<u32>,
    active_components: u32,
}

impl LevelDsu {
    fn new() -> Self {
        LevelDsu {
            parent: Vec::new(),
            hint: Vec::new(),
            birth: Vec::new(),
            members: Vec::new(),
            popcount: Vec::new(),
            snapshot: Vec::new(),
            snap_popcount: Vec::new(),
            touched: Vec::new(),
            root_live: Vec::new(),
            active_components: 0,
        }
    }

    fn register(&mut self, members: Bitset, birth: f64) -> u32 {
        let idx = self.parent.len() as u32;
        self.parent.push(idx);
        self.hint.push(idx);
        self.birth.push(birth);
        self.popcount.push(members.count());
        self.members.push(Some(members));
        self.snapshot.push(None);
        self.snap_popcount.push(0);
        self.root_live.push(1);
        self.active_components += 1;
        idx
    }

    fn on_object_ghosted(&mut self, slot: u32) {
        let root = self.find(slot) as usize;
        self.root_live[root] -= 1;
        if self.root_live[root] == 0 {
            self.active_components -= 1;
        }
    }

    fn find(&mut self, mut i: u32) -> u32 {
        while self.parent[i as usize] != i {
            let grandparent = self.parent[self.parent[i as usize] as usize];
            self.parent[i as usize] = grandparent;
            i = grandparent;
        }
        i
    }

    /// Saves the pre-batch composition of a root born before the current threshold.
    fn stash(&mut self, root: u32, threshold: f64) {
        let root_us = root as usize;
        if self.birth[root_us] < threshold && self.snapshot[root_us].is_none() {
            let bits = self.members[root_us].as_ref().expect("root has members");
            self.snapshot[root_us] = Some(bits.clone());
            self.snap_popcount[root_us] = self.popcount[root_us];
            self.touched.push(root);
        }
    }

    /// Composition and size of a root as observable before the current batch.
    fn observable_stats(&self, root: u32, threshold: f64) -> (u32, &Bitset) {
        let root_us = root as usize;
        if self.birth[root_us] < threshold {
            (
                self.snap_popcount[root_us],
                self.snapshot[root_us]
                    .as_ref()
                    .expect("stashed before compare"),
            )
        } else {
            (
                self.popcount[root_us],
                self.members[root_us].as_ref().expect("root has members"),
            )
        }
    }

    /// Elder rule: smaller birth, then larger pre-batch size, then lexicographically
    /// smaller member set, then earlier slot.
    fn survives_over(&self, a: u32, b: u32, threshold: f64) -> bool {
        let (birth_a, birth_b) = (self.birth[a as usize], self.birth[b as usize]);
        if birth_a != birth_b {
            return birth_a < birth_b;
        }
        let (pop_a, set_a) = self.observable_stats(a, threshold);
        let (pop_b, set_b) = self.observable_stats(b, threshold);
        if pop_a != pop_b {
            return pop_a > pop_b;
        }
        if set_a != set_b {
            return set_a.lex_less(set_b);
        }
        a < b
    }

    /// Returns false when both slots were already in one component.
    fn union(
        &mut self,
        a: u32,
        b: u32,
        threshold: f64,
        q: i32,
        diagram: &mut PersistenceDiagram,
    ) -> bool {
        let hint_a = self.hint[a as usize];
        if hint_a == self.hint[b as usize] && self.parent[hint_a as usize] == hint_a {
            return false;
        }
        let root_a = self.find(a);
        let root_b = self.find(b);
        self.hint[a as usize] = root_a;
        self.hint[b as usize] = root_b;
        if root_a == root_b {
            return false;
        }
        self.stash(root_a, threshold);
        self.stash(root_b, threshold);

        let (winner, loser) = if self.survives_over(root_a, root_b, threshold) {
            (root_a, root_b)
        } else {
            (root_b, root_a)
        };
        let loser_us = loser as usize;

        if self.birth[loser_us] < threshold {
            let snapshot = self.snapshot[loser_us].take().expect("old root was stashed");
            diagram.push(q, self.birth[loser_us], threshold, snapshot.iter_ones());
        }

        let loser_members = self.members[loser_us].take().expect("loser is a root");
        let winner_members = self.members[winner as usize]
            .as_mut()
            .expect("winner is a root");
        winner_members.union_with(&loser_members);
        self.popcount[winner as usize] = winner_members.count();
        self.parent[loser_us] = winner;
        self.hint[a as usize] = winner;
        self.hint[b as usize] = winner;
        self.root_live[winner as usize] += self.root_live[loser_us];
        self.active_components -= 1;
        true
    }

    fn drop_snapshots(&mut self) {
        for idx in self.touched.drain(..) {
            self.snapshot[idx as usize] = None;
        }
    }

    fn emit_survivors(&self, q: i32, diagram: &mut PersistenceDiagram) {
        for idx in 0..self.parent.len() {
            if self.parent[idx] == idx as u32 {
                if let Some(bits) = &self.members[idx] {
                    diagram.push(q, self.birth[idx], f64::INFINITY, bits.iter_ones());
                }
            }
        }
    }
}

const UNREGISTERED: u32 = u32::MAX;

#[derive(Default)]
struct EngineStats {
    scanned_live: u64,
    scan_pairs: u64,
    union_calls: u64,
    scan_time: std::time::Duration,
    union_time: std::time::Duration,
}

struct Engine {
    store: CliqueStore,
    /// q = 0 is plain graph connectivity, tracked over vertices.
    vertex_level: LevelDsu,
    vertex_slot: Vec<u32>,
    /// clique_levels[i] is the DSU for q = i + 1.
    clique_levels: Vec<LevelDsu>,
    clique_slots: Vec<Vec<u32>>,
    max_q: Option<usize>,
    num_vertices: usize,
    diagram: PersistenceDiagram,
    scratch: ScanScratch,
    neighbors: Vec<(CliqueId, u32)>,
    clique_buf: Vec<Vec<VertexId>>,
    dedup: BatchDedup,
    stats: EngineStats,
}

impl Engine {
    fn new(num_vertices: usize, max_q: Option<usize>) -> Self {
        Engine {
            store: CliqueStore::new(num_vertices),
            vertex_level: LevelDsu::new(),
            vertex_slot: vec![UNREGISTERED; num_vertices],
            clique_levels: Vec::new(),
            clique_slots: Vec::new(),
            max_q,
            num_vertices,
            diagram: PersistenceDiagram::new(),
            scratch: ScanScratch::default(),
            neighbors: Vec::new(),
            clique_buf: Vec::new(),
            dedup: BatchDedup::default(),
            stats: EngineStats::default(),
        }
    }

    fn top_clique_level(&self, dimension: usize) -> usize {
        match self.max_q {
            Some(cap) => dimension.min(cap),
            None => dimension,
        }
    }

    fn process_batch(&mut self, batch: &[(VertexId, VertexId)], weight: f64) {
        for &(u, v) in batch {
            self.store.add_edge_to_graph(u, v);
            self.union_vertices(u, v, weight);
        }
        self.dedup.clear();
        for &(u, v) in batch {
            let mut clique_buf = std::mem::take(&mut self.clique_buf);
            self.store.cliques_of_edge(u, v, &mut clique_buf);
            for clique in clique_buf.drain(..) {
                if self.dedup.is_new(&clique) {
                    self.add_clique(clique, weight);
                }
            }
            self.clique_buf = clique_buf;
        }
        self.vertex_level.drop_snapshots();
        for level in &mut self.clique_levels {
            level.drop_snapshots();
        }
    }

    fn union_vertices(&mut self, u: VertexId, v: VertexId, weight: f64) {
        let slot_u = self.ensure_vertex(u, weight);
        let slot_v = self.ensure_vertex(v, weight);
        self.vertex_level
            .union(slot_u, slot_v, weight, 0, &mut self.diagram);
    }

    fn ensure_vertex(&mut self, v: VertexId, weight: f64) -> u32 {
        if self.vertex_slot[v as usize] == UNREGISTERED {
            let mut bits = Bitset::empty(self.num_vertices);
            bits.insert(v);
            self.vertex_slot[v as usize] = self.vertex_level.register(bits, weight);
        }
        self.vertex_slot[v as usize]
    }

    fn add_clique(&mut self, members: Vec<VertexId>, weight: f64) {
        let id = self.store.create_clique(members);
        self.register_clique(id, weight);

        let mut neighbors = std::mem::take(&mut self.neighbors);
        let scan_start = std::time::Instant::now();
        self.store.scan_intersections(id, &mut self.scratch, &mut neighbors);
        self.stats.scan_time += scan_start.elapsed();
        self.stats.scanned_live += self.store.num_live() as u64;
        self.stats.scan_pairs += neighbors.len() as u64;

        let union_start = std::time::Instant::now();
        let mut quiet = self.all_levels_quiet();
        for &(other, shared) in &neighbors {
            if shared >= 2 && !quiet {
                self.stats.union_calls += 1;
                if self.union_cliques(id, other, shared as usize, weight) {
                    quiet = self.all_levels_quiet();
                }
            }
            if shared as usize == self.store.size(other) {
                self.ghost_clique(other);
            }
        }
        self.stats.union_time += union_start.elapsed();
        self.neighbors = neighbors;
        self.store.make_visible(id);
    }

    /// With one live component on every level, any further union is a no-op,
    /// and no union can happen until the next clique registers.
    fn all_levels_quiet(&self) -> bool {
        self.clique_levels
            .iter()
            .all(|level| level.active_components <= 1)
    }

    fn ghost_clique(&mut self, id: CliqueId) {
        self.store.ghost(id);
        for (level_idx, &slot) in self.clique_slots[id as usize].iter().enumerate() {
            self.clique_levels[level_idx].on_object_ghosted(slot);
        }
    }

    fn register_clique(&mut self, id: CliqueId, weight: f64) {
        let top = self.top_clique_level(self.store.size(id) - 1);
        while self.clique_levels.len() < top {
            self.clique_levels.push(LevelDsu::new());
        }
        let mut slots = Vec::with_capacity(top);
        for level in &mut self.clique_levels[..top] {
            let mut bits = Bitset::empty(self.num_vertices);
            for &v in self.store.members(id) {
                bits.insert(v);
            }
            slots.push(level.register(bits, weight));
        }
        debug_assert_eq!(id as usize, self.clique_slots.len());
        self.clique_slots.push(slots);
    }

    /// Unions two live cliques at every level they are q-near, descending from the
    /// top. Components equal at level q are equal at every level below, so the first
    /// already-merged level ends the walk; a level with a single live component
    /// already holds both cliques, which ends it likewise.
    fn union_cliques(&mut self, a: CliqueId, b: CliqueId, shared: usize, weight: f64) -> bool {
        let top = self.top_clique_level(shared - 1);
        let mut any_merge = false;
        for q in (1..=top).rev() {
            let level = &mut self.clique_levels[q - 1];
            if level.active_components == 1 {
                break;
            }
            let slot_a = self.clique_slots[a as usize][q - 1];
            let slot_b = self.clique_slots[b as usize][q - 1];
            let merged = level.union(slot_a, slot_b, weight, q as i32, &mut self.diagram);
            if !merged {
                break;
            }
            any_merge = true;
        }
        any_merge
    }

    fn finish(mut self) -> PersistenceDiagram {
        self.vertex_level.emit_survivors(0, &mut self.diagram);
        for (i, level) in self.clique_levels.iter().enumerate() {
            level.emit_survivors(i as i32 + 1, &mut self.diagram);
        }
        if std::env::var_os("QA_PERSISTENCE_STATS").is_some() {
            eprintln!(
                "cliques created={} levels={} intervals={} scanned_live={} scan_pairs={} union_calls={} | scan={:?} union={:?}",
                self.store.num_cliques_created(),
                self.clique_levels.len() + 1,
                self.diagram.q.len(),
                self.stats.scanned_live,
                self.stats.scan_pairs,
                self.stats.union_calls,
                self.stats.scan_time,
                self.stats.union_time,
            );
        }
        self.diagram
    }
}

/// Computes elder-rule persistence intervals of q-connected components over the
/// ascending edge-weight filtration of a graph's clique complex.
pub fn persistent_q_communities(
    num_vertices: usize,
    edges: &[(VertexId, VertexId, f64)],
    max_q: Option<usize>,
) -> PersistenceDiagram {
    let mut sorted: Vec<(VertexId, VertexId, f64)> = edges
        .iter()
        .filter(|&&(u, v, w)| w.is_finite() && u != v)
        .map(|&(u, v, w)| (u.min(v), u.max(v), w))
        .collect();
    sorted.sort_unstable_by(|a, b| {
        a.2.total_cmp(&b.2).then(a.0.cmp(&b.0)).then(a.1.cmp(&b.1))
    });
    let mut seen: HashSet<(VertexId, VertexId)> = HashSet::new();
    sorted.retain(|&(u, v, _)| seen.insert((u, v)));

    let mut engine = Engine::new(num_vertices, max_q);
    let mut batch: Vec<(VertexId, VertexId)> = Vec::new();
    let mut start = 0;
    while start < sorted.len() {
        let weight = sorted[start].2;
        batch.clear();
        let mut end = start;
        while end < sorted.len() && sorted[end].2 == weight {
            batch.push((sorted[end].0, sorted[end].1));
            end += 1;
        }
        engine.process_batch(&batch, weight);
        start = end;
    }

    engine.finish()
}
