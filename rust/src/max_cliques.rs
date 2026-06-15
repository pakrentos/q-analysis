use hashbrown::HashSet;

use crate::bitset::Bitset;

pub type VertexId = u32;
pub type CliqueId = u32;

/// Maintains the maximal cliques of a graph under batched edge insertions.
///
/// Cliques absorbed by a larger one become "ghosts": their id stays valid for
/// union-find bookkeeping, but they leave the live set and free their storage.
pub struct CliqueStore {
    adjacency: Vec<Bitset>,
    members: Vec<Vec<VertexId>>,
    bits: Vec<Option<Bitset>>,
    live: Vec<CliqueId>,
    live_pos: Vec<u32>,
    vertex_index: Vec<Vec<CliqueId>>,
    use_vertex_index: bool,
}

const LIVE_POS_GHOST: u32 = u32::MAX;
/// Below this vertex count a full bitset row scan beats per-vertex index lists.
const DENSE_SCAN_VERTEX_LIMIT: usize = 1024;

impl CliqueStore {
    pub fn new(num_vertices: usize) -> Self {
        let use_vertex_index = num_vertices > DENSE_SCAN_VERTEX_LIMIT;
        CliqueStore {
            adjacency: vec![Bitset::empty(num_vertices); num_vertices],
            members: Vec::new(),
            bits: Vec::new(),
            live: Vec::new(),
            live_pos: Vec::new(),
            vertex_index: if use_vertex_index {
                vec![Vec::new(); num_vertices]
            } else {
                Vec::new()
            },
            use_vertex_index,
        }
    }

    pub fn members(&self, id: CliqueId) -> &[VertexId] {
        &self.members[id as usize]
    }

    pub fn size(&self, id: CliqueId) -> usize {
        self.members[id as usize].len()
    }

    pub fn add_edge_to_graph(&mut self, u: VertexId, v: VertexId) {
        self.adjacency[u as usize].insert(v);
        self.adjacency[v as usize].insert(u);
    }

    /// Maximal cliques containing edge (u, v) in the current graph.
    /// Must be called after all edges of the batch were added to the graph.
    pub fn cliques_of_edge(&self, u: VertexId, v: VertexId, out: &mut Vec<Vec<VertexId>>) {
        out.clear();
        let common = self.adjacency[u as usize].intersection(&self.adjacency[v as usize]);
        if common.is_empty() {
            out.push(vec![u.min(v), u.max(v)]);
            return;
        }
        let mut extensions = Vec::new();
        maximal_cliques_within(&self.adjacency, common, &mut extensions);
        for ext in extensions {
            let mut clique: Vec<VertexId> = ext.iter_ones().collect();
            clique.push(u);
            clique.push(v);
            clique.sort_unstable();
            out.push(clique);
        }
    }

    /// Registers a new maximal clique, not yet visible to scans.
    pub fn create_clique(&mut self, members_sorted: Vec<VertexId>) -> CliqueId {
        let id = self.members.len() as CliqueId;
        let mut bits = Bitset::empty(self.adjacency.len());
        for &v in &members_sorted {
            bits.insert(v);
        }
        self.members.push(members_sorted);
        self.bits.push(Some(bits));
        self.live_pos.push(LIVE_POS_GHOST);
        id
    }

    pub fn make_visible(&mut self, id: CliqueId) {
        self.live_pos[id as usize] = self.live.len() as u32;
        self.live.push(id);
        if self.use_vertex_index {
            for &x in &self.members[id as usize] {
                self.vertex_index[x as usize].push(id);
            }
        }
    }

    /// Intersection sizes of `id` with every live clique sharing >= 1 vertex.
    pub fn scan_intersections(
        &self,
        id: CliqueId,
        scratch: &mut ScanScratch,
        out: &mut Vec<(CliqueId, u32)>,
    ) {
        out.clear();
        if self.use_vertex_index {
            self.scan_via_index(id, scratch, out);
        } else {
            self.scan_via_bitsets(id, out);
        }
    }

    fn scan_via_bitsets(&self, id: CliqueId, out: &mut Vec<(CliqueId, u32)>) {
        let my_bits = self.bits[id as usize].as_ref().expect("live clique has bits");
        for &other in &self.live {
            if other == id {
                continue;
            }
            let other_bits = self.bits[other as usize].as_ref().expect("live clique has bits");
            let shared = my_bits.intersection_count(other_bits);
            if shared > 0 {
                out.push((other, shared));
            }
        }
    }

    pub fn num_live(&self) -> usize {
        self.live.len()
    }

    fn scan_via_index(&self, id: CliqueId, scratch: &mut ScanScratch, out: &mut Vec<(CliqueId, u32)>) {
        scratch.counts.resize(self.members.len(), 0);
        for &x in &self.members[id as usize] {
            for &k in &self.vertex_index[x as usize] {
                if k == id {
                    continue;
                }
                if scratch.counts[k as usize] == 0 {
                    scratch.touched.push(k);
                }
                scratch.counts[k as usize] += 1;
            }
        }
        for &k in &scratch.touched {
            out.push((k, scratch.counts[k as usize]));
            scratch.counts[k as usize] = 0;
        }
        scratch.touched.clear();
    }

    /// Demotes an absorbed clique to a ghost.
    pub fn ghost(&mut self, id: CliqueId) {
        let pos = self.live_pos[id as usize];
        debug_assert_ne!(pos, LIVE_POS_GHOST, "ghosting a clique that is not live");
        self.live.swap_remove(pos as usize);
        if let Some(&moved) = self.live.get(pos as usize) {
            self.live_pos[moved as usize] = pos;
        }
        self.live_pos[id as usize] = LIVE_POS_GHOST;

        let members = std::mem::take(&mut self.members[id as usize]);
        if self.use_vertex_index {
            for &x in &members {
                let list = &mut self.vertex_index[x as usize];
                if let Some(list_pos) = list.iter().position(|&c| c == id) {
                    list.swap_remove(list_pos);
                }
            }
        }
        self.bits[id as usize] = None;
    }

    pub fn num_cliques_created(&self) -> usize {
        self.members.len()
    }
}

#[derive(Default)]
pub struct ScanScratch {
    counts: Vec<u32>,
    touched: Vec<CliqueId>,
}

/// Tracks cliques already created within the current threshold batch:
/// the same maximal clique can host several new edges of the batch.
#[derive(Default)]
pub struct BatchDedup {
    seen: HashSet<Vec<VertexId>>,
}

impl BatchDedup {
    pub fn clear(&mut self) {
        self.seen.clear();
    }

    pub fn is_new(&mut self, clique: &[VertexId]) -> bool {
        if self.seen.contains(clique) {
            return false;
        }
        self.seen.insert(clique.to_vec());
        true
    }
}

/// Bron–Kerbosch with pivoting over the subgraph induced on `candidates`.
fn maximal_cliques_within(adjacency: &[Bitset], candidates: Bitset, out: &mut Vec<Bitset>) {
    let mut current = Bitset::empty(adjacency.len());
    let excluded = Bitset::empty(adjacency.len());
    bron_kerbosch(adjacency, &mut current, candidates, excluded, out);
}

fn bron_kerbosch(
    adjacency: &[Bitset],
    current: &mut Bitset,
    mut candidates: Bitset,
    mut excluded: Bitset,
    out: &mut Vec<Bitset>,
) {
    if candidates.is_empty() && excluded.is_empty() {
        out.push(current.clone());
        return;
    }

    let pivot = candidates
        .iter_ones()
        .chain(excluded.iter_ones())
        .max_by_key(|&u| candidates.intersection_count(&adjacency[u as usize]))
        .expect("P or X is non-empty here");

    let branch_vertices: Vec<u32> = candidates
        .difference(&adjacency[pivot as usize])
        .iter_ones()
        .collect();

    for v in branch_vertices {
        let neighbors = &adjacency[v as usize];
        current.insert(v);
        bron_kerbosch(
            adjacency,
            current,
            candidates.intersection(neighbors),
            excluded.intersection(neighbors),
            out,
        );
        current.remove(v);
        candidates.remove(v);
        excluded.insert(v);
    }
}
