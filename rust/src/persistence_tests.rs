use crate::persistence::{persistent_q_communities, PersistenceDiagram};

type Entry = (i32, u64, u64, Vec<u32>);

fn canonical(diagram: &PersistenceDiagram) -> Vec<Entry> {
    let mut entries: Vec<Entry> = (0..diagram.q.len())
        .map(|i| {
            let lo = diagram.offsets[i] as usize;
            let hi = diagram.offsets[i + 1] as usize;
            (
                diagram.q[i],
                diagram.birth[i].to_bits(),
                diagram.death[i].to_bits(),
                diagram.members[lo..hi].to_vec(),
            )
        })
        .collect();
    entries.sort();
    entries
}

fn entry(q: i32, birth: f64, death: f64, members: &[u32]) -> Entry {
    (q, birth.to_bits(), death.to_bits(), members.to_vec())
}

// --- Naive reference: rebuild everything at every threshold, track lineages. ---

#[derive(Clone)]
struct NaiveComp {
    vertices: Vec<u32>,
    birth: f64,
    witness: Vec<u32>,
}

fn is_clique(vertices: &[u32], adj: &[Vec<bool>]) -> bool {
    vertices.iter().enumerate().all(|(i, &a)| {
        vertices[i + 1..]
            .iter()
            .all(|&b| adj[a as usize][b as usize])
    })
}

fn naive_maximal_cliques(n: usize, adj: &[Vec<bool>]) -> Vec<Vec<u32>> {
    let active: Vec<u32> = (0..n as u32)
        .filter(|&v| adj[v as usize].iter().any(|&x| x))
        .collect();
    let mut cliques: Vec<Vec<u32>> = Vec::new();
    for mask in 1u64..(1 << active.len()) {
        let vertices: Vec<u32> = active
            .iter()
            .enumerate()
            .filter(|&(i, _)| mask & (1 << i) != 0)
            .map(|(_, &v)| v)
            .collect();
        if vertices.len() >= 2 && is_clique(&vertices, adj) {
            cliques.push(vertices);
        }
    }
    cliques
        .iter()
        .filter(|c| {
            !cliques
                .iter()
                .any(|other| other.len() > c.len() && is_superset(other, c))
        })
        .cloned()
        .collect()
}

fn is_superset(big: &[u32], small: &[u32]) -> bool {
    small.iter().all(|x| big.binary_search(x).is_ok())
}

fn intersection_size(a: &[u32], b: &[u32]) -> usize {
    let (mut i, mut j, mut count) = (0, 0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                count += 1;
                i += 1;
                j += 1;
            }
        }
    }
    count
}

struct NaiveQComp {
    vertices: Vec<u32>,
    clique_indices: Vec<usize>,
}

fn naive_q_components(cliques: &[Vec<u32>], q: usize) -> Vec<NaiveQComp> {
    let eligible: Vec<usize> = (0..cliques.len())
        .filter(|&i| cliques[i].len() >= q + 1)
        .collect();
    let mut visited = vec![false; cliques.len()];
    let mut components = Vec::new();
    for &start in &eligible {
        if visited[start] {
            continue;
        }
        let mut stack = vec![start];
        visited[start] = true;
        let mut clique_indices = Vec::new();
        while let Some(current) = stack.pop() {
            clique_indices.push(current);
            for &other in &eligible {
                if !visited[other]
                    && intersection_size(&cliques[current], &cliques[other]) >= q + 1
                {
                    visited[other] = true;
                    stack.push(other);
                }
            }
        }
        clique_indices.sort_unstable();
        let mut vertices: Vec<u32> = clique_indices
            .iter()
            .flat_map(|&i| cliques[i].iter().copied())
            .collect();
        vertices.sort_unstable();
        vertices.dedup();
        components.push(NaiveQComp {
            vertices,
            clique_indices,
        });
    }
    components
}

/// Same elder order as the engine: birth, then size desc, then lexicographic members.
fn elder_order(a: &NaiveComp, b: &NaiveComp) -> std::cmp::Ordering {
    a.birth
        .total_cmp(&b.birth)
        .then(b.vertices.len().cmp(&a.vertices.len()))
        .then_with(|| a.vertices.cmp(&b.vertices))
}

fn prepare_edges(edges: &[(u32, u32, f64)]) -> Vec<(u32, u32, f64)> {
    let mut sorted: Vec<(u32, u32, f64)> = edges
        .iter()
        .filter(|&&(u, v, w)| w.is_finite() && u != v)
        .map(|&(u, v, w)| (u.min(v), u.max(v), w))
        .collect();
    sorted.sort_unstable_by(|a, b| a.2.total_cmp(&b.2).then(a.0.cmp(&b.0)).then(a.1.cmp(&b.1)));
    let mut seen = std::collections::HashSet::new();
    sorted.retain(|&(u, v, _)| seen.insert((u, v)));
    sorted
}

fn naive_persistence(
    n: usize,
    edges: &[(u32, u32, f64)],
    max_q: Option<usize>,
) -> Vec<(i32, f64, f64, Vec<u32>)> {
    let filtered = prepare_edges(edges);
    let mut adj = vec![vec![false; n]; n];
    let mut state: Vec<Vec<NaiveComp>> = Vec::new();
    let mut out = Vec::new();

    let mut i = 0;
    while i < filtered.len() {
        let w = filtered[i].2;
        while i < filtered.len() && filtered[i].2 == w {
            let (u, v, _) = filtered[i];
            adj[u as usize][v as usize] = true;
            adj[v as usize][u as usize] = true;
            i += 1;
        }

        let cliques = naive_maximal_cliques(n, &adj);
        let max_dim = cliques.iter().map(|c| c.len() - 1).max().unwrap_or(0);
        let top = match max_q {
            Some(cap) => max_dim.min(cap),
            None => max_dim,
        };
        while state.len() <= top {
            state.push(Vec::new());
        }

        for q in 0..=top {
            let comps = naive_q_components(&cliques, q);
            let mut preds: Vec<Vec<usize>> = vec![Vec::new(); comps.len()];
            for (old_idx, old) in state[q].iter().enumerate() {
                let matches: Vec<usize> = comps
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| {
                        c.clique_indices
                            .iter()
                            .any(|&k| is_superset(&cliques[k], &old.witness))
                    })
                    .map(|(ci, _)| ci)
                    .collect();
                assert_eq!(
                    matches.len(),
                    1,
                    "an old component must land in exactly one new component"
                );
                preds[matches[0]].push(old_idx);
            }

            let mut next: Vec<NaiveComp> = Vec::with_capacity(comps.len());
            for (ci, comp) in comps.into_iter().enumerate() {
                let witness = cliques[comp.clique_indices[0]].clone();
                if preds[ci].is_empty() {
                    next.push(NaiveComp {
                        vertices: comp.vertices,
                        birth: w,
                        witness,
                    });
                    continue;
                }
                let winner = *preds[ci]
                    .iter()
                    .min_by(|&&x, &&y| elder_order(&state[q][x], &state[q][y]))
                    .unwrap();
                for &loser in preds[ci].iter().filter(|&&p| p != winner) {
                    let dead = &state[q][loser];
                    out.push((q as i32, dead.birth, w, dead.vertices.clone()));
                }
                next.push(NaiveComp {
                    vertices: comp.vertices,
                    birth: state[q][winner].birth,
                    witness,
                });
            }
            state[q] = next;
        }
    }

    for (q, comps) in state.iter().enumerate() {
        for comp in comps {
            out.push((q as i32, comp.birth, f64::INFINITY, comp.vertices.clone()));
        }
    }
    out
}

fn canonical_naive(entries: Vec<(i32, f64, f64, Vec<u32>)>) -> Vec<Entry> {
    let mut result: Vec<Entry> = entries
        .into_iter()
        .map(|(q, b, d, m)| (q, b.to_bits(), d.to_bits(), m))
        .collect();
    result.sort();
    result
}

fn assert_matches_naive(n: usize, edges: &[(u32, u32, f64)], max_q: Option<usize>) {
    let engine = canonical(&persistent_q_communities(n, edges, max_q));
    let naive = canonical_naive(naive_persistence(n, edges, max_q));
    assert_eq!(engine, naive, "n={n} max_q={max_q:?} edges={edges:?}");
}

// --- Unit tests ---

#[test]
fn empty_input() {
    let diagram = persistent_q_communities(0, &[], None);
    assert!(diagram.q.is_empty());
    assert_eq!(diagram.offsets, vec![0]);
}

#[test]
fn single_edge() {
    let diagram = persistent_q_communities(2, &[(0, 1, 0.5)], None);
    let expected = vec![
        entry(0, 0.5, f64::INFINITY, &[0, 1]),
        entry(1, 0.5, f64::INFINITY, &[0, 1]),
    ];
    assert_eq!(canonical(&diagram), expected);
}

#[test]
fn triangle_distinct_weights() {
    let edges = [(0, 1, 1.0), (1, 2, 2.0), (0, 2, 3.0)];
    let diagram = persistent_q_communities(3, &edges, None);
    let expected = vec![
        entry(0, 1.0, f64::INFINITY, &[0, 1, 2]),
        entry(1, 1.0, f64::INFINITY, &[0, 1, 2]),
        entry(1, 2.0, 3.0, &[1, 2]),
        entry(2, 3.0, f64::INFINITY, &[0, 1, 2]),
    ];
    assert_eq!(canonical(&diagram), expected);
    assert_matches_naive(3, &edges, None);
}

#[test]
fn triangle_capped_q_keeps_top_level_connected() {
    // With the old all-k-cliques core, max_q=1 left the three edges disconnected at q=1.
    let edges = [(0, 1, 1.0), (1, 2, 2.0), (0, 2, 3.0)];
    let diagram = persistent_q_communities(3, &edges, Some(1));
    let expected = vec![
        entry(0, 1.0, f64::INFINITY, &[0, 1, 2]),
        entry(1, 1.0, f64::INFINITY, &[0, 1, 2]),
        entry(1, 2.0, 3.0, &[1, 2]),
    ];
    assert_eq!(canonical(&diagram), expected);
    assert_matches_naive(3, &edges, Some(1));
}

#[test]
fn equal_weight_batch_suppresses_zero_persistence() {
    // Square at w=1, then one diagonal at w=2.
    let edges = [
        (0, 1, 1.0),
        (1, 2, 1.0),
        (2, 3, 1.0),
        (0, 3, 1.0),
        (0, 2, 2.0),
    ];
    let diagram = persistent_q_communities(4, &edges, None);
    let expected = vec![
        entry(0, 1.0, f64::INFINITY, &[0, 1, 2, 3]),
        entry(1, 1.0, 2.0, &[0, 3]),
        entry(1, 1.0, 2.0, &[1, 2]),
        entry(1, 1.0, 2.0, &[2, 3]),
        entry(1, 1.0, f64::INFINITY, &[0, 1, 2, 3]),
        entry(2, 2.0, f64::INFINITY, &[0, 1, 2]),
        entry(2, 2.0, f64::INFINITY, &[0, 2, 3]),
    ];
    assert_eq!(canonical(&diagram), expected);
    assert_matches_naive(4, &edges, None);
}

#[test]
fn clique_growth_keeps_lineage() {
    // K4 at w=1, then vertex 4 attaches one spoke at a time.
    let mut edges = vec![
        (0, 1, 1.0),
        (0, 2, 1.0),
        (0, 3, 1.0),
        (1, 2, 1.0),
        (1, 3, 1.0),
        (2, 3, 1.0),
        (4, 0, 2.0),
        (4, 1, 3.0),
        (4, 2, 4.0),
        (4, 3, 5.0),
    ];
    let diagram = persistent_q_communities(5, &edges, None);
    let entries = canonical(&diagram);
    // The K4 community keeps its birth through absorption into K5.
    assert!(entries.contains(&entry(3, 1.0, f64::INFINITY, &[0, 1, 2, 3, 4])));
    assert!(entries.contains(&entry(4, 5.0, f64::INFINITY, &[0, 1, 2, 3, 4])));
    // {0,1,2,4} lived at q=3 from w=4 until the K5 merge at w=5.
    assert!(entries.contains(&entry(3, 4.0, 5.0, &[0, 1, 2, 4])));
    assert_matches_naive(5, &edges, None);

    edges.reverse();
    assert_matches_naive(5, &edges, None);
}

#[test]
fn duplicate_edges_and_self_loops_are_ignored() {
    let edges = [
        (1, 1, 0.5),
        (0, 1, 1.0),
        (1, 0, 2.0),
        (0, 1, 3.0),
        (1, 2, f64::NAN),
    ];
    let diagram = persistent_q_communities(3, &edges, None);
    let expected = vec![
        entry(0, 1.0, f64::INFINITY, &[0, 1]),
        entry(1, 1.0, f64::INFINITY, &[0, 1]),
    ];
    assert_eq!(canonical(&diagram), expected);
}

// --- Property test against the naive reference ---

struct Lcg(u64);

impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[test]
fn property_matches_naive_reference() {
    for seed in 0..96u64 {
        let n = 5 + (seed % 6) as usize;
        let density = [0.25, 0.5, 0.85, 1.0][(seed % 4) as usize];
        let max_q = [None, Some(2), Some(0), Some(4)][(seed % 4) as usize];
        // Coarse weights produce many equal-weight batches, fine weights none.
        let weight_steps = [10.0, 4.0, 1000.0][(seed % 3) as usize];
        let mut rng = Lcg(seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1));

        let mut edges = Vec::new();
        for u in 0..n as u32 {
            for v in (u + 1)..n as u32 {
                if rng.next_f64() < density {
                    let weight = (rng.next_f64() * weight_steps).round() / weight_steps;
                    edges.push((u, v, weight));
                }
            }
        }
        assert_matches_naive(n, &edges, max_q);
    }
}
