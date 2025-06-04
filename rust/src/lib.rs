use hashbrown::hash_set::HashSet;
use graph_q_components::find_all_q_connected_components;
use std::cmp::Ordering;
use std::collections::VecDeque;

use std::collections::{BTreeSet, HashMap};

use crate::types::*;

mod graph_q_components;
mod pybindings;
#[cfg(test)]
mod tests;
mod types;


fn calculate_simplex_dimension(simplex: &Simplex) -> SimplexDimension {
    if simplex.is_empty() {
        -1
    } else {
        simplex.len() as SimplexDimension - 1
    }
}

pub fn compare_f64_nan_first(a: &f64, b: &f64) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (false, false) => a.partial_cmp(b).unwrap(),
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
        (true, true) => Ordering::Equal,
    }
}

fn get_shared_face_dimension(
    s_idx1: SimplexIndex,
    s_idx2: SimplexIndex,
    simplices_ref: &[Simplex],
    simplex_dims_ref: &[SimplexDimension],
    q_for_context: usize,
) -> SimplexDimension {
    if simplex_dims_ref[s_idx1] < q_for_context as SimplexDimension
        || simplex_dims_ref[s_idx2] < q_for_context as SimplexDimension
    {
        return -1;
    }

    let simplex1 = &simplices_ref[s_idx1];
    let simplex2 = &simplices_ref[s_idx2];
    let shared_vertices_count = simplex1.intersection(simplex2).count();

    let shared_face_dim = if shared_vertices_count == 0 {
        -1
    } else {
        shared_vertices_count as SimplexDimension - 1
    };

    shared_face_dim
}

/// Iteratively finds components for a given q-level and scope, and queues further work.
fn find_components_and_queue_further_work(
    scope_simplex_indices: Vec<SimplexIndex>,
    current_q: usize,
    all_simplices: &[Simplex],
    all_simplex_dims: &[SimplexDimension],
    results: &mut Vec<Vec<HashSet<SimplexIndex>>>,
    max_overall_dim: SimplexDimension,
    work_queue: &mut VecDeque<(Vec<SimplexIndex>, usize)>,
) {
    if current_q as SimplexDimension > max_overall_dim {
        return;
    }

    let eligible_indices: Vec<SimplexIndex> = scope_simplex_indices
        .into_iter()
        .filter(|&idx| all_simplex_dims[idx] >= current_q as SimplexDimension)
        .collect();

    if eligible_indices.is_empty() {
        return;
    }

    let mut visited_in_scope: HashSet<SimplexIndex> =
        HashSet::with_capacity(eligible_indices.len());
    let mut component_storage: HashSet<SimplexIndex> =
        HashSet::with_capacity(eligible_indices.len());
    let mut stack_storage: Vec<SimplexIndex> = vec![0 as SimplexIndex; eligible_indices.len()];

    for &start_idx in &eligible_indices {
        if visited_in_scope.contains(&start_idx) {
            continue;
        }

        component_storage.clear();
        let mut stack_top: usize = 0;
        let mut min_shared_dim_active_in_component: SimplexDimension = std::isize::MAX;

        if stack_top < stack_storage.len() {
            stack_storage[stack_top] = start_idx;
            stack_top += 1;
        }

        visited_in_scope.insert(start_idx);
        component_storage.insert(start_idx);

        while stack_top > 0 {
            // DFS
            stack_top -= 1;
            let current_s_idx = stack_storage[stack_top];

            for &neighbor_idx in &eligible_indices {
                if current_s_idx == neighbor_idx || visited_in_scope.contains(&neighbor_idx) {
                    continue;
                }

                let shared_dim = get_shared_face_dimension(
                    current_s_idx,
                    neighbor_idx,
                    all_simplices,
                    all_simplex_dims,
                    current_q,
                );

                if shared_dim >= current_q as SimplexDimension {
                    visited_in_scope.insert(neighbor_idx);
                    component_storage.insert(neighbor_idx);
                    min_shared_dim_active_in_component =
                        min_shared_dim_active_in_component.min(shared_dim);

                    if stack_top < stack_storage.len() {
                        stack_storage[stack_top] = neighbor_idx;
                        stack_top += 1;
                    }
                }
            }
        }

        if !component_storage.is_empty() {
            let max_q_this_component_satisfies: SimplexDimension = if component_storage.len() == 1 {
                let simplex_idx = *component_storage.iter().next().unwrap();
                all_simplex_dims[simplex_idx]
            } else {
                if min_shared_dim_active_in_component == std::isize::MAX {
                    current_q as SimplexDimension
                } else {
                    min_shared_dim_active_in_component
                }
            };

            let start_q_for_results = current_q as isize;
            let end_q_for_results = max_q_this_component_satisfies;

            if end_q_for_results >= start_q_for_results {
                for q_isize in start_q_for_results..=end_q_for_results {
                    if q_isize < 0 {
                        continue;
                    }
                    let q_usize = q_isize as usize;
                    if results.len() <= q_usize {
                        results.resize_with(q_usize + 1, Vec::new);
                    }
                    results[q_usize].push(
                        component_storage
                            .iter()
                            .cloned()
                            .collect::<HashSet<SimplexIndex>>(),
                    );
                }
            }

            let next_q_to_explore_isize = max_q_this_component_satisfies + 1;
            if next_q_to_explore_isize <= max_overall_dim && next_q_to_explore_isize >= 0 {
                let next_q_to_explore_usize = next_q_to_explore_isize as usize;
                let next_scope_for_recursion: Vec<SimplexIndex> = component_storage
                    .iter()
                    .cloned()
                    .filter(|&idx| all_simplex_dims[idx] >= next_q_to_explore_isize)
                    .collect();
                if !next_scope_for_recursion.is_empty() {
                    work_queue.push_back((next_scope_for_recursion, next_q_to_explore_usize));
                }
            }
        }
    }
}

pub fn find_hierarchical_q_components(simplices: Vec<Simplex>) -> Vec<Vec<HashSet<SimplexIndex>>> {
    if simplices.is_empty() {
        return Vec::new();
    }

    let simplex_dims: Vec<SimplexDimension> =
        simplices.iter().map(calculate_simplex_dimension).collect();

    let max_overall_dim = simplex_dims.iter().max().cloned().unwrap_or(-1);
    if max_overall_dim < 0 {
        let mut results = Vec::new();
        if !simplices.is_empty() {
            results.resize_with(1, Vec::new);
        }
        return results;
    }

    let results_vec_size = (max_overall_dim + 1) as usize;
    let mut results: Vec<Vec<HashSet<SimplexIndex>>> = vec![Vec::new(); results_vec_size];

    let initial_scope_indices: Vec<SimplexIndex> = (0..simplices.len()).collect();

    let mut work_queue: VecDeque<(Vec<SimplexIndex>, usize)> = VecDeque::new();
    if !initial_scope_indices.is_empty() {
        work_queue.push_back((initial_scope_indices, 0));
    }

    while let Some((current_processing_scope, q)) = work_queue.pop_front() {
        find_components_and_queue_further_work(
            current_processing_scope,
            q,
            &simplices,
            &simplex_dims,
            &mut results,
            max_overall_dim,
            &mut work_queue,
        );
    }

    results
}

/// Calculates persistent q-connected components from a distance matrix and thresholds.
pub fn calculate_persistent_q_components(
    edges: Vec<(VertexId, VertexId, f64)>,
    max_q: Option<usize>,
) -> Vec<PersistenceEntry> {
    if edges.is_empty() {
        return Vec::new();
    }

    let mut sorted_edges = edges
        .into_iter()
        .filter(|(_, _, weight)| weight.is_finite())
        .collect::<Vec<_>>();

    sorted_edges
        .sort_by(|(_, _, weight_a), (_, _, weight_b)| compare_f64_nan_first(weight_a, weight_b));

    let mut persistence_results: Vec<PersistenceEntry> = Vec::new();
    let mut active_components: HashMap<CanonicalComponentId, f64> = HashMap::new();
    let mut adjacency_map: HashMap<VertexId, BTreeSet<VertexId>> = HashMap::new();

    for current_edge in &sorted_edges {
        adjacency_map
            .entry(current_edge.0)
            .and_modify(|set| {
                set.insert(current_edge.1);
            })
            .or_insert(BTreeSet::from([current_edge.1]));

        adjacency_map
            .entry(current_edge.1)
            .and_modify(|set| {
                set.insert(current_edge.0);
            })
            .or_insert(BTreeSet::from([current_edge.0]));

        let hierarchical_q_components_indices: HashSet<(isize, BTreeSet<VertexId>)> =
            find_all_q_connected_components(&adjacency_map, max_q)
                .into_iter()
                .flat_map(|(key, vec_of_components)| {
                    vec_of_components
                        .into_iter()
                        .map(move |btreeset| {
                            (key, btreeset)
                        })
                })
                .collect();

        let (old_components, new_components): (HashSet<_>, HashSet<_>) =
            hierarchical_q_components_indices
                .into_iter()
                .partition(|key| active_components.contains_key(key));

        let (preserved_active_components, dead_components) = active_components
            .clone()
            .into_iter()
            .partition(|(key, _value)| old_components.contains(key));

        active_components = preserved_active_components;
        for new_component in new_components.into_iter() {
            active_components.insert(new_component, current_edge.2);
        }

        for ((q, component_vertices), dead_component_birth) in dead_components.into_iter() {
            persistence_results.push((
                (q, component_vertices.into_iter().collect()),
                dead_component_birth,
                current_edge.2,
            ));
        }
    }

    // Handle components that survive all thresholds
    for ((q, component_vertices), dead_component_birth) in active_components.into_iter() {
        persistence_results.push((
            (q, component_vertices.into_iter().collect()),
            dead_component_birth,
            sorted_edges.last().unwrap().2,
        ));
    }

    persistence_results
}
