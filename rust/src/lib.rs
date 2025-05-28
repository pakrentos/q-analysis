use hashbrown::hash_set::HashSet;
use std::collections::VecDeque;
use std::num::NonZeroUsize;

// PyO3 imports
use pyo3::prelude::*;
use pyo3::types::{PyList, PySet};

// Petgraph imports for persistence
use petgraph::algo;
use petgraph::graph::{Graph, NodeIndex, UnGraph};
use std::collections::{BTreeSet, HashMap};
use utils::compare_f64_nan_first;
use std::time::Instant;
mod utils;

type VertexId = usize;
type Simplex = HashSet<VertexId>;
type SimplexIndex = usize;
type SimplexDimension = isize;

type ComponentVertices = BTreeSet<VertexId>; // Representation of vertices in a component
type CanonicalComponentId = (usize, ComponentVertices); // (q_level, vertices)
type PersistenceEntry = ((usize, Vec<VertexId>), f64, f64); // ((q, sorted_vertices), birth_threshold, death_threshold)

fn calculate_simplex_dimension(simplex: &Simplex) -> SimplexDimension {
    if simplex.is_empty() {
        -1
    } else {
        simplex.len() as SimplexDimension - 1
    }
}

fn get_shared_face_dimension(
    s_idx1: SimplexIndex,
    s_idx2: SimplexIndex,
    simplices_ref: &[Simplex],
    simplex_dims_ref: &[SimplexDimension],
    q_for_context: usize, // The q-level being currently processed
) -> SimplexDimension {
    // Early exit: if either simplex's dimension is less than the current q,
    // they cannot be q-near or part of a q-chain at this q-level.
    if simplex_dims_ref[s_idx1] < q_for_context as SimplexDimension
        || simplex_dims_ref[s_idx2] < q_for_context as SimplexDimension
    {
        return -1; // Indicates they cannot be q-near at this q_for_context
    }


    // Calculate it directly
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
    scope_simplex_indices: Vec<SimplexIndex>, // Takes ownership
    current_q: usize,
    all_simplices: &[Simplex],
    all_simplex_dims: &[SimplexDimension],
    results: &mut Vec<Vec<HashSet<SimplexIndex>>>,
    max_overall_dim: SimplexDimension,
    work_queue: &mut VecDeque<(Vec<SimplexIndex>, usize)>, // For queueing next scopes
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
            // Initial push
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

pub fn find_hierarchical_q_components(
    simplices: Vec<Simplex>,
) -> Vec<Vec<HashSet<SimplexIndex>>> {
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

#[pyfunction]
fn py_find_hierarchical_q_components(
    py: Python,
    py_simplices: PyObject,
) -> PyResult<PyObject> {
    let rust_simplices: Vec<Simplex> = py_simplices
        .extract::<Vec<Vec<VertexId>>>(py)?
        .into_iter()
        .map(|vertex_list| vertex_list.into_iter().collect::<HashSet<VertexId>>())
        .collect();

    let rust_result: Vec<Vec<HashSet<SimplexIndex>>> =
        find_hierarchical_q_components(rust_simplices);

    let py_outer_list = PyList::empty_bound(py);
    for q_level_components_rust in rust_result {
        let py_inner_list_for_q_level = PyList::empty_bound(py);
        for component_rust in q_level_components_rust {
            let py_set_for_component =
                PySet::new_bound(py, &component_rust.into_iter().collect::<Vec<_>>())?;
            py_inner_list_for_q_level.append(py_set_for_component)?;
        }
        py_outer_list.append(py_inner_list_for_q_level)?;
    }

    Ok(py_outer_list.into_py(py))
}

/// Calculates persistent q-connected components from a distance matrix and thresholds.
pub fn calculate_persistent_q_components(
    edges: Vec<(VertexId, VertexId, f64)>,
) -> Vec<PersistenceEntry> {
    if edges.is_empty() {
        return Vec::new();
    }

    let mut thresholds = edges
        .iter()
        .filter_map(|(_, _, weight)| weight.is_finite().then_some(*weight))
        .collect::<Vec<_>>();

    thresholds.sort_by(utils::compare_f64_nan_first);

    let mut persistence_results: Vec<PersistenceEntry> = Vec::new();
    let mut active_components: HashMap<CanonicalComponentId, f64> = HashMap::new();

    for current_threshold in &thresholds {
        let present_nodes: HashSet<_> = edges
            .iter()
            .filter(|(_, _, weight)| weight.is_finite())
            .filter(|(_, _, weight)| current_threshold.partial_cmp(weight) == Some(std::cmp::Ordering::Greater))
            .cloned()
            .map(|(node_a, node_b, _)| NodeIndex::new(node_a.max(node_b) as usize))
            .collect();

        let graph = UnGraph::<VertexId, ()>::from_edges(edges.iter().filter_map(
            |(node_a, node_b, weight)| {
                (weight.is_finite()
                    && current_threshold.partial_cmp(weight) == Some(std::cmp::Ordering::Greater))
                .then(|| (NodeIndex::new(*node_a), NodeIndex::new(*node_b)))
            },
        ));

        let start_time = Instant::now();
        let cliques = algo::maximal_cliques(&graph)
            .into_iter()
            .filter_map(|clique| {
                present_nodes.intersection(&clique).any(|_| true).then(|| {
                    clique
                        .into_iter()
                        .map(|node_idx| node_idx.index())
                        .collect::<HashSet<_>>()
                })
            })
            .collect::<Vec<_>>();
        let duration = start_time.elapsed();
        
        let start_time = Instant::now();
        let hierarchical_q_components_indices = find_hierarchical_q_components(cliques.clone())
            .into_iter()
            .enumerate()
            .map(|(q, components_list)| {
                components_list
                    .into_iter()
                    .map(|list_of_cliques_indecies| {
                        (
                            q,
                            list_of_cliques_indecies
                                .into_iter()
                                .filter_map(|clique_index| cliques.get(clique_index).cloned())
                                .flatten()
                                .collect::<BTreeSet<_>>(),
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .flatten()
            .collect::<HashSet<_>>();
        let duration_components = start_time.elapsed();
        println!("current_threshold: {}, num_cliques: {}, bron-kerbosh took {:?}, components took {:?}", current_threshold, cliques.len(), duration, duration_components);

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
            active_components.insert(new_component, *current_threshold);
        }

        for ((q, component_vertices), dead_component_birth) in dead_components.into_iter() {
            persistence_results.push((
                (q, component_vertices.into_iter().collect()),
                dead_component_birth,
                *current_threshold,
            ));
        }
    }

    // Handle components that survive all thresholds
    for ((q, component_vertices), dead_component_birth) in active_components.into_iter() {
        persistence_results.push((
            (q, component_vertices.into_iter().collect()),
            dead_component_birth,
            *(thresholds.last().unwrap()),
        ));
    }

    persistence_results
}

#[pyfunction]
fn py_calculate_persistent_q_components(
    py: Python,
    py_edges: PyObject,
) -> PyResult<PyObject> {
    let rust_edges: Vec<(VertexId, VertexId, f64)> = py_edges.extract(py)?;

    let rust_result: Vec<PersistenceEntry> =
        calculate_persistent_q_components(rust_edges);

    let py_results_list = PyList::empty_bound(py);
    for ((q_level, vertices), birth, death) in rust_result {
        let py_vertices = PyList::new_bound(py, &vertices); // vertices is already Vec<VertexId>
        let py_component_id = (q_level.to_object(py), py_vertices.to_object(py)).to_object(py);
        let py_entry = (py_component_id, birth.to_object(py), death.to_object(py)).to_object(py);
        py_results_list.append(py_entry)?;
    }
    Ok(py_results_list.into_py(py))
}

#[pymodule]
fn q_analysis(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_find_hierarchical_q_components, m)?)?;
    m.add_function(wrap_pyfunction!(py_calculate_persistent_q_components, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hashbrown::hash_set::HashSet;

    fn set(vertices: &[VertexId]) -> Simplex {
        vertices.iter().cloned().collect()
    }

    #[test]
    fn it_works_empty_input() {
        let simplices = Vec::new();
        let result = find_hierarchical_q_components(simplices); // No Some(1)
        assert!(result.is_empty());
    }

    #[test]
    fn it_works_single_simplex_vertex() {
        let simplices = vec![set(&[0])]; // dim 0
        let result = find_hierarchical_q_components(simplices);
        // Expected: q=0: [[{0}]]
        assert_eq!(result.len(), 1); // Only for q=0
        assert_eq!(result[0].len(), 1);
        assert_eq!(result[0][0], vec![0 as SimplexIndex].into_iter().collect());
    }

    #[test]
    fn it_works_single_simplex_edge() {
        let simplices = vec![set(&[0, 1])]; // dim 1
        let result = find_hierarchical_q_components(simplices);
        // Expected: q=0: [[{0}]], q=1: [[{0}]]
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].len(), 1);
        assert_eq!(result[0][0], vec![0 as SimplexIndex].into_iter().collect());
        assert_eq!(result[1].len(), 1);
        assert_eq!(result[1][0], vec![0 as SimplexIndex].into_iter().collect());
    }

    #[test]
    fn two_disjoint_vertices() {
        let simplices = vec![set(&[0]), set(&[1])]; // both dim 0
        let result = find_hierarchical_q_components(simplices);
        // Expected: q=0: [[{0}], [{1}]] (or vice versa)
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 2);
        let mut comp0_sorted: Vec<HashSet<SimplexIndex>> = result[0].clone();
        comp0_sorted.sort_by_key(|s| *(s.iter().next().unwrap_or(&0)));

        let expected_q0_comp0: HashSet<SimplexIndex> = vec![0].into_iter().collect();
        let expected_q0_comp1: HashSet<SimplexIndex> = vec![1].into_iter().collect();
        assert!(comp0_sorted.contains(&expected_q0_comp0));
        assert!(comp0_sorted.contains(&expected_q0_comp1));
    }

    #[test]
    fn two_0_near_simplices_forming_one_0_component() {
        let simplices = vec![set(&[0, 1]), set(&[1, 2])]; // s0 dim 1, s1 dim 1
        let result = find_hierarchical_q_components(simplices);
        // q=0: [[{0, 1}]] (s0 and s1 are 0-near via {1})
        // q=1: [[{0}], [{1}]] (s0 not 1-near s1 as shared face {1} is dim 0)

        assert_eq!(result.len(), 2);

        assert_eq!(result[0].len(), 1);
        let expected_q0: HashSet<SimplexIndex> = vec![0, 1].into_iter().collect();
        assert_eq!(result[0][0], expected_q0);

        assert_eq!(result[1].len(), 2);
        let mut comp1_sorted: Vec<HashSet<SimplexIndex>> = result[1].clone();
        comp1_sorted.sort_by_key(|s| *(s.iter().next().unwrap_or(&0)));

        let expected_q1_comp0: HashSet<SimplexIndex> = vec![0].into_iter().collect();
        let expected_q1_comp1: HashSet<SimplexIndex> = vec![1].into_iter().collect();
        assert_eq!(comp1_sorted[0], expected_q1_comp0);
        assert_eq!(comp1_sorted[1], expected_q1_comp1);
    }

    #[test]
    fn complex_case_q0_q1_q2() {
        // s0: {0,1,2} (2-simplex)
        // s1: {1,2,3} (2-simplex) -> s0, s1 are 1-near (share {1,2}, dim 1)
        // s2: {3,4,5} (2-simplex) -> s1, s2 are 0-near (share {3}, dim 0)
        // s3: {6,7}   (1-simplex) -> disjoint
        let simplices = vec![
            set(&[0, 1, 2]), // 0: dim 2
            set(&[1, 2, 3]), // 1: dim 2. s0,s1 share {1,2} (dim 1) -> 1-near
            set(&[3, 4, 5]), // 2: dim 2. s1,s2 share {3} (dim 0) -> 0-near
            set(&[6, 7]),    // 3: dim 1. Disjoint.
        ];
        let result = find_hierarchical_q_components(simplices);

        assert_eq!(result.len(), 3); // Max dim 2 -> q=0,1,2

        // Q=0: [[{0,1,2}], [{3}]]
        assert_eq!(result[0].len(), 2);
        let mut q0_comps: Vec<HashSet<SimplexIndex>> = result[0].clone();
        q0_comps.sort_by_key(|c| c.iter().cloned().min().unwrap_or(SimplexIndex::MAX));
        assert_eq!(q0_comps[0], vec![0, 1, 2].into_iter().collect());
        assert_eq!(q0_comps[1], vec![3].into_iter().collect());

        // Q=1: [[{0,1}], [{2}], [{3}]]
        assert_eq!(result[1].len(), 3);
        let mut q1_comps: Vec<HashSet<SimplexIndex>> = result[1].clone();
        q1_comps.sort_by_key(|c| c.iter().cloned().min().unwrap_or(SimplexIndex::MAX));
        assert_eq!(q1_comps[0], vec![0, 1].into_iter().collect());
        assert_eq!(q1_comps[1], vec![2].into_iter().collect());
        assert_eq!(q1_comps[2], vec![3].into_iter().collect());

        // Q=2: [[{0}], [{1}], [{2}]]
        assert_eq!(result[2].len(), 3);
        let mut q2_comps: Vec<HashSet<SimplexIndex>> = result[2].clone();
        q2_comps.sort_by_key(|c| c.iter().cloned().min().unwrap_or(SimplexIndex::MAX));
        assert_eq!(q2_comps[0], vec![0].into_iter().collect());
        assert_eq!(q2_comps[1], vec![1].into_iter().collect());
        assert_eq!(q2_comps[2], vec![2].into_iter().collect());
    }

    #[test]
    fn test_fsv_complex_example() {
        let simplices = vec![
            set(&[0, 3, 4, 6, 8]), // 0: dim 4
            set(&[0, 4, 7]),       // 1: dim 2
            set(&[2]),             // 2: dim 0
            set(&[0, 3]),          // 3: dim 1
            set(&[3]),             // 4: dim 0
            set(&[4]),             // 5: dim 0
            set(&[2, 4, 8]),       // 6: dim 2
            set(&[1, 7]),          // 7: dim 1
            set(&[0, 1, 3, 7]),    // 8: dim 3
            set(&[2]),             // 9: dim 0
            set(&[2, 5, 7, 9]),    // 10: dim 3
            set(&[1, 4, 7]),       // 11: dim 2
            set(&[2, 4]),          // 12: dim 1
            set(&[7]),             // 13: dim 0
            set(&[7]),             // 14: dim 0
            set(&[8]),             // 15: dim 0
            set(&[5, 6, 7, 8]),    // 16: dim 3
            set(&[3, 5, 6, 8]),    // 17: dim 3
            set(&[4, 5, 6, 9]),    // 18: dim 3
            set(&[4, 6]),          // 19: dim 1
        ];

        // Expected FSV: Q_0=1, Q_1=1, Q_2=7, Q_3=6, Q_4=1
        // Corresponding to result[0].len(), result[1].len(), ... result[4].len()
        let expected_fsv = vec![1, 1, 7, 6, 1];
        let result = find_hierarchical_q_components(simplices);

        assert_eq!(result.len(), expected_fsv.len(), "Number of q-levels");
        for q in 0..expected_fsv.len() {
            assert_eq!(
                result[q].len(),
                expected_fsv[q],
                "Number of components for q={}",
                q
            );
        }
    }
}
