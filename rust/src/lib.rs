use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use lru::LruCache;
use rayon::ThreadPoolBuilder; // Using ThreadPoolBuilder to configure and build the pool
use std::num::NonZeroUsize;

// PyO3 imports
use pyo3::prelude::*;
use pyo3::types::{PyList, PySet};

// Type aliases for clarity
type VertexId = usize;
type Simplex = HashSet<VertexId>;
type SimplexIndex = usize;
type SimplexDimension = isize;

type CacheKey = (SimplexIndex, SimplexIndex);
type SharedFaceDimensionCache = LruCache<CacheKey, SimplexDimension>;

/// Calculates the dimension of a single simplex.
/// dim(simplex) = |simplex| - 1
fn calculate_simplex_dimension(simplex: &Simplex) -> SimplexDimension {
    if simplex.is_empty() {
        -1 // Dimension of an empty simplex
    } else {
        simplex.len() as SimplexDimension - 1
    }
}

/// Gets the dimension of the shared face between two simplices.
/// Uses an LRU cache to store results of this computation.
/// The `q_for_context` parameter helps in early exit if simplices are too small for the current q-level.
fn get_shared_face_dimension(
    s_idx1: SimplexIndex,
    s_idx2: SimplexIndex,
    simplices_ref: &[Simplex],
    simplex_dims_ref: &[SimplexDimension],
    q_for_context: usize, // The q-level being currently processed
    cache_mutex: &Arc<Mutex<SharedFaceDimensionCache>>,
) -> SimplexDimension {
    // Early exit: if either simplex's dimension is less than the current q,
    // they cannot be q-near or part of a q-chain at this q-level.
    if simplex_dims_ref[s_idx1] < q_for_context as SimplexDimension || 
       simplex_dims_ref[s_idx2] < q_for_context as SimplexDimension {
        return -1; // Indicates they cannot be q-near at this q_for_context
    }

    let key = (s_idx1.min(s_idx2), s_idx1.max(s_idx2));
    
    // Try to get from cache first (read lock)
    // To avoid holding lock too long during computation, we can check, then compute, then lock again to put.
    // Or, simpler: lock, check, compute if needed, put, unlock.
    let mut cache_guard = cache_mutex.lock().unwrap();

    if let Some(shared_dim) = cache_guard.get(&key) {
        return *shared_dim;
    }

    // Not in cache, calculate it
    let simplex1 = &simplices_ref[s_idx1];
    let simplex2 = &simplices_ref[s_idx2];
    let shared_vertices_count = simplex1.intersection(simplex2).count();

    let shared_face_dim = if shared_vertices_count == 0 {
        -1 // No shared vertices means no shared face
    } else {
        shared_vertices_count as SimplexDimension - 1
    };

    cache_guard.put(key, shared_face_dim);
    shared_face_dim
}

/// Recursive function to find q-connected components within a given scope of simplices.
fn find_components_recursive<'scope_lifetime>(
    current_q: usize,
    scope_simplex_indices: Vec<SimplexIndex>,
    max_overall_dim: SimplexDimension,
    simplices_arc: Arc<Vec<Simplex>>,
    simplex_dims_arc: Arc<Vec<SimplexDimension>>,
    cache_arc: Arc<Mutex<SharedFaceDimensionCache>>,
    results_aggregator_arc: Arc<Mutex<Vec<Vec<HashSet<SimplexIndex>>>>>,
    rayon_task_scope: &rayon::Scope<'scope_lifetime>,
) {
    if current_q as SimplexDimension > max_overall_dim {
        return;
    }

    let eligible_indices: Vec<SimplexIndex> = scope_simplex_indices
        .into_iter()
        .filter(|&idx| simplex_dims_arc[idx] >= current_q as SimplexDimension)
        .collect();

    if eligible_indices.is_empty() {
        return;
    }

    let mut visited_in_scope: HashSet<SimplexIndex> = HashSet::new();
    let mut current_q_level_components_for_this_scope: Vec<HashSet<SimplexIndex>> = Vec::new();

    // Pre-allocate storage for DFS stack and component building
    let mut component_storage: HashSet<SimplexIndex> = HashSet::with_capacity(eligible_indices.len());
    let mut stack_storage: Vec<SimplexIndex> = vec![0 as SimplexIndex; eligible_indices.len()]; // Initialize with a default

    for &start_idx in &eligible_indices {
        if visited_in_scope.contains(&start_idx) {
            continue;
        }

        // Flush/reset storage for the new DFS
        component_storage.clear();
        let mut stack_top: usize = 0;

        // Initial push onto manual stack
        if stack_top < stack_storage.len() { // Should always be true here if eligible_indices is not empty
            stack_storage[stack_top] = start_idx;
            stack_top += 1;
        } else if eligible_indices.len() > 0 {
            // This case (eligible_indices.len() > 0 but stack_storage.len() == 0) should not happen
            // Or if stack_top somehow became >= stack_storage.len() before the first push
            // Potentially panic or handle error if eligible_indices.len() > 0 implies stack_storage.len() > 0
        }
        
        visited_in_scope.insert(start_idx);
        component_storage.insert(start_idx);

        while stack_top > 0 {
            stack_top -= 1;
            let current_s_idx = stack_storage[stack_top];

            for &neighbor_idx in &eligible_indices {
                if current_s_idx == neighbor_idx || visited_in_scope.contains(&neighbor_idx) {
                    continue;
                }

                let shared_dim = get_shared_face_dimension(
                    current_s_idx,
                    neighbor_idx,
                    &*simplices_arc,
                    &*simplex_dims_arc,
                    current_q,
                    &cache_arc,
                );

                if shared_dim >= current_q as SimplexDimension {
                    visited_in_scope.insert(neighbor_idx);
                    component_storage.insert(neighbor_idx);
                    // Push onto manual stack
                    if stack_top < stack_storage.len() {
                        stack_storage[stack_top] = neighbor_idx;
                        stack_top += 1;
                    } else {
                        // Error: stack overflow. This implies eligible_indices.len() was too small,
                        // or there's a bug in logic. For a connected component, stack depth shouldn't exceed num_nodes.
                        // Consider panicking or logging an error.
                    }
                }
            }
        }

        if !component_storage.is_empty() {
            // Store the found component for the current q-level
            current_q_level_components_for_this_scope.push(component_storage.clone());

            // If there's a next q-level to explore, spawn a task for it
            if (current_q as SimplexDimension + 1) <= max_overall_dim {
                let next_q_scope = component_storage.iter().cloned().collect::<Vec<_>>();
                let simplices_clone = Arc::clone(&simplices_arc);
                let simplex_dims_clone = Arc::clone(&simplex_dims_arc);
                let cache_clone = Arc::clone(&cache_arc);
                let results_clone = Arc::clone(&results_aggregator_arc);
                
                rayon_task_scope.spawn(move |s_next_task| {
                    find_components_recursive(
                        current_q + 1,
                        next_q_scope,
                        max_overall_dim,
                        simplices_clone,
                        simplex_dims_clone,
                        cache_clone,
                        results_clone,
                        s_next_task,
                    );
                });
            }
        }
    }

    if !current_q_level_components_for_this_scope.is_empty() {
        let mut results_guard = results_aggregator_arc.lock().unwrap();
        // Ensure the vector for current_q exists and is large enough
        if results_guard.len() <= current_q {
            results_guard.resize_with(current_q + 1, Vec::new);
        }
        results_guard[current_q].extend(current_q_level_components_for_this_scope);
    }
}

/// Finds hierarchical q-connected components in a list of simplices.
///
/// # Arguments
/// * `simplices`: A vector of simplices, where each simplex is a HashSet of vertex IDs.
/// * `max_threads_opt`: An optional maximum number of threads for parallel processing. Defaults to 8.
///
/// # Returns
/// A vector of vectors of HashSets of simplex indices.
/// The outer vector is indexed by q-level.
/// Each inner vector contains all q-connected components found at that q-level.
/// Each component is a HashSet of indices referring to the original `simplices` vector.
pub fn find_hierarchical_q_components(
    simplices: Vec<Simplex>,
    max_threads_opt: Option<usize>,
) -> Vec<Vec<HashSet<SimplexIndex>>> {
    if simplices.is_empty() {
        return Vec::new();
    }

    let num_threads = max_threads_opt.unwrap_or(8).max(1);

    let simplex_dims: Vec<SimplexDimension> = simplices
        .iter()
        .map(calculate_simplex_dimension)
        .collect();

    let max_overall_dim = simplex_dims.iter().max().cloned().unwrap_or(-1);

    let results_vec_size = if max_overall_dim < 0 { 1 } else { (max_overall_dim + 1) as usize };
    let results_aggregator_arc = Arc::new(Mutex::new(vec![Vec::new(); results_vec_size]));
    
    let simplices_arc = Arc::new(simplices); // Simplices are moved here
    let simplex_dims_arc = Arc::new(simplex_dims);
    
    let cache_capacity = NonZeroUsize::new(1000000).expect("Cache capacity must be non-zero");
    let cache_arc = Arc::new(Mutex::new(SharedFaceDimensionCache::new(cache_capacity)));
    
    let initial_scope_indices: Vec<SimplexIndex> = (0..simplices_arc.len()).collect();

    let pool = ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .expect("Failed to build thread pool");

    let simplices_for_initial_task = Arc::clone(&simplices_arc);
    let simplex_dims_for_initial_task = Arc::clone(&simplex_dims_arc);
    let cache_for_initial_task = Arc::clone(&cache_arc);
    let results_aggregator_for_initial_task = Arc::clone(&results_aggregator_arc);

    pool.scope(move |s| {
        s.spawn(move |s_task| {
            find_components_recursive(
                0,
                initial_scope_indices,
                max_overall_dim,
                simplices_for_initial_task,
                simplex_dims_for_initial_task,
                cache_for_initial_task,
                results_aggregator_for_initial_task,
                s_task,
            );
        });
    }); 

    // Extract results using the original `results_aggregator_arc` which was not moved.
    match Arc::try_unwrap(results_aggregator_arc) { // No .clone() needed if this is the last use.
        Ok(mutex) => mutex.into_inner().unwrap_or_else(|_| panic!("Mutex poisoned")),
        Err(arc_still_has_refs) => {
            eprintln!("Warning: Could not exclusively unwrap results aggregator after scope. Cloning data.");
            arc_still_has_refs.lock().unwrap().clone()
        }
    }
}

/// Python wrapper for `find_hierarchical_q_components`.
#[pyfunction]
fn py_find_hierarchical_q_components(py: Python, py_simplices: PyObject, max_threads_opt: Option<usize>) -> PyResult<PyObject> {
    // 1. Convert Python input (list of lists/tuples/sets of vertices) to Rust Vec<Simplex>
    let rust_simplices: Vec<Simplex> = py_simplices.extract::<Vec<Vec<VertexId>>>(py)?
        .into_iter()
        .map(|vertex_list| vertex_list.into_iter().collect::<HashSet<VertexId>>())
        .collect();

    // 2. Call the core Rust function
    let rust_result: Vec<Vec<HashSet<SimplexIndex>>> = find_hierarchical_q_components(rust_simplices, max_threads_opt);

    // 3. Convert Rust output (Vec<Vec<HashSet<SimplexIndex>>>) to Python list of lists of sets
    let py_outer_list = PyList::empty_bound(py);
    for q_level_components_rust in rust_result {
        let py_inner_list_for_q_level = PyList::empty_bound(py);
        for component_rust in q_level_components_rust {
            // Convert HashSet<SimplexIndex> to PySet
            let py_set_for_component = PySet::new_bound(py, &component_rust.into_iter().collect::<Vec<_>>())?;
            py_inner_list_for_q_level.append(py_set_for_component)?;
        }
        py_outer_list.append(py_inner_list_for_q_level)?;
    }
    
    Ok(py_outer_list.into_py(py))
}

/// This defines the Python module and its functions.
#[pymodule]
fn q_analysis(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_find_hierarchical_q_components, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn set(vertices: &[VertexId]) -> Simplex {
        vertices.iter().cloned().collect()
    }

    #[test]
    fn it_works_empty_input() {
        let simplices = Vec::new();
        let result = find_hierarchical_q_components(simplices, Some(1));
        assert!(result.is_empty());
    }

    #[test]
    fn it_works_single_simplex_vertex() {
        // A single 0-simplex (vertex)
        let simplices = vec![set(&[0])];
        let result = find_hierarchical_q_components(simplices.clone(), Some(1));
        
        // Expected: q=0: [[{0}]] (component containing simplex 0)
        //           q>0: []
        assert_eq!(result.len(), 1); // Only for q=0
        assert_eq!(result[0].len(), 1); // One 0-component
        assert_eq!(result[0][0], vec![0 as SimplexIndex].into_iter().collect());
    }

    #[test]
    fn it_works_single_simplex_edge() {
        // A single 1-simplex (edge)
        let simplices = vec![set(&[0, 1])];
        let result = find_hierarchical_q_components(simplices.clone(), Some(1));
        // Expected: q=0: [[{0}]]
        //           q=1: [[{0}]]
        assert_eq!(result.len(), 2); // For q=0 and q=1
        assert_eq!(result[0].len(), 1);
        assert_eq!(result[0][0], vec![0 as SimplexIndex].into_iter().collect());
        assert_eq!(result[1].len(), 1);
        assert_eq!(result[1][0], vec![0 as SimplexIndex].into_iter().collect());
    }

    #[test]
    fn two_disjoint_vertices() {
        let simplices = vec![set(&[0]), set(&[1])];
        let result = find_hierarchical_q_components(simplices.clone(), Some(1));
        // Expected: q=0: [[{0}], [{1}]] (or [{1}], [{0}])
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 2); // Two 0-components
        // Convert to something sortable for comparison if order isn't guaranteed
        let mut comp0_sorted: Vec<HashSet<SimplexIndex>> = result[0].clone();
        comp0_sorted.sort_by_key(|s| *(s.iter().next().unwrap_or(&0))); // Corrected: dereference to get owned value for key

        let expected_q0_comp0: HashSet<SimplexIndex> = vec![0].into_iter().collect();
        let expected_q0_comp1: HashSet<SimplexIndex> = vec![1].into_iter().collect();
        assert!(comp0_sorted.contains(&expected_q0_comp0));
        assert!(comp0_sorted.contains(&expected_q0_comp1));
    }

    #[test]
    fn two_0_near_simplices_forming_one_0_component() {
        // s0 = {0,1}, s1 = {1,2}. They share vertex 1. Both are 1-simplices.
        let simplices = vec![set(&[0, 1]), set(&[1, 2])];
        let result = find_hierarchical_q_components(simplices.clone(), Some(1));
        // q=0: [[{0, 1}]] (s0 and s1 are 0-near)
        // q=1: [[{0}], [{1}]] (s0 and s1 are not 1-near, shared face {1} has dim 0)
        
        assert_eq!(result.len(), 2); // q=0, q=1

        // q=0 components
        assert_eq!(result[0].len(), 1); // One 0-component
        let expected_q0: HashSet<SimplexIndex> = vec![0, 1].into_iter().collect();
        assert_eq!(result[0][0], expected_q0);

        // q=1 components
        assert_eq!(result[1].len(), 2); // Two 1-components (each simplex itself)
        let mut comp1_sorted: Vec<HashSet<SimplexIndex>> = result[1].clone();
        comp1_sorted.sort_by_key(|s| *(s.iter().next().unwrap_or(&0))); // Corrected: dereference to get owned value for key

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
            set(&[0, 1, 2]), // 0
            set(&[1, 2, 3]), // 1
            set(&[3, 4, 5]), // 2
            set(&[6, 7]),    // 3
        ];
        let result = find_hierarchical_q_components(simplices.clone(), Some(2));

        // Max dim is 2. Result length should be 3 (for q=0, q=1, q=2)
        assert_eq!(result.len(), 3);

        // Q=0:
        // s0,s1,s2 are 0-connected. s3 is separate.
        // Expected: [[{0,1,2}], [{3}]]
        assert_eq!(result[0].len(), 2);
        let mut q0_comps: Vec<HashSet<SimplexIndex>> = result[0].clone();
        // Sort components by size or first element to make comparison deterministic
        q0_comps.sort_by_key(|c| c.iter().cloned().min().unwrap_or(SimplexIndex::MAX));

        let expected_q0_c1: HashSet<SimplexIndex> = vec![0,1,2].into_iter().collect();
        let expected_q0_c2: HashSet<SimplexIndex> = vec![3].into_iter().collect();
        assert_eq!(q0_comps[0], expected_q0_c1);
        assert_eq!(q0_comps[1], expected_q0_c2);
        
        // Q=1:
        // Within {0,1,2}: s0, s1 are 1-near. s2 is not 1-near to s0 or s1.
        //   So, sub-components are [{0,1}], [{2}]
        // Within {3}: s3 is a 1-simplex, so [{3}]
        // Expected: [[{0,1}], [{2}], [{3}]]
        assert_eq!(result[1].len(), 3);
        let mut q1_comps: Vec<HashSet<SimplexIndex>> = result[1].clone();
        q1_comps.sort_by_key(|c| c.iter().cloned().min().unwrap_or(SimplexIndex::MAX));
        
        let expected_q1_c1: HashSet<SimplexIndex> = vec![0,1].into_iter().collect();
        let expected_q1_c2: HashSet<SimplexIndex> = vec![2].into_iter().collect();
        let expected_q1_c3: HashSet<SimplexIndex> = vec![3].into_iter().collect();
        assert_eq!(q1_comps[0], expected_q1_c1);
        assert_eq!(q1_comps[1], expected_q1_c2);
        assert_eq!(q1_comps[2], expected_q1_c3);

        // Q=2:
        // Within {0,1} (from q=1): s0 is 2-dim, s1 is 2-dim. They are not 2-near.
        //   So, sub-components are [{0}], [{1}]
        // Within {2} (from q=1): s2 is 2-dim. So [{2}]
        // Within {3} (from q=1): s3 is 1-dim (dim < q=2). So empty from this branch.
        // Expected: [[{0}], [{1}], [{2}]]
        assert_eq!(result[2].len(), 3);
        let mut q2_comps: Vec<HashSet<SimplexIndex>> = result[2].clone();
        q2_comps.sort_by_key(|c| c.iter().cloned().min().unwrap_or(SimplexIndex::MAX));

        let expected_q2_c1: HashSet<SimplexIndex> = vec![0].into_iter().collect();
        let expected_q2_c2: HashSet<SimplexIndex> = vec![1].into_iter().collect();
        let expected_q2_c3: HashSet<SimplexIndex> = vec![2].into_iter().collect();
        assert_eq!(q2_comps[0], expected_q2_c1);
        assert_eq!(q2_comps[1], expected_q2_c2);
        assert_eq!(q2_comps[2], expected_q2_c3);
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

        let result = find_hierarchical_q_components(simplices.clone(), Some(4)); // Use 4 threads for this larger test

        // Max dimension in the complex is 4 (from simplex 0).
        // So, results should be available for q=0, 1, 2, 3, 4.
        // The length of the result vector should be max_dim + 1 = 5.
        assert_eq!(result.len(), expected_fsv.len(), "Number of q-levels in result does not match expected FSV length");

        for q in 0..expected_fsv.len() {
            assert_eq!(
                result[q].len(), 
                expected_fsv[q], 
                "Number of components for q={} does not match. Expected: {}, Got: {}. Components: {:?}", 
                q, 
                expected_fsv[q], 
                result[q].len(),
                result[q]
            );
        }
    }
}
