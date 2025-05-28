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

