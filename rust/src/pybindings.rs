use std::collections::{BTreeSet, HashMap};

use pyo3::prelude::*;
use pyo3::types::{PyList, PySet};

use hashbrown::HashSet;

use crate::graph_q_components::find_all_q_connected_components;
use crate::types::{Simplex, VertexId, SimplexIndex, PersistenceEntry};
use crate::{find_hierarchical_q_components, calculate_persistent_q_components};

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

#[pyfunction]
fn py_find_hierarchical_q_components_within_graph(
    py: Python,
    edges: PyObject,
    max_q: PyObject
) -> PyResult<PyObject> {
    let rust_edges: Vec<(VertexId, VertexId)> = edges.extract(py)?;

    let mut adjacency: HashMap<VertexId, BTreeSet<VertexId>> = HashMap::new();

    for current_edge in rust_edges {
        adjacency
            .entry(current_edge.0)
            .and_modify(|set| {
                set.insert(current_edge.1);
            })
            .or_insert(BTreeSet::from([current_edge.1]));

        adjacency
            .entry(current_edge.1)
            .and_modify(|set| {
                set.insert(current_edge.0);
            })
            .or_insert(BTreeSet::from([current_edge.0]));
    }

    let mut result =
        find_all_q_connected_components(&adjacency, max_q.extract(py)?).into_iter().collect::<Vec<_>>();
    result.sort_by_key(|(k, _)| *k);

    let py_outer_list = PyList::empty_bound(py);
    for (_, q_level_components_rust) in result {
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

#[pyfunction]
fn py_calculate_persistent_q_components(
    py: Python,
    edges: PyObject,
    max_q: PyObject,
) -> PyResult<PyObject> {
    let rust_edges: Vec<(VertexId, VertexId, f64)> = edges.extract(py)?;

    let rust_result: Vec<PersistenceEntry> =
        calculate_persistent_q_components(rust_edges, max_q.extract(py)?);

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
    m.add_function(wrap_pyfunction!(py_find_hierarchical_q_components_within_graph, m)?)?;
    Ok(())
}
