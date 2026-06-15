use std::collections::{BTreeSet, HashMap};

use numpy::{IntoPyArray, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PySet};

use hashbrown::HashSet;

use crate::find_hierarchical_q_components;
use crate::graph_q_components::find_all_q_connected_components;
use crate::max_cliques::VertexId as PersistenceVertexId;
use crate::persistence::{persistent_q_communities, PersistenceDiagram};
use crate::types::{Simplex, SimplexIndex, VertexId};

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

fn diagram_to_py(py: Python, diagram: PersistenceDiagram) -> PyResult<PyObject> {
    Ok((
        diagram.q.into_pyarray_bound(py),
        diagram.birth.into_pyarray_bound(py),
        diagram.death.into_pyarray_bound(py),
        diagram.offsets.into_pyarray_bound(py),
        diagram.members.into_pyarray_bound(py),
    )
        .into_py(py))
}

#[pyfunction]
#[pyo3(signature = (matrix, max_q=None))]
fn py_persistent_q_communities_matrix(
    py: Python,
    matrix: PyReadonlyArray2<f64>,
    max_q: Option<usize>,
) -> PyResult<PyObject> {
    let matrix = matrix.as_array();
    if matrix.nrows() != matrix.ncols() {
        return Err(PyValueError::new_err("distance matrix must be square"));
    }
    let num_vertices = matrix.nrows();

    let mut edges = Vec::new();
    for i in 0..num_vertices {
        for j in (i + 1)..num_vertices {
            let weight = matrix[[i, j]];
            if weight.is_finite() {
                edges.push((i as PersistenceVertexId, j as PersistenceVertexId, weight));
            }
        }
    }

    let diagram = py.allow_threads(move || persistent_q_communities(num_vertices, &edges, max_q));
    diagram_to_py(py, diagram)
}

#[pyfunction]
#[pyo3(signature = (edges, max_q=None))]
fn py_persistent_q_communities_edges(
    py: Python,
    edges: PyReadonlyArray2<f64>,
    max_q: Option<usize>,
) -> PyResult<PyObject> {
    let edges = edges.as_array();
    if edges.ncols() != 3 {
        return Err(PyValueError::new_err(
            "edges must have shape (E, 3): rows of [u, v, weight]",
        ));
    }

    let mut parsed = Vec::with_capacity(edges.nrows());
    let mut max_vertex: i64 = -1;
    for row in edges.rows() {
        let (u, v, weight) = (row[0], row[1], row[2]);
        let ids_valid = u >= 0.0
            && v >= 0.0
            && u.fract() == 0.0
            && v.fract() == 0.0
            && u <= u32::MAX as f64
            && v <= u32::MAX as f64;
        if !ids_valid {
            return Err(PyValueError::new_err(
                "vertex ids must be non-negative integers fitting into u32",
            ));
        }
        max_vertex = max_vertex.max(u as i64).max(v as i64);
        parsed.push((u as PersistenceVertexId, v as PersistenceVertexId, weight));
    }
    let num_vertices = (max_vertex + 1).max(0) as usize;

    let diagram = py.allow_threads(move || persistent_q_communities(num_vertices, &parsed, max_q));
    diagram_to_py(py, diagram)
}

#[pymodule]
fn q_analysis(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_find_hierarchical_q_components, m)?)?;
    m.add_function(wrap_pyfunction!(py_find_hierarchical_q_components_within_graph, m)?)?;
    m.add_function(wrap_pyfunction!(py_persistent_q_communities_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(py_persistent_q_communities_edges, m)?)?;
    Ok(())
}
