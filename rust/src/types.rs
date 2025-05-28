use std::collections::BTreeSet;
use hashbrown::hash_set::HashSet;

pub(crate) type VertexId = usize;
pub(crate) type Simplex = HashSet<VertexId>;
pub(crate) type SimplexIndex = usize;
pub(crate) type SimplexDimension = isize;

pub(crate) type ComponentVertices = BTreeSet<VertexId>; // Representation of vertices in a component
pub(crate) type CanonicalComponentId = (usize, ComponentVertices); // (q_level, vertices)
pub(crate) type PersistenceEntry = ((usize, Vec<VertexId>), f64, f64); // ((q, sorted_vertices), birth_threshold, death_threshold)

pub(crate) type Clique = BTreeSet<VertexId>;
pub(crate) type CliqueId = usize;
