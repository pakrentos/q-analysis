use std::collections::BTreeSet;
use hashbrown::hash_set::HashSet;

pub(crate) type VertexId = usize;
pub(crate) type Simplex = HashSet<VertexId>;
pub(crate) type SimplexIndex = usize;
pub(crate) type SimplexDimension = isize;

pub(crate) type Clique = BTreeSet<VertexId>;
pub(crate) type CliqueId = usize;
