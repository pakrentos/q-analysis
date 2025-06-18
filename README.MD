# Q-analysis Package

This package provides tools for performing [Q-analysis](https://en.wikipedia.org/wiki/Q-analysis) on complex networks. It implements methods for constructing simplicial complexes, computing Q-analysis metrics, and includes statistical and visualization utilities. The core computations are accelerated with Rust, and the package provides scikit-learn compatible interfaces.

## Installation

To install the package, run:

```
pip install q-analysis
```

## Package Structure

```
q_analysis/
├── __init__.py
├── simplicial_complex.py
├── stat.py
├── transformers.py
├── datasets.py
├── viz.py
└── examples/
    └── scale_free_configurational.py
scripts/
├──usage_example.py
README.md
pyproject.toml
```

## Usage

This example demonstrates how to compare two ensembles of networks (Scale-Free vs. Configurational), compute their structure vectors, and visualize the results.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product

from q_analysis.examples.scale_free_configurational import generate_networks
from q_analysis.simplicial_complex import SimplicialComplex
from q_analysis.viz import plot_q_analysis_vectors

# 1. Generate sample networks
N_SAMPLES, N_NODES, M_PARAMETER = 100, 100, 8
scale_free_networks, configurational_networks = generate_networks(
    N_NODES, M_PARAMETER, N_SAMPLES
)
networks = np.concatenate([scale_free_networks, configurational_networks])

# 2. Compute graded parameters for each network
index = product(['Scale free', 'Configurational'], range(N_SAMPLES))
simplicial_complex_metrics = [
    SimplicialComplex.from_adjacency_matrix(network)
    .graded_parameters()
    .to_dataframe()
    .assign(Network=net_type, Sample=sample_id)
    for network, (net_type, sample_id) in zip(networks, index)
]
structure_vectors_df = pd.concat(simplicial_complex_metrics, ignore_index=True)

# 3. Visualize the aggregated results
plot_q_analysis_vectors(
    structure_vectors_df, 
    hue="Network", 
    height=3,
    col_wrap=2,
    legend_out=False
)
plt.show()
```

You can find other code snippets in scripts/

## Features

- **Simplicial Complex Analysis**: Create simplicial complexes from adjacency matrices by finding maximal cliques.
- **Q-Analysis Metrics**: Compute various graded parameters including:
  - First, Second, and Third Structure Vectors (FSV, SSV, TSV)
  - Topological Entropy
  - Simplex Counts
  - Shared Faces Counts
- **Statistical Tools**: Includes utilities for network comparison and statistical analysis.
- **Visualization**: Plot Q-analysis vectors and other topological properties.
- **Scikit-learn Compatibility**: Provides transformers for network transformations compatible with scikit-learn pipelines.
- **Rust Backend**: Core algorithms are implemented in Rust for high performance.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0).