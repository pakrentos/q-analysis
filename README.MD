# Q-analysis Package

This package provides tools for performing Q-analysis on complex networks, implementing key Q-analysis metrics, visualization tools, and statistical utilities with scikit-learn compatible interfaces.

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
├── utils.py
├── viz.py
├── connected_components.py
└── examples/
    └── scale_free_configurational.py
README.md
pyproject.toml
requirements.txt
```

## Usage

Here's a basic example of how to use the package:

```python
import numpy as np
from q_analysis.simplicial_complex import IncidenceSimplicialComplex
from q_analysis.transformers import QReducer
from q_analysis.utils import get_incidence

# Create an adjacency matrix
adj_matrix = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])

# Create an IncidenceSimplicialComplex
incidence = get_incidence(adj_matrix)
complex = IncidenceSimplicialComplex(incidence)

# Calculate the Q-analysis vectors
q_vectors = complex.q_analysis_vectors(as_dataframe=True)
print("Q-analysis vectors:\n", q_vectors)

# Calculate topological entropy
entropy = complex.topological_entropy(q=1)
print("Topological Entropy (q=1):", entropy)

# Use the QReducer transformer
reducer = QReducer(q=1)
reduced_adj = reducer.fit_transform([adj_matrix])[0]
print("Reduced Adjacency Matrix:\n", reduced_adj)
```

### Advanced Example: Network Comparison

Here's a more advanced example comparing scale-free and configurational networks:

```python
from q_analysis.examples.scale_free_configurational import generate_networks
from q_analysis.simplicial_complex import IncidenceSimplicialComplex
from q_analysis.viz import plot_q_analysis_vectors
from q_analysis.stat import consensus_statistic
from q_analysis.utils import calculate_consensus_adjacency_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Generate sample networks
N_SAMPLES, N_NODES, M_PARAMETER = 100, 100, 8
scale_free_networks, configurational_networks = generate_networks(
    N_NODES, M_PARAMETER, N_SAMPLES
)

# Create simplicial complexes and extract Q-analysis vectors
networks = np.concatenate([scale_free_networks, configurational_networks])
from itertools import product
index = product(['Scale free', 'Configurational'], range(N_SAMPLES))

# Create simplicial complexes and get structure vectors
structure_vectors_dfs = [
    IncidenceSimplicialComplex
        .from_adjacency_matrix(network)
        .q_analysis_vectors(as_dataframe=True)
        .assign(Network=net_type, Sample=sample_id)
    for network, (net_type, sample_id) in zip(networks, index)
]

# Combine results and visualize
structure_vectors_df = pd.concat(structure_vectors_dfs, ignore_index=True)
plot_q_analysis_vectors(
    structure_vectors_df, 
    hue="Network", 
    height=3,
    col_wrap=2,
    legend_out=False
)
plt.show()
```

## Features

- **Simplicial Complex Analysis**: Create and analyze simplicial complexes from adjacency matrices
- **Q-Analysis Vectors**: Compute first and second structure vectors
- **Statistical Tools**: Perform statistical tests on network comparisons
- **Visualization**: Plot Q-analysis vectors and other topological properties
- **Connected Components**: Analyze connected components in networks
- **Transformers**: Scikit-learn compatible interfaces for network transformations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0).