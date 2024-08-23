# Q-analysis Package

This package provides tools for performing Q-analysis on complex networks, including implementations of key Q-analysis metrics and scikit-learn compatible interfaces.

## Installation

To install the package, run:

```
pip install q-analysis
```

## Package Structure

```
q-analysis/
│
├── q_analysis/
│   ├── __init__.py
│   ├── simplicial_complex.py
│   ├── metrics.py
│   ├── transformers.py
│   └── utils.py
├── README.md
├── setup.py
└── requirements.txt
```

## Usage

Here's a basic example of how to use the package:

```python
import numpy as np
from q_analysis import IncidenceSimplicialComplex, QReducer

# Create an adjacency matrix
adj_matrix = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])

# Create an IncidenceSimplicialComplex
incidence = get_incidence(adj_matrix)
complex = IncidenceSimplicialComplex(incidence)

# Calculate the First Structure Vector
fsv = complex.first_structure_vector()
print("First Structure Vector:", fsv)

# Calculate topological entropy
entropy = complex.topological_entropy(q=1)
print("Topological Entropy (q=1):", entropy)

# Use the QReducer transformer
reducer = QReducer(q=1)
reduced_adj = reducer.fit_transform([adj_matrix])[0]
print("Reduced Adjacency Matrix:\n", reduced_adj)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.
