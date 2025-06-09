"""
Q-analysis Package
Copyright (C) 2024 Nikita Smirnov

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
import networkx as nx

def havel_hakimi_generator(degrees):
    """Construct adjacency matrix for a simple graph with given degrees"""
    n = len(degrees)
    adj_matrix = np.zeros((n, n), dtype=int)
    nodes = [(i, deg) for i, deg in enumerate(degrees)]
    
    while any(deg > 0 for _, deg in nodes):
        nodes.sort(key=lambda x: x[1], reverse=True)
        
        current, deg = nodes[0]
        
        for i in range(1, deg + 1):
            if i >= len(nodes):
                return None
            
            neighbor = nodes[i][0]
            
            if current != neighbor and adj_matrix[current][neighbor] == 0:
                adj_matrix[current][neighbor] = 1
                adj_matrix[neighbor][current] = 1
                
                nodes[0] = (nodes[0][0], nodes[0][1] - 1)
                nodes[i] = (nodes[i][0], nodes[i][1] - 1)
    
    return adj_matrix

def generate_networks(n, m, n_samples):
    barabasi_adj_matrices = np.array([
        nx.to_numpy_array(
            nx.barabasi_albert_graph(n, m, seed=i)
        )
        for i in range(n_samples)
    ])
    barabasi_degrees = np.array([
        adj.sum(axis=1) for adj in barabasi_adj_matrices
    ]).astype(int)

    configuration_adj_matrices = np.array([
        havel_hakimi_generator(degrees)
        for degrees in barabasi_degrees
    ])

    return barabasi_adj_matrices, configuration_adj_matrices
