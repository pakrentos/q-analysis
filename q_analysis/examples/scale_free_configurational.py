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
    
    # Create empty adjacency matrix
    adj_matrix = np.zeros((n, n), dtype=int)
    
    # Create list of nodes with their desired degrees
    nodes = [(i, deg) for i, deg in enumerate(degrees)]
    
    while any(deg > 0 for _, deg in nodes):
        # Sort nodes by remaining degree (descending)
        nodes.sort(key=lambda x: x[1], reverse=True)
        
        # Take node with highest remaining degree
        current, deg = nodes[0]
        
        # Connect to deg nodes with next highest degrees
        for i in range(1, deg + 1):
            if i >= len(nodes):
                return None  # Should not happen if sequence is graphical
            
            neighbor = nodes[i][0]
            
            # Add edge if it doesn't create self-loop or parallel edge
            if current != neighbor and adj_matrix[current][neighbor] == 0:
                adj_matrix[current][neighbor] = 1
                adj_matrix[neighbor][current] = 1
                
                # Update remaining degrees
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
