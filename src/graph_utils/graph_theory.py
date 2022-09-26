"""
Script with all functions necessary to work with graphs.
"""

from typing import List

import networkx as nx
import numpy as np


def build_graph_at_given_state(
    state: np.array, n_vertices: int
) -> nx.Graph: 
    """
    This function builds a graph from an array representing the state. 
    """
    graph = nx.Graph() 
    vertices_list: List[int] = list(range(n_vertices))
    graph.add_nodes_from(vertices_list)

    state_index: int = 0

    for i in range(0, n_vertices):
        for j in range(i+1, n_vertices):
            if state[state_index] == 1:
                graph.add_edge(i,j)
            state_index += 1

    return graph


def calculate_matching_number(graph) -> int: 
    """
    This function calculates all matchings for a given graph and
    it returns the matching number (i.e. the length of the maximum
    matching).
    """
    max_matching = nx.max_weight_matching(graph)
    return len(max_matching)


def calculate_max_abs_val_eigenvalue(graph) -> float:
    """
    This function computes the eigenvalues of the adjacency matrix
    that corresponds to a specific graph. It returns the largest
    eigenvalue in absolute value.
    """
    adjacency_matrix = nx.adjacency_matrix(graph).todense() 
    eigenvals = np.linalg.eigvalsh(adjacency_matrix) 
    eigenvals_abs = abs(eigenvals)
    return max(eigenvals_abs)
