"""
Script with all functions necessary to work with graphs.
"""

import logging
import sys
from typing import List

import networkx as nx
import numpy as np
import scipy as sp

logging.basicConfig(
    stream=sys.stdout,
    datefmt='%Y-%m-%d %H:%M',
    format='%(asctime)s | %(message)s'
)
log = logging.getLogger(__name__)


def build_graph_from_array(
    array: np.ndarray, n_vertices: int,
) -> nx.Graph: 
    """
    This function builds a graph from an array representing the state. 
    """
    graph = nx.Graph() 
    vertices_list: List[int] = list(range(n_vertices))
    graph.add_nodes_from(vertices_list)

    index: int = 0

    for i in range(0, n_vertices):
        for j in range(i+1, n_vertices):
            if array[index] == 1:
                graph.add_edge(i,j)
            index += 1

    return graph


def calculate_matching_number(graph: nx.Graph) -> int: 
    """
    This function calculates all matchings for a given graph and
    it returns the matching number (i.e. the length of the maximum
    matching).
    """
    max_matching = nx.max_weight_matching(graph)
    return len(max_matching)


def calculate_max_abs_val_eigenvalue(graph: nx.Graph) -> float:
    """
    This function computes the eigenvalues of the adjacency matrix
    that corresponds to a specific graph. It returns the largest
    eigenvalue in absolute value.
    """
    adjacency_matrix = nx.adjacency_matrix(graph).todense() 
    eigenvals = np.linalg.eigvalsh(adjacency_matrix) 
    eigenvals_abs = abs(eigenvals)
    return max(eigenvals_abs)


def signless_laplacian_matrix(G, nodelist=None, weight='weight'):
    """
    Returns the signless Laplacian matrix of G, L = D + A, where
    A is the adjacency matrix and D is the diagonal matrix of node degrees.
    """
    if nodelist is None:
        nodelist = list(G)
    adj_mat = nx.to_scipy_sparse_matrix(
        G, nodelist=nodelist, weight=weight, format='csr'
    )
    n, m = adj_mat.shape
    diags = adj_mat.sum(axis=1)
    deg_mat = sp.sparse.spdiags(diags.flatten(), [0], m, n, format='csr')
    return deg_mat + adj_mat


def calculate_laplacian_eigenvalues(
    graph: nx.Graph, signless_laplacian: bool,
) -> List[float]:
    """
    This function computes the eigenvalues of the laplacian
    matrix that corresponds to a specific graph.
    """
    laplacian_matrix = nx.laplacian_matrix(graph).todense()
    if signless_laplacian: 
        laplacian_matrix = signless_laplacian_matrix(graph).todense()
    eigenvals = np.linalg.eigvalsh(laplacian_matrix)
    return eigenvals


def print_counterexample_to_file(
    method: str, conjecture: str, counterexample: np.ndarray, 
) -> None:
    """
    This function prints the graph that has been identified as a
    counterexample into a .txt file in the output/ directory.
    """
    log.info(
        f'A counterexample to {conjecture}\'s conjecture'
        f'has been found using {method}!'
    )

    # We write the graph into a text file and exit the program
    file = open(f'../output/{method}/counterexample.txt', 'w+') 
    content = str(counterexample)
    file.write(content)
    file.close()
    exit()