"""
Script with all functions necessary to work with graphs.
"""

import logging
import sys
from typing import List

import networkx as nx
import numpy as np

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


def calculate_laplacian_eigenvalues(
    graph: nx.Graph, signless_laplacian: bool,
) -> List[float]:
    """
    This function computes the eigenvalues of the laplacian
    matrix that corresponds to a specific graph.
    """
    laplacian_matrix = nx.laplacian_matrix(graph).todense()
    if signless_laplacian: 
        laplacian_matrix = np.abs(laplacian_matrix)
    eigenvals = np.linalg.eigvalsh(laplacian_matrix)
    return eigenvals


def print_counterexample_to_file(
    method: str, counterexample: np.ndarray,
) -> None:
    """
    This function prints the graph that has been identified as a
    counterexample into a .txt file in the output/ directory.
    """
    log.info('A counterexample has been found!')

    # We write the graph into a text file and exit the program
    file = open(f'../output/{method}/counterexample.txt', 'w+') 
    content = str(counterexample)
    file.write(content)
    file.close()
    exit()