from MCN.MCN_exact.mip import solve_mip
from MCN.MCN_exact.attack_protect import AP


def solve_mcn_exact(graph, Omega, Phi, Lambda):
    r"""The MCN algorithm to solve the MCN problem, from the paper
    https://cerc-datascience.polymtl.ca/wp-content/uploads/2017/11/Technical-Report_DS4DM-2017-012.pdf

    Parameters:
    ----------
    graph: networkx Digraph,
           for every undirected edge (u,v), contains two directed edges (u,v) and (v,u)
    Lambda: int,
            protection budget
    Phi: int,
         attack budget
    Omega: int,
           vaccination budget

    Returns:
    -------
    D: list,
       nodes to vaccinate
    best: int,
          number of saved nodes
    I: list,
       nodes to attack
    P: list,
       nodes to protect
    """

    # Initialization
    Q = []
    best = 0
    is_weighted = False
    for v in graph.nodes():
        # if the graph is weighted, best = sum_v weight_v
        if 'weight' in graph.nodes[v].keys():
            best += float(graph.nodes[v]['weight'])
            is_weighted = True
        # else, best = len(V)
        else:
            best += 1.0
    D = []

    while True:
        Graph = graph.copy()
        if is_weighted:
            target = best - sum([graph.nodes[v]['weight'] for v in D])
        else:
            target = best - len(D)
        # remove the vaccinated nodes from the graph
        Graph.remove_nodes_from(D)
        I, status, P, _ = AP(Graph, Phi, Lambda, target)

        if status == "opt":
            return (best, D, I, P)

        elif status == "goal":
            Q.append(I)
            best, D = solve_mip(Q, graph, Lambda, Omega)
