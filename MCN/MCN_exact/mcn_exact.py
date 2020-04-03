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
    best = len(graph.nodes())
    D = []

    while True:
        Graph = graph.copy()
        target = best - len(D)
        # remove the vaccinated nodes from the graph
        Graph.remove_nodes_from(D)
        I, status, P = AP(Graph.nodes(), Graph.edges(), Phi, Lambda, target)

        if status == "opt":
            return (best, D, I, P)

        elif status == "goal":
            Q.append(I)
            best, D = solve_mip(Q, graph.nodes(), graph.edges(), Lambda, Omega)