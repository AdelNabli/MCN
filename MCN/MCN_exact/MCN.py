from MCN.MCN_exact.onelvlMIPQ import solve_1lvlMIP_Q
from MCN.MCN_exact.AttackProtect import AP


def MCN(graph, Omega, Phi, Lambda):
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
            return (D, best, I, P)

        elif status == "goal":
            Q.append(I)
            best, D = solve_1lvlMIP_Q(Q, graph.nodes(), graph.edges(), Lambda, Omega)
