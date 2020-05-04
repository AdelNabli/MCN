from MCN.MCN_exact.rlx_ap import solve_rlxAP
from MCN.MCN_exact.defender import solve_defender


def AP(G, Phi, Lambda, target, J=[]):

    r"""Solve the Attack-Protection problem as described in the paper
    https://cerc-datascience.polymtl.ca/wp-content/uploads/2017/11/Technical-Report_DS4DM-2017-012.pdf

    Parameters:
    ----------
    G: networkx graph,
    Lambda: int,
            protection budget
    Phi: int,
         attack budget
    Omega: int,
           vaccination budget
    target: int,
            the highest number of infected nodes so far

    Returns:
    -------
    I: list,
       list of the ids of the attacked nodes
    status: str,
            "goal" if I is better than the best attack so far
            "opt" if I is the best attack possible
    P: list,
       list of the ids of the protected nodes"""

    # Initialization
    V = G.nodes()
    S = [list(V)]
    best = 0
    for v in V:
        # if the graph is weighted, best = sum_v weight_v
        if 'weight' in G.node[v].keys():
            best += float(G.node[v]['weight'])
        # else, best = len(V)
        else:
            best += 1.0
    I_best = J
    P = []
    P_best = []

    while True:

        try:
            # if the problem is feasible
            value, I = solve_rlxAP(S, G, Lambda, Phi, J=J)
            I += J
        except:
            # if not, we stop the loop
            break

        if value <= target - 1:
            return (I, "goal", P, value)

        len_S, new_S, P = solve_defender(I, G, Lambda)

        if len_S <= target - 1:
            return (I, "goal", P, len_S)

        if len_S < best:

            best = len_S
            P_best = P
            I_best = I

        S.append(new_S)
    return (I_best, "opt", P_best, best)
