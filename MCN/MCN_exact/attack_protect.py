from MCN.MCN_exact.rlx_ap import solve_rlxAP
from MCN.MCN_exact.defender import solve_defender


def AP(V, E, Phi, Lambda, target, J=[]):

    r"""Solve the Attack-Protection problem as described in the paper
    https://cerc-datascience.polymtl.ca/wp-content/uploads/2017/11/Technical-Report_DS4DM-2017-012.pdf

    Parameters:
    ----------
    V: list of ints,
       list of the vertices of the graph
    E: list of tuples of ints,
       list of the edges of the graph
       if (v,u) \in E, then (u,v) must be too
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
    S = [list(V)]
    best = len(V)
    I_best = J
    P = []
    P_best = []

    while True:

        try:
            # if the problem is feasible
            value, I = solve_rlxAP(S, V, E, Lambda, Phi, J=J)
            I += J
        except:
            # if not, we stop the loop
            break

        if value <= target - 1:
            return (I, "goal", P)

        len_S, new_S, P = solve_defender(I, V, E, Lambda)

        if len_S <= target - 1:
            return (I, "goal", P)

        if len_S < best:

            best = len_S
            P_best = P
            I_best = I

        S.append(new_S)
    return (I_best, "opt", P_best)
