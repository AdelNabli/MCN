from MCN.MCN_exact.mcn_exact import solve_mcn_exact
from MCN.MCN_exact.attack_protect import AP
from MCN.MCN_exact.defender import solve_defender
from MCN_curriculum.mcn_heuristic import solve_mcn_heuristic
from MCN.utils import load_saved_experts, get_player


def solve_mcn(G, Omega, Phi, Lambda, J=[], Omega_max=0, Phi_max=0, Lambda_max=0, exact=False,
              path="", list_experts=[] ):
    """Solve the mcn instance given with the chosen method:
     - either the exact one and we apply the procedure described in
       https://cerc-datascience.polymtl.ca/wp-content/uploads/2017/11/Technical-Report_DS4DM-2017-012.pdf
    - either we use the trained list of neural networks.
      If this method is chosen, the budget_max are required as well as either
      the path to the directory containing the target nets, or the list of
      experts already loaded

    Parameters:
    ----------
    G: networkx Digraph,
    Omega, Phi, Lambda: int,
    exact: bool,
           whether to apply the exact algorithm or not
    path: str,
          path to the directory containing the target nets
    list_experts: list of pytorch neural nets,
                  loaded list of experts
    Returns:
    -------
    value: int,
           number of saved nodes
    D, I, P: lists,
             respectively, list of the vaccinated, attacked, protected nodes"""

    if exact:
        player = get_player(Omega, Phi, Lambda)
        if player == 0:
            return solve_mcn_exact(G, Omega, Phi, Lambda)
        elif player == 1:
            I, _, P, value = AP(G.nodes(), G.edges(), Phi, Lambda, target=1)
            return (value, [], I, P)
        elif player == 0:
            value, _, P = solve_defender(J, G.nodes(), G.edges(), Lambda)
            return (value, [], [], P)
    else:
        if list_experts == []:
            list_experts = load_saved_experts(path)
        return solve_mcn_heuristic(
            list_experts, G, Omega, Phi, Lambda, Omega_max, Phi_max, Lambda_max, J
        )
