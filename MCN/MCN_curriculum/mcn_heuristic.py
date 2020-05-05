from MCN.solve_mcn import solve_mcn
from MCN.MCN_curriculum.environment import Environment
from MCN.utils import get_target_net, take_action_deterministic, get_player


def original_names_actions_episode(actions_episode, Phi, Lambda, exact_protection):

    """Given the budgets and the list of ids of the actions taken during the episode,
    find the ids of the actions in the original graph and returns the sets D, I, P

    Parameters:
    ----------
    actions_episode: list,
                     the actions taken by the experts during the episode
                     must be "first action taken" at position 0
    Phi, Lambda: int
    exact_protection: bool,
                      whether or not the exact algorithm was used for the protection phase

    Returns:
    -------
    D, I, P: lists,
             nodes to vaccinate, attack, protect respectively """

    # reverse the order of the list
    all_actions = [action for action in reversed(actions_episode)]
    # for all actions, starting from the last taken
    # we iteratively rename the list of previous actions
    # until we are at the first position
    begin_rewrite = exact_protection * (Lambda + 1)
    for k in range(begin_rewrite, len(actions_episode)):
        current_action = all_actions[k]
        previous_actions = all_actions[:k]
        previous_actions = [
            action if action < current_action else action + 1
            for action in previous_actions
        ]
        all_actions[:k] = previous_actions

    return (
        all_actions[Lambda + Phi :],
        all_actions[Lambda:Lambda + Phi],
        all_actions[:Lambda],
    )


def solve_mcn_heuristic(list_experts, G, Omega, Phi, Lambda, Omega_max, Phi_max, Lambda_max, J=[],
                        exact_protection=False):

    """Given the list of target nets, an instance of the MCN problem and the maximum budgets
    allowed, solves the MCN problem using the list of experts"""

    # Get the player
    player = get_player(Omega, Phi, Lambda)
    # if it's the protector turn and we are to use the exact protector agent
    if player == 2 and exact_protection:
        return solve_mcn(G, Omega, Phi, Lambda, J=J, exact=True)
    else:
        # Initialize the environment
        env = Environment(G, Omega, Phi, Lambda, J=J)
        # list of actions for the episode
        actions_episode = []

        while env.Budget >= 1:

            # if the next player is the protector and we use the exact first attack
            if env.Budget == Lambda + 1 and exact_protection:
                J_att = env.list_J[env.action]
                G_att = env.list_G_nx[env.action]
                value, _, I, P = solve_mcn(G_att, Omega=0, Phi=1, Lambda=Lambda, J=J_att, exact=True)
                actions_episode += I + P
                break

            env.compute_current_situation()
            target_net = get_target_net(
                list_experts,
                env.next_Omega,
                env.next_Phi,
                env.next_Lambda,
                Omega_max,
                Phi_max,
                Lambda_max,
            )
            # Take an action
            action, targets, value = take_action_deterministic(
                target_net,
                env.player,
                env.next_player,
                env.next_rewards,
                env.next_list_G_torch,
                n_nodes = env.next_n_nodes_tensor,
                Omegas=env.next_Omega_tensor,
                Phis=env.next_Phi_tensor,
                Lambdas=env.next_Lambda_tensor,
                Omegas_norm=env.next_Omega_norm,
                Phis_norm=env.next_Phi_norm,
                Lambdas_norm=env.next_Lambda_norm,
                J=env.next_J_tensor,
                saved_nodes=env.next_saved_tensor,
                infected_nodes=env.next_infected_tensor,
                size_connected=env.next_size_connected_tensor,
            )
            # save the action to the memory of actions
            actions_episode.append(action)
            # Update the environment
            env.step(action)

        D, I, P = original_names_actions_episode(actions_episode, Phi, Lambda, exact_protection)

        return (value, D, I, P)
