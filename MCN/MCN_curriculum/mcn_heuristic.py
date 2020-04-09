from MCN.MCN_curriculum.environment import Environment
from MCN.utils import get_target_net, take_action_deterministic


def original_names_actions_episode(actions_episode, Phi, Lambda):

    """Given the budgets and the list of ids of the actions taken during the episode,
    find the ids of the actions in the original graph and returns the sets D, I, P

    Parameters:
    ----------
    actions_episode: list,
                     the actions taken by the experts during the episode
                     must be "first action taken" at position 0
    Phi, Lambda: int

    Returns:
    -------
    D, I, P: lists,
             nodes to vaccinate, attack, protect respectively """

    # reverse the order of the list
    all_actions = [action for action in reversed(actions_episode)]
    # for all actions, starting from the last taken
    # we iteratively rename the list of previous actions
    # until we are at the first position
    for k in range(len(actions_episode)):
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


def solve_mcn_heuristic(list_experts, G, Omega, Phi, Lambda, Omega_max, Phi_max, Lambda_max, J=[]):

    """Given the list of target nets, an instance of the MCN problem and the maximum budgets
    allowed, solves the MCN problem using the list of experts"""

    # Initialize the environment
    env = Environment(G, Omega, Phi, Lambda, J=J)
    # list of actions for the episode
    actions_episode = []

    while env.Budget >= 1:

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
            Omegas=env.next_Omega_tensor,
            Phis=env.next_Phi_tensor,
            Lambdas=env.next_Lambda_tensor,
            J=env.next_J_tensor,
            saved_nodes=env.next_saved_tensor,
            infected_nodes=env.next_infected_tensor,
            size_connected=env.next_size_connected_tensor,
        )
        # save the action to the memory of actions
        actions_episode.append(action)
        # Update the environment
        env.step(action)

    D, I, P = original_names_actions_episode(actions_episode, Phi, Lambda)

    return (value, D, I, P)
