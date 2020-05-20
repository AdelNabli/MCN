import networkx as nx
import numpy as np
from MCN.MCN_curriculum.environment import Environment
from MCN.utils import get_target_net, take_action_deterministic, take_action_deterministic_batch, get_player, Instance
from MCN.MCN_exact.attack_protect import AP
from MCN.MCN_exact.defender import solve_defender

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

    player = get_player(Omega, Phi, Lambda)
    # if it's the protector turn and we are to use the exact protector agent
    if player == 2 and exact_protection:
        value, _, P = solve_defender(J, G, Lambda)
        return value, [], [], P
    else:
        # Initialize the environment
        instance = Instance(G, Omega, Phi, Lambda, J, 0)
        env = Environment([instance])
        # list of actions for the episode
        actions_episode = []
        val_actions = 0

        while env.Budget >= 1:

            # if the next player is the protector and we use the exact first attack
            if env.Budget == Lambda + 1 and exact_protection:
                J_att = env.list_J[env.actions[0]]
                G_att = env.list_G_nx[env.actions[0]]
                I, _, P, value = AP(G_att, 1, Lambda, target=1, J=J_att)
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
            )
            # save the action to the memory of actions
            actions_episode.append(action)
            # Update the environment
            env.step([action])
            val_actions += np.sum(env.action_values)

        D, I, P = original_names_actions_episode(actions_episode, Phi, Lambda, exact_protection)
        value += val_actions

        return (value, D, I, P)


def solve_mcn_heuristic_batch(list_experts, list_instances, Omega_max, Phi_max, Lambda_max):

    """Given the list of target nets, an instance of the MCN problem and the maximum budgets
    allowed, solves the MCN problem using the list of experts"""


    # Initialize the environment
    env = Environment(list_instances)
    val_actions = np.zeros(len(list_instances))

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
        action, targets, value = take_action_deterministic_batch(
            target_net,
            env.player,
            env.next_player,
            env.next_rewards,
            env.next_list_G_torch,
            env.id_graphs,
            n_nodes=env.next_n_nodes_tensor,
            Omegas=env.next_Omega_tensor,
            Phis=env.next_Phi_tensor,
            Lambdas=env.next_Lambda_tensor,
            Omegas_norm=env.next_Omega_norm,
            Phis_norm=env.next_Phi_norm,
            Lambdas_norm=env.next_Lambda_norm,
            J=env.next_J_tensor,
        )
        env.step(action)
        val_actions += env.action_values

    value = np.array(value) + val_actions

    return value.tolist()
