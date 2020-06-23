import os
import pickle
import torch
import time
import networkx as nx
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Batch
from MCN.utils import instance_to_torch, get_target_net, Instance, InstanceTorch, new_graph, get_player, graph_weights
from MCN.MCN_heur.data import generate_test_set
from MCN.solve_mcn import solve_mcn
from MCN.MCN_heur.train_dqn import solve_greedy_dqn
from MCN.test_performances.metrics import opt_gap, approx_ratio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def metrics_each_stage_cur(Omega_max, Phi_max, Lambda_max, list_experts, exact_protection=False,
                           path_test_data=None, DQN=False, **kwargs):

    """Compute the optimality gap on test sets of exactly solved instances.
    print the average optimality over all the test sets,
    the optimality gaps for each player and for each learning stage.
    Return the results of the computation if return_computation is set to True"""

    # if the test set was not given
    if path_test_data is None:
        # generate the test set in the 'data\test' directory
        generate_test_set(Omega_max=Omega_max,
                          Phi_max=Phi_max,
                          Lambda_max=Lambda_max,
                          **kwargs)
        path_test_data = os.path.join('data', 'test_data')
    path_test_set = os.path.join(path_test_data, 'test_set.gz')
    test_set = pickle.load(open(path_test_set, "rb"))

    print("==========================================================================")
    print("Computing the values using the heuristic... \n")
    # Initialize the variables
    player_values_true = [[],[],[]]
    player_values_heuristic = [[],[],[]]
    budget_values_true = []
    budget_values_heuristic = []

    df_player = dict()
    df_budget = dict()
    df_player['$\eta(\%)$'] = []
    df_player['$\zeta$'] = []
    df_budget['$\eta(\%)$'] = []
    df_budget['$\zeta$'] = []

    # For each learning stage, there is a corresponding test set
    for k in tqdm(range(len(test_set))):
        # Initialize the variables of the learning stage
        budget = k + 1
        dataset = test_set[k]
        budget_values_true.append([])
        budget_values_heuristic.append([])
        # Iterate over the instances in the dataset
        for instance in dataset:
            if not DQN:
                value_heuristic, D_heur, I_heur, P_heur = solve_mcn(instance.G, instance.Omega, instance.Phi, instance.Lambda,
                                                                    J=instance.J, Omega_max=Omega_max, Phi_max=Phi_max,
                                                                    Lambda_max=Lambda_max, exact=False, list_experts=list_experts,
                                                                    exact_protection=exact_protection)
            else:
                value_heuristic = solve_greedy_dqn(instance, list_experts[0])
            value_exact = instance.value
            # add the values to memory
            budget_values_true[k].append(value_exact)
            budget_values_heuristic[k].append(value_heuristic)
            if budget <= Lambda_max:
                player_values_true[2].append(value_exact)
                player_values_heuristic[2].append(value_heuristic)
            elif budget <= Lambda_max + Phi_max:
                player_values_true[1].append(value_exact)
                player_values_heuristic[1].append(value_heuristic)
            elif budget > Lambda_max + Phi_max:
                player_values_true[0].append(value_exact)
                player_values_heuristic[0].append(value_heuristic)
        # compute the budget's metrics
        df_budget['$\eta(\%)$'].append(opt_gap(budget_values_true[k], budget_values_heuristic[k])*100)
        df_budget['$\zeta$'].append(approx_ratio(budget_values_true[k], budget_values_heuristic[k]))
    # Compute the player's metrics
    for player in [0,1,2]:
        df_player['$\eta(\%)$'].append(opt_gap(player_values_true[player], player_values_heuristic[player])*100)
        df_player['$\zeta$'].append(approx_ratio(player_values_true[player], player_values_heuristic[player]))

    # Compute the average metrics over all datasets
    # doesn't take into account the values already solved exactly in the heuristic method
    # (e.g for Lambda = 1 if exact_protection=False, the instances are solved exactly)
    first_budget = 1 + Lambda_max*exact_protection
    tot_val_approx = [value for k in range(first_budget, len(test_set)) for value in budget_values_heuristic[k]]
    tot_val_true = [value for k in range(first_budget, len(test_set)) for value in budget_values_true[k]]
    opt_gap_mean = opt_gap(tot_val_true, tot_val_approx)
    approx_ratio_mean = approx_ratio(tot_val_true, tot_val_approx)

    print('Average Approx Ratio : ', approx_ratio_mean)
    print("Average optimality gap : %f %%" % (opt_gap_mean * 100))

    # Compute the index of the budget's dataframe
    index_budget = []
    for k in range(Omega_max + Phi_max + Lambda_max):
        budget = k + 1
        if budget <= Lambda_max:
            index_budget.append('$\Omega = 0, \Phi = 0 , \Lambda = %d $' % budget)
        elif budget <= Lambda_max + Phi_max:
            index_budget.append('$\Omega = 0, \Phi = %d , \Lambda \in [\![0, %d]\!]$' % (budget - Lambda_max, Lambda_max))
        elif budget > Lambda_max + Phi_max:
            index_budget.append(
                '$\Omega = %d, \Phi \in [\![1, %d]\!] , \Lambda \in [\![0, %d]\!] $' % (
                    budget - Lambda_max - Phi_max, Phi_max, Lambda_max))

    return df_budget, df_player, index_budget, ['Vaccinator', 'Attacker', 'Protector']


def metrics_test_set(path_test_set, Omega_max, Phi_max, Lambda_max, list_experts, path_times=None):

    dict_instances = pickle.load(open(path_test_set, "rb"))
    n_nodes = list(dict_instances.keys())
    df_n = dict()
    df_n['$\#V$'] = n_nodes
    df_n['$\eta(\%)$'] = []
    df_n['$\zeta$'] = []
    df_n['$t_{exact}(s)$'] = []
    df_n['$t_{heur}(s)$'] = []

    if path_times is not None:
        exact_times = pickle.load(open(path_times, "rb"))
    else:
        exact_times = dict()
        for n in n_nodes:
            exact_times[n] = [np.nan]*len(dict_instances[n])

    print("==========================================================================")
    print("Computing the values using the heuristic... \n")

    for n in tqdm(n_nodes):
        val_exact_n = []
        val_heur_n = []
        time_heur_n = []

        for instance in dict_instances[n]:
            ta = time.time()
            value_heuristic, D_heur, I_heur, P_heur = solve_mcn(instance.G, instance.Omega, instance.Phi,
                                                                instance.Lambda,
                                                                J=instance.J, Omega_max=Omega_max, Phi_max=Phi_max,
                                                                Lambda_max=Lambda_max, exact=False,
                                                                list_experts=list_experts)
            tb = time.time()
            time_heur_n.append(tb - ta)
            val_heur_n.append(value_heuristic)
            val_exact_n.append(instance.value)

        df_n['$\eta(\%)$'].append(opt_gap(val_exact_n, val_heur_n)*100)
        df_n['$\zeta$'].append(approx_ratio(val_exact_n, val_heur_n))
        df_n['$t_{heur}(s)$'].append(np.mean(np.array(time_heur_n)))
        df_n['$t_{exact}(s)$'].append(np.mean(np.array(exact_times[n])))

    return df_n


def compute_node_values(G, J, Omega, Phi, Lambda, exact=True, Omega_max=None, Phi_max=None, Lambda_max=None,
                        list_experts=None):

    """Compute the value of each node of the graph given the budgets and already attacked nodes."""

    value_nodes = dict()
    weights = graph_weights(G)
    is_weighted = len(nx.get_node_attributes(G, 'weight').values()) != 0
    is_directed = False in [(v, u) in G.edges() for (u, v) in G.edges()]
    # for every node possible
    for k in G.nodes():
        # if the node is already attacked
        if k in J:
            # its value is null
            value_nodes[k] = 0
        else:
            G1 = G.copy()
            # get the player whose turn it is to play
            player = get_player(Omega, Phi, Lambda)
            # if it is the defender's turn
            if player == 0 or player == 2:
                # remove the node from the graph
                next_G, mapping = new_graph(G1, k)
                next_J = [mapping[node] for node in J]
                reward = weights[k]
            # if it is the attacker's turn
            else:
                # attack the node
                next_J = J + [k]
                next_G = G1
                reward = 0
            # compute the next budgets
            next_Omega = Omega
            next_Phi = Phi
            next_Lambda = Lambda
            if player == 0:
                next_Omega = Omega - 1
            elif player == 1:
                next_Phi = Phi - 1
            elif player == 2:
                next_Lambda = Lambda - 1

            if exact:
                # compute the value of the afterstate
                value, D, I, P = solve_mcn(next_G, next_Omega, next_Phi, next_Lambda, J=next_J, exact=True)
                # the value of the node is: reward + value of the afterstate
                value_nodes[k] = int(reward + value)
            else:
                # format the instance so that it can be read by our neural network
                instance = Instance(next_G, next_Omega, next_Phi, next_Lambda, next_J, 0)
                instance_torch = instance_to_torch(instance)
                batch_torch = Batch.from_data_list([instance_torch.G_torch]).to(device)
                # get the right expert
                target_net = get_target_net(list_experts, next_Omega, next_Phi, next_Lambda, Omega_max, Phi_max, Lambda_max)
                # compute the value of the afterstate
                value_approx = float(target_net(
                    batch_torch,
                    instance_torch.n_nodes,
                    instance_torch.Omegas,
                    instance_torch.Phis,
                    instance_torch.Lambdas,
                    instance_torch.Omegas_norm,
                    instance_torch.Phis_norm,
                    instance_torch.Lambdas_norm,
                    instance_torch.J
                ))
                if is_weighted:
                    value_nodes[k] = round(reward + value_approx, 1)
                else:
                    value_nodes[k] = round(Omega + Lambda + value_approx, 1)

    # plot the values
    nx.draw_spring(G, with_labels=True, node_size=600, node_color=np.array(list(value_nodes.values())),
                   cmap='viridis_r', alpha=1.0, edge_color='gray', arrows=is_directed, width=3, labels=value_nodes,
                   font_size=12, font_color='white')

