import os
import pickle
import numpy as np
import networkx as nx
from tqdm import tqdm
from MCN.utils import Instance
from MCN.MCN_curriculum.data import generate_test_set
from MCN.solve_mcn import solve_mcn


def opt_gap(list_exact, list_approx):

    """Function used to compute the optimality gap:
    mean((val_exact - val_approx) / val_exact)"""

    vals_exact = np.array(list_exact)
    vals_approx = np.array(list_approx)
    # replace 0s with 1s in the denominator to prevent errors
    vals_exact_denom = np.where(vals_exact==0,1,vals_exact)
    gap = np.mean(np.abs(vals_approx - vals_exact)/ vals_exact_denom)
    return gap


def print_opt_gap(Omega_max, Phi_max, Lambda_max, og_mean, og_budget, og_player):

    """Print the optimality gaps computed with the compute_optimality_gap function"""

    # print the average optimality gap
    print("average optimality gap : %f %%" % (og_mean * 100))
    # print the optimality gap for each budget
    for k in range(Omega_max + Phi_max + Lambda_max):
        budget = k + 1
        if budget <= Lambda_max:
            print("optimality gap for instances with budget Omega = 0, Phi = 0 , Lambda = %d : %f %%" % (
            budget, og_budget[k] * 100))
        elif budget <= Lambda_max + Phi_max:
            print("optimality gap for instances with budget Omega = 0, Phi = %d , Lambda \in [0, %d] : %f %%" % (
            budget - Lambda_max, Lambda_max, og_budget[k] * 100))
        elif budget > Lambda_max + Phi_max:
            print(
                "optimality gap for instances with budget Omega = %d, Phi \in [1, %d] , Lambda \in [0, %d] : %f %%" % (
                budget - Lambda_max - Phi_max, Phi_max, Lambda_max, og_budget[k] * 100))
    # print the optimality gap of each player
    print("optimality gap for the vaccinator : %f %%" % (og_player[0] * 100))
    print("optimality gap for the attacker : %f %%" % (og_player[1] * 100))
    print("optimality gap for the protector : %f %%" % (og_player[2] * 100))


def compute_optimality_gap(Omega_max, Phi_max, Lambda_max, list_experts, exact_protection=False,
                           path_test_set="", return_computation=False, **kwargs):

    """Compute the optimality gap on a test set of exactly solved instances.
    print the average optimality over all the test sets,
    the optimality gaps for each player and for each learning stage.
    Return the results of the computation if return_computation is set to True"""

    # if the test set was not given
    if ".gz" not in path_test_set:
        # generate the test set in the 'data\test' directory
        generate_test_set(Omega_max=Omega_max,
                          Phi_max=Phi_max,
                          Lambda_max=Lambda_max,
                          **kwargs)
        path_test_set = os.path.join('data', 'test_data', 'test_set.gz')
    test_set = pickle.load(open(path_test_set, "rb"))

    print("==========================================================================")
    print("Computing the values using the heuristic... \n")
    # Initialize the variables
    player_values_true = [[],[],[]]
    player_values_heuristic = [[],[],[]]
    budget_values_true = []
    budget_values_heuristic = []
    opt_gap_budget = []
    opt_gap_player = []
    # For each learning stage, there is a corresponding test set
    for k in tqdm(range(len(test_set))):
        # Initialize the variables of the learning stage
        budget = k + 1
        dataset = test_set[k]
        budget_values_true.append([])
        budget_values_heuristic.append([])
        # Iterate over the instances in the dataset
        for instance in dataset:
            value_heuristic, _,_,_ = solve_mcn(instance.G, instance.Omega, instance.Phi, instance.Lambda,
                                               J=instance.J, Omega_max=Omega_max, Phi_max=Phi_max,
                                               Lambda_max=Lambda_max, exact=False, list_experts=list_experts,
                                               exact_protection=exact_protection)
            # re-add the values of the nodes removed with the defender's moves
            is_weighted = len(nx.get_node_attributes(instance.G, 'weight').values()) != 0
            if is_weighted:
                weights = np.array([instance.G.nodes[node]['weight'] for node in instance.G.nodes()])
            else:
                weights = np.ones(len(instance.G))
            value_heuristic += np.sum(weights[instance.D]) + np.sum(weights[instance.P])
            value_exact = instance.value + np.sum(weights[instance.D]) + np.sum(weights[instance.P])
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
        # compute the budget's optimality gap
        opt_gap_budget.append(opt_gap(budget_values_true[k], budget_values_heuristic[k]))
    # Compute the player's optimality gap
    for player in [0,1,2]:
        opt_gap_player.append(opt_gap(player_values_true[player], player_values_heuristic[player]))
    # Compute the average optimality gap over all datasets
    # doesn't take into account the values already solved exactly in the heuristic method
    # (e.g for Lambda = 1 if exact_protection=False, the instances are solved exactly)
    first_budget = 1 + Lambda_max*exact_protection
    tot_val_approx = [value for k in range(first_budget, len(test_set)) for value in budget_values_heuristic[k]]
    tot_val_true = [value for k in range(first_budget, len(test_set)) for value in budget_values_true[k]]
    opt_gap_mean = opt_gap(tot_val_true, tot_val_approx)

    print_opt_gap(Omega_max, Phi_max, Lambda_max, opt_gap_mean, opt_gap_budget, opt_gap_player)

    if return_computation:
        return(opt_gap_mean, opt_gap_budget, opt_gap_player)
