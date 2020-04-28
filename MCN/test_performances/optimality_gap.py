import os
import pickle
import numpy as np
from tqdm import tqdm
from MCN.utils import generate_random_instance, Instance, instance_to_torch
from MCN.solve_mcn import solve_mcn


def generate_test_set(n_free_min, n_free_max, d_edge_min, d_edge_max, Omega_max, Phi_max, Lambda_max,
                      size_test_set, directory_path, to_torch=False):

    """Generates a set of random instances that are solved exactly with the MCN_exact algorithm.
    Each budget possible in [1, Omega_max + Phi_max + Lambda_max] is equally represented in
    the test set. The dataset is then dumped in a .gz file inside the directory_path"""

    # Initialize the variables
    Budget_max = Omega_max + Phi_max + Lambda_max
    # for each budget possible, generate the same number of instances
    n_budget = size_test_set // Budget_max
    test_set = []

    print("==========================================================================")
    print("Generates the test set... \n")

    # for all budgets

    for budget in tqdm(range(1, Budget_max + 1)):
        # initialize the budget's instances list
        test_set_budget = []
        for k in range(n_budget):
            # generate a random instance
            instance_budget_k = generate_random_instance(
                n_free_min,
                n_free_max,
                d_edge_min,
                d_edge_max,
                Omega_max,
                Phi_max,
                Lambda_max,
                Budget_target=budget,
            )
            G = instance_budget_k.G
            Omega = instance_budget_k.Omega
            Phi = instance_budget_k.Phi
            Lambda = instance_budget_k.Lambda
            J = instance_budget_k.J
            # if there is an attacker's move,
            # we empty J
            if Phi > 0:
                J = []
            # solve the instance
            value, D, I, P = solve_mcn(G, Omega, Phi, Lambda, J=J, exact=True)
            # save the value in the Instance object
            instance_budget_k.value = value
            # pushes it to memory
            if to_torch:
                instance_budget_k = instance_to_torch(instance_budget_k)
            test_set_budget.append(instance_budget_k)
        test_set.append(test_set_budget)

    if not os.path.exists(directory_path):
        os.mkdir(directory_path)
    if to_torch:
        file_path = os.path.join(directory_path, "test_set_torch.gz")
    else:
        file_path = os.path.join(directory_path, "test_set.gz")
    # save the test set
    pickle.dump(test_set, open(file_path, "wb"))


def opt_gap(list_exact, list_approx):

    vals_exact = np.array(list_exact)
    vals_approx = np.array(list_approx)
    # replace 0s with 1s in the denominator to prevent errors
    vals_exact_denom = np.where(vals_exact==0,1,vals_exact)
    gap = np.mean(np.abs(vals_approx - vals_exact)/ vals_exact_denom)
    return gap


def compute_optimality_gap(Omega_max, Phi_max, Lambda_max, list_experts, path_test_set="", **kwargs):

    # if the test set was not given
    if ".gz" not in path_test_set:
        # generate the test set in the 'data\test' directory
        if not os.path.exists('data'):
            os.mkdir('data')
        path_test_data = os.path.join('data', 'test_data')
        if not os.path.exists(path_test_data):
            os.mkdir(path_test_data)
        generate_test_set(directory_path=path_test_data,
                          Omega_max=Omega_max,
                          Phi_max=Phi_max,
                          Lambda_max=Lambda_max,
                          **kwargs)
        path_test_set = os.path.join(path_test_data, 'test_set.gz')
    test_set = pickle.load(open(path_test_set, "rb"))

    print("==========================================================================")
    print("Computing the values using the heuristic... \n")

    player_values_true = [[],[],[]]
    player_values_heuristic = [[],[],[]]
    budget_values_true = []
    budget_values_heuristic = []
    opt_gap_budget = []
    opt_gap_player = []
    for k in tqdm(range(len(test_set))):
        budget = k + 1
        dataset = test_set[k]
        budget_values_true.append([])
        budget_values_heuristic.append([])
        for instance in dataset:
            value_heuristic, _,_,_ = solve_mcn(instance.G, instance.Omega, instance.Phi, instance.Lambda,
                                               J=instance.J, Omega_max=Omega_max, Phi_max=Phi_max,
                                               Lambda_max=Lambda_max, exact=False, list_experts=list_experts)
            value_exact = instance.value
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

    for player in [0,1,2]:
        opt_gap_player.append(opt_gap(player_values_true[player], player_values_heuristic[player]))

    return(opt_gap_budget, opt_gap_player)
