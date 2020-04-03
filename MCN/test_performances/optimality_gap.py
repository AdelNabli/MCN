import os
import pickle
import numpy as np
from tqdm import tqdm
from MCN.utils import generate_random_instance, load_saved_experts
from MCN.solve_mcn import solve_mcn


class Instance:

    """Creates an instance object to store everything needed in the same place"""

    def __init__(self, G, Omega, Phi, Lambda, J, value):
        self.G = G
        self.Omega = Omega
        self.Phi = Phi
        self.Lambda = Lambda
        self.J = J
        self.value = value


def generate_test_set(n_free_min, n_free_max, Omega_max, Phi_max, Lambda_max, size_test_set, directory_path):

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
            G, J, Omega, Phi, Lambda = generate_random_instance(n_free_min, n_free_max, Omega_max,
                                                                Phi_max, Lambda_max, Budget_target=budget)
            # if there is an attacker's move,
            # we empty J
            if Phi > 0:
                J = []
            # solve the instance
            value, D, I, P = solve_mcn(G, Omega, Phi, Lambda, J=J, exact=True)
            # save everything in the Instance object
            instance_budget_k = Instance(G, Omega, Phi, Lambda, J, value)
            # pushes it to memory
            test_set_budget.append(instance_budget_k)
        test_set.append(test_set_budget)

    if not os.path.exists(directory_path):
        os.mkdir(directory_path)
    file_path = os.path.join(directory_path, "test_set.gz")
    # save the test set
    pickle.dump(test_set, open(file_path, "wb"))


def opt_gap(list_exact, list_approx):

    vals_exact = np.array(list_exact)
    vals_approx = np.array(list_approx)
    gap = np.sum(np.abs(vals_approx - vals_exact)) / np.sum(vals_exact)
    return gap


def compute_optimality_gap(Omega_max, Phi_max, Lambda_max, list_experts=[], test_set=[],
                           directory_experts="", path_test_set="", **kwargs):

    # if the list of experts have not been loaded in memory
    if list_experts == []:
        list_experts = load_saved_experts(directory_experts)
        if list_experts == []:
            raise ValueError("no models found in the directory specified")
    # if the test set is not loaded in memory
    if test_set == []:
        # if the test set was not given
        if ".gz" not in path_test_set:
            # generate the test set in the 'test_performances' directory
            if not os.path.exists("test_performances"):
                os.mkdir("test_performances")
                test_directory = "test_performances"
            generate_test_set(directory_path=test_directory,
                              Omega_max=Omega_max,
                              Phi_max=Phi_max,
                              Lambda_max=Lambda_max,
                              **kwargs)
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
            value_heuristic += instance.Omega + instance.Lambda
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
