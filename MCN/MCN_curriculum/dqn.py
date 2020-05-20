import os
import pickle
from tqdm import tqdm
import math
import random
import numpy as np
from MCN.MCN_curriculum.value_nn import DQN
from MCN.solve_mcn import solve_mcn
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from datetime import datetime
from torch_scatter import scatter_min, scatter_max, scatter
from MCN.MCN_curriculum.data import collate_fn, MCNDataset, DataLoader
from MCN.utils import (
    new_graph,
    get_player,
    compute_saved_nodes,
    Instance,
    InstanceTorch,
    instance_to_torch,
    get_target_net,
    load_saved_experts,
    generate_random_batch_instance,
    generate_random_instance,
    save_models,
    load_training_param,
    count_param_NN,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EnvironmentDQN(object):

    """Defines the environment's sequential behavior"""

    def __init__(self, list_instances):

        """Initialize all the variables of the environment given the starting state.

        Parameters:
        ----------
        list_instances: list of Instance object"""

        self.batch_instance = list_instances
        self.batch_instance_torch = None
        self.mappings = []
        self.batch_size = len(list_instances)
        self.rewards = [0]*self.batch_size
        self.Omega = list_instances[0].Omega
        self.Phi = list_instances[0].Phi
        self.Lambda = list_instances[0].Lambda
        self.Budget = self.Omega + self.Phi + self.Lambda
        self.player = get_player(self.Omega, self.Phi, self.Lambda)
        self.batch_torch()
        self.compute_mappings()
        self.update_budgets()
        self.next_player = get_player(self.next_Omega, self.next_Phi, self.next_Lambda)

    def batch_torch(self):

        # initialize a list of instance torch
        self.batch_instance_torch = []
        # transform the list of instances to a list of instance torch
        for instance in self.batch_instance:
            instance_torch = instance_to_torch(instance)
            self.batch_instance_torch.append(instance_torch)
        # create a proper batch from the list
        self.batch_instance_torch = collate_fn(self.batch_instance_torch)

    def compute_mappings(self):
        """Compute a mapping 'id_free_node_batch': 'true_name_of_node_in_graph' for each
        instance in the list of instances and saves that in a list"""

        self.mappings = []

        count_free = 0
        for instance in self.batch_instance:
            map_i = dict()
            for node in instance.G.nodes():
                if node not in instance.J:
                    map_i[count_free] = node
                    count_free += 1
            self.mappings.append(map_i)

    def update_budgets(self):

        """Compute the next triplet of budgets given the current one
            and the player whose turn it is to play"""

        # Init the variables
        self.next_Omega = self.Omega
        self.next_Phi = self.Phi
        self.next_Lambda = self.Lambda
        # Update the one to be updated
        if self.player == 0:
            self.next_Omega = self.Omega - 1
        elif self.player == 1:
            self.next_Phi = self.Phi - 1
        elif self.player == 2:
            self.next_Lambda = self.Lambda - 1

    def step(self, actions):

        next_instances = []
        rewards = [0]*self.batch_size
        for k in range(self.batch_size):
            action_k = int(actions[k])
            instance_k = self.batch_instance[k]
            node_k = self.mappings[k][action_k]
            if self.player == 1:
                J_k = instance_k.J + [node_k]
                G_k = instance_k.G
            else:
                G_k, mapping = new_graph(instance_k.G, node_k)
                J_k = [mapping[j] for j in instance_k.J]
            # if we are not at the end of the episode
            # we need to compute the new states
            if self.next_player != 3:
                new_instance = Instance(G_k, self.Omega, self.Phi, self.Lambda, J_k, 0)
                next_instances.append(new_instance)
            # else, we need to compute the reward of taking the action
            else:
                rewards[k] = compute_saved_nodes(G_k, J_k)
        # update the variables of the environment
        self.rewards = rewards
        self.batch_instance = next_instances
        self.Budget -= 1
        self.Omega = self.next_Omega
        self.Phi = self.next_Phi
        self.Lambda = self.next_Lambda
        self.player = self.next_player
        if self.next_player != 3:
            self.batch_torch()
            self.compute_mappings()
            self.update_budgets()
            self.next_player = get_player(self.next_Omega, self.next_Phi, self.next_Lambda)


def take_action_deterministic_batch_dqn(target_net, player, batch_instances):

    with torch.no_grad():
        # We compute the target values
        batch = batch_instances.batch
        mask_values = batch_instances.J.eq(0)[:, 0]
        action_values = target_net(batch_instances.G_torch,
                                   batch_instances.n_nodes,
                                   batch_instances.Omegas,
                                   batch_instances.Phis,
                                   batch_instances.Lambdas,
                                   batch_instances.Omegas_norm,
                                   batch_instances.Phis_norm,
                                   batch_instances.Lambdas_norm,
                                   batch_instances.J,
                                   )
        action_values = action_values[mask_values]
        batch = batch[mask_values]
    # if it's the turn of the attacker
    if player == 1:
        # we take the argmin
        values, actions = scatter_min(action_values, batch, dim=0)
    else:
        # we take the argmax
        values, actions = scatter_max(action_values, batch, dim=0)

    return actions.view(-1).tolist()


def sample_action_batch_dqn(neural_net, player, batch_instances,
                            eps_end, eps_decay, eps_start, count_steps):

    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * np.exp(-1. * count_steps / eps_decay)
    if sample > eps_threshold:
        return take_action_deterministic_batch_dqn(neural_net, player, batch_instances)
    else:
        actions = []
        mask_actions = batch_instances.J.eq(0)[:, 0]
        batch = batch_instances.G_torch.batch[mask_actions]
        n_batches = int(batch[-1]) + 1
        n_actions_already_taken = 0
        for i in range(n_batches):
            actions_possible = int(torch.sum(batch.eq(i)))
            random_action = int(np.random.randint(0, actions_possible)) + n_actions_already_taken
            n_actions_already_taken += actions_possible
            actions.append(random_action)

        return actions


def solve_mcn_heuristic_batch_dqn(list_experts, list_instances, Omega_max, Phi_max, Lambda_max):

    """Given the list of target nets, an instance of the MCN problem and the maximum budgets
    allowed, solves the MCN problem using the list of experts"""

    # Get the current budget and the player whose turn it is to play
    Omega = list_instances[0].Omega
    Phi = list_instances[0].Phi
    Lambda = list_instances[0].Lambda
    Budget = Omega + Phi + Lambda
    player = get_player(Omega, Phi, Lambda)
    # If there is only 1 of budget,
    # we can solve the instance exactly
    if Budget == 1:
        rewards_batch = []
        batch_size = len(list_instances)
        free_nodes = [[x for x in list_instances[k].G.nodes() if x not in list_instances[k].J] for k in
                           range(batch_size)]
        id_graphs = torch.tensor(
            [k for k in range(batch_size) for i in range(len(free_nodes[k]))],
            dtype=torch.int64).to(device)

        for k in range(batch_size):
            for node in free_nodes[k]:
                G_k = list_instances[k].G.copy()
                J_k = list_instances[k].J.copy()
                if player == 1:
                    J_k += [node]
                else:
                    G_k, mapping = new_graph(G_k, node)
                    J_k = [mapping[j] for j in J_k]
                reward_k = compute_saved_nodes(G_k, J_k)
                rewards_batch.append(reward_k)
        rewards_batch = torch.tensor(rewards_batch, dtype=torch.float).view([len(rewards_batch), 1]).to(device)
        if player == 1:
            value, _ = scatter_min(rewards_batch, id_graphs, dim=0).view(-1).tolist()
        else:
            value, _ = scatter_max(rewards_batch, id_graphs, dim=0).view(-1).tolist()

    # Else, we need to unroll the experts
    else:
        # Initialize the environment
        env = EnvironmentDQN(list_instances)

        while env.Budget >= 1:

            target_net = get_target_net(
                list_experts,
                env.Omega,
                env.Phi,
                env.Lambda,
                Omega_max,
                Phi_max,
                Lambda_max,
            )
            # Take an action
            action = take_action_deterministic_batch_dqn(
                target_net,
                env.player,
                env.batch_instance_torch,
            )
            env.step(action)
            value = env.rewards

    return value


def load_create_datasets_dqn(size_train_data, size_val_data, batch_size, num_workers, n_free_min, n_free_max,
                             d_edge_min, d_edge_max, Omega_max, Phi_max, Lambda_max, weighted, w_max, directed, Budget,
                             list_experts, path_data, solve_exact=False, exact_protection=False, batch_unroll=None):

    """Create or load the training and validation sets.
    Return two dataloaders to access both datasets.
    Dump the datasets in a .gz file in data/train_data and data/val_data"""

    print("\n==========================================================================")
    print("Creating or Loading the Training and Validation sets for Budget = %2d \n" % Budget)

    # Initialize the dataset and number of instances to generate
    data = []
    len_data_train = 0
    total_size = size_train_data + size_val_data
    # If there is a data folder
    if path_data is not None:
        # we check whether there is already a training set
        # corresponding to the budget we want
        path_train_data_budget = os.path.join(path_data, 'train_data', 'data_'+str(Budget)+'.gz')
        # if it's the case, we load it
        if os.path.exists(path_train_data_budget):
            data += pickle.load(open(path_train_data_budget, "rb"))
            len_data_train = len(data)
        # similarly, we check whether there is a validation set available
        path_val_data_budget = os.path.join(path_data, 'val_data', 'data_' + str(Budget) + '.gz')
        # if it's the case, we load it
        if os.path.exists(path_val_data_budget):
            data += pickle.load(open(path_val_data_budget, "rb"))

    # Update the number of instances that needs to be created
    total_size = total_size - len(data)
    # We create the instances that are currently lacking in the datasets
    # If we need the exact protection, we solve one instance at a time
    # Compute the number of batches necessary to fill the memory
    if batch_unroll is None:
        min_size_instance = n_free_min + Budget
        max_size_instance = n_free_max + Budget
        mean_size_instance = min_size_instance + (max_size_instance - min_size_instance) // 2
        batch_instances = batch_size // mean_size_instance
    else:
        batch_instances = batch_unroll
    n_iterations = total_size // batch_instances + 1 * (total_size % batch_instances > 0)
    for k in tqdm(range(n_iterations)):
        # Sample a batch of random instance
        list_instances = generate_random_batch_instance(
            batch_instances,
            n_free_min,
            n_free_max,
            d_edge_min,
            d_edge_max,
            Omega_max,
            Phi_max,
            Lambda_max,
            Budget,
            weighted,
            w_max,
            directed,
        )
        # Solves the mcn problem for the batch using the heuristic
        values = solve_mcn_heuristic_batch_dqn(
            list_experts,
            list_instances,
            Omega_max,
            Phi_max,
            Lambda_max,
        )
        for i in range(batch_instances):
            list_instances[i].value = values[i]
            # Transform the instance to a InstanceTorch object
            instance_torch = instance_to_torch(list_instances[i])
            # add the instance to the data
            data.append(instance_torch)


    # Save the data if there is a change in the dataset
    if len_data_train != size_train_data or total_size > 0:
        if path_data is None:
            path_data = 'data'
        if not os.path.exists(path_data):
            os.mkdir(path_data)
        path_train = os.path.join(path_data, 'train_data')
        if not os.path.exists(path_train):
            os.mkdir(path_train)
        path_val = os.path.join(path_data, 'val_data')
        if not os.path.exists(path_val):
            os.mkdir(path_val)
        path_train_data_budget = os.path.join(path_train, 'data_' + str(Budget) + '.gz')
        path_val_data_budget = os.path.join(path_val, 'data_' + str(Budget) + '.gz')
        pickle.dump(data[:size_train_data], open(path_train_data_budget, "wb"))
        pickle.dump(data[size_train_data:], open(path_val_data_budget, "wb"))
        print("\nSaved datasets in " + path_data, '\n')

    # Create the datasets used during training and validation
    val_data = MCNDataset(data[size_train_data:size_train_data + size_val_data])
    train_data = MCNDataset(data[:size_train_data])
    train_loader = DataLoader(
        train_data,
        collate_fn=collate_fn,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_data,
        collate_fn=collate_fn,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    return train_loader, val_loader


def compute_loss_test_dqn(test_set_generators, list_players, value_net=None, list_experts=None, id_to_test=None):

    """Compute the list of losses of the value_net or the list_of_experts
    over the list of exactly solved datasets that constitutes the test set"""

    list_losses = []
    with torch.no_grad():
        if id_to_test is None:
            iterator = range(len(test_set_generators))
        else:
            iterator = [id_to_test]
        for k in iterator:
            target = []
            val_approx = []
            player = list_players[k]
            if list_experts is not None:
                try:
                    target_net = list_experts[k]
                except IndexError:
                    target_net = None
            elif value_net is not None:
                target_net = value_net
            if target_net is None:
                list_losses.append(0)
            else:
                for i_batch, batch_instances in enumerate(test_set_generators[k]):
                    batch = batch_instances.batch
                    mask_values = batch_instances.J.eq(0)[:, 0]
                    action_values = target_net(batch_instances.G_torch,
                                               batch_instances.n_nodes,
                                               batch_instances.Omegas,
                                               batch_instances.Phis,
                                               batch_instances.Lambdas,
                                               batch_instances.Omegas_norm,
                                               batch_instances.Phis_norm,
                                               batch_instances.Lambdas_norm,
                                               batch_instances.J,
                                               )
                    action_values = action_values[mask_values]
                    batch = batch[mask_values]
                    # if it's the turn of the attacker
                if player == 1:
                    # we take the argmin
                    values, actions = scatter_min(action_values, batch, dim=0)
                else:
                    # we take the argmax
                    values, actions = scatter_max(action_values, batch, dim=0)

                val_approx.append(values)
                target.append(batch_instances.target)
                # Compute the loss
                target = torch.cat(target)
                val_approx = torch.cat(val_approx)
                loss_target_net = float(torch.sqrt(torch.mean((val_approx[:, 0] - target[:, 0]) ** 2)))
                list_losses.append(loss_target_net)

    return list_losses


def generate_test_set_dqn(n_free_min, n_free_max, d_edge_min, d_edge_max, Omega_max, Phi_max, Lambda_max,
                          weighted, w_max, directed, size_test_set, to_torch=False):

    """Generates a set of random instances that are solved exactly with the MCN_exact algorithm.
    Each budget possible in [1, Omega_max + Phi_max + Lambda_max] is equally represented in
    the test set. The dataset is then dumped in a .gz file inside data\test_data"""

    # Initialize the variables
    Budget_max = Omega_max + Phi_max + Lambda_max
    test_set = []
    if to_torch:
        test_set_torch = []

    print("==========================================================================")
    print("Generates the test set... \n")

    # for all budgets
    for budget in tqdm(range(1, Budget_max + 1)):
        # initialize the budget's instances list
        test_set_budget = []
        if to_torch:
            test_set_budget_torch = []
        for k in range(size_test_set):
            # generate a random instance
            instance_budget_k = generate_random_instance(
                n_free_min,
                n_free_max,
                d_edge_min,
                d_edge_max,
                Omega_max,
                Phi_max,
                Lambda_max,
                weighted=weighted,
                w_max=w_max,
                Budget_target=budget,
                directed=directed,
            )
            G = instance_budget_k.G
            Omega = instance_budget_k.Omega
            Phi = instance_budget_k.Phi
            Lambda = instance_budget_k.Lambda
            J = instance_budget_k.J
            # solve the instance
            value, D, I, P = solve_mcn(G, Omega, Phi, Lambda, J=J, exact=True)
            # save the value, P, D in the Instance object
            instance_budget_k.value = value
            instance_budget_k.D = D
            instance_budget_k.I = I
            instance_budget_k.P = P
            # pushes it to memory
            test_set_budget.append(instance_budget_k)
            # if we want to save the corresponding InstanceTorch
            # to evaluate the training, we stop at Budget_max - 1
            if to_torch:
                instance_budget_k_torch = instance_to_torch(instance_budget_k)
                test_set_budget_torch.append(instance_budget_k_torch)
        test_set.append(test_set_budget)
        if to_torch:
            test_set_torch.append(test_set_budget_torch)

    if not os.path.exists('data'):
        os.mkdir('data')
    folder_name = 'test_data'
    if weighted:
        folder_name += '_w'
    if directed:
        folder_name += '_dir'
    path_test_data = os.path.join('data', folder_name)
    if not os.path.exists(path_test_data):
        os.mkdir(path_test_data)
    # Save the test sets
    file_path = os.path.join(path_test_data, "test_set.gz")
    pickle.dump(test_set, open(file_path, "wb"))
    if to_torch:
        file_path_torch = os.path.join(path_test_data, "test_set_torch.gz")
        pickle.dump(test_set_torch, open(file_path_torch, "wb"))


def load_create_test_set_dqn(n_free_min, n_free_max, d_edge_min, d_edge_max, Omega_max, Phi_max, Lambda_max,
                             weighted, w_max, directed, size_test_set, path_test_data, batch_size, num_workers):

    """Load or create the test sets and returns a list of Dataloaders to access each test set """

    test_set_generators = []
    if size_test_set > 0 :
        if path_test_data is None:
            generate_test_set_dqn(n_free_min, n_free_max, d_edge_min, d_edge_max, Omega_max, Phi_max, Lambda_max,
                                  weighted, w_max, directed, size_test_set, to_torch=True)
            folder_name = 'test_data'
            if weighted:
                folder_name += '_w'
            if directed:
                folder_name += '_dir'
            path_test_set = os.path.join('data', folder_name, 'test_set_torch.gz')
        else:
            path_test_set = os.path.join(path_test_data, 'test_set_torch.gz')
        # load the test set
        test_set = pickle.load(open(path_test_set, "rb"))
        # create a dataloader object for each dataset in the test set
        for k in range(len(test_set)):
            test_set_k = MCNDataset(test_set[k])
            test_gen_k = DataLoader(
                test_set_k,
                collate_fn=collate_fn,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
            )
            test_set_generators.append(test_gen_k)

    return test_set_generators


class TargetExpertsDQN(object):

    """Object containing the target nets and updating them during learning"""

    def __init__(self, dim_input, dim_embedding, dim_values, dim_hidden, n_heads, n_att_layers, n_pool, K, alpha,
                 weighted, Omega_max, Phi_max, Lambda_max, path_experts, path_data, exact_protection):

        # Initialize the parameters of the neural network
        self.dim_input = dim_input
        self.dim_embedding = dim_embedding
        self.dim_values = dim_values
        self.dim_hidden = dim_hidden
        self.n_att_layers = n_att_layers
        self.n_pool = n_pool
        self.n_heads = n_heads
        self.K = K
        self.alpha = alpha
        self.weighted = weighted
        # Initialize the parameters of the list of experts
        self.n_max = Omega_max + Phi_max + Lambda_max
        self.Lambda_max = Lambda_max
        self.list_target_nets = [None] * (self.n_max)
        self.losses_validation_sets = [math.inf] * (self.n_max)
        self.losses_test_set = [0] * (self.n_max)
        self.loss_value_net = math.inf
        self.Budget_target = 1
        self.list_players = [0]*Omega_max + [1]*Phi_max + [2]*Lambda_max
        # If use the exact algorithm for protection, update the parameters
        self.exact_protection = exact_protection
        if exact_protection:
            self.losses_validation_sets[:Lambda_max] = [0] * Lambda_max
            self.Budget_target = Lambda_max + 1

        self.resume_training(path_experts, path_data)

    def resume_training(self, path_experts, path_data):

        """Load the targets nets that are already available"""

        # If pre-trained experts are available
        if path_experts is not None:
            # load them
            list_trained_experts = load_saved_experts(path_experts)
            Budget_trained = len(list_trained_experts)
            # update the TargetExperts object
            self.list_target_nets[:Budget_trained] = list_trained_experts

            # If there is a data folder
            if path_data is not None:
                Budget_begin = 1 + self.exact_protection * self.Lambda_max
                # for every budget already trained
                for Budget in range(Budget_begin, Budget_trained + 1):
                    path_val_data_budget = os.path.join(path_data, 'val_data', 'data_' + str(Budget) + '.gz')
                    # we check whether there is a validation set available
                    if os.path.exists(path_val_data_budget):
                        # if it's the case, we load it
                        val_data = pickle.load(open(path_val_data_budget, "rb"))
                        val_data = MCNDataset(val_data)
                        val_loader = DataLoader(
                            val_data,
                            collate_fn=collate_fn,
                            batch_size=128,
                            shuffle=True,
                            num_workers=0,
                        )
                        # then, we test the target net on this validation set
                        self.Budget_target = Budget
                        self.test_update_target_nets(self.list_target_nets[Budget - 1], val_loader)

            self.Budget_target = Budget_trained + 1

    def test_update_target_nets(self, value_net, val_generator, test_generator=None):

        """Test the current value net against the saved expert on the current validation set
        and keep the best of both as the current target net"""

        # Create a target net from the current value net
        new_target_net = DQN(
            dim_input=self.dim_input,
            dim_embedding=self.dim_embedding,
            dim_values=self.dim_values,
            dim_hidden=self.dim_hidden,
            n_heads=self.n_heads,
            n_att_layers=self.n_att_layers,
            n_pool=self.n_pool,
            K=self.K,
            alpha=self.alpha,
            p=0,
            weighted=self.weighted,
        ).to(device)
        new_target_net.load_state_dict(value_net.state_dict())
        new_target_net.eval()
        # init the values approx and the target
        target = []
        val_approx = []
        id_slot = self.Budget_target - 1
        player = self.list_players[id_slot]
        with torch.no_grad():
            # Compute the approximate values given
            # by the current value net on the validation set
            # for every batch
            for i_batch, batch_instances in enumerate(val_generator):
                batch = batch_instances.batch
                mask_values = batch_instances.J.eq(0)[:, 0]
                action_values = new_target_net(batch_instances.G_torch,
                                               batch_instances.n_nodes,
                                               batch_instances.Omegas,
                                               batch_instances.Phis,
                                               batch_instances.Lambdas,
                                               batch_instances.Omegas_norm,
                                               batch_instances.Phis_norm,
                                               batch_instances.Lambdas_norm,
                                               batch_instances.J,
                                           )
                action_values = action_values[mask_values]
                batch = batch[mask_values]
                # if it's the turn of the attacker
            if player == 1:
                # we take the argmin
                values, actions = scatter_min(action_values, batch, dim=0)
            else:
                # we take the argmax
                values, actions = scatter_max(action_values, batch, dim=0)
                val_approx.append(values)
                target.append(batch_instances.target)
            # Compute the loss
            target = torch.cat(target)
            val_approx = torch.cat(val_approx)
            loss_value_net = float(torch.sqrt(torch.mean((val_approx[:, 0] - target[:, 0]) ** 2)))
            self.loss_value_net = loss_value_net
        # If the current loss is less than the best loss so far
        if loss_value_net < self.losses_validation_sets[id_slot]:
            # we update both the current target net and loss
            self.list_target_nets[id_slot] = new_target_net
            self.losses_validation_sets[id_slot] = loss_value_net
            if test_generator is not None:
                self.losses_test_set[id_slot] = compute_loss_test_dqn(
                    test_generator,
                    self.list_players,
                    list_experts=self.list_target_nets,
                    id_to_test=id_slot,
                )[0]


def train_dqn(batch_size, size_train_data, size_val_data, size_test_data, lr, betas, n_epoch, update_experts,
              dim_embedding, dim_values, dim_hidden, n_heads, n_att_layers, n_pool, alpha, p,
              n_free_min, n_free_max, d_edge_min, d_edge_max, Omega_max, Phi_max, Lambda_max,
              weighted, w_max=1, directed=False,
              num_workers=0, path_experts=None, path_data=None, resume_training=False, path_train="",
              path_test_data=None, exact_protection=False, n_epoch_already_trained=0, batch_unroll=None):

    # Gather the hyperparameters
    dict_args = locals()
    # Gather the date as a string
    date_str = (
            datetime.now().strftime('%b')
            + str(datetime.now().day)
            + "_"
            + str(datetime.now().hour)
            + "-"
            + str(datetime.now().minute)
            + "-"
            + str(datetime.now().second)
    )
    # Tensorboard init
    writer = SummaryWriter()
    # Init the step count
    count = 0
    # Compute n_max
    n_max = n_free_max + Omega_max + Phi_max + Lambda_max
    # Initialize the Value neural network
    value_net = DQN(
        dim_input=5,
        dim_embedding=dim_embedding,
        dim_values=dim_values,
        dim_hidden=dim_hidden,
        n_heads=n_heads,
        n_att_layers=n_att_layers,
        n_pool=n_pool,
        K=n_max,
        alpha=alpha,
        p=p,
        weighted=weighted,
    ).to(device)
    # Initialize the pool of experts (target nets)
    targets_experts = TargetExpertsDQN(
        dim_input=5,
        dim_embedding=dim_embedding,
        dim_values=dim_values,
        dim_hidden=dim_hidden,
        n_heads=n_heads,
        n_att_layers=n_att_layers,
        n_pool=n_pool,
        K=n_max,
        alpha=alpha,
        weighted=weighted,
        Omega_max=Omega_max,
        Phi_max=Phi_max,
        Lambda_max=Lambda_max,
        path_experts=path_experts,
        path_data=path_data,
        exact_protection=exact_protection,
    )
    # Initialize the optimizer
    optimizer = optim.Adam(value_net.parameters(), lr=lr, betas=betas)
    # If resume training
    first_epoch = False
    if resume_training:
        # load the state dicts of the optimizer and value_net
        value_net, optimizer = load_training_param(value_net, optimizer, path_train)
        first_epoch = True
    # generate the test set
    test_set_generators = load_create_test_set_dqn(n_free_min, n_free_max, d_edge_min, d_edge_max, Omega_max, Phi_max,
                                                   Lambda_max, weighted, w_max, directed, size_test_data, path_test_data,
                                                   batch_size, num_workers)

    print("Number of parameters to train = %2d \n" % count_param_NN(value_net))

    # While all the target nets are not trained
    Budget_max = Omega_max + Phi_max + Lambda_max
    while targets_experts.Budget_target <= Budget_max:

        print("\n==========================================================================")
        print("Training for Budget = %2d \n" % targets_experts.Budget_target)

        # Load or Create the training and validation datasets
        training_generator, val_generator = load_create_datasets_dqn(
            size_train_data,
            size_val_data,
            batch_size,
            num_workers,
            n_free_min,
            n_free_max,
            d_edge_min,
            d_edge_max,
            Omega_max,
            Phi_max,
            Lambda_max,
            weighted,
            w_max,
            directed,
            targets_experts.Budget_target,
            targets_experts.list_target_nets,
            path_data,
            solve_exact=False,
            exact_protection=exact_protection,
            batch_unroll=batch_unroll,
        )
        # Init the player
        player = targets_experts.list_players[targets_experts.Budget_target-1]
        # Loop over epochs
        if first_epoch:
            n_loops = n_epoch - n_epoch_already_trained
            first_epoch = False
        else:
            n_loops = n_epoch
        for epoch in tqdm(range(n_loops)):
            # Training for all batches in the training set
            for i_batch, batch_instances in enumerate(training_generator):
                # Compute the approximate values
                batch = batch_instances.batch
                mask_values = batch_instances.J.eq(0)[:, 0]
                action_values = value_net(batch_instances.G_torch,
                                          batch_instances.n_nodes,
                                          batch_instances.Omegas,
                                          batch_instances.Phis,
                                          batch_instances.Lambdas,
                                          batch_instances.Omegas_norm,
                                          batch_instances.Phis_norm,
                                          batch_instances.Lambdas_norm,
                                          batch_instances.J,
                                          )
                action_values = action_values[mask_values]
                batch = batch[mask_values]
                # if it's the turn of the attacker
            if player == 1:
                # we take the argmin
                values_approx, actions = scatter_min(action_values, batch, dim=0)
            else:
                # we take the argmax
                values_approx, actions = scatter_max(action_values, batch, dim=0)
                # Init the optimizer
                optimizer.zero_grad()
                # Compute the loss of the batch
                loss = torch.sqrt(torch.mean((values_approx[:, 0] - batch_instances.target[:, 0]) ** 2))
                # Compute the loss on the Validation set
                if count % update_experts == 0:
                    targets_experts.test_update_target_nets(value_net, val_generator, test_set_generators)
                # Update the parameters of the Value_net
                loss.backward()
                optimizer.step()
                # Update the tensorboard
                writer.add_scalar("Loss", float(loss), count)
                writer.add_scalar("Loss validation best",
                                  targets_experts.losses_validation_sets[targets_experts.Budget_target - 1],
                                  count,
                                  )
                writer.add_scalar("Loss validation", targets_experts.loss_value_net, count)
                for k in range(len(targets_experts.losses_test_set)):
                    name_loss = 'Loss test budget ' + str(k + 1)
                    writer.add_scalar(name_loss, float(targets_experts.losses_test_set[k]), count)
                count += 1

            # Print the information of the epoch
            print(
                " \n Budget target : %2d/%2d" % (targets_experts.Budget_target, Budget_max - 1),
                " \n Epoch: %2d/%2d" % (epoch + 1, n_epoch),
                " \n Loss of the current value net: %f" % float(loss),
                " \n Loss val of the current value net: %f" % targets_experts.loss_value_net,
                " \n Losses of the experts on val set: ", targets_experts.losses_validation_sets,
                " \n Losses on test set : ", targets_experts.losses_test_set,
            )
            # Saves model
            save_models(date_str, dict_args, value_net, optimizer, count, targets_experts)

        # Update the target budget
        targets_experts.Budget_target += 1


def train_dqn_mc(batch_size, size_test_data, lr, betas, n_episode, update_target, n_time_instance_seen,
                 eps_end, eps_decay, eps_start,
                 dim_embedding, dim_values, dim_hidden, n_heads, n_att_layers, n_pool, alpha, p,
                 n_free_min, n_free_max, d_edge_min, d_edge_max, Omega_max, Phi_max, Lambda_max, weighted,
                 w_max=1, directed=False,
                 num_workers=0, resume_training=False, path_train="", path_test_data=None,
                 exact_protection=False, rate_display=200, batch_unroll=None):

    """Train a neural network to solve the MCN problem either using Monte Carlo samples"""

    # Gather the hyperparameters
    dict_args = locals()
    # Gather the date as a string
    date_str = (
            datetime.now().strftime('%b')
            + str(datetime.now().day)
            + "_"
            + str(datetime.now().hour)
            + "-"
            + str(datetime.now().minute)
            + "-"
            + str(datetime.now().second)
    )
    # Tensorboard init
    writer = SummaryWriter()
    # Init the counts
    count_steps = 0
    count_instances = 0
    # Compute n_max
    n_max = n_free_max + Omega_max + Phi_max + Lambda_max
    max_budget = Omega_max + Phi_max + Lambda_max
    list_players = [0]*Omega_max + [1]*Phi_max + [2]*Lambda_max
    # Compute the size of the memory and the rate of epoch over it
    # depending on the number of time we want to 'see' each instance, the
    # total number of episodes to generate and the batch size
    size_memory = batch_size * n_time_instance_seen
    n_instance_before_epoch = batch_size
    size_batch_instances = batch_size
    n_episode_batch = n_episode // size_batch_instances
    # Init the value net
    value_net = DQN(
        dim_input=5,
        dim_embedding=dim_embedding,
        dim_values=dim_values,
        dim_hidden=dim_hidden,
        n_heads=n_heads,
        n_att_layers=n_att_layers,
        n_pool=n_pool,
        K=n_max,
        alpha=alpha,
        p=p,
        weighted=weighted,
    ).to(device)
    # Initialize the optimizer
    optimizer = optim.Adam(value_net.parameters(), lr=lr, betas=betas)
    # Initialize the memory
    replay_memory = []
    count_memory = 0
    # If resume training
    if resume_training:
        # load the state dicts of the optimizer and value_net
        value_net, optimizer = load_training_param(value_net, optimizer, path_train)
    # Init the target net
    target_net = DQN(
        dim_input=5,
        dim_embedding=dim_embedding,
        dim_values=dim_values,
        dim_hidden=dim_hidden,
        n_heads=n_heads,
        n_att_layers=n_att_layers,
        n_pool=n_pool,
        K=n_max,
        alpha=alpha,
        p=p,
        weighted=weighted,
    ).to(device)
    target_net.load_state_dict(value_net.state_dict())
    target_net.eval()
    # in order to use the current value_net during training for an evaluation task,
    # we first create a second instance of ValueNet in which we will load the
    # state_dicts of the learning value_net before each use
    value_net_bis = DQN(
        dim_input=5,
        dim_embedding=dim_embedding,
        dim_values=dim_values,
        dim_hidden=dim_hidden,
        n_heads=n_heads,
        n_att_layers=n_att_layers,
        n_pool=n_pool,
        K=n_max,
        alpha=alpha,
        p=p,
        weighted=weighted,
    ).to(device)
    # generate the test set
    test_set_generators = load_create_test_set_dqn(n_free_min, n_free_max, d_edge_min, d_edge_max, Omega_max, Phi_max,
                                                   Lambda_max, weighted, w_max, directed, size_test_data, path_test_data,
                                                   batch_size, num_workers)
    losses_test = [0]*max_budget

    print("Number of parameters to train = %2d \n" % count_param_NN(value_net))

    for episode in tqdm(range(n_episode_batch)):
        # Sample a random batch of instances from where to begin
        list_instances = generate_random_batch_instance(
            batch_size,
            n_free_min,
            n_free_max,
            d_edge_min,
            d_edge_max,
            Omega_max,
            Phi_max,
            Lambda_max,
            Budget_target=max_budget,
            weighted=weighted,
            w_max=w_max,
            directed=directed,
        )
        # Initialize the environment
        env = EnvironmentDQN(list_instances)
        # Init the list of instances for the episode
        instances_episode = []
        # Unroll the episode
        while env.Budget >= 1:
            instances_episode.append(env.batch_instance)
            action = sample_action_batch_dqn(target_net,
                                             env.player,
                                             env.batch_instance_torch,
                                             eps_end,
                                             eps_decay,
                                             eps_start,
                                             count_steps
                                             )
            env.step(action)
            value = env.rewards
        # Add the instances from the episode to memory
        for batch_instance in instances_episode:
            for k in range(len(batch_instance)):
                instance_k = batch_instance[k]
                instance_k.value = value[k]
                instance_k_torch = instance_k_torch(instance_k)
                if len(replay_memory) < size_memory:
                    replay_memory.append(None)
                replay_memory[count_memory % size_memory] = instance_k_torch
                count_memory += 1

        # perform an epoch over the replay memory
        # if there is enough new instances in memory
        if count_memory > size_memory:
            # create a list of randomly shuffled indices to sample batches from
            memory_size = len(replay_memory)
            n_batch = memory_size // batch_size + 1 * (memory_size % batch_size > 0)
            ids_batch = random.sample(range(memory_size), memory_size)
            # sample the batches from the memory in the order defined with ids_batch
            for i_batch in range(n_batch):
                if i_batch == n_batch - 1:
                    id_batch = ids_batch[i_batch*batch_size:]
                else:
                    id_batch = ids_batch[i_batch*batch_size: (i_batch + 1)*batch_size]
                batch_data = [replay_memory[k] for k in id_batch]
                batch_instances = collate_fn(batch_data)
                # Compute the approximate values
                batch = batch_instances.batch
                mask_values = batch_instances.J.eq(0)[:, 0]
                action_values = value_net(batch_instances.G_torch,
                                          batch_instances.n_nodes,
                                          batch_instances.Omegas,
                                          batch_instances.Phis,
                                          batch_instances.Lambdas,
                                          batch_instances.Omegas_norm,
                                          batch_instances.Phis_norm,
                                          batch_instances.Lambdas_norm,
                                          batch_instances.J,
                                          )
                action_values = action_values[mask_values]
                batch = batch[mask_values]
                # Compute the masks to apply
                mask_attack = batch_instances.player.eq(1)[:, 0]
                mask_defend = torch.logical_not(mask_attack)
                # compute both the max and min of the action values
                val_max, _ = scatter_max(action_values, batch, dim=0)
                val_min, _ = scatter_min(action_values, batch, dim=0)
                val_max = val_max[mask_defend]
                val_min = val_min[mask_attack]
                values_approx = torch.cat(val_max, val_min)[:, 0]
                # compute both the max and the min of the targets
                target_max = batch_instances.target[mask_defend]
                target_min = batch_instances.target[mask_attack]
                batch_target = torch.cat(target_max, target_min)[:, 0]
                # Init the optimizer
                optimizer.zero_grad()
                # Compute the loss of the batch
                loss = torch.sqrt(torch.mean((values_approx[:, 0] - batch_target) ** 2))
                # compute the loss on the test set using the value_net_bis
                value_net_bis.load_state_dict(value_net.state_dict())
                value_net_bis.eval()
                # Check the test losses every 20 steps
                if count_steps % 20 == 0:
                    losses_test = compute_loss_test_dqn(test_set_generators, list_players, value_net=value_net_bis)
                for k in range(len(losses_test)):
                    name_loss = 'Loss test budget ' + str(k+1)
                    writer.add_scalar(name_loss, float(losses_test[k]), count_steps)
                # Update the parameters of the Value_net
                loss.backward()
                optimizer.step()
                # Update the tensorboard
                writer.add_scalar("Loss", float(loss), count_steps)
                count_steps += 1

                # Update the target net
                if count_steps % update_target == 0:
                    target_net.load_state_dict(value_net.state_dict())
                    target_net.eval()

                # Saves model every rate_display steps
                if count_steps % rate_display == 0:
                    save_models(date_str, dict_args, value_net, optimizer, count_steps)
                    print(
                        " \n Episode: %2d/%2d" % (episode*size_batch_instances, n_episode),
                        " \n Loss of the current value net: %f" % float(loss),
                        " \n Losses on test set : ", losses_test,
                    )
