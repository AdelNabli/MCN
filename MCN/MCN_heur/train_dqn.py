import os
import pickle
from tqdm import tqdm
import random
import numpy as np
from MCN.MCN_heur.neural_networks import DQN
from MCN.solve_mcn import solve_mcn
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from datetime import datetime
from torch_scatter import scatter_min, scatter_max
from MCN.MCN_heur.data import collate_fn, MCNDataset, DataLoader
from MCN.utils import (
    new_graph,
    get_player,
    compute_saved_nodes,
    Instance,
    InstanceTorch,
    instance_to_torch,
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
        self.mappings = []
        self.batch_instance_torch = None
        self.list_instance_torch = None
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
        batch_instance_torch = []
        # transform the list of instances to a list of instance torch
        for instance in self.batch_instance:
            instance_torch = instance_to_torch(instance)
            batch_instance_torch.append(instance_torch)
        self.list_instance_torch = batch_instance_torch
        # create a proper batch from the list
        self.batch_instance_torch = collate_fn(batch_instance_torch)

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
            J_k = instance_k.J.copy()
            G_k = instance_k.G.copy()
            node_k = self.mappings[k][action_k]
            if self.player == 1:
                J_k += [node_k]
            else:
                G_k, mapping = new_graph(G_k, node_k)
                J_k = [mapping[j] for j in J_k]
            # if we are not at the end of the episode
            # we need to compute the new states
            if self.next_player != 3:
                new_instance = Instance(G_k, self.next_Omega, self.next_Phi, self.next_Lambda, J_k, 0)
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
        batch = batch_instances.G_torch.batch
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
                    batch = batch_instances.G_torch.batch
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


def train_dqn(batch_size, size_test_data, lr, betas, n_episode, update_target, n_time_instance_seen,
              eps_end, eps_decay, eps_start,
              dim_embedding, dim_values, dim_hidden, n_heads, n_att_layers, n_pool, alpha, p,
              n_free_min, n_free_max, d_edge_min, d_edge_max, Omega_max, Phi_max, Lambda_max, weighted,
              w_max=1, directed=False,
              num_workers=0, resume_training=False, path_train="", path_test_data=None,
              exact_protection=False, rate_display=200, batch_unroll=128):

    """Train a DQN to solve the MCN problem"""

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
    list_players = [2]*Lambda_max + [1]*Phi_max + [0]*Omega_max
    # Compute the size of the memory
    size_memory = batch_size * n_time_instance_seen
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
    replay_memory_states = []
    replay_memory_actions = []
    replay_memory_afterstates = []
    replay_memory_rewards = []
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

    for episode in tqdm(range(n_episode)):
        # Sample a random batch of instances from where to begin
        list_instances = generate_random_batch_instance(
            batch_unroll,
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
        current_states = None
        current_actions = None
        current_rewards = None
        cpt_budget = 0
        # Unroll the episode
        while env.Budget >= 1:
            last_states = current_states
            current_states = env.list_instance_torch
            action = sample_action_batch_dqn(value_net,
                                             env.player,
                                             env.batch_instance_torch,
                                             eps_end,
                                             eps_decay,
                                             eps_start,
                                             count_steps
                                             )
            env.step(action)
            last_actions = current_actions
            current_actions = action
            last_rewards = current_rewards
            current_rewards = env.rewards
            cpt_budget += 1

            # if we have the couples (state, afterstates) available
            if cpt_budget > 1:
                n_visited = 0
                for k in range(batch_unroll):
                    if len(replay_memory_states) < size_memory:
                        replay_memory_states.append(None)
                        replay_memory_afterstates.append(None)
                        replay_memory_actions.append(None)
                        replay_memory_rewards.append(None)
                    replay_memory_states[count_memory % size_memory] = last_states[k]
                    replay_memory_afterstates[count_memory % size_memory] = current_states[k]
                    replay_memory_rewards[count_memory % size_memory] = last_rewards[k]
                    n_free = int(torch.sum(last_states[k].J.eq(0)[:,0]))
                    replay_memory_actions[count_memory % size_memory] = last_actions[k] - n_visited
                    n_visited += n_free
                    count_memory += 1
            # If we are in the last step, we push to memory the end rewards
            if env.Budget == 0 and cpt_budget > 1:
                n_visited = 0
                for k in range(batch_unroll):
                    if len(replay_memory_states) < size_memory:
                        replay_memory_states.append(None)
                        replay_memory_afterstates.append(None)
                        replay_memory_actions.append(None)
                        replay_memory_rewards.append(None)
                    replay_memory_states[count_memory % size_memory] = current_states[k]
                    # doesn't matter what we put in the afterstates here
                    replay_memory_afterstates[count_memory % size_memory] = current_states[k]
                    replay_memory_rewards[count_memory % size_memory] = current_rewards[k]
                    n_free = int(torch.sum(current_states[k].J.eq(0)[:, 0]))
                    replay_memory_actions[count_memory % size_memory] = current_actions[k] - n_visited
                    n_visited += n_free
                    count_memory += 1

            # if there is enough new instances in memory
            if count_memory > size_memory:
                # create a list of randomly shuffled indices to sample a batch from
                memory_size = len(replay_memory_states)
                id_batch = random.sample(range(memory_size), batch_size)
                # gather the states, afterstates, actions and rewards
                list_states = [replay_memory_states[k] for k in id_batch]
                list_afterstates = [replay_memory_afterstates[k] for k in id_batch]
                list_actions = [replay_memory_actions[k] for k in id_batch]
                list_rewards = [replay_memory_rewards[k] for k in id_batch]
                # recover the actions id in the batch
                n_visited = 0
                list_actions_new = []
                for k in range(len(list_actions)):
                    n_free = int(torch.sum(list_states[k].J.eq(0)[:,0]))
                    list_actions_new.append(list_actions[k]+n_visited)
                    n_visited += n_free
                # create the tensors
                batch_states = collate_fn(list_states)
                batch_afterstates = collate_fn(list_afterstates)
                batch_actions = torch.tensor(list_actions_new, dtype=torch.long).view([len(list_actions), 1]).to(device)
                batch_rewards = torch.tensor(list_rewards, dtype=torch.float).view([len(list_rewards), 1]).to(device)
                # Compute the approximate values
                action_values = value_net(batch_states.G_torch,
                                          batch_states.n_nodes,
                                          batch_states.Omegas,
                                          batch_states.Phis,
                                          batch_states.Lambdas,
                                          batch_states.Omegas_norm,
                                          batch_states.Phis_norm,
                                          batch_states.Lambdas_norm,
                                          batch_states.J,
                                          )
                # mask the attacked nodes
                mask_values = batch_states.J.eq(0)[:, 0]
                action_values = action_values[mask_values]
                # Gather the approximate values
                approx_values = action_values.gather(0, batch_actions)
                # compute the masks to apply to the target
                mask_attack = batch_states.next_player.eq(1)[:, 0]
                mask_exact = batch_states.next_player.eq(3)[:, 0]

                # Compute the approximate targets
                with torch.no_grad():
                    target_values = target_net(batch_afterstates.G_torch,
                                               batch_afterstates.n_nodes,
                                               batch_afterstates.Omegas,
                                               batch_afterstates.Phis,
                                               batch_afterstates.Lambdas,
                                               batch_afterstates.Omegas_norm,
                                               batch_afterstates.Phis_norm,
                                               batch_afterstates.Lambdas_norm,
                                               batch_afterstates.J,
                                               ).detach()
                        
                    batch = batch_afterstates.G_torch.batch
                    mask_J = batch_afterstates.J.eq(0)[:, 0]
                    # mask the attacked nodes
                    batch = batch[mask_J]
                    target_values = target_values[mask_J]
                    # Compute the min and max
                    val_min, _ = scatter_min(target_values, batch, dim=0)
                    val_max, _ = scatter_max(target_values, batch, dim=0)
                    # create the target tensor
                    target = val_max
                    target[mask_attack] = val_min[mask_attack]
                    target[mask_exact] = batch_rewards[mask_exact]

                # Init the optimizer
                optimizer.zero_grad()
                # Compute the loss of the batch
                loss = torch.sqrt(torch.mean((approx_values - target) ** 2))
                # Update the parameters of the Value_net
                loss.backward()
                optimizer.step()
                # compute the loss on the test set using the value_net_bis
                value_net_bis.load_state_dict(value_net.state_dict())
                value_net_bis.eval()
                # Check the test losses every 20 steps
                if count_steps % 20 == 0:
                    losses_test = compute_loss_test_dqn(test_set_generators, list_players, value_net=value_net_bis)
                for k in range(len(losses_test)):
                    name_loss = 'Loss test budget ' + str(k+1)
                    writer.add_scalar(name_loss, float(losses_test[k]), count_steps)
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
                        " \n Episode: %2d/%2d" % (episode*batch_size, n_episode),
                        " \n Loss of the current value net: %f" % float(loss),
                        " \n Losses on test set : ", losses_test,
                    )

def solve_greedy_dqn(instance, value_net):
    env = EnvironmentDQN([instance])
    Omega = env.Omega
    Lambda = env.Lambda
    # Unroll the episode
    while env.Budget >= 1:
        action = take_action_deterministic_batch_dqn(value_net, env.player, env.batch_instance_torch)
        env.step(action)
        rewards = env.rewards
    return Omega + Lambda + rewards[0]

