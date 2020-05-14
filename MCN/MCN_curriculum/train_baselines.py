import os
import torch
import random
import torch.optim as optim
from tensorboardX import SummaryWriter
from tqdm import tqdm
from datetime import datetime
from MCN.utils import (
    save_models,
    load_training_param,
    count_param_NN,
    generate_random_batch_instance,
    sample_action,
    Instance,
    InstanceTorch,
    instance_to_torch,
    compute_loss_test,
)
from MCN.MCN_curriculum.data import collate_fn, load_create_test_set
from MCN.MCN_curriculum.environment import Environment
from MCN.MCN_curriculum.value_nn import ValueNet
from MCN.test_performances.optimality_gap import compute_optimality_gap
from torch_scatter import scatter_max, scatter_min

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_targets_dqn(target_net, instances_batch, targets, players, next_players):

    """In DQN, the targets used in the loss are the approximate values of the next state
    given by the neural network"""

    n = len(instances_batch)
    # Initialize the masks:
    # we have to compute the approximate values of each possible afterstate,
    # but the target is either the min or the max of all these possibilities depending
    # on whose player it is the turn to play. So we compute all the approximate values
    # first and then gather the min or the max depending on the player.
    players_torch = torch.tensor(players)
    next_players_torch = torch.tensor(next_players)
    mask_approx = next_players_torch.le(2)
    mask_exact = next_players_torch.eq(3)
    players_approx = players_torch[mask_approx]
    mask_approx_min = players_approx.eq(1)
    mask_approx_max = torch.logical_not(mask_approx_min)
    players_exact = players_torch[mask_exact]
    mask_exact_min = players_exact.eq(1)
    mask_exact_max = torch.logical_not(mask_exact_min)
    # Initializes the variables
    instances_torch = [instances_batch[k] for k in range(n) if mask_approx[k]]
    id_graph_approx = torch.tensor(
        [i for i in range(len(instances_torch)) for k in range(len(instances_torch[i].G_torch))]
    ).to(device)
    batch_instances = collate_fn(instances_torch, for_dqn=True)
    # Compute the approximate values
    values_approx = target_net(
        batch_instances.G_torch,
        batch_instances.n_nodes,
        batch_instances.Omegas,
        batch_instances.Phis,
        batch_instances.Lambdas,
        batch_instances.Omegas_norm,
        batch_instances.Phis_norm,
        batch_instances.Lambdas_norm,
        batch_instances.J,
        batch_instances.saved_nodes,
        batch_instances.infected_nodes,
        batch_instances.size_connected,
    )[:, 0]
    targets_approx_min = scatter_min(values_approx, id_graph_approx)[0]
    targets_approx_max = scatter_max(values_approx, id_graph_approx)[0]
    # Gather the exact values when the next player is 3
    values_exact = [targets[k] for k in range(n) if mask_exact[k]]
    id_graph_exact = torch.tensor(
        [i for i in range(len(values_exact)) for k in range(values_exact[i].size()[0])]
    ).to(device)
    values_exact = torch.cat(values_exact)[:, 0]
    targets_exact_min = scatter_min(values_exact, id_graph_exact)[0]
    targets_exact_max = scatter_max(values_exact, id_graph_exact)[0]
    # Collate the different targets corresponding to each possible case
    # into a single target tensor
    targets = torch.tensor([0]*n, dtype=torch.float).to(device)
    targets[mask_exact][mask_exact_min] = targets_exact_min[mask_exact_min]
    targets[mask_exact][mask_exact_max] = targets_exact_max[mask_exact_max]
    targets[mask_approx][mask_approx_min] = targets_approx_min[mask_approx_min]
    targets[mask_approx][mask_approx_max] = targets_approx_max[mask_approx_max]

    return targets


def train_value_net_baseline(batch_size, size_test_data, lr, betas, n_episode, update_target, n_time_instance_seen,
                             eps_end, eps_decay, eps_start,
                             dim_embedding, dim_values, dim_hidden, n_heads, n_att_layers, n_pool, alpha, p,
                             n_free_min, n_free_max, d_edge_min, d_edge_max, Omega_max, Phi_max, Lambda_max, weighted,
                             w_max=1, directed=False,
                             num_workers=0, resume_training=False, path_train="", path_test_data=None,
                             training_method='MC', exact_protection=False, rate_display=200):

    """Train a neural network to solve the MCN problem either using Monte Carlo samples or with Q-learning"""

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
    max_budget = Omega_max + Phi_max + Lambda_max - 1
    # Compute the size of the memory and the rate of epoch over it
    # depending on the number of time we want to 'see' each instance, the
    # total number of episodes to generate and the batch size
    size_memory = batch_size * n_time_instance_seen
    n_instance_before_epoch = batch_size
    # Compute the size of the batch of instances we can generate
    # and unroll in parallel
    min_size_instance = n_free_min + 1
    max_size_instance = n_free_max + max_budget
    size_batch_instances = (max_size_instance - min_size_instance) // 2
    n_episode_batch = n_episode // size_batch_instances
    # Init the value net
    value_net = ValueNet(
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
    if training_method == 'DQN':
        memory_player = []
        memory_next_player = []
        memory_next_state = []
        memory_targets = []
    # If resume training
    if resume_training:
        # load the state dicts of the optimizer and value_net
        value_net, optimizer = load_training_param(value_net, optimizer, path_train)
    # Init the target net
    target_net = ValueNet(
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
    value_net_bis = ValueNet(
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
    test_set_generators = load_create_test_set(n_free_min, n_free_max, d_edge_min, d_edge_max, Omega_max, Phi_max,
                                               Lambda_max, weighted, w_max, directed, size_test_data, path_test_data,
                                               batch_size, num_workers)

    print("Number of parameters to train = %2d \n" % count_param_NN(value_net))

    for episode in tqdm(range(n_episode_batch)):
        # Sample a random instance from where to begin
        list_instances = generate_random_batch_instance(
            size_batch_instances,
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
        env = Environment(list_instances)
        # Init the list of instances for the episode
        instances_episode = []
        if training_method == 'DQN':
            player_episode = []
            next_player_episode = []
            next_state_episode = []
            targets_episode = []
        # Unroll the episode
        while env.Budget >= 1:
            # update the environment
            env.compute_current_situation()
            # save the current instance
            current_instances = Instance(env.next_G, env.Omega, env.Phi, env.Lambda, env.next_J, 0)
            instances_episode.append(current_instance)
            # Take an action
            # begin by choosing which neural network is used as a policy network
            # depending on the method of training
            if training_method == 'DQN':
                value_net_bis.load_state_dict(value_net.state_dict())
                value_net_bis.eval()
                neural_net_policy = value_net_bis
            else:
                neural_net_policy = target_net
            action, targets, value = sample_action(
                neural_net_policy,
                env.player,
                env.next_player,
                env.next_rewards,
                env.next_list_G_torch,
                eps_end,
                eps_decay,
                eps_start,
                count_steps,
                n_nodes=env.next_n_nodes_tensor,
                Omegas=env.next_Omega_tensor,
                Phis=env.next_Phi_tensor,
                Lambdas=env.next_Lambda_tensor,
                Omegas_norm=env.next_Omega_norm,
                Phis_norm=env.next_Phi_norm,
                Lambdas_norm=env.next_Lambda_norm,
                J=env.next_J_tensor,
            )
            if training_method == 'DQN':
                # save the the parameters necessary to compute the
                # approximate values of the next afterstates
                next_instance_torch = InstanceTorch(
                    G_torch=env.next_list_G_torch,
                    n_nodes=env.next_n_nodes_tensor,
                    Omegas=env.next_Omega_tensor,
                    Phis=env.next_Phi_tensor,
                    Lambdas=env.next_Lambda_tensor,
                    Omegas_norm=env.next_Omega_norm,
                    Phis_norm=env.next_Phi_norm,
                    Lambdas_norm=env.next_Lambda_norm,
                    J=env.next_J_tensor,
                    target=torch.tensor(value).view([1,1]),
                )
                player_episode.append(env.player)
                next_player_episode.append(env.next_player)
                next_state_episode.append(next_instance_torch)
                targets_episode.append(targets)

                count_instances += 1
                # Update the environment
                env.step(action)

        # perform an epoch over the replay memory
        # if there is enough new instances in memory
        if count_instances % n_instance_before_epoch == 0 and count_memory > batch_size:
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
                values_approx = value_net(
                    batch_instances.G_torch,
                    batch_instances.n_nodes,
                    batch_instances.Omegas,
                    batch_instances.Phis,
                    batch_instances.Lambdas,
                    batch_instances.Omegas_norm,
                    batch_instances.Phis_norm,
                    batch_instances.Lambdas_norm,
                    batch_instances.J,
                    batch_instances.saved_nodes,
                    batch_instances.infected_nodes,
                    batch_instances.size_connected,
                )
                if training_method == 'DQN':
                    instances_batch = [memory_next_state[k] for k in id_batch]
                    targets_batch = [memory_targets[k] for k in id_batch]
                    players = [memory_player[k] for k in id_batch]
                    next_players = [memory_next_player[k] for k in id_batch]
                    # compute the target for the batch
                    batch_target = compute_targets_dqn(target_net, instances_batch, targets_batch,
                                                       players, next_players)
                else:
                    batch_target = batch_instances.target[:, 0]
                # Init the optimizer
                optimizer.zero_grad()
                # Compute the loss of the batch
                loss = torch.sqrt(torch.mean((values_approx[:, 0] - batch_target) ** 2))
                # compute the loss on the test set using the value_net_bis
                value_net_bis.load_state_dict(value_net.state_dict())
                value_net_bis.eval()
                if exact_protection:
                    list_experts = [None]*Lambda_max
                    list_experts += [value_net_bis]*(Omega_max + Phi_max - 1)
                    losses_test = compute_loss_test(test_set_generators, list_experts=list_experts)
                else:
                    losses_test = compute_loss_test(test_set_generators, value_net=value_net_bis)
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
                        " \n Episode: %2d/%2d" % (episode, n_episode),
                        " \n Loss of the current value net: %f" % float(loss),
                        " \n Losses on test set : ", losses_test,
                    )

        # add the instances from the episode to memory
        for k in range(len(instances_episode)):
            instance = instances_episode[k]
            instance.value = value
            instance_torch = instance_to_torch(instance)
            if len(replay_memory) < size_memory:
                replay_memory.append(None)
                if training_method == 'DQN':
                    memory_player.append(None)
                    memory_next_player.append(None)
                    memory_next_state.append(None)
                    memory_targets.append(None)
            replay_memory[count_memory % size_memory] = instance_torch
            if training_method == 'DQN':
                memory_player[count_memory % size_memory] = player_episode[k]
                memory_next_player[count_memory % size_memory] = next_player_episode[k]
                memory_next_state[count_memory % size_memory] = next_state_episode[k]
                memory_targets[count_memory % size_memory] = targets_episode[k]
            count_memory += 1

    # Compute how the neural networks we trained perform on the test set
    if size_test_data > 0:
        if path_test_data is None:
            # the test data has been generated before the training
            folder_name = 'test_data'
            if weighted:
                folder_name += '_w'
            if directed:
                folder_name += '_dir'
            path_test_data = os.path.join('data', folder_name)
        value_net_bis.load_state_dict(value_net.state_dict())
        value_net_bis.eval()
        list_experts = [value_net_bis] * max_budget
        compute_optimality_gap(Omega_max, Phi_max, Lambda_max, list_experts,
                               exact_protection=exact_protection, path_test_data=path_test_data)
