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
    generate_random_instance,
    sample_action,
    Instance,
    InstanceTorch,
    instance_to_torch,
    compute_loss_test,
)
from MCN.MCN_curriculum.data import collate_fn, load_create_test_set
from MCN.MCN_curriculum.environment import Environment
from MCN.MCN_curriculum.value_nn import ValueNet
from torch_scatter import scatter_max, scatter_min

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_targets_dqn(target_net, instances_batch, targets, players, next_players):

    n = len(instances_batch)

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

    instances_torch = [instances_batch[k] for k in range(n) if mask_approx[k]]
    id_graph_approx = torch.tensor(
        [i for i in range(len(instances_torch)) for k in range(len(instances_torch[i].G_torch))]
    ).to(device)
    batch_instances = collate_fn(instances_torch, for_dqn=True)
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

    values_exact = [targets[k] for k in range(n) if mask_exact[k]]
    id_graph_exact = torch.tensor(
        [i for i in range(len(values_exact)) for k in range(values_exact[i].size()[0])]
    ).to(device)
    values_exact = torch.cat(values_exact)[:, 0]
    targets_exact_min = scatter_min(values_exact, id_graph_exact)[0]
    targets_exact_max = scatter_max(values_exact, id_graph_exact)[0]

    targets = torch.tensor([0]*n, dtype=torch.float).to(device)
    targets[mask_exact][mask_exact_min] = targets_exact_min[mask_exact_min]
    targets[mask_exact][mask_exact_max] = targets_exact_max[mask_exact_max]
    targets[mask_approx][mask_approx_min] = targets_approx_min[mask_approx_min]
    targets[mask_approx][mask_approx_max] = targets_approx_max[mask_approx_max]

    return targets


def train_value_net_baseline(batch_size, size_memory, size_test_data, lr, betas, n_instances, update_target, count_step,
                             eps_end, eps_decay, eps_start,
                             dim_embedding, dim_values, dim_hidden, n_heads, n_att_layers, n_pool, alpha, p,
                             n_free_min, n_free_max, d_edge_min, d_edge_max, Omega_max, Phi_max, Lambda_max,
                             num_workers=0, resume_training=False, path_train="", path_test_data=None,
                             training_method='MC'):

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
    ).to(device)
    target_net.load_state_dict(value_net.state_dict())
    target_net.eval()
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
    ).to(device)
    # generate the test set
    test_set_generators = load_create_test_set(n_free_min, n_free_max, d_edge_min, d_edge_max, Omega_max, Phi_max,
                                               Lambda_max, size_test_data, path_test_data, batch_size, num_workers)

    print("Number of parameters to train = %2d \n" % count_param_NN(value_net))

    for episode in tqdm(range(n_instances//max_budget + 1)):
        # Sample a random instance from where to begin
        instance = generate_random_instance(
            n_free_min,
            n_free_max,
            d_edge_min,
            d_edge_max,
            Omega_max,
            Phi_max,
            Lambda_max,
            Budget_target=max_budget,
        )
        # Initialize the environment
        env = Environment(instance.G, instance.Omega, instance.Phi, instance.Lambda, J=instance.J)
        # Init the list of instances for the episode
        instances_episode = []
        # Unroll the episode
        while env.Budget >= 1:
            # update the environment
            env.compute_current_situation()
            # save the current instance
            current_instance = Instance(env.next_G, env.Omega, env.Phi, env.Lambda, env.next_J, 0)
            instances_episode.append(current_instance)
            # Take an action
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
                saved_nodes=env.next_saved_tensor,
                infected_nodes=env.next_infected_tensor,
                size_connected=env.next_size_connected_tensor,
            )
            if training_method == 'DQN':
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
                    saved_nodes=env.next_saved_tensor,
                    infected_nodes=env.next_infected_tensor,
                    size_connected=env.next_size_connected_tensor,
                    target=torch.tensor(value).view([1,1]),
                )
                if len(memory_player) < size_memory:
                    memory_player.append(None)
                    memory_next_player.append(None)
                    memory_next_state.append(None)
                    memory_targets.append(None)
                memory_player[count_instances % size_memory] = env.player
                memory_next_player[count_instances % size_memory] = env.next_player
                memory_next_state[count_instances % size_memory] = next_instance_torch
                memory_targets[count_instances % size_memory] = targets

            count_instances += 1
            # Update the environment
            env.step(action)

            # perform an epoch over the replay memory
            if count_memory % count_step == 0 and count_memory > batch_size:
                memory_size = len(replay_memory)
                n_batch = memory_size // batch_size + 1 * (memory_size % batch_size > 0)
                ids_batch = random.sample(range(memory_size), memory_size)
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
                        batch_target = compute_targets_dqn(target_net, instances_batch, targets_batch, players, next_players)
                    else:
                        batch_target = batch_instances.target[:, 0]
                    # Init the optimizer
                    optimizer.zero_grad()
                    # Compute the loss of the batch
                    loss = torch.sqrt(torch.mean((values_approx[:, 0] - batch_target) ** 2))
                    # compute the loss on the test set
                    value_net_bis.load_state_dict(value_net.state_dict())
                    value_net_bis.eval()
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

                # Saves model
                save_models(date_str, dict_args, value_net, optimizer, count_steps)
                print(
                    " \n Instances: %2d/%2d" % (count_instances, n_instances),
                    " \n Loss of the current value net: %f" % float(loss),
                    " \n Losses on test set : ", losses_test,
                )

        # transform the instances to instance torch
        for instance in instances_episode:
            instance.value = value
            instance_torch = instance_to_torch(instance)
            if len(replay_memory) < size_memory:
                replay_memory.append(None)
            replay_memory[count_memory % size_memory] = instance_torch
            count_memory += 1
