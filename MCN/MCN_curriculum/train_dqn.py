import torch
import os
import random
import pickle
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from MCN.test_performances.optimality_gap import generate_test_set
from MCN.utils import (
    save_models,
    load_training_param,
    count_param_NN,
    generate_random_instance,
    take_action_deterministic,
    Instance,
    instance_to_torch,
)
from MCN.MCN_curriculum.data import collate_fn, MCNDataset
from MCN.MCN_curriculum.environment import Environment
from MCN.MCN_curriculum.value_nn import ValueNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_loss_test(test_set_generators, value_net=None, list_experts=None):

    list_losses = []
    with torch.no_grad():
        for k in range(len(test_set_generators)):
            target = []
            val_approx = []
            if list_experts is not None:
                target_net = list_experts[k]
            elif value_net is not None:
                target_net = value_net
            for i_batch, batch_instances in enumerate(test_set_generators[k]):
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
                )
                val_approx.append(values_approx)
                target.append(batch_instances.target)
            # Compute the loss
            target = torch.cat(target)
            val_approx = torch.cat(val_approx)
            loss_target_net = float(torch.sqrt(torch.mean((val_approx[:, 0] - target[:, 0]) ** 2)))
            list_losses.append(loss_target_net)

    return list_losses


def train_value_net_dqn(batch_size, size_memory, size_test_data, lr, betas, n_instances, update_target, count_step,
                        dim_embedding, dim_values, dim_hidden, n_heads, n_att_layers, n_pool, alpha, p,
                        n_free_min, n_free_max, d_edge_min, d_edge_max, Omega_max, Phi_max, Lambda_max,
                        num_workers=0, resume_training=False, path_train="", path_test_data=None):

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
    replay_memory = [None]*size_memory
    count_memory = 0
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
    target_net_test = ValueNet(
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
    if path_test_data is None:
        generate_test_set(n_free_min, n_free_max, d_edge_min, d_edge_max, Omega_max-1, Phi_max, Lambda_max,
                          size_test_data, to_torch=True)
        path_test_set = os.path.join('data', 'test_data', 'test_set_torch.gz')
    else:
        path_test_set = path_test_data
    # load the test set
    test_set = pickle.load(open(path_test_set, "rb"))
    # create a dataloader object for each dataset in the test set
    test_set_generators = []
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
            count_instances += 1
            # Take an action
            action, targets, value = take_action_deterministic(
                target_net,
                env.player,
                env.next_player,
                env.next_rewards,
                env.next_list_G_torch,
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
            # Update the environment
            env.step(action)

            # take an optim step
            if count_instances % count_step == 0 and count_memory > batch_size:
                batch_data = random.sample(replay_memory, batch_size)
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
                # Init the optimizer
                optimizer.zero_grad()
                # Compute the loss of the batch
                loss = torch.sqrt(torch.mean((values_approx[:, 0] - batch_instances.target[:, 0]) ** 2))
                # compute the loss on the test set
                target_net_test.load_state_dict(value_net.state_dict())
                target_net_test.eval()
                losses_test = compute_loss_test(test_set_generators, value_net=target_net_test)
                for k in range(len(losses_test)):
                    name_loss = 'Loss test budget = ' + str(k+1)
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

                if count_steps % 200 == 0:
                    print(
                        " \n Instances: %2d/%2d" % (count_instances, n_instances),
                        " \n Loss of the current value net: %f" % float(loss),
                        " \n Losses on test set : ", losses_test,
                    )

        # transform the instances to instance torch
        for instance in instances_episode:
            instance.value = value
            instance_torch = instance_to_torch(instance)
            replay_memory[count_memory % size_memory] = instance_torch
            count_memory += 1
