import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from tqdm import tqdm
from datetime import datetime
from collections import namedtuple
from MCN.utils import (
    ReplayMemory,
    generate_random_instance,
    sample_memory,
    compute_loss,
    update_training_memory,
    get_target_net,
    take_action_deterministic,
    save_models,
    load_saved_experts,
)
from .environment import Environment
from .value_nn import ValueNet
from .experts import TargetExperts

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_value_net(batch_size, memory_size, lr, betas, E, target_update, h1, h2, n_heads, alpha, tolerance,
                    n_free_min, n_free_max, Omega_max, Phi_max, Lambda_max, path_experts=""):
    """Training procedure. Follows the evolution of the training using tensorboard.
    Stores the neural networks each time a new task is learnt.

    Parameters:
    ----------
    batch_size: int,
                size of the batch used to compute the loss
    memory_size: int,
                 size of the ReplayMemory used
    lr: float,
        learning rate of the optimizer
    betas: tuple of floats,
           betas used for the optimizer
    E: int,
       total number of episodes to train on
    target_update: int,
                   number of steps after which to test
                   the current value net on the Validation Dataset
    h1: int,
        first hidden dim of the ValueNet
    h2: int,
        second hidden dim of the ValueNet
    n_heads: int,
             number of heads used in the GATs of the ValueNet
    alpha: float,
           alpha value of the APPNP of the ValueNet
    tolerance: float,
               value of the loss under which the value net
               is considered an expert at the task currently at hand
    n_free_min: int,
                minimum number of free nodes of the instances
                we are considering
    n_free_max: int,
                maximum number of free nodes of the instances
                we are considering
    Omega_max: int,
               maximum value of Omega we want the instances to have
    Phi_max: int,
             maximum value of Phi we want the instances to have
    Lambda_max: int,
                maximum value of Lambda we want the instances to have
    path_experts: str (default=""),
                  path the directory containing saved experts models
                  if given, begin the training at the first step after
                  the last trained expert

    Returns:
    -------
    value_net: neural net (pytorch module),
               the trained value network
    targets_experts: TargetExperts object,
                     contains the Validation Dataset,
                     and the list of trained experts """

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
    count = 0
    # Initialize the memory
    Transition = namedtuple(
        "Transition",
        (
            "afterstates",
            "Omegas",
            "Phis",
            "Lambdas",
            "J",
            "saved_nodes",
            "infected_nodes",
            "size_connected",
            "id_loss",
            "targets",
            "next_n_free",
        ),
    )
    memory = ReplayMemory(memory_size, Transition)
    # Compute n_max
    n_max = n_free_max + Omega_max + Phi_max + Lambda_max
    # Initialize the Value neural network
    value_net = ValueNet(
        input_dim=5,
        hidden_dim1=h1,
        hidden_dim2=h2,
        n_heads=n_heads,
        K=n_max,
        alpha=alpha,
    ).to(device)
    # Initialize the pool of experts (target nets)
    targets_experts = TargetExperts(
        input_dim=5,
        hidden_dim1=h1,
        hidden_dim2=h2,
        n_heads=n_heads,
        K=n_max,
        alpha=alpha,
        n_free_min=n_free_min,
        n_free_max=n_free_max,
        Omega_max=Omega_max,
        Phi_max=Phi_max,
        Lambda_max=Lambda_max,
        memory_size=50,
        tolerance=tolerance,
    )
    # If pre-trained experts are available
    if path_experts != "":
        # load them
        list_trained_experts = load_saved_experts(path_experts)
        Budget_trained = len(list_trained_experts)
        # update the TargetExperts object
        targets_experts.list_target_nets[:Budget_trained] = list_trained_experts
        if Budget_trained < targets_experts.n_max:
            targets_experts.Budget_target = Budget_trained + 1
    # Initialize the optimizer
    optimizer = optim.Adam(value_net.parameters(), lr=lr, betas=betas)
    # Initialize the loss memory:
    memory_loss = [100*(tolerance + 1)] * 100

    print("==========================================================================")
    print("Beginning training... \n")

    for episode in tqdm(range(E)):

        # Sample a random instance
        G_nx, I, Omega, Phi, Lambda = generate_random_instance(
            n_free_min,
            n_free_max,
            Omega_max,
            Phi_max,
            Lambda_max,
            targets_experts.Budget_target,
        )
        # Initialize the environment
        env = Environment(G_nx, Omega, Phi, Lambda, J=I)
        # Get the initial player
        initial_player = env.player
        # Initialize the memory of the episode and the sets of actions
        memory_episode = []
        # one list of action for each player
        actions_episode = [[], [], []]

        # While there is some budget to spend
        while env.Budget >= 1:

            # Get the current situation
            env.compute_current_situation()
            # Get the appropriate target net
            list_target_nets = targets_experts.list_target_nets
            target_net = get_target_net(
                list_target_nets,
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
                Omegas=env.next_Omega_tensor,
                Phis=env.next_Phi_tensor,
                Lambdas=env.next_Lambda_tensor,
                J=env.next_J_tensor,
                saved_nodes=env.next_saved_tensor,
                infected_nodes=env.next_infected_tensor,
                size_connected=env.next_size_connected_tensor,
            )
            # save the action to the memory of actions
            actions_episode[env.player].append(action)
            # Save the situation to memory of the episode
            memory_episode.append(
                (
                    env.list_G_torch,
                    env.Omega_tensor,
                    env.Phi_tensor,
                    env.Lambda_tensor,
                    env.J_tensor,
                    env.saved_tensor,
                    env.infected_tensor,
                    env.size_connected_tensor,
                    env.id_loss,
                    targets,
                    env.next_n_free,
                )
            )
            # Update the environment
            env.step(action)

            if len(memory) < batch_size:
                pass
            else:
                # Init the optimizer
                optimizer.zero_grad()
                # Sample the memory
                (
                    afterstates,
                    Omegas,
                    Phis,
                    Lambdas,
                    J,
                    saved_nodes,
                    infected_nodes,
                    size_connected,
                    id_loss,
                    targets,
                    id_graphs,
                ) = sample_memory(memory, Transition, batch_size)
                # Compute the loss
                loss = compute_loss(
                    value_net,
                    id_loss,
                    targets,
                    id_graphs,
                    G_torch=afterstates,
                    Omegas=Omegas,
                    Phis=Phis,
                    Lambdas=Lambdas,
                    J=J,
                    saved_nodes=saved_nodes,
                    infected_nodes=infected_nodes,
                    size_connected=size_connected,
                )
                # Update the parameters of the Value_net
                loss.backward()
                for param in value_net.parameters():
                    param.grad.data.clamp_(-1, 1)
                optimizer.step()

                # Update the tensorboard
                writer.add_scalar("Loss", float(loss), count)

                # update memory loss
                memory_loss[count % 100] = float(loss)
                count += 1

        # update the memory
        memory = update_training_memory(memory, memory_episode, actions_episode, value, initial_player)

        if (
                sum(memory_loss) / 100 < tolerance
                and len(memory) >= batch_size
                and count % target_update == 0
        ):
            print(
                " \n Budget target : %2d" % targets_experts.Budget_target,
                " \n episode: %2d/%2d" % (episode, E),
                " \n loss : %f" % float(loss),
            )
            # test and update the target networks
            targets_experts.test_update_target_nets(value_net)
            # print the losses of the experts
            print(
                "Losses of the experts : " , targets_experts.losses_validation_sets,
                "\n losses of the current value net : " , targets_experts.losses_value_net
            )
            # Saves model
            save_models(date_str, dict_args, value_net, optimizer, count, targets_experts)
            # reset memory loss
            memory_loss = [100*(tolerance + 1)] * 100
    # Saves model
    save_models(date_str, dict_args, value_net, optimizer, count, targets_experts)
    writer.close()

    return (value_net, targets_experts)

