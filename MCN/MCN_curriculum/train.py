import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from tqdm import tqdm
from datetime import datetime
from MCN.utils import save_models,load_training_param
from MCN.MCN_curriculum.value_nn import ValueNet
from MCN.MCN_curriculum.experts import TargetExperts
from MCN.MCN_curriculum.data import load_create_datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_value_net(batch_size, size_train_data, size_val_data, lr, betas, n_epoch, h1, h2, n_heads, alpha, p,
                    n_free_min, n_free_max, Omega_max, Phi_max, Lambda_max,
                    num_workers=0, path_experts=None, path_data=None, resume_training=False, path_train=""):

    r"""Training procedure. Follows the evolution of the training using tensorboard.
    Stores the neural networks each time a new task is learnt.

    Parameters:
    ----------
    batch_size: int,
                size of the batch used to compute the loss
    size_train_data: int,
                     size of the training set used to train each target net
    size_val_data: int,
                   size of the validation dataset used for each target net
    lr: float,
        learning rate of the optimizer
    betas: tuple of floats,
           betas used for the optimizer
    n_epoch: int,
             number of epoch used to train each target net
    h1: int,
        first hidden dim of the ValueNet
    h2: int,
        second hidden dim of the ValueNet
    n_heads: int,
             number of heads used in the GATs of the ValueNet
    alpha: float,
           alpha value of the APPNP of the ValueNet
    p: float \in [0,1]
       dropout probability
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
    num_workers: int (default 0),
                 num of workers used for the dataloader object during training
    path_experts: str (default None),
                  path the directory containing saved experts models
                  if given, begin the training at the first step after
                  the last trained expert
    path_data: str (default None),
               path the directory containing the saved datasets
    resume_training: bool (default False),
                     if True, then the parameters of a previous training session
                     is loaded thanks to path_train
    path_train: str (default ""),
                path of the file containing the parameters of a previous
                training session"""

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
    value_net = ValueNet(
        input_dim=5,
        hidden_dim1=h1,
        hidden_dim2=h2,
        n_heads=n_heads,
        K=n_max,
        alpha=alpha,
        p=p
    ).to(device)
    # Initialize the pool of experts (target nets)
    targets_experts = TargetExperts(
        input_dim=5,
        hidden_dim1=h1,
        hidden_dim2=h2,
        n_heads=n_heads,
        K=n_max,
        alpha=alpha,
        Omega_max=Omega_max,
        Phi_max=Phi_max,
        Lambda_max=Lambda_max,
        path_experts=path_experts,
        path_data=path_data,
    )
    # Initialize the optimizer
    optimizer = optim.Adam(value_net.parameters(), lr=lr, betas=betas)
    # If resume training
    if resume_training:
        # load the state dicts of the optimizer and value_net
        value_net, optimizer = load_training_param(value_net, optimizer, path_train)

    # While all the target nets are not trained
    Budget_max = Omega_max + Phi_max + Lambda_max
    while targets_experts.Budget_target < Budget_max:

        print("\n==========================================================================")
        print("Begin the training for Budget = %2d \n" % targets_experts.Budget_target)

        # Load or Create the training and validation datasets
        training_generator, val_data = load_create_datasets(
            size_train_data,
            size_val_data,
            batch_size,
            num_workers,
            n_free_min,
            n_free_max,
            Omega_max,
            Phi_max,
            Lambda_max,
            targets_experts.Budget_target,
            targets_experts.list_target_nets,
            path_data,
            solve_exact=False,
        )
        # Loop over epochs
        for epoch in tqdm(range(n_epoch)):
            # Training for all batchs in the training set
            for i_batch, batch_instances in enumerate(training_generator):
                # Compute the approximate values
                values_approx = value_net(
                    batch_instances.G_torch,
                    batch_instances.Omegas,
                    batch_instances.Phis,
                    batch_instances.Lambdas,
                    batch_instances.J,
                    batch_instances.saved_nodes,
                    batch_instances.infected_nodes,
                    batch_instances.size_connected,
                )
                # Init the optimizer
                optimizer.zero_grad()
                # Compute the loss of the batch
                loss = torch.sqrt(torch.mean((values_approx[:, 0] - batch_instances.target[:, 0]) ** 2))
                # Compute the loss on the Validation set
                targets_experts.test_update_target_nets(value_net, val_data)
                # Update the parameters of the Value_net
                loss.backward()
                optimizer.step()
                # Update the tensorboard
                writer.add_scalar("Loss", float(loss), count)
                writer.add_scalar("Loss validation",
                                  targets_experts.losses_validation_sets[targets_experts.Budget_target - 1],
                                  count,
                                  )
                count += 1

            # Print the information of the epoch
            print(
                " \n Budget target : %2d/%2d" % (targets_experts.Budget_target, Budget_max - 1),
                " \n Epoch: %2d/%2d" % (epoch + 1, n_epoch),
                " \n Loss of the current value net: %f" % float(loss),
                " \n Losses of the experts : ", targets_experts.losses_validation_sets,
            )
            # Saves model
            save_models(date_str, dict_args, value_net, optimizer, count, targets_experts)

        # Update the target budget
        targets_experts.Budget_target += 1
