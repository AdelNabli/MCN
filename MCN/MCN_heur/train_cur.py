import os
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from tqdm import tqdm
from datetime import datetime
from MCN.utils import save_models, load_training_param, count_param_NN
from MCN.test_performances.optimality_gap import compute_optimality_gap
from MCN.MCN_heur.neural_networks import ValueNet
from MCN.MCN_heur.experts import TargetExperts
from MCN.MCN_heur.data import load_create_datasets, load_create_test_set

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_value_net(batch_size, size_train_data, size_val_data, size_test_data, lr, betas, n_epoch, update_experts,
                    dim_embedding, dim_values, dim_hidden, n_heads, n_att_layers, n_pool, alpha, p,
                    n_free_min, n_free_max, d_edge_min, d_edge_max, Omega_max, Phi_max, Lambda_max,
                    weighted, w_max=1, directed=False,
                    num_workers=0, path_experts=None, path_data=None, resume_training=False, path_train="",
                    path_test_data=None, exact_protection=False, n_epoch_already_trained=0, batch_unroll=None):

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
    size_test_data: int,
                   size of the test dataset used
    lr: float,
        learning rate of the optimizer
    betas: tuple of floats,
           betas used for the optimizer
    n_epoch: int,
             number of epoch used to train each target net
    dim_embedding: int,
                   dimension of the node and graph embedding used in the ValueNet
    dim_values: int,
                dimension of the values embedding in the attention heads of the ValueNet
    dim_hidden: int,
                dimension of the first hidden layer used in the fully connected layers of the ValueNet
    n_heads: int,
             number of heads used in the attention layers of the ValueNet
    n_att_layers: int,
                  number of attention layers in the ValueNet
    n_pool: int,
            number of different graph embeddings used in the ValueNet
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
    d_edge_min: float \in [0,1],
                minimal edge density of the graphs considered
    d_edge_max: float \in [0,1],
                maximal edge density of the graphs considered
    Omega_max: int,
               maximum value of Omega we want the instances to have
    Phi_max: int,
             maximum value of Phi we want the instances to have
    Lambda_max: int,
                maximum value of Lambda we want the instances to have
    weighted: bool (default False),
              whether or not to train on weighted graphs
    w_max: int (default 1),
           if weighted, the maximum weight for a node to have
    directed: bool (default False),
              whether or not to train on directed graphs
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
                training session
    path_test_data: str (default ""),
                    path of the file containing the test data
    exact_protection: bool (default False),
                      whether or not to use the exact algorithm as expert
                      for the protection phase
    n_epoch_already_trained: int (default 0)
                             if resume training, begin the training where it was stopped"""

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
    targets_experts = TargetExperts(
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
    test_set_generators = load_create_test_set(n_free_min, n_free_max, d_edge_min, d_edge_max, Omega_max, Phi_max,
                                               Lambda_max, weighted, w_max, directed, size_test_data, path_test_data,
                                               batch_size, num_workers)

    print("Number of parameters to train = %2d \n" % count_param_NN(value_net))

    # While all the target nets are not trained
    Budget_max = Omega_max + Phi_max + Lambda_max
    while targets_experts.Budget_target < Budget_max:

        print("\n==========================================================================")
        print("Training for Budget = %2d \n" % targets_experts.Budget_target)

        # Load or Create the training and validation datasets
        training_generator, val_generator = load_create_datasets(
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
                )
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
        compute_optimality_gap(Omega_max, Phi_max, Lambda_max, targets_experts.list_target_nets,
                               exact_protection=exact_protection, path_test_data=path_test_data)