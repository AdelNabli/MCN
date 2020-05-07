import os
from MCN.MCN_curriculum.train import train_value_net


def run_experiments():

    # parameters of the neural networks used
    dim_embedding = 20
    dim_values = 10
    dim_hidden = 40
    n_heads = 3
    n_att_layers = 2
    n_pool = 2
    p = 0.2
    alpha = 0.2

    # parameters of the optimizer used
    lr = 1.3e-4
    betas = (0.8, 0.98)

    # EXPERIMENT 2: Train the experts on bigger graphs.
    #               We train for longer to cope with the increase in graph size.
    #               There are 3 experiments to run, one for each possible type of graph (undirected, directed, weighted)

    # parameters of the distribution of instances
    n_free_min = 3
    n_free_max = 5
    d_edge_min = 0.5
    d_edge_max = 0.6
    Omega_max = 2
    Phi_max = 2
    Lambda_max = 2
    w_max = 5

    # training parameters for the curriculum
    batch_size = 12
    size_train_data = 120
    size_val_data = 20
    size_test_data = 10
    n_epoch = 5
    exact_protection = False
    # basic case
    weighted = False
    directed = False
    folder_name = 'test_data'
    path_test_data = os.path.join('data', folder_name)
    # Train with curriculum
    train_value_net(batch_size, size_train_data, size_val_data, size_test_data, lr, betas, n_epoch,
                    dim_embedding, dim_values, dim_hidden, n_heads, n_att_layers, n_pool, alpha, p,
                    n_free_min, n_free_max, d_edge_min, d_edge_max, Omega_max, Phi_max, Lambda_max,
                    weighted, w_max, directed, path_test_data=path_test_data, exact_protection=exact_protection)


if __name__ == "__main__":
    run_experiments()
