import os
from MCN.MCN_curriculum.train import train_value_net
from MCN.MCN_curriculum.train_baselines import train_value_net_baseline


def run_experiments():

    # parameters of the neural networks used
    dim_embedding = 200
    dim_values = 100
    dim_hidden = 400
    n_heads = 3
    n_att_layers = 7
    n_pool = 3
    p = 0.2
    alpha = 0.2

    # parameters of the optimizer used
    lr = 1.3e-4
    betas = (0.8, 0.98)

    # EXPERIMENT 1: CURRICULUM VS BASELINES
    #               compare the curriculum with the baselines methods
    #               (Monte carlo samples and Q-learning) on small instances such that
    #               the comparison with the exact results can be done to assess the learning

    # parameters of the distribution of instances
    n_free_min = 5
    n_free_max = 15
    d_edge_min = 0.2
    d_edge_max = 0.5
    Omega_max = 3
    Phi_max = 3
    Lambda_max = 3
    weighted = False
    w_max = 1
    directed = False

    # training parameters for the curriculum
    batch_size = 256
    size_train_data = 100000
    size_val_data = 1000
    size_test_data = 1000
    n_epoch = 60

    train_value_net(batch_size, size_train_data, size_val_data, size_test_data, lr, betas, n_epoch,
                    dim_embedding, dim_values, dim_hidden, n_heads, n_att_layers, n_pool, alpha, p,
                    n_free_min, n_free_max, d_edge_min, d_edge_max, Omega_max, Phi_max, Lambda_max,
                    weighted, w_max, directed)

    # training parameters for the baselines
    # Monte Carlo
    n_episode = 450000
    update_target = 100
    n_time_instance_seen = 14
    eps_end = 0.01
    eps_start = 0.9
    eps_decay = 10000
    path_test_data = os.path.join('data', 'test_data')
    training_method = 'MC'
    train_value_net_baseline(batch_size, size_test_data, lr, betas, n_episode, update_target, n_time_instance_seen,
                             eps_end, eps_decay, eps_start,
                             dim_embedding, dim_values, dim_hidden, n_heads, n_att_layers, n_pool, alpha, p,
                             n_free_min, n_free_max, d_edge_min, d_edge_max, Omega_max, Phi_max, Lambda_max, weighted,
                             w_max, directed, path_test_data=path_test_data, training_method=training_method)
    # DQN
    training_method = 'DQN'
    train_value_net_baseline(batch_size, size_test_data, lr, betas, n_episode, update_target, n_time_instance_seen,
                             eps_end, eps_decay, eps_start,
                             dim_embedding, dim_values, dim_hidden, n_heads, n_att_layers, n_pool, alpha, p,
                             n_free_min, n_free_max, d_edge_min, d_edge_max, Omega_max, Phi_max, Lambda_max, weighted,
                             w_max, directed, path_test_data=path_test_data, training_method=training_method)
