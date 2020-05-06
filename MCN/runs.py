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
    #               the comparison with the exact results can be done to assess the learning.
    #               In total, there are 3 x 3 x 2 experiments run for this part

    # parameters of the distribution of instances
    n_free_min = 9
    n_free_max = 14
    d_edge_min = 0.2
    d_edge_max = 0.5
    Omega_max = 3
    Phi_max = 3
    Lambda_max = 3
    w_max = 5

    # training parameters for the curriculum
    batch_size = 256
    size_train_data = 100000
    size_val_data = 1000
    size_test_data = 1000
    n_epoch = 60

    # training parameters for the baselines
    update_target = 100
    eps_end = 0.01
    eps_start = 0.9
    eps_decay = 10000

    # RUN THE EXPERIMENTS
    # For each of the 3 possible distribution of graphs (undirected, directed, weighted)
    for k in range(3):
        if k == 0:
            # basic case
            weighted = False
            directed = False
            folder_name = 'test_data'
        elif k == 1:
            # directed case
            weighted = False
            directed = True
            folder_name = 'test_data_dir'
        elif k == 2:
            # weighted case
            weighted = True
            directed = False
            folder_name = 'test_data_w'
        path_test_data = os.path.join('data', folder_name)
        # Either use the exact protector or not
        for exact_protection in [True, False]:
            # In order to fairly compare the baselines with the curriculum,
            # we enforce the equality of optimization steps between the two methods
            # as well as the same total number of instances generated throughout the process
            if exact_protection:
                n_episode = 300000
                n_time_instance_seen = 20
            else:
                n_episode = 450000
                n_time_instance_seen = 14
            # Train with curriculum
            train_value_net(batch_size, size_train_data, size_val_data, size_test_data, lr, betas, n_epoch,
                            dim_embedding, dim_values, dim_hidden, n_heads, n_att_layers, n_pool, alpha, p,
                            n_free_min, n_free_max, d_edge_min, d_edge_max, Omega_max, Phi_max, Lambda_max,
                            weighted, w_max, directed, path_test_data=path_test_data, exact_protection=exact_protection)
            # Train a baseline Monte Carlo method
            train_value_net_baseline(batch_size, size_test_data, lr, betas, n_episode, update_target,
                                     n_time_instance_seen,
                                     eps_end, eps_decay, eps_start,
                                     dim_embedding, dim_values, dim_hidden, n_heads, n_att_layers, n_pool, alpha, p,
                                     n_free_min, n_free_max, d_edge_min, d_edge_max, Omega_max, Phi_max, Lambda_max,
                                     weighted, w_max, directed,
                                     path_test_data=path_test_data, training_method='MC',
                                     exact_protection=exact_protection)
            # Train a baseline Q-learning method
            train_value_net_baseline(batch_size, size_test_data, lr, betas, n_episode, update_target,
                                     n_time_instance_seen,
                                     eps_end, eps_decay, eps_start,
                                     dim_embedding, dim_values, dim_hidden, n_heads, n_att_layers, n_pool, alpha, p,
                                     n_free_min, n_free_max, d_edge_min, d_edge_max, Omega_max, Phi_max, Lambda_max,
                                     weighted, w_max, directed,
                                     path_test_data=path_test_data, training_method='DQN',
                                     exact_protection=exact_protection)

    # EXPERIMENT 2: Train the experts on bigger graphs.
    #               We train for longer to cope with the increase in graph size.
    #               There are 3 experiments to run, one for each possible type of graph (undirected, directed, weighted)

    # parameters of the distribution of instances
    n_free_min = 19
    n_free_max = 51
    d_edge_min = 0.1
    d_edge_max = 0.2
    Omega_max = 3
    Phi_max = 3
    Lambda_max = 3
    w_max = 5

    # training parameters for the curriculum
    batch_size = 256
    size_train_data = 120000
    size_val_data = 2000
    size_test_data = 1000
    n_epoch = 120
    exact_protection = False

    # RUN THE EXPERIMENTS
    # For each of the 3 possible distribution of graphs (undirected, directed, weighted)
    for k in range(3):
        if k == 0:
            # basic case
            weighted = False
            directed = False
            folder_name = 'test_data'
        elif k == 1:
            # directed case
            weighted = False
            directed = True
            folder_name = 'test_data_dir'
        elif k == 2:
            # weighted case
            weighted = True
            directed = False
            folder_name = 'test_data_w'
        path_test_data = os.path.join('data', folder_name)
        # Train with curriculum
        train_value_net(batch_size, size_train_data, size_val_data, size_test_data, lr, betas, n_epoch,
                        dim_embedding, dim_values, dim_hidden, n_heads, n_att_layers, n_pool, alpha, p,
                        n_free_min, n_free_max, d_edge_min, d_edge_max, Omega_max, Phi_max, Lambda_max,
                        weighted, w_max, directed, path_test_data=path_test_data, exact_protection=exact_protection)


if __name__ == "__main__":
    run_experiments()
