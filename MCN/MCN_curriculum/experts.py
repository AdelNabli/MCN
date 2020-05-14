import torch
import math
import os
import pickle
from torch.utils.data import DataLoader
from MCN.utils import load_saved_experts, compute_loss_test
from MCN.MCN_curriculum.value_nn import ValueNet
from MCN.MCN_curriculum.data import MCNDataset, collate_fn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TargetExperts(object):

    """Object containing the target nets and updating them during learning"""

    def __init__(self, dim_input, dim_embedding, dim_values, dim_hidden, n_heads, n_att_layers, n_pool, K, alpha,
                 weighted, Omega_max, Phi_max, Lambda_max, path_experts, path_data, exact_protection):

        # Initialize the parameters of the neural network
        self.dim_input = dim_input
        self.dim_embedding = dim_embedding
        self.dim_values = dim_values
        self.dim_hidden = dim_hidden
        self.n_att_layers = n_att_layers
        self.n_pool = n_pool
        self.n_heads = n_heads
        self.K = K
        self.alpha = alpha
        self.weighted = weighted
        # Initialize the parameters of the list of experts
        self.n_max = Omega_max + Phi_max + Lambda_max
        self.Lambda_max = Lambda_max
        self.list_target_nets = [None] * (self.n_max - 1)
        self.losses_validation_sets = [math.inf] * (self.n_max - 1)
        self.losses_test_set = [None] * (self.n_max - 1)
        self.loss_value_net = math.inf
        self.Budget_target = 1
        # If use the exact algorithm for protection, update the parameters
        self.exact_protection = exact_protection
        if exact_protection:
            self.losses_validation_sets[:Lambda_max] = [0] * Lambda_max
            self.Budget_target = Lambda_max + 1

        self.resume_training(path_experts, path_data)

    def resume_training(self, path_experts, path_data):

        """Load the targets nets that are already available"""

        # If pre-trained experts are available
        if path_experts is not None:
            # load them
            list_trained_experts = load_saved_experts(path_experts)
            Budget_trained = len(list_trained_experts)
            # update the TargetExperts object
            self.list_target_nets[:Budget_trained] = list_trained_experts

            # If there is a data folder
            if path_data is not None:
                Budget_begin = 1 + self.exact_protection * self.Lambda_max
                # for every budget already trained
                for Budget in range(Budget_begin, Budget_trained + 1):
                    path_val_data_budget = os.path.join(path_data, 'val_data', 'data_' + str(Budget) + '.gz')
                    # we check whether there is a validation set available
                    if os.path.exists(path_val_data_budget):
                        # if it's the case, we load it
                        val_data = pickle.load(open(path_val_data_budget, "rb"))
                        val_data = MCNDataset(val_data)
                        val_loader = DataLoader(
                            val_data,
                            collate_fn=collate_fn,
                            batch_size=128,
                            shuffle=True,
                            num_workers=0,
                        )
                        # then, we test the target net on this validation set
                        self.Budget_target = Budget
                        self.test_update_target_nets(self.list_target_nets[Budget - 1], val_loader)

            self.Budget_target = Budget_trained + 1

    def test_update_target_nets(self, value_net, val_generator, test_generator):

        """Test the current value net against the saved expert on the current validation set
        and keep the best of both as the current target net"""

        # Create a target net from the current value net
        new_target_net = ValueNet(
            dim_input=self.dim_input,
            dim_embedding=self.dim_embedding,
            dim_values=self.dim_values,
            dim_hidden=self.dim_hidden,
            n_heads=self.n_heads,
            n_att_layers=self.n_att_layers,
            n_pool=self.n_pool,
            K=self.K,
            alpha=self.alpha,
            p=0,
            weighted=self.weighted,
        ).to(device)
        new_target_net.load_state_dict(value_net.state_dict())
        new_target_net.eval()
        # init the values approx and the target
        target = []
        val_approx = []
        with torch.no_grad():
            # Compute the approximate values given
            # by the current value net on the validation set
            # for every batch
            for i_batch, batch_instances in enumerate(val_generator):
                values_approx = new_target_net(
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
                val_approx.append(values_approx)
                target.append(batch_instances.target)
            # Compute the loss
            target = torch.cat(target)
            val_approx = torch.cat(val_approx)
            loss_value_net = float(torch.sqrt(torch.mean((val_approx[:, 0] - target[:, 0]) ** 2)))
            self.loss_value_net = loss_value_net
        id_slot = self.Budget_target - 1
        # If the current loss is less than the best loss so far
        if loss_value_net < self.losses_validation_sets[id_slot]:
            # we update both the current target net and loss
            self.list_target_nets[id_slot] = new_target_net
            self.losses_validation_sets[id_slot] = loss_value_net
            self.losses_test_set[id_slot] = compute_loss_test(
                test_generator,
                list_experts=self.list_target_nets,
                id_to_test=id_slot,
            )[0]
