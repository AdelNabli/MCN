import torch
import math
import os
import pickle
from MCN.utils import load_saved_experts
from MCN.MCN_curriculum.value_nn import ValueNet
from MCN.MCN_curriculum.data import collate_fn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TargetExperts(object):

    """Object containing the target nets and updating them during learning"""

    def __init__(self, input_dim, hidden_dim1, hidden_dim2, n_heads, K, alpha,
                 Omega_max, Phi_max, Lambda_max, path_experts, path_data):

        # Initialize the parameters of the neural network
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.n_heads = n_heads
        self.K = K
        self.alpha = alpha
        # Initialize the parameters of the list of experts
        self.n_max = Omega_max + Phi_max + Lambda_max
        self.list_target_nets = [None] * (self.n_max - 1)
        self.losses_validation_sets = [math.inf] * (self.n_max - 1)
        self.Budget_target = 1

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
                # for every budget already trained
                for Budget in range(1, Budget_trained + 1):
                    path_val_data_budget = os.path.join(path_data, 'val_data', 'data_' + str(Budget) + '.gz')
                    # we check whether there is a validation set available
                    if os.path.exists(path_val_data_budget):
                        # if it's the case, we load it
                        val_data = pickle.load(open(path_val_data_budget, "rb"))
                        val_data = collate_fn(val_data)
                        # then, we test the target net on this validation set
                        self.Budget_target = Budget
                        self.test_update_target_nets(self.list_target_nets[Budget - 1], val_data)

            self.Budget_target = Budget_trained + 1

    def test_update_target_nets(self, value_net, val_data):

        """Test the current value net against the saved expert on the current validation set
        and keep the best of both as the current target net"""

        # Create a target net from the current value net
        new_target_net = ValueNet(
            self.input_dim,
            self.hidden_dim1,
            self.hidden_dim2,
            self.n_heads,
            self.K,
            self.alpha,
            p=0,
        ).to(device)
        new_target_net.load_state_dict(value_net.state_dict())
        new_target_net.eval()
        with torch.no_grad():
            # Compute the approximate values given
            # by the current value net on the validation set
            values_approx = new_target_net(
                        val_data.G_torch,
                        val_data.n_nodes,
                        val_data.Omegas,
                        val_data.Phis,
                        val_data.Lambdas,
                        val_data.Omegas_norm,
                        val_data.Phis_norm,
                        val_data.Lambdas_norm,
                        val_data.J,
                        val_data.saved_nodes,
                        val_data.infected_nodes,
                        val_data.size_connected,
            )
            # Compute the loss
            loss_value_net = float(torch.sqrt(torch.mean((values_approx[:, 0] - val_data.target[:, 0]) ** 2)))
        id_slot = self.Budget_target - 1
        # If the current loss is less than the best loss so far
        if loss_value_net < self.losses_validation_sets[id_slot]:
            # we update both the current target net and loss
            self.list_target_nets[id_slot] = new_target_net
            self.losses_validation_sets[id_slot] = loss_value_net
