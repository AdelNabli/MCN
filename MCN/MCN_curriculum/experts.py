import torch
from tqdm import tqdm
from torch_geometric.data import Batch
from MCN.utils import generate_random_instance, Instance, graph_torch, features_connected_comp
from MCN.MCN_curriculum.value_nn import ValueNet
from MCN.MCN_curriculum.mcn_heuristic import solve_mcn_heuristic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Slot:
    def __init__(self, tolerance):

        self.instances = []
        self.loss = tolerance
        self.compute_target = True
        self.afterstates = None
        self.Omegas = None
        self.Phis = None
        self.Lambdas = None
        self.J = None
        self.saved_nodes = None
        self.infected_nodes = None
        self.size_connected = None
        self.targets = None


class TargetExperts(object):

    """Object containing everything related to the target nets:

    - Build the Validation Dataset
    - Store the target nets
    - Test the current value net and update the targets nets"""

    def __init__(self, input_dim, hidden_dim1, hidden_dim2, n_heads, K, alpha, p, n_free_min, n_free_max,
                 Omega_max, Phi_max, Lambda_max, memory_size, tolerance):

        """Initiatialization of the target experts and creation of the Validation Dataset.

        Parameters:
        ----------
        - Parameters of the ValueNet
        - memory_size: int,
                       size the Dataset specific to each subtask
        - tolerance: float,
                     value of the loss under which the value_net
                     is considered an expert for the subtask it is
                     currently trained on"""

        # Initialize the parameters of the neural network
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.n_heads = n_heads
        self.K = K
        self.alpha = alpha
        self.p = p
        # Initialize the parameters of the instances
        self.n_free_min = n_free_min
        self.n_free_max = n_free_max
        self.Omega_max = Omega_max
        self.Phi_max = Phi_max
        self.Lambda_max = Lambda_max
        # Initialize the parameters of the Validation Dataset
        self.memory_size = memory_size
        self.tolerance = tolerance
        self.n_max = Omega_max + Phi_max + Lambda_max
        self.Dataset = []
        self.create_dataset()
        # Initialize the parameters of the list of experts
        self.list_target_nets = [None] * (self.n_max - 1)
        self.Budget_target = 1
        self.losses_validation_sets = [None] * (self.n_max - 1)
        self.losses_value_net = [None] * (self.n_max - 1)
        self.train_on_every_task = False

    def create_dataset(self):

        """Create the Validation Dataset by generating memory_size instances for each
        subtask appearing during the training of the value net"""

        print("\n==========================================================================")
        print("Creation of the Validation Dataset ...", "\n")

        for id_slot in tqdm(range(self.n_max - 1)):

            # we are considering the phase where the Budget = id_slot +1
            Budget = id_slot + 1
            # create a slot for each id_slot
            slot = Slot(self.tolerance)
            self.Dataset.append(slot)

            for k in range(self.memory_size):

                # generate a random instance
                G, J, Omega, Phi, Lambda = generate_random_instance(
                    self.n_free_min,
                    self.n_free_max,
                    self.Omega_max,
                    self.Phi_max,
                    self.Lambda_max,
                    Budget,
                )
                # save everything in the Instance object
                instance_k = Instance(G, Omega, Phi, Lambda, J, None)
                # save the instance in the memory of the slot
                self.Dataset[id_slot].instances.append(instance_k)

    def test_update_target_nets(self, value_net):

        """Test the current value net against the saved experts,
        if it performs well enough, update the list of target nets"""

        print("\n==========================================================================")
        print("Test the Value net on Validation Dataset and Update the target nets ... \n")

        # for all the previously solved cases where an expert is available
        for id_slot in range(self.Budget_target):
            # STEP 1 : We compute the target or gather it if it is already available
            # STEP 2 : Then, we compute the losses
            # STEP 3 : Depending on them, we update the list of experts

            # STEP 1: COMPUTE / GATHER TARGETS
            # if we need to compute the target for the loss
            if self.Dataset[id_slot].compute_target:
                # Initialize the lists for the slot
                afterstates = []
                Omegas = []
                Phis = []
                Lambdas = []
                J_slot = []
                saved_nodes_slot = []
                infected_nodes_slot = []
                size_connected_slot = []
                targets = []

                for instance in self.Dataset[id_slot].instances:
                    # gather the values and the tensors
                    (value, _, _, _) = solve_mcn_heuristic(
                        self.list_target_nets,
                        instance.G,
                        instance.Omega,
                        instance.Phi,
                        instance.Lambda,
                        self.Omega_max,
                        self.Phi_max,
                        self.Lambda_max,
                        J=instance.J,
                    )
                    G_torch = graph_torch(instance.G)
                    (
                        _,
                        J,
                        saved_nodes,
                        infected_nodes,
                        size_connected,
                    ) = features_connected_comp(instance.G, instance.J)
                    # update the lists of the slot
                    n_nodes = len(instance.G)
                    afterstates.append(G_torch)
                    Omegas.append(instance.Omega / n_nodes)
                    Phis.append(instance.Phi / n_nodes)
                    Lambdas.append(instance.Lambda / n_nodes)
                    J_slot.append(J)
                    saved_nodes_slot.append(saved_nodes)
                    infected_nodes_slot.append(infected_nodes)
                    size_connected_slot.append(size_connected)
                    targets.append(value)

                # concatenate everything into tensors that can be fed to a ValueNet
                # and saved everything into the Slot object
                self.Dataset[id_slot].afterstates = Batch.from_data_list(afterstates).to(device)
                self.Dataset[id_slot].Omegas = (
                    torch.tensor(Omegas, dtype=torch.float)
                    .view([self.memory_size, 1])
                    .to(device)
                )
                self.Dataset[id_slot].Phis = (
                    torch.tensor(Phis, dtype=torch.float)
                    .view([self.memory_size, 1])
                    .to(device)
                )
                self.Dataset[id_slot].Lambdas = (
                    torch.tensor(Lambdas, dtype=torch.float)
                    .view([self.memory_size, 1])
                    .to(device)
                )
                self.Dataset[id_slot].J = torch.cat(J_slot)
                self.Dataset[id_slot].saved_nodes = torch.cat(saved_nodes_slot)
                self.Dataset[id_slot].infected_nodes = torch.cat(infected_nodes_slot)
                self.Dataset[id_slot].size_connected = torch.cat(size_connected_slot)
                self.Dataset[id_slot].targets = (
                    torch.tensor(targets, dtype=torch.float)
                    .view([self.memory_size, 1])
                    .to(device)
                )
                self.Dataset[id_slot].compute_target = False

            # Gather the parameters necessary to compute the loss
            afterstates = self.Dataset[id_slot].afterstates
            Omegas = self.Dataset[id_slot].Omegas
            Phis = self.Dataset[id_slot].Phis
            Lambdas = self.Dataset[id_slot].Lambdas
            J = self.Dataset[id_slot].J
            saved_nodes = self.Dataset[id_slot].saved_nodes
            infected_nodes = self.Dataset[id_slot].infected_nodes
            size_connected = self.Dataset[id_slot].size_connected
            targets = self.Dataset[id_slot].targets

            # STEP 2: COMPUTE THE LOSSES
            with torch.no_grad():
                value_approx = value_net(
                    afterstates,
                    Omegas,
                    Phis,
                    Lambdas,
                    J,
                    saved_nodes,
                    infected_nodes,
                    size_connected,
                )
                loss_value = float(torch.sqrt(torch.mean((value_approx[:, 0] - targets[:, 0]) ** 2)))
                # if there exists an expert to compete with the value net
                if self.list_target_nets[id_slot] is not None:
                    target_net = self.list_target_nets[id_slot]
                    value_target = target_net(
                        afterstates,
                        Omegas,
                        Phis,
                        Lambdas,
                        J,
                        saved_nodes,
                        infected_nodes,
                        size_connected,
                    )
                    loss_target = float(torch.sqrt(torch.mean((value_target[:, 0] - targets[:, 0]) ** 2)))
                    # update the loss in memory
                    # usefull in the case where the target nets where loaded
                    # at the beginning of the training instead of trained
                    self.Dataset[id_slot].loss = loss_target
                    self.losses_validation_sets[id_slot] = loss_target
                    update_Budget_target = False
                # if there is no expert to compete with the value net
                else:
                    # then the losses we want to compare are the one
                    # from the value net and the tolerance
                    loss_target = self.Dataset[id_slot].loss
                    update_Budget_target = True

            self.losses_value_net[id_slot] = loss_value

            # STEP 3: update the list of experts
            # if the value net performed better than the expert
            if loss_value < loss_target:
                # update the loss
                self.Dataset[id_slot].loss = loss_value
                self.losses_validation_sets[id_slot] = loss_value
                # Update the target net
                new_target_net = ValueNet(
                    self.input_dim,
                    self.hidden_dim1,
                    self.hidden_dim2,
                    self.n_heads,
                    self.K,
                    self.alpha,
                    self.p
                ).to(device)
                new_target_net.load_state_dict(value_net.state_dict())
                new_target_net.eval()
                self.list_target_nets[id_slot] = new_target_net
                # if there is a slot after this one
                if id_slot < self.n_max - 2:
                    # we set the boolean of the next slot to True
                    # as we changed the target net, so the next targets need to be updated
                    self.Dataset[id_slot + 1].compute_target = True
                # if this slot is before the first attack
                if id_slot < self.Lambda_max <= self.n_max - 2:
                    # we will need to re-compute the targets of the first attack
                    self.Dataset[self.Lambda_max].compute_target = True
                # if this slot is an "attacker" one
                elif self.Lambda_max <= id_slot < self.Lambda_max + self.Phi_max <= self.n_max - 2:
                    # we will need to re-compute the targets of the first attack
                    self.Dataset[self.Lambda_max + self.Phi_max].compute_target = True
                # if there was no expert and the value net performed well, we can pass to the next task
                if update_Budget_target:
                    # if it's not already the maximum of Budget
                    if self.Budget_target < self.n_max - 1:
                        self.Budget_target += 1
                # if the last expert needed is trained sufficiently well,
                # we can train the value net on all tasks simultaneously
                if id_slot == self.n_max - 2:
                    self.train_on_every_task = True

