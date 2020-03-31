import torch
from collections import namedtuple
from tqdm import tqdm
from MCN.utils import (
    ReplayMemory,
    generate_random_instance,
    sample_memory_validation,
    compute_loss,
)
from .environment import Environment
from .value_nn import ValueNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TargetExperts(object):

    """Object containing everything related to the target nets:

    - Build the Validation Dataset
    - Store the target nets
    - Test the current value net and update the targets nets"""

    def __init__( self, input_dim, hidden_dim1, hidden_dim2, n_heads, K, alpha, n_free_min,
                  n_free_max, Omega_max, Phi_max, Lambda_max, memory_size, tolerance):

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
        self.Transition = namedtuple(
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
                "next_afterstates",
                "next_Omegas",
                "next_Phis",
                "next_Lambdas",
                "next_J",
                "next_saved_nodes",
                "next_infected_nodes",
                "next_size_connected",
                "next_reward",
                "id_graphs",
            ),
        )

        self.create_dataset()
        # Initialize the parameters of the list of experts
        self.list_target_nets = [None] * self.n_max
        self.Budget_target = 1
        self.losses_validation_sets = [None] * self.n_max
        self.losses_value_net = [None] * self.n_max

    def create_dataset(self):

        """Create the Validation Dataset by generating memory_size instances for each
        subtask appearing during the training of the value net"""

        print(
            "=========================================================================="
        )
        print("Creation of the Validation Dataset ...", "\n")

        for id_slot in tqdm(range(self.n_max)):

            # we are considering the phase where the Budget = id_slot +1
            Budget = id_slot + 1
            # create a named tuple in each slot of the list
            self.Dataset.append(namedtuple("slot", ("Data", "loss", "compute_target")))
            # initiatlize the loss
            self.Dataset[id_slot].loss = self.tolerance
            # initialize the boolean variable
            # if id_loss == 0, we are at the "end state" and the targets
            # are the exact rewards, thus we do not need to compute them
            self.Dataset[id_slot].compute_target = id_slot > 0
            # initiatlize a replay memory
            Data = ReplayMemory(self.memory_size, self.Transition)

            for k in range(self.memory_size):

                # generate a random instance
                G, I, Omega, Phi, Lambda = generate_random_instance(
                    self.n_free_min,
                    self.n_free_max,
                    self.Omega_max,
                    self.Phi_max,
                    self.Lambda_max,
                    Budget,
                )
                # Initialize the environment
                env = Environment(G, Omega, Phi, Lambda, J=I)
                # Compute the afterstates
                env.compute_current_situation()
                # save the data in memory
                Data.push(
                    env.list_G_torch,
                    env.Omega_tensor,
                    env.Phi_tensor,
                    env.Lambda_tensor,
                    env.J_tensor,
                    env.saved_tensor,
                    env.infected_tensor,
                    env.size_connected_tensor,
                    env.id_loss,
                    env.next_list_G_torch,
                    env.next_Omega_tensor,
                    env.next_Phi_tensor,
                    env.next_Lambda_tensor,
                    env.next_J_tensor,
                    env.next_saved_tensor,
                    env.next_infected_tensor,
                    env.next_size_connected_tensor,
                    env.next_rewards,
                    None,
                )

            # if we are in the tricky case of considering the first move of the vaccinator
            # or the first from the attacker, and the previsous stage consisted in several steps
            # we need to keep all the random instances separated as their afterstates may call several "experts"
            test_first_Phi = Budget == self.Lambda_max + 1 and self.Lambda_max >= 1
            test_first_Omega = (
                Budget == self.Lambda_max + self.Phi_max + 1 and self.Phi_max > 1
            )
            if test_first_Phi or test_first_Omega:
                self.Dataset[id_slot].Data = Data
            # else, we are in a simple case, and only one expert is needed to
            # compute the values of the afterstates, thus we concatenate everything
            # and save the variables in a dataset of size 1
            else:
                Data_1 = ReplayMemory(1, self.Transition)
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
                    next_afterstates,
                    next_Omegas,
                    next_Phis,
                    next_Lambdas,
                    next_J,
                    next_saved_nodes,
                    next_infected_nodes,
                    next_size_connected,
                    rewards,
                    id_graphs,
                ) = sample_memory_validation(Data, self.Transition, self.memory_size)
                Data_1.push(
                    afterstates,
                    Omegas,
                    Phis,
                    Lambdas,
                    J,
                    saved_nodes,
                    infected_nodes,
                    size_connected,
                    id_loss,
                    next_afterstates,
                    next_Omegas,
                    next_Phis,
                    next_Lambdas,
                    next_J,
                    next_saved_nodes,
                    next_infected_nodes,
                    next_size_connected,
                    rewards,
                    id_graphs,
                )
                self.Dataset[id_slot].Data = Data_1

    def test_update_target_nets(self, value_net):

        """Test the current value net against the saved experts,
        if it performs well enough, update the list of target nets"""

        print("==========================================================================")
        print("Test the Value net on Validation Dataset and Update the target nets ... \n")

        # for all the previously solved cases where an expert is available
        for id_slot in range(self.Budget_target):
            # STEP 1 : We compute the target or gather it if it is already available
            # STEP 2 : Then, we compute the losses
            # STEP 3 : Depending on them, we update the list of experts

            # STEP 1: COMPUTE / GATHER TARGETS
            # if we need to compute the target for the loss
            if self.Dataset[id_slot].compute_target:
                # if it's the simple case where the target is simply
                # computed thanks to one expert
                Budget = id_slot + 1
                test_first_Phi = Budget == self.Lambda_max + 1 and self.Lambda_max >= 1
                test_first_Omega = (
                    Budget == self.Lambda_max + self.Phi_max + 1 and self.Phi_max > 1
                )
                if not test_first_Phi and not test_first_Omega:
                    # call the expert
                    target_net = self.list_target_nets[id_slot - 1]
                    # retrieve the validation data
                    env = self.Dataset[id_slot].Data.sample(1)[0]
                    # compute the target
                    with torch.no_grad():
                        target = target_net(
                            env.next_afterstates,
                            env.next_Omegas,
                            env.next_Phis,
                            env.next_Lambdas,
                            env.next_J,
                            env.next_saved_nodes,
                            env.next_infected_nodes,
                            env.next_size_connected,
                        )
                    # gather all the other variables necessary to compute the losses
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
                        id_graphs,
                    ) = (
                        env.afterstates,
                        env.Omegas,
                        env.Phis,
                        env.Lambdas,
                        env.J,
                        env.saved_nodes,
                        env.infected_nodes,
                        env.size_connected,
                        env.id_loss,
                        env.id_graphs,
                    )
                    # we update the target as it was asked to compute it
                    # we do that by pushing a whole new dataset as the size of the memory is 1
                    self.Dataset[id_slot].Data.push(
                        env.afterstates,
                        env.Omegas,
                        env.Phis,
                        env.Lambdas,
                        env.J,
                        env.saved_nodes,
                        env.infected_nodes,
                        env.size_connected,
                        env.id_loss,
                        env.next_afterstates,
                        env.next_Omegas,
                        env.next_Phis,
                        env.next_Lambdas,
                        env.next_J,
                        env.next_saved_nodes,
                        env.next_infected_nodes,
                        env.next_size_connected,
                        target,
                        env.id_graphs,
                    )
                    # update the boolean value
                    self.Dataset[id_slot].compute_target = False
                # else, it means we are in the more complicated case of the first attack or first vaccination
                else:
                    memory = self.Dataset[id_slot].Data
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
                        target,
                        id_graphs,
                    ) = sample_memory_validation(
                        memory,
                        self.Transition,
                        self.memory_size,
                        multiple_targets=True,
                        list_target_nets=self.list_target_nets,
                        Omega_max=self.Omega_max,
                        Phi_max=self.Phi_max,
                        Lambda_max=self.Lambda_max,
                    )
            # else, the targets are already available
            else:
                env = self.Dataset[id_slot].Data.sample(1)[0]
                # gather all the variables necessary to compute the losses, including the target
                (
                    afterstates,
                    Omegas,
                    Phis,
                    Lambdas,
                    J,
                    saved_nodes,
                    infected_nodes,
                    size_connected,
                    target,
                    id_loss,
                    id_graphs,
                ) = (
                    env.afterstates,
                    env.Omegas,
                    env.Phis,
                    env.Lambdas,
                    env.J,
                    env.saved_nodes,
                    env.infected_nodes,
                    env.size_connected,
                    env.next_reward,
                    env.id_loss,
                    env.id_graphs,
                )

            # STEP 2: COMPUTE THE LOSSES
            with torch.no_grad():
                loss_value = float(
                    compute_loss(
                        value_net,
                        id_loss,
                        target,
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
                )
                # if there exists an expert to compete with the value net
                if self.list_target_nets[id_slot] is not None:
                    loss_target = float(
                        compute_loss(
                            self.list_target_nets[id_slot],
                            id_loss,
                            target,
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
                    )
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
                ).to(device)
                new_target_net.load_state_dict(value_net.state_dict())
                new_target_net.eval()
                self.list_target_nets[id_slot] = new_target_net
                # if there is a slot after this one
                if id_slot < self.n_max - 1:
                    # we set the boolean of the next slot to True
                    # as we changed the target net, so the next targets need to be updated
                    self.Dataset[id_slot + 1].compute_target = True
                # if there was no expert and the value net performed well, we can pass to the next task
                if update_Budget_target:
                    # if it's not already the maximum of Budget
                    if self.Budget_target < self.n_max:
                        self.Budget_target += 1
