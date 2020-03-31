import torch
from MCN.utils import graph_torch, new_graph, features_connected_comp, get_player

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Environment(object):

    """Defines the environment's sequential behavior"""

    def __init__(self, G_nx, Omega, Phi, Lambda, J=[]):

        """Initialize all the variables of the environment given the starting state.

        Parameters:
        ----------
        Omega: int,
               the budget for vaccination
        Phi: int,
             the budget of attack
        Lambda: int,
                the budget for protection
        G_nx: networkx graph
        J: list of ints,
           list of the ids of the infected nodes"""

        self.Omega = Omega
        self.Phi = Phi
        self.Lambda = Lambda
        self.Budget = Omega + Phi + Lambda
        self.n_nodes = len(G_nx)
        self.n_free = self.n_nodes + 1 - len(J)
        self.list_G_nx = [G_nx] * self.n_free
        self.list_J = [J] * self.n_free
        self.action = self.n_free - 1  # last index of list_G_nx
        self.player = get_player(Omega, Phi, Lambda)
        self.id_loss = (
            torch.tensor([5] * self.n_free, dtype=torch.float)
            .view([self.n_free, 1])
            .to(device)
        )
        G_torch = graph_torch(G_nx)
        # init the tensors describing the state
        self.list_G_torch = [G_torch] * self.n_free
        # we normalize all the budgets
        self.Omega_tensor = (
            torch.tensor([self.Omega / self.n_nodes] * self.n_free, dtype=torch.float)
            .view([self.n_free, 1])
            .to(device)
        )
        self.Phi_tensor = (
            torch.tensor([self.Phi / self.n_nodes] * self.n_free, dtype=torch.float)
            .view([self.n_free, 1])
            .to(device)
        )
        self.Lambda_tensor = (
            torch.tensor([self.Lambda / self.n_nodes] * self.n_free, dtype=torch.float)
            .view([self.n_free, 1])
            .to(device)
        )
        # compute the other variables
        (
            _,
            J_tensor,
            saved_tensor,
            infected_tensor,
            size_connected_tensor,
        ) = features_connected_comp(G_nx, J)
        self.J_tensor = torch.cat([J_tensor] * self.n_free)
        self.saved_tensor = torch.cat([saved_tensor] * self.n_free)
        self.infected_tensor = torch.cat([infected_tensor] * self.n_free)
        self.size_connected_tensor = torch.cat([size_connected_tensor] * self.n_free)

    def compute_current_situation(self):

        self.update_id_loss()
        self.next_G = self.list_G_nx[self.action]
        self.next_J = self.list_J[self.action]
        self.next_n_free = self.n_free - 1
        self.next_Budget = self.Budget - 1
        self.update_budgets()
        self.next_player = get_player(self.next_Omega, self.next_Phi, self.next_Lambda)
        self.compute_all_possible_afterstates()
        self.next_n_nodes = len(self.next_list_G_nx[0])
        self.compute_next_state_tensors()

    def step(self, action):

        self.action = action
        self.Omega = self.next_Omega
        self.Phi = self.next_Phi
        self.Lambda = self.next_Lambda
        self.Budget = self.next_Budget
        self.n_free = self.next_n_free
        self.list_G_nx = self.next_list_G_nx
        self.list_J = self.next_list_J
        self.player = self.next_player
        self.id_loss = self.next_id_loss
        # the next state tensors
        self.list_G_torch = self.next_list_G_torch
        self.Omega_tensor = self.next_Omega_tensor
        self.Phi_tensor = self.next_Phi_tensor
        self.Lambda_tensor = self.next_Lambda_tensor
        self.J_tensor = self.next_J_tensor
        self.saved_tensor = self.next_saved_tensor
        self.infected_tensor = self.next_infected_tensor
        self.size_connected_tensor = self.next_size_connected_tensor

    def compute_next_state_tensors(self):

        self.next_Omega_tensor = (
            torch.tensor(
                [self.next_Omega / self.next_n_nodes] * self.next_n_free,
                dtype=torch.float,
            )
            .view([self.next_n_free, 1])
            .to(device)
        )
        self.next_Phi_tensor = (
            torch.tensor(
                [self.next_Phi / self.next_n_nodes] * self.next_n_free,
                dtype=torch.float,
            )
            .view([self.next_n_free, 1])
            .to(device)
        )
        self.next_Lambda_tensor = (
            torch.tensor(
                [self.next_Lambda / self.next_n_nodes] * self.next_n_free,
                dtype=torch.float,
            )
            .view([self.next_n_free, 1])
            .to(device)
        )
        self.next_player_tensor = (
            torch.tensor([self.next_player] * self.next_n_free, dtype=torch.float)
            .view([self.next_n_free, 1])
            .to(device)
        )

    def update_id_loss(self):

        """Update the tensor id_loss by putting a 0 or 1 in the action's index.
            id_loss is a tensor of size n_free containing the id of the loss to apply on the list
            of possible afterstates.

            Nomenclature for the id_loss:
                - 0: \hat{s} = max \tilde{s}
                - 1: \hat{s} = min \tilde{s}
                - 2: \hat{s} >= \tilde{s}
                - 3: \hat{s} <= max \tilde{s}
                - 4: \hat{s} = \tilde{s}
                - 5: no loss associated with the afterstate"""

        # if the player is a defender
        # we want to find the max of the next afterstates
        if self.player == 0 or self.player == 2:
            self.id_loss[self.action] = 0
        # if the player is the attacker
        # we want to find the min of the next afterstates
        elif self.player == 1:
            self.id_loss[self.action] = 1

    def update_budgets(self):

        """Compute the next triplet of budgets given the current one
            and the player whose turn it is to play"""

        # Init the variables
        self.next_Omega = self.Omega
        self.next_Phi = self.Phi
        self.next_Lambda = self.Lambda
        # Update the one to be updated
        if self.player == 0:
            self.next_Omega = self.Omega - 1
        elif self.player == 1:
            self.next_Phi = self.Phi - 1
        elif self.player == 2:
            self.next_Lambda = self.Lambda - 1

    def compute_all_possible_afterstates(self):

        """Compute all the possible afterstates given the current player and the current state"""

        free_nodes = [x for x in list(self.next_G.nodes) if x not in self.next_J]
        # init variables
        self.next_list_G_nx = []
        self.next_list_G_torch = []
        self.next_list_J = []
        list_next_J_tensor = []
        list_next_saved_tensor = []
        list_next_infected_tensor = []
        list_next_size_connected_tensor = []
        next_rewards = []
        next_id_loss = []

        for node in free_nodes:

            G = self.next_G.copy()
            J = list(self.next_J.copy())

            if self.player == 0 or self.player == 2:

                # remove one free node from G
                G_new, mapping = new_graph(G, node)
                # update the name of the nodes in J:
                # as we removed a node, the vertices in G_new changed names
                # compared to G, and thus the vertices infected changed of index
                J_new = [mapping[j] for j in J]

            elif self.player == 1:

                # add one node to the set of attacked nodes
                J_new = J + [node]
                G_new = G

            self.next_list_J.append(J_new)
            self.next_list_G_nx.append(G_new)
            # compute the corresponding G_torch graph
            G_torch_new = graph_torch(G_new)
            self.next_list_G_torch.append(G_torch_new)
            # compute the features of the connected components
            (
                next_reward,
                next_J_tensor,
                next_saved_tensor,
                next_infected_tensor,
                next_size_connected_tensor,
            ) = features_connected_comp(G_new, J_new)
            list_next_J_tensor.append(next_J_tensor)
            list_next_saved_tensor.append(next_saved_tensor)
            list_next_infected_tensor.append(next_infected_tensor)
            list_next_size_connected_tensor.append(next_size_connected_tensor)
            # if it's the last action of the game
            # the end reward is available
            if self.next_player == 3:
                reward = next_reward
                next_id_loss.append(5)

            elif self.next_player == 0 or self.next_player == 2:
                # else, we put the reward to 0
                reward = 0
                next_id_loss.append(2)

            elif self.next_player == 1:
                reward = 0
                next_id_loss.append(3)
            next_rewards.append(reward)

        self.next_rewards = (
            torch.tensor(next_rewards, dtype=torch.float)
            .view([self.next_n_free, 1])
            .to(device)
        )
        self.next_id_loss = (
            torch.tensor(next_id_loss, dtype=torch.float)
            .view([self.next_n_free, 1])
            .to(device)
        )
        self.next_J_tensor = torch.cat(list_next_J_tensor)
        self.next_saved_tensor = torch.cat(list_next_saved_tensor)
        self.next_infected_tensor = torch.cat(list_next_infected_tensor)
        self.next_size_connected_tensor = torch.cat(list_next_size_connected_tensor)
