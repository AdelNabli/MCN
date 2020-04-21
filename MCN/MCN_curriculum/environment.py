import torch
from MCN.utils import graph_torch, new_graph, features_connected_comp, get_player, compute_saved_nodes

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
        self.n_free = len(G_nx) + 1 - len(J)
        self.list_G_nx = [G_nx]
        self.list_J = [J]
        self.action = 0
        self.player = get_player(Omega, Phi, Lambda)


    def compute_current_situation(self):

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

    def compute_next_state_tensors(self):

        self.next_Omega_norm = (
            torch.tensor(
                [self.next_Omega / self.next_n_nodes] * self.next_n_free,
                dtype=torch.float,
            )
            .view([self.next_n_free, 1])
            .to(device)
        )
        self.next_Phi_norm = (
            torch.tensor(
                [self.next_Phi / self.next_n_nodes] * self.next_n_free,
                dtype=torch.float,
            )
            .view([self.next_n_free, 1])
            .to(device)
        )
        self.next_Lambda_norm = (
            torch.tensor(
                [self.next_Lambda / self.next_n_nodes] * self.next_n_free,
                dtype=torch.float,
            )
            .view([self.next_n_free, 1])
            .to(device)
        )
        self.next_Omega_tensor = (
            torch.tensor(
                [self.next_Omega] * self.next_n_free,
                dtype=torch.float,
            )
                .view([self.next_n_free, 1])
                .to(device)
        )
        self.next_Phi_tensor = (
            torch.tensor(
                [self.next_Phi] * self.next_n_free,
                dtype=torch.float,
            )
                .view([self.next_n_free, 1])
                .to(device)
        )
        self.next_Lambda_tensor = (
            torch.tensor(
                [self.next_Lambda] * self.next_n_free,
                dtype=torch.float,
            )
                .view([self.next_n_free, 1])
                .to(device)
        )
        self.next_n_nodes_tensor = (
            torch.tensor(
                [self.next_n_nodes] * self.next_n_free,
                dtype=torch.float,
            )
                .view([self.next_n_free, 1])
                .to(device)
        )


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
            # if the next player is not 3,
            # we need to compute the torch tensors
            if self.next_player != 3:
                # compute the corresponding G_torch graph
                G_torch_new = graph_torch(G_new)
                self.next_list_G_torch.append(G_torch_new)
                # compute the features of the connected components
                (
                    _,
                    next_J_tensor,
                    next_saved_tensor,
                    next_infected_tensor,
                    next_size_connected_tensor,
                ) = features_connected_comp(G_new, J_new)
                # put the reward to 0
                next_reward = 0
            # else, it's the end of the game
            else:
                # we need to compute the true reward
                next_reward = compute_saved_nodes(G_new, J_new)
                # the other tensors aren't necessary
                next_J_tensor = torch.tensor([])
                next_saved_tensor = torch.tensor([])
                next_infected_tensor = torch.tensor([])
                next_size_connected_tensor = torch.tensor([])

            list_next_J_tensor.append(next_J_tensor)
            list_next_saved_tensor.append(next_saved_tensor)
            list_next_infected_tensor.append(next_infected_tensor)
            list_next_size_connected_tensor.append(next_size_connected_tensor)
            next_rewards.append(next_reward)

        self.next_rewards = (
            torch.tensor(next_rewards, dtype=torch.float)
            .view([self.next_n_free, 1])
            .to(device)
        )
        self.next_J_tensor = torch.cat(list_next_J_tensor)
        self.next_saved_tensor = torch.cat(list_next_saved_tensor)
        self.next_infected_tensor = torch.cat(list_next_infected_tensor)
        self.next_size_connected_tensor = torch.cat(list_next_size_connected_tensor)
