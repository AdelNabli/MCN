import torch
from MCN.utils import graph_torch, new_graph, get_player, compute_saved_nodes, features_connected_comp, Instance

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Environment(object):

    """Defines the environment's sequential behavior"""

    def __init__(self, list_instances):

        """Initialize all the variables of the environment given the starting state.

        Parameters:
        ----------
        list_instances: list of Instance object"""

        self.batch_size = len(list_instances)
        self.Omega = list_instances[0].Omega
        self.Phi = list_instances[0].Phi
        self.Lambda = list_instances[0].Lambda
        self.Budget = self.Omega + self.Phi + self.Lambda
        self.n_free = [len(instance.G) + 1 - len(instance.J) for instance in list_instances]
        self.list_G_nx = [instance.G for instance in list_instances]
        self.list_J = [instance.J for instance in list_instances]
        self.actions = list(range(self.batch_size))
        self.player = get_player(self.Omega, self.Phi, self.Lambda)


    def compute_current_situation(self):

        self.next_G = [self.list_G_nx[action] for action in self.actions]
        self.next_J = [self.list_J[action] for action in self.actions]
        self.next_n_free = [n_free - 1 for n_free in self.n_free]
        self.next_Budget = self.Budget - 1
        self.update_budgets()
        self.next_player = get_player(self.next_Omega, self.next_Phi, self.next_Lambda)
        self.compute_all_possible_afterstates()
        self.next_n_nodes = [len(G) for G in self.next_list_G_nx]
        self.compute_next_state_tensors()

    def step(self, actions):

        self.actions = actions
        self.Omega = self.next_Omega
        self.Phi = self.next_Phi
        self.Lambda = self.next_Lambda
        self.Budget = self.next_Budget
        self.n_free = self.next_n_free
        self.list_G_nx = self.next_list_G_nx
        self.list_J = self.next_list_J
        self.player = self.next_player

    def compute_next_state_tensors(self):

        self.next_n_nodes_tensor = (
            torch.tensor(
                self.next_n_nodes,
                dtype=torch.float,
            )
                .view([len(self.next_n_nodes), 1])
                .to(device)
        )
        self.next_Omega_tensor = self.next_Omega * torch.ones(self.next_n_nodes_tensor.size(), dtype=torch.float).to(device)
        self.next_Phi_tensor = self.next_Phi * torch.ones(self.next_n_nodes_tensor.size(), dtype=torch.float).to(device)
        self.next_Lambda_tensor = self.next_Lambda * torch.ones(self.next_n_nodes_tensor.size(), dtype=torch.float).to(device)
        self.next_Omega_norm = self.next_Omega_tensor / self.next_n_nodes_tensor
        self.next_Phi_norm = self.next_Phi_tensor / self.next_n_nodes_tensor
        self.next_Lambda_norm = self.next_Lambda_tensor / self.next_n_nodes_tensor

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

        """Compute all the possible afterstates given the current player and the current states"""

        self.free_nodes = [[x for x in list(self.next_G[k].nodes) if x not in self.next_J[k]] for k in range(self.batch_size)]
        self.id_graphs = torch.tensor(
            [k for k in range(self.batch_size) for i in range(len(self.free_nodes[k]))], dtype=torch.int64).to(device)
        # init variables
        self.next_list_G_nx = []
        self.next_list_G_torch = []
        self.next_list_J = []
        self.list_next_J_tensor = []
        self.list_next_saved_tensor = []
        self.list_next_infected_tensor = []
        self.list_next_size_connected_tensor = []
        next_rewards = []

        for k in range(self.batch_size):

            for node in self.free_nodes[k]:

                G = self.next_G[k].copy()
                J = list(self.next_J[k].copy())

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

                self.list_next_J_tensor.append(next_J_tensor)
                self.list_next_saved_tensor.append(next_saved_tensor)
                self.list_next_infected_tensor.append(next_infected_tensor)
                self.list_next_size_connected_tensor.append(next_size_connected_tensor)
                next_rewards.append(next_reward)

        self.next_rewards = (
            torch.tensor(next_rewards, dtype=torch.float)
            .view([len(next_rewards), 1])
            .to(device)
        )
        self.next_J_tensor = torch.cat(self.list_next_J_tensor)
        self.next_saved_tensor = torch.cat(self.list_next_saved_tensor)
        self.next_infected_tensor = torch.cat(self.list_next_infected_tensor)
        self.next_size_connected_tensor = torch.cat(self.list_next_size_connected_tensor)
