import os
import numpy as np
import torch
import networkx as nx
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.utils import from_networkx, scatter_
from torch_geometric.transforms import LocalDegreeProfile

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_graph(graph, color_type='fabulous', id_colors=None):
    """Given a networkx graph, plots it.

    Parameters:
    ----------
    graph: networkx graph
    color_type: str,
                either 'fabulous' or 'DAD'
    id_colors: list of ints (size = len(graph)),
               for each nodes, the id of the corresponding color"""

    n_nodes = graph.number_of_nodes()
    # Arbitrary formula to adapt the size of the image displayed to the size of the graph
    size = int(9 * np.log(n_nodes) / np.log(50))
    plt.figure(1, figsize=(size, size))

    if color_type == "fabulous":

        # Draw the graph with lots of colors because it's FA-BU-LOUS
        nx.draw(
            graph,
            with_labels=True,
            font_weight="bold",
            node_color=range(n_nodes),
            edge_color="magenta",
            cmap=plt.cm.hsv,
        )

    elif color_type == "DAD":

        # We map the id_colors of each node to a color,
        # each color representing a 'state' of the node:
        # 0 --> green   --> saved
        # 1 --> blue    --> vaccinated
        # 2 --> red     --> attacked
        # 3 --> magenta --> protected
        # 4 --> orange  --> infected

        colors = ["green", "blue", "red", "magenta", "orange"]
        node_colors = []
        for ids in id_colors:
            node_colors.append(colors[int(ids)])

        nx.draw(
            graph,
            with_labels=True,
            font_weight="bold",
            node_color=node_colors,
            edge_color="magenta",
        )

    plt.show()


def generate_random_graph(n_nodes, density, is_tree=False, seed=None, draw=False):
    r"""Generate a random graph with the desired number of nodes and density.
    Draw the graph if asked. Returns the graph as a networkx object.

    Parameters:
    ----------
    n_nodes: int,
             number of nodes
    density: float (\in [0,1]),
             density of edges
    is_tree: bool,
             whether to generate a tree or not
    seed: int,
          the random seed to use
    draw: bool,
          whether to draw the graph or not

    Returns:
    -------
    graph: networkx Digraph"""

    # Compute the number of edges corresponding to the given density
    n_edges = int(density * n_nodes * (n_nodes - 1) / 2)
    # Create the graph
    if is_tree:
        graph = nx.random_tree(n_nodes, seed=seed)
    else:
        graph = nx.gnm_random_graph(n_nodes, n_edges, seed)
    # Add the edge (v,u) for each edge (u,v) of the graph
    graph_2 = nx.DiGraph(graph)
    graph_2.add_edges_from([(v, u) for (u, v) in graph.edges()])

    if draw:
        plot_graph(graph, color_type="fabulous")

    return graph_2


def generate_random_instance(n_free_min, n_free_max, Omega_max, Phi_max, Lambda_max, Budget_target):
    r"""Generate a random instance of the MCN problem
    corresponding to the stage of the training we are in.
    The parameters of the distribution of instances are
    the number of free nodes in the graph (i.e nodes that won't
    be defended nor attacked) and the maximum budget we allow
    for each player (given that an instance of MCN should have
    at least one unit of budget reserved for the attacker).

    Parameters:
    ----------
    n_free_min: int,
                the minimal number of free nodes
                in the graph we will generate
    n_free_max: int,
                the maximal number of free nodes
    Omega_max: int,
               the maximal vaccination budget
    Phi_max: int,
             the maximal attack budget
    Lambda_max: int,
                the maximal protection budget
    Budget_target: int,
                   the total budget we are considering
                   at this stage of the training procedure
                   (\in [1, Omega_max+Phi_max+Lambda_max])

    Returns:
    -------
    G: networkx graph
    I: list of ints,
       the list of attacked nodes
    Omega: int,
           the budget allocated to the vaccinator
    Phi: int,
         the budget allocated to the attacker
    Lambda: int,
            the budget allocated to the protector"""

    # if the target net is learning the protection values
    if Budget_target <= Lambda_max:
        Omega = 0
        Phi = 0
        Lambda = Budget_target
        # we need to attack some nodes in order
        # to learn the protection
        Phi_attacked = np.random.randint(1, Phi_max + 1)
    # if the target net is learning the attack values
    elif Budget_target <= Phi_max + Lambda_max:
        Omega = 0
        Phi = Budget_target - Lambda_max
        Lambda = np.random.randint(0, Lambda_max + 1)
        remaining_attack_budget = Phi_max - Phi
        Phi_attacked = np.random.randint(0, remaining_attack_budget + 1)
    # else, the target net is learning the vaccination values
    elif Budget_target <= Omega_max + Phi_max + Lambda_max:
        Omega = Budget_target - (Phi_max + Lambda_max)
        # we oblige that at least one node is attacked
        Phi = np.random.randint(1, Phi_max + 1)
        Lambda = np.random.randint(0, Lambda_max + 1)
        Phi_attacked = 0

    # random number of nodes
    n_free = random.randrange(n_free_min, n_free_max)
    n = n_free + Omega + Phi + Lambda + Phi_attacked
    # random number of components
    min_size_comp = np.random.randint(1, n + 1)
    max_n_comp = n // min_size_comp + 1 * (n % min_size_comp > 0)
    n_comp = np.random.randint(1, max_n_comp + 1)
    partition = list(
        np.sort(np.random.choice(range(1, n + 1), n_comp - 1, replace=False))
    )
    # Generate the graphs
    G = nx.DiGraph()
    partition = [0] + partition + [n]
    for k in range(n_comp):
        n_k = partition[k + 1] - partition[k]
        d_k = 0.05 + np.exp(-n_k / 9)
        G_k = generate_random_graph(n_k, d_k, draw=False)
        G = nx.union(G, G_k, rename=("G-", "H-"))
    # Generate the attack
    I = list(np.random.choice(range(n), Phi_attacked, replace=False))
    G = nx.convert_node_labels_to_integers(G)

    return (G, I, Omega, Phi, Lambda)


def graph_torch(G_networkx):
    """Create the Pytorch Geometric graph with 5 simple node features:
    the node degree, the min and max neighbor degree,
    the mean and std of the neighbors degree.

    Parameters:
    ----------
    G_networkx: networkx graph

    Returns:
    -------
    G_torch: Pytorch Geometric Data object"""

    # transform the networkx object to pytorch geometric data
    G_torch = from_networkx(G_networkx.to_undirected()).to(device)
    G_torch.edge_index = G_torch.edge_index.type(torch.LongTensor).to(device)
    # Add features
    with torch.no_grad():
        add_features = LocalDegreeProfile()
        G_torch = add_features(G_torch)
    # if there is a nan, put it to 0
    G_torch.x[torch.isnan(G_torch.x)] = 0
    # normalize the 4 first features by the max possible degree in the graph
    n = len(G_networkx)
    G_torch.x[:, :4] /= n - 1
    # normalize the std of degree by the max std possible
    G_torch.x[:, -1] /= (n - 1) / 2

    return G_torch


def new_graph(G, action):
    """Updates a graph after removing the node 'action'.

    Parameters:
    ----------
    G: networkx graph
    action: int,
            the id of a node in G

    Returns:
    -------
    G_new: networkx graph
    mapping: dict,
             the mapping from the old nodes id
             to their new ids in G_new"""

    G.remove_nodes_from([action])
    labels_old = sorted(G)
    mapping = dict()
    for k in range(len(G)):
        mapping[labels_old[k]] = k
    G_new = nx.relabel_nodes(G, mapping)

    return (G_new, mapping)


def compute_saved_nodes(G, I):
    """Compute the number of saved node given a graph and
    a list of attacked nodes.

    Parameters:
    ----------
    G: networkx graph
    I: list of ints,
       list of the ids of the attacked nodes of G

    Returns:
    -------
    n_saved: int,
             the number of saved nodes"""

    # insure G is undirected
    G1 = G.to_undirected()
    n_saved = 0
    for c in nx.connected_components(G1):
        # a connected component is saved if
        # it doesn't countain any attacked node
        if set(I).intersection(c) == set():
            n_saved += len(c)

    return n_saved


def features_connected_comp(G, I):
    """Compute the features of a given instance (G,I) related to
    the connected components of G

    Parameters:
    ----------
    G: networkx graph
    I: list of ints,
       list of the ids of the attacked nodes of G

    Returns:
    -------
    n_saved: int,
             the number of saved nodes
    J_tensor: float tensor,
              indicator 1_{node \in I}
    indic_saved: float tensor,
                 indicator 1_{node saved}
    indic_infected: float tensor,
                    indicator 1_{node infected}
    size_connected_tensor: float tensor,
                           the size of the connected component
                           each node belongs to
                           (divided by the size of G)"""

    n = len(G)
    # insure G is undirected
    G1 = G.to_undirected()
    # Initialize the variables
    n_saved = 0
    connected_infected = set()
    connected_saved = set()
    size_connected = [1] * n

    for c in nx.connected_components(G1):
        size_c = len(c)
        # update the vector of size of the connected comp
        for node in c:
            # we normalize the size by the total size of the graph
            size_connected[node] = size_c / n
        # a connected component is saved if
        # there is not any attacked node inside it
        if set(I).intersection(c) == set():
            n_saved += size_c
            connected_saved = connected_saved.union(c)
        else:
            connected_infected = connected_infected.union(c)

    # transform the variables to tensor
    # init the tensors
    J_tensor = np.zeros(n)
    indic_saved = np.zeros(n)
    indic_infected = np.zeros(n)
    # compute the one hot encoding
    J_tensor[I] = 1
    indic_saved[list(connected_saved)] = 1
    indic_infected[list(connected_infected)] = 1
    J_tensor = torch.tensor(J_tensor, dtype=torch.float).view([n, 1]).to(device)
    indic_saved = torch.tensor(indic_saved, dtype=torch.float).view([n, 1]).to(device)
    indic_infected = (
        torch.tensor(indic_infected, dtype=torch.float).view([n, 1]).to(device)
    )
    size_connected_tensor = (
        torch.tensor(size_connected, dtype=torch.float).view([n, 1]).to(device)
    )

    return (n_saved, J_tensor, indic_saved, indic_infected, size_connected_tensor)


def get_player(Omega, Phi, Lambda):
    """Returns the player whose turn it is to play
    given the budgets.

    Parameters:
    ----------
    Omega: int,
           budget of the vaccinator
    Phi: int,
         budget of the attacker
    Lambda: int,
            budget of the protector

    Returns:
    -------
    id_player: int,
               the id of the player whose turn it is to play
               0: if the player is the vaccinator
               1: if the player is the attacker
               2: if the player is the protector
               3: if it's the end of the game and there is no more budget to spend"""

    # if there is some vaccination budget, it's the vaccinator's turn
    if Omega >= 1:
        return 0
    # otherwise, if there is still some budget for attack, it's the attacker's turn
    elif Phi >= 1:
        return 1
    # otherwise, it's the protector's turn
    elif Lambda >= 1:
        return 2
    # if no budget is left, it's the end of the game
    else:
        return 3


def take_action_deterministic(target_net, player, next_player, rewards, next_afterstates, **kwargs):
    """Given the current situation and the player whose turn it is to play,
    decides of the best action to take with respect to the exact or approximate
    values of all of the possible afterstates.

    Parameters:
    ----------
    target_net: Pytorch neural network (module),
                the value network for the possible afterstates
    player: int,
            id of the current player:
            0 --> vaccinator
            1 --> attacker
            2 --> protector
            3 --> end of the game
    next_player: int,
                 id of the next player
    rewards: float tensor,
             the reward of each afterstate
    next_afterstates: list of Pytorch Geometric Data objects,
                      list of the possible afterstates

    Returns:
    -------
    action: int,
            the id of the optimal afterstate amoung all the ones possible
    targets: float tensor,
             the values of each of the possible afterstates
    value: float,
           the value of the afterstate chosen with action"""

    # if the game is finished in the next turn
    # we know what is the best action to take
    # because we have the true rewards available
    if next_player == 3:
        # the targets are the true values
        targets = rewards
    # if it's not the end state,
    # we sample from the values
    else:
        with torch.no_grad():
            # Create a Batch of graphs
            G_torch = Batch.from_data_list(next_afterstates).to(device)
            # We compute the target values
            targets = target_net(G_torch, **kwargs)
    # if it's the turn of the attacker
    if player == 1:
        # we take the argmin
        action = int(targets.argmin())
        value = float(targets.min())
    else:
        # we take the argmax
        action = int(targets.argmax())
        value = float(targets.max())

    return action, targets, value


class ReplayMemory(object):
    """ The Replay Memory for the training procedure.
    Keep in memory 'capacity' transitions to sample from.

    Parameters:
    ----------
    capacity: int,
              the capacity of the memory
    Transition: namedtuple,
                the parameters to remember from the transition

    code from: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html """

    def __init__(self, capacity, Transition):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.Transition = Transition

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def compute_loss(value_net, id_loss, targets, id_graphs, **kwargs):
    r"""Given the value net, a batch of afterstates, the id of the loss to apply to each afterstate and the targets,
    compute the total loss for the batch.

    Parameters:
    ----------
    value_net: neural network (pytorch module)
    id_loss: float tensor (size = nb of current afterstates x 1),
             the id of the loss to apply to each afterstate
             0 -> \hat{s} = max target
             1 -> \hat{s} = min target
             2 -> \hat{s} >= target
             3 -> \hat{s} <= target
             4 -> \hat{s} = target
             5 -> no loss associated with the afterstate
    targets: float tensor (size = nb of possible next afterstates x 1),
             values of each of the next possible afterstates
    id_graphs: tensor (size = nb of possible next afterstates x 1),
               id of the graph each next afterstate is computed from

    Returns:
    -------
    loss: float tensor (size = 1),
          computed as follows:
          sqrt( mse_{parts where the is an equality} +
                mse_{parts where there is an inequality} )"""

    # compute the values and the targets
    S = value_net(**kwargs)[:, 0]

    # create the masks
    mask_01 = id_loss.le(1)[:, 0]
    mask_0 = id_loss[mask_01].eq(0)[:, 0]
    mask_1 = id_loss[mask_01].eq(1)[:, 0]
    mask_23 = id_loss.ge(2)[:, 0]
    mask_2 = id_loss[mask_23].eq(2)[:, 0]
    mask_3 = id_loss[mask_23].eq(3)[:, 0]
    mask_4 = id_loss[mask_23].eq(4)[:, 0]

    # create the targets
    target = targets[:, 0]
    # get the targets "min" and "max"
    targets_max = scatter_("max", target, id_graphs)
    targets_min = scatter_("min", target, id_graphs)
    # Compute the losses
    # S = max
    loss_max = torch.sum((S[mask_01][mask_0] - targets_max[mask_0]) ** 2)
    # S = min
    loss_min = torch.sum((S[mask_01][mask_1] - targets_min[mask_1]) ** 2)
    # S >= target
    loss_sup = torch.sum((F.relu(target[mask_2] - S[mask_23][mask_2])) ** 2)
    # S <= target
    loss_inf = torch.sum((F.relu(S[mask_23][mask_3] - target[mask_3])) ** 2)
    # S = target
    loss_equal = torch.sum((S[mask_23][mask_4] - target[mask_4]) ** 2)
    # get the normalization constants
    n_max_min_equal = len(S[mask_01]) + len(S[mask_23][mask_4])
    n_sup_inf = len(S[mask_23][mask_2]) + len(S[mask_23][mask_3])
    # normalize the loss
    loss = 0
    if n_max_min_equal > 0:
        loss += (loss_max + loss_min + loss_equal) / n_max_min_equal
        # print('loss max min', loss)
    if n_sup_inf > 0:
        loss += (loss_sup + loss_inf) / n_sup_inf
    # take the square root
    loss = torch.sqrt(loss)

    return loss


def get_target_net(list_target_nets, Omega, Phi, Lambda, Omega_max, Phi_max, Lambda_max):
    """Returns the expert specialized in the subtask determined by the value of the budgets

    Parameters:
    ----------
    list_target_nets: list of neural nets (pytorch modules),
                      size = Omega_max + Phi_max + Lambda_max
    everything else: int

    Returns:
    -------
    expert: neural network (pytorch module) or None"""

    if Omega > 0:
        target_id = int(Omega + Phi_max + Lambda_max - 1)
    elif Phi > 0:
        target_id = int(Phi + Lambda_max - 1)
    elif Lambda > 0:
        target_id = int(Lambda - 1)
    elif Lambda == 0 and Omega == 0 and Phi == 0:
        return None

    return list_target_nets[target_id]


def sample_memory(memory, Transition, batch_size):
    """Sample the Replay Memory and returns all the necessary to compute the loss.

    Parameters:
    ----------
    memory: ReplayMemory object,
    Transition: named tuple used to build the ReplayMemory,
    batch size: int,
                size of the sample

    Returns:
    -------
    Parameters of the loss function"""

    # sample the memory
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))
    # create the torch geometric graphs
    afterstates = Batch.from_data_list(
        [graph for k in range(batch_size) for graph in batch.afterstates[k]]
    ).to(device)
    # concatenate the other variables
    id_loss = torch.cat(batch.id_loss)
    targets = torch.cat(batch.targets)
    n = len(id_loss)
    # create the budgets
    Omegas = torch.cat(batch.Omegas).view([n, 1])
    Phis = torch.cat(batch.Phis).view([n, 1])
    Lambdas = torch.cat(batch.Lambdas).view([n, 1])
    # create the id_graph
    id_graphs = torch.tensor(
        [k for k in range(batch_size) for i in range(batch.next_n_free[k])]
    ).to(device)
    # concatenate the connected component features
    J = torch.cat(batch.J)
    saved_nodes = torch.cat(batch.saved_nodes)
    infected_nodes = torch.cat(batch.infected_nodes)
    size_connected = torch.cat(batch.size_connected)

    return (
        afterstates,
        Omegas,
        Phis,
        Lambdas,
        J,
        saved_nodes,
        infected_nodes,
        size_connected,
        id_loss,
        targets,
        id_graphs,
    )


def sample_memory_validation(memory, Transition, batch_size, multiple_targets=False, **kwargs):
    """Sample the Validation dataset and returns all the necessary to compute the validation loss.

    Parameters:
    ----------
    memory: ReplayMemory object,
    Transition: named tuple used to build the ReplayMemory,
    batch size: int,
                size of the sample
    multiple_targets: bool,
                      whether it is necessary to compute the values
                      of the targets from multiple target nets

    Returns:
    -------
    Parameters of the loss function if multiple_target, else
    Parameters of the loss function and the necessary
    parameters to compute the values of the targets using
    one target net"""

    # sample the memory
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))
    # create the torch geometric graphs
    afterstates = Batch.from_data_list(
        [graph for k in range(batch_size) for graph in batch.afterstates[k]]
    ).to(device)
    # concatenate the other variables
    id_loss = torch.cat(batch.id_loss)
    # create the budgets
    Omegas = torch.cat(batch.Omegas)
    Phis = torch.cat(batch.Phis)
    Lambdas = torch.cat(batch.Lambdas)
    # concatenate the connected component features
    J = torch.cat(batch.J)
    saved_nodes = torch.cat(batch.saved_nodes)
    infected_nodes = torch.cat(batch.infected_nodes)
    size_connected = torch.cat(batch.size_connected)

    if multiple_targets:

        list_next_Omegas = [
            float(next_Omegas_tensor[0]) for next_Omegas_tensor in batch.next_Omegas
        ]
        list_next_Phis = [
            float(next_Phis_tensor[0]) for next_Phis_tensor in batch.next_Phis
        ]
        list_next_Lambdas = [
            float(next_Lambdas_tensor[0]) for next_Lambdas_tensor in batch.next_Lambdas
        ]
        target_list = []
        id_graph_list = []

        for k in range(len(list_next_Omegas)):

            id_graph_list += [k for i in range(len(batch.next_afterstates[k]))]
            Omega = list_next_Omegas[k]
            Phi = list_next_Phis[k]
            Lambda = list_next_Lambdas[k]
            target_net = get_target_net(Omega=Omega, Phi=Phi, Lambda=Lambda, **kwargs)
            # if it's the end
            if target_net is None:
                # the target is the true reward
                target_list.append(batch.next_reward[k])

            else:
                # gather all the variables for the next turn
                G_torch = Batch.from_data_list(
                    [graph for graph in batch.next_afterstates[k]]
                ).to(device)
                next_Omega = batch.next_Omegas[k]
                next_Phi = batch.next_Phis[k]
                next_Lambda = batch.next_Lambdas[k]
                next_J = batch.next_J[k]
                next_saved = batch.next_saved_nodes[k]
                next_infected = batch.next_infected_nodes[k]
                next_size_connected = batch.next_size_connected[k]
                # we compute the values of the target
                with torch.no_grad():
                    target_values = target_net(
                        G_torch,
                        next_Omega,
                        next_Phi,
                        next_Lambda,
                        next_J,
                        next_saved,
                        next_infected,
                        next_size_connected,
                    )
                target_list.append(target_values)
        target = torch.cat(target_list)
        id_graphs = torch.tensor(id_graph_list).to(device)

        return (
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
        )

    else:
        next_afterstates = Batch.from_data_list(
            [graph for k in range(batch_size) for graph in batch.next_afterstates[k]]
        ).to(device)
        rewards = torch.cat(batch.next_reward)
        next_Omegas = torch.cat(batch.next_Omegas)
        next_Phis = torch.cat(batch.next_Phis)
        next_Lambdas = torch.cat(batch.next_Lambdas)
        # create the id_graph
        id_graphs = torch.tensor(
            [
                k
                for k in range(batch_size)
                for i in range(len(batch.next_afterstates[k]))
            ]
        ).to(device)
        next_J = torch.cat(batch.next_J)
        next_saved_nodes = torch.cat(batch.next_saved_nodes)
        next_infected_nodes = torch.cat(batch.next_infected_nodes)
        next_size_connected = torch.cat(batch.next_size_connected)

        return (
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


def update_training_memory(memory, memory_episode, actions, value, initial_player):
    """Backtracks the end reward obtained using the experts to
    take the actions at each middle step appearing during the
    unrolling of the episode and push all of these experiences
    in the ReplayMemory dataset

    Parameters:
    ----------
    memory: ReplayMemory object,
            the main replay memory used in training
    memory_episode: list,
                    contains the values of the parameters
                    we need to store in the memory
                    during the unrolling of one episode
    actions: list of list (of the form [[], [], []]),
             contains the id of the action taken
             by each player (0:vaccinator, 1:attacker, 2:protector)
             during the episode
    value: float,
           end reward obtained at the end of the episode
    initial_player: int,
                    the player we are currently focused on in the
                    learning procedure

    Returns:
    -------
    memory: ReplayMemory object,
            the main replay memory updated
            with the transitions experienced
            during the episode"""

    # initialize the count
    count = 1

    # for each player, beginning with the last to play
    for player in [2, 1, 0]:

        # initialize its set of actions
        actions_player = []

        # for each actions he took, beggining with the last
        for action in reversed(actions[player]):

            # if the memory is empty
            if len(actions_player) == 0:

                # put the last action in memory
                actions_player.append(action)
                # get the variables from the memory of the episode
                (
                    list_G_torch,
                    Omega_tensor,
                    Phi_tensor,
                    Lambda_tensor,
                    J_tensor,
                    saved_tensor,
                    infected_tensor,
                    size_connected_tensor,
                    id_loss,
                    targets,
                    next_n_free,
                ) = memory_episode[-count]
            else:
                # rename the actions with the correct id
                actions_player = [
                    action_p if action_p < action else action_p + 1
                    for action_p in actions_player
                ]
                # update the id_loss such that the loss for all the actions
                # is the one (value_net - target)^2
                id_loss[actions_player] = 4
                # if it is the turn of the player we are currently focused on,
                # we push the transition to memory
                if player == initial_player:
                    memory.push(
                        list_G_torch,
                        Omega_tensor,
                        Phi_tensor,
                        Lambda_tensor,
                        J_tensor,
                        saved_tensor,
                        infected_tensor,
                        size_connected_tensor,
                        id_loss,
                        targets,
                        next_n_free,
                    )
                actions_player.append(action)
                # get the variables from the memory of the episode
                (
                    list_G_torch,
                    Omega_tensor,
                    Phi_tensor,
                    Lambda_tensor,
                    J_tensor,
                    saved_tensor,
                    infected_tensor,
                    size_connected_tensor,
                    id_loss,
                    targets,
                    next_n_free,
                ) = memory_episode[-count]
                # update the target such that instead of an estimation
                # the target for all the optimal actions is the true end reward
                targets[actions_player] = value

            # update the count variable
            count += 1

        # if the list of actions is non empty
        # and if it is the turn of the player we want
        if len(actions_player) > 0 and player == initial_player:
            # we push the last modification to the training memory
            memory.push(
                list_G_torch,
                Omega_tensor,
                Phi_tensor,
                Lambda_tensor,
                J_tensor,
                saved_tensor,
                infected_tensor,
                size_connected_tensor,
                id_loss,
                targets,
                next_n_free,
            )

    return memory


def save_models(date_str, dict_args, value_net, optimizer, count, targets_experts):

    """Saves the models and the hyperparameters

    Parameters:
    ----------
    date_str: str,
              the date of the beginning of the training
    dict_args: dict,
               countains the hyperparameters
    value_net: neural network (pytorch module)
    optimizer: torch optimizer
    count: int,
           the number of steps
    targets_experts: TargetExperts object,
                     countains the list of experts"""

    # if there is no directory for the saved models
    # creates it
    if not os.path.exists("models"):
        os.mkdir("models")
    # inside this models directory
    # creates one specific to the run if it doesn't exist yet
    path = "models/" + date_str
    if not os.path.exists(path):
        os.mkdir(path)
    # inside this directory, saves the hyperparameters
    # used in a txt file if it doesn't already exist
    if not os.path.exists(path + "/hyperparameters.txt"):
        f = open(path + "/hyperparameters.txt", "w")
        for key, values in dict_args.items():
            f.write(key + ": " + str(values) + "\n")
        f.close()
    # saves the value net
    torch.save(
        {
            "epoch": count,
            "model_state_dict": value_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path + "/value_net.tar",
    )
    # create a directory for the experts
    path += "/experts"
    if not os.path.exists(path):
        os.mkdir(path)
    # saves every expert in the list of experts
    count = 0
    for target_net in targets_experts.list_target_nets:
        if target_net is not None:
            name = path + "/expert_" + str(count) + ".pt"
            torch.save(target_net, name)
            count += 1
