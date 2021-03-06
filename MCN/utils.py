import os
import re
import numpy as np
import torch
import networkx as nx
import random
import matplotlib.pyplot as plt
from torch_scatter import scatter_max, scatter_min
from torch_geometric.data import Batch
from torch_geometric.utils import from_networkx
from torch_geometric.transforms import LocalDegreeProfile
from MCN.MCN_heur.neural_networks import ValueNet


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
    is_directed = False in [(v, u) in graph.edges() for (u, v) in graph.edges()]

    if color_type == "fabulous":

        # Draw the graph with a colormap
        nx.draw_spring(graph,
                       with_labels=True,
                       node_size=600,
                       node_color=np.array(list(graph.nodes())),
                       cmap='viridis_r',
                       alpha=1.0,
                       edge_color='gray',
                       arrows=is_directed,
                       width=2,
                       font_size=15,
                       font_color='white')

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

        nx.draw_spring(graph,
                       with_labels=True,
                       node_size=600,
                       node_color=node_colors,
                       edge_color='gray',
                       arrows=is_directed,
                       width=2,
                       font_size=15,
                       font_color='white')

    plt.show()


def generate_random_graph(n_nodes, density, directed=False, is_tree=False, seed=None, draw=False):
    r"""Generate a random graph with the desired number of nodes and density.
    Draw the graph if asked. Returns the graph as a networkx object.

    Parameters:
    ----------
    n_nodes: int,
             number of nodes
    density: float (\in [0,1]),
             density of edges
    directed: bool,
              whether or not the graph is directed
    is_tree: bool,
             whether to generate a tree or not
    seed: int,
          the random seed to use
    draw: bool,
          whether to draw the graph or not

    Returns:
    -------
    graph: networkx Digraph"""

    # if we want to generate a directed graph, we generate two undirected graphs
    # of the same size and assemble them in one single directed graph by saying the
    # vertices of the first are in the direction left -> right and the inverse for the second

    # Compute the number of edges corresponding to the given density
    n_edges = int(density * n_nodes * (n_nodes - 1) / 2)
    # Create the graph
    seed_1 = None if seed is None else seed + 1
    if is_tree:
        graph_0 = nx.random_tree(n_nodes, seed=seed)
        if directed:
            graph_1 = nx.random_tree(n_nodes, seed=seed_1)
        else:
            graph_1 = graph_0
    else:
        graph_0 = nx.gnm_random_graph(n_nodes, n_edges, seed)
        if directed:
            graph_1 = nx.gnm_random_graph(n_nodes, n_edges, seed_1)
        else:
            graph_1 = graph_0
    graph_2 = nx.DiGraph()
    graph_2.add_nodes_from(graph_0.nodes())
    graph_2.add_edges_from([(u, v) for (u, v) in graph_0.edges()])
    graph_2.add_edges_from([(v, u) for (u, v) in graph_1.edges()])

    if draw:
        plot_graph(graph_2, color_type="fabulous")

    return graph_2


def generate_random_instance(n_free_min, n_free_max, d_edge_min, d_edge_max,
                             Omega_max, Phi_max, Lambda_max, Budget_target=np.nan,
                             weighted=False, w_max=1, directed=False):
    r"""Generate a random instance of the MCN problem corresponding
    to the stage of the training we are in if Budget_target is defined.
    Else, we generate a random instance of the MCN problem.
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
    d_edge_min: float \in [0,1],
                minimal edge density of the graphs considered
    d_edge_max: float \in [0,1],
                maximal edge density of the graphs considered
    Omega_max: int,
               the maximal vaccination budget
    Phi_max: int,
             the maximal attack budget
             must be > 0, otherwise there is no MCN problem
    Lambda_max: int,
                the maximal protection budget
    Budget_target: int,
                   the total budget we are considering
                   at this stage of the training procedure
                   (\in [1, Omega_max+Phi_max+Lambda_max])
    weighted: bool,
              whether to create weights for the nodes or not
    w_max: int,
           the maximum weight a node can have
    directed: bool,
              whether or not the graph is directed

    Returns:
    -------
    instance: Instance object"""

    # if we generate an instance for the training procedure
    if Budget_target is not np.nan:
        # if the target net is learning the protection values
        if Budget_target <= Lambda_max:
            Omega = 0
            Phi = 0
            Lambda = Budget_target
            # we need to attack some nodes in order
            # to learn the protection
            Phi_attacked = np.random.randint(1, Phi_max + 1)
            # add some random "already defended" nodes
            # that we will remove from the graph
            Lambda_del = np.random.randint(0, Lambda_max - Lambda + 1)
            Omega_del = np.random.randint(0, Omega_max + 1)
        # if the target net is learning the attack values
        elif Budget_target <= Phi_max + Lambda_max:
            Omega = 0
            Phi = Budget_target - Lambda_max
            Lambda = np.random.randint(0, Lambda_max + 1)
            remaining_attack_budget = Phi_max - Phi
            Phi_attacked = np.random.randint(0, remaining_attack_budget + 1)
            # add some random "already defended" nodes
            # that we will remove from the graph
            Lambda_del = 0
            Omega_del = np.random.randint(0, Omega_max + 1)
        # else, the target net is learning the vaccination values
        elif Budget_target <= Omega_max + Phi_max + Lambda_max:
            Omega = Budget_target - (Phi_max + Lambda_max)
            # we oblige that at least one node is attacked
            Phi = np.random.randint(1, Phi_max + 1)
            Lambda = np.random.randint(0, Lambda_max + 1)
            Phi_attacked = 0
            # add some random "already defended" nodes
            # that we will remove from the graph
            Lambda_del = 0
            Omega_del = np.random.randint(0, Omega_max - Omega + 1)
    # else, we are not in the training procedure
    else:
        # sample a random player
        if Omega_max == 0 and Lambda_max == 0:
            player = 1
        elif Omega_max == 0:
            player = np.random.randint(1,3)
        elif Lambda_max == 0:
            player = np.random.randint(0,2)
        elif Omega_max > 0 and Lambda_max > 0:
            player = np.random.randint(0,3)
        # if vaccinator
        if player == 0:
            # means Omega > 0
            Omega = np.random.randint(1, Omega_max + 1)
            Phi = np.random.randint(1, Phi_max + 1)
            Lambda = np.random.randint(0, Lambda_max + 1)
            # no nodes pre-attacked
            Phi_attacked = 0
            # already defended nodes
            Lambda_del = 0
            Omega_del = np.random.randint(0, Omega_max - Omega + 1)
        # if attacker
        elif player == 1:
            Omega = 0
            Phi = np.random.randint(1, Phi_max + 1)
            Lambda = np.random.randint(0, Lambda_max + 1)
            # no nodes pre-attacked
            Phi_attacked = 0
            # already defended nodes
            Lambda_del = 0
            Omega_del = np.random.randint(0, Omega_max + 1)
        # if protector
        elif player == 2:
            Omega = 0
            Phi = 0
            Lambda = np.random.randint(1, Lambda_max + 1)
            # some nodes are pre-attacked
            Phi_attacked = np.random.randint(1, Phi_max + 1)
            # already defended nodes
            Lambda_del = np.random.randint(0, Lambda_max - Lambda + 1)
            Omega_del = np.random.randint(0, Omega_max + 1)

    # random number of nodes
    n_free = random.randrange(n_free_min, n_free_max)
    n = n_free + Omega + Phi + Lambda + Omega_del + Phi_attacked + Lambda_del
    # random density
    d = d_edge_min + (d_edge_max - d_edge_min)*np.random.random()
    # Generate the graph
    G = generate_random_graph(n, d, directed=directed, draw=False)
    # Generate a random defense
    defended_nodes = np.random.choice(n, Omega_del + Lambda_del, replace=False)
    if Omega_del + Lambda_del > 0:
        # delete the nodes defended
        G.remove_nodes_from(defended_nodes)
        # relabel the nodes so that they are labeled from 0 to nb_nodes
        G = nx.convert_node_labels_to_integers(G)
    # Generate the attack
    I = list(np.random.choice(range(n - Omega_del - Lambda_del), Phi_attacked, replace=False))
    # Generate the weights
    if weighted:
        for node in G.nodes():
            G.nodes[node]['weight'] = random.randint(1, w_max)
    # Create the instance
    instance = Instance(G, Omega, Phi, Lambda, I, value=0)

    return instance


def generate_random_batch_instance(batch_size, n_free_min, n_free_max, d_edge_min, d_edge_max,
                                   Omega_max, Phi_max, Lambda_max, Budget_target=np.nan,
                                   weighted=False, w_max=1, directed=False):
    r"""Generate a random batch (list) of instances of the MCN problem. """

    # if we generate an instance for the training procedure
    if Budget_target is not np.nan:
        # if the target net is learning the protection values
        if Budget_target <= Lambda_max:
            Omega = np.zeros(batch_size)
            Phi = np.zeros(batch_size)
            Lambda = Budget_target * np.ones(batch_size)
            # we need to attack some nodes in order
            # to learn the protection
            Phi_attacked = np.random.randint(1, Phi_max + 1, size=batch_size)
            # add some random "already defended" nodes
            # that we will remove from the graph
            Lambda_del = np.random.randint(0, Lambda_max - Budget_target + 1, size=batch_size)
            Omega_del = np.random.randint(0, Omega_max + 1, size=batch_size)
        # if the target net is learning the attack values
        elif Budget_target <= Phi_max + Lambda_max:
            Omega = np.zeros(batch_size)
            Phi = (Budget_target - Lambda_max) * np.ones(batch_size)
            Lambda = np.random.randint(0, Lambda_max + 1) * np.ones(batch_size)
            remaining_attack_budget = Phi_max - Phi
            Phi_attacked = np.random.randint(0, remaining_attack_budget + 1, size=batch_size)
            # add some random "already defended" nodes
            # that we will remove from the graph
            Lambda_del = np.zeros(batch_size)
            Omega_del = np.random.randint(0, Omega_max + 1, size=batch_size)
        # else, the target net is learning the vaccination values
        elif Budget_target <= Omega_max + Phi_max + Lambda_max:
            Omega = (Budget_target - (Phi_max + Lambda_max)) * np.ones(batch_size)
            # we oblige that at least one node is attacked
            Phi = np.random.randint(1, Phi_max + 1) * np.ones(batch_size)
            Lambda = np.random.randint(0, Lambda_max + 1) * np.ones(batch_size)
            Phi_attacked = np.zeros(batch_size)
            # add some random "already defended" nodes
            # that we will remove from the graph
            Lambda_del = np.zeros(batch_size)
            Omega_del = np.random.randint(0, Omega_max - Omega[0] + 1, size=batch_size)


    # random number of nodes
    n_free = np.random.randint(n_free_min, n_free_max + 1, size=batch_size)
    n = n_free + Omega + Phi + Lambda + Omega_del + Phi_attacked + Lambda_del
    # random density
    d = d_edge_min + (d_edge_max - d_edge_min)*np.random.random(size=batch_size)
    # Generate a batch of instances
    instance_batch = []
    for k in range(batch_size):
        # Generate the graph
        G = generate_random_graph(int(n[k]), d[k], directed=directed, draw=False)
        # Generate a random defense
        defended_nodes = np.random.choice(int(n[k]), int(Omega_del[k] + Lambda_del[k]), replace=False)
        if Omega_del[k] + Lambda_del[k] > 0:
            # delete the nodes defended
            G.remove_nodes_from(defended_nodes)
            # relabel the nodes so that they are labeled from 0 to nb_nodes
            G = nx.convert_node_labels_to_integers(G)
        # Generate the attack
        I = list(np.random.choice(range(int(n[k] - Omega_del[k] - Lambda_del[k])), int(Phi_attacked[k]), replace=False))
        # Generate the weights
        if weighted:
            for node in G.nodes():
                G.nodes[node]['weight'] = random.randint(1, w_max)
        # Create the instance
        instance_k = Instance(G, Omega[k], Phi[k], Lambda[k], I, value=0)
        instance_batch.append(instance_k)

    return instance_batch


class Instance:

    """Creates an instance object to store the parameters defining an instance"""

    def __init__(self, G, Omega, Phi, Lambda, J, value, D=None, I=None, P=None):
        """
        Parameters:
        ----------
        G: networkx graph
        I: list of ints,
           the list of attacked nodes
        Omega: int,
               the budget allocated to the vaccinator
        Phi: int,
             the budget allocated to the attacker
        Lambda: int,
                the budget allocated to the protector
        value: int,
               if known, the value of the instance
        D: list of ints,
           the list of saved nodes
        P: list of ints,
           list of protected nodes"""

        self.G = G
        self.Omega = Omega
        self.Phi = Phi
        self.Lambda = Lambda
        self.J = J
        self.value = value
        self.D = D
        self.I = I
        self.P = P


class InstanceTorch:

    """Creates an instance object to store all the tensors necessary to compute
    the approximate values with the ValueNet"""

    def __init__(self, G_torch, n_nodes, Omegas, Phis, Lambdas, Omegas_norm, Phis_norm, Lambdas_norm, J,
                 target=None, next_player=None):

        self.G_torch = G_torch
        self.n_nodes = n_nodes
        self.Omegas = Omegas
        self.Phis = Phis
        self.Lambdas = Lambdas
        self.Omegas_norm = Omegas_norm
        self.Phis_norm = Phis_norm
        self.Lambdas_norm = Lambdas_norm
        self.J = J
        self.target = target
        self.next_player = next_player


def instance_to_torch(instance):

    """Transform an Instance object to an InstanceTorch one"""

    # Transform the graph
    G_torch = graph_torch(instance.G)
    n = len(instance.G)
    # Compute J
    J= np.zeros(n)
    J[instance.J] = 1
    J = torch.tensor(J, dtype=torch.float).view([n, 1]).to(device)
    # Put the number of nodes into a tensor
    n_nodes = torch.tensor([n], dtype=torch.float).view([1, 1]).to(device)
    # Put the budgets into tensors
    Omega_tensor = torch.tensor([instance.Omega], dtype=torch.float).view([1, 1]).to(device)
    Lambda_tensor = torch.tensor([instance.Lambda], dtype=torch.float).view([1, 1]).to(device)
    Phi_tensor = torch.tensor([instance.Phi], dtype=torch.float).view([1, 1]).to(device)
    # Put the normalized budgets into tensors
    Omega_norm = Omega_tensor / n
    Lambda_norm = Lambda_tensor / n
    Phi_norm = Phi_tensor / n

    # Put the value into a tensor
    target = torch.tensor([instance.value], dtype=torch.float).view([1, 1]).to(device)
    # Get the player
    next_player = get_next_player(instance.Omega, instance.Phi, instance.Lambda)
    next_player = torch.tensor([next_player], dtype=torch.float).view([1, 1]).to(device)
    # Gather everything into a single InstanceTorch object
    instance_torch = InstanceTorch(
        G_torch,
        n_nodes,
        Omega_tensor,
        Phi_tensor,
        Lambda_tensor,
        Omega_norm,
        Phi_norm,
        Lambda_norm,
        J,
        target,
        next_player
    )

    return instance_torch


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
    G_torch = from_networkx(G_networkx).to(device)
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


def graph_weights(G):

    V = list(G.nodes())
    w = dict()
    for v in V:
        # if the graph is weighted, gather the weights
        if 'weight' in G.nodes[v].keys():
            w[v] = float(G.nodes[v]['weight'])
        # else, all weights are 1
        else:
            w[v] = 1.0
    return w


def compute_saved_nodes(G, I):
    """Compute the values of the saved node given a graph and
    a list of attacked nodes.

    Parameters:
    ----------
    G: networkx graph
    I: list of ints,
       list of the ids of the attacked nodes of G

    Returns:
    -------
    value: int,
           the values of the saved nodes"""

    n = len(G)
    connected_infected = set()
    is_weighted = len(nx.get_node_attributes(G, 'weight').values()) != 0
    is_directed = False in [(v, u) in G.edges() for (u, v) in G.edges()]
    # Gather the weights
    if is_weighted:
        weights = np.array([G.nodes[node]['weight'] for node in G.nodes()])
    else:
        weights = np.ones(n)
    # Compute the infected nodes in the directed case
    if is_directed:
        for node in G.nodes():
            connected = set(u for u in nx.dfs_preorder_nodes(G, node))
            if node in I:
                connected_infected = connected_infected.union(connected)
    else:
        # insure G is undirected
        G1 = G.to_undirected()
        for c in nx.connected_components(G1):
            if set(I).intersection(c) != set():
                connected_infected = connected_infected.union(c)

    value = np.sum(weights[list(set(G.nodes()) - connected_infected)])

    return value


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


def get_next_player(Omega, Phi, Lambda):
    """Returns the next player given the budgets."""
    next_Omega = Omega
    next_Phi = Phi
    next_Lambda = Lambda
    player = get_player(Omega, Phi, Lambda)
    if player == 0:
        next_Omega = Omega - 1
    elif player == 1:
        next_Phi = Phi - 1
    elif player == 2:
        next_Lambda = Lambda - 1

    return get_player(next_Omega, next_Phi, next_Lambda)


def take_action_deterministic(target_net, player, next_player, rewards, next_afterstates, weights, **kwargs):
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
    weights: list,
             the weights attributed to each action possible

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

    weights_tensor = torch.tensor(weights, dtype=torch.float).view(targets.size()).to(device)
    target_decision = targets + weights_tensor
    # if it's the turn of the attacker
    if player == 1:
        # we take the argmin
        action = int(target_decision.argmin())
    else:
        # we take the argmax
        action = int(target_decision.argmax())
    value = float(targets[action])

    return action, targets, value


def take_action_deterministic_batch(target_net, player, next_player, rewards, next_afterstates, weights=None,
                                    id_graphs=None, **kwargs):
    """Take actions in batch"""

    if id_graphs is None:
        n_nodes = sum([len(afterstate) for afterstate in next_afterstates])
        id_graphs = torch.zeros(size=(n_nodes,), dtype=torch.int64).to(device)
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
    if weights is not None:
        weights_tensor = torch.tensor(weights, dtype=torch.float).view(targets.size()).to(device)
        target_decision = targets + weights_tensor
    else:
        target_decision = targets
    # if it's the turn of the attacker
    if player == 1:
        # we take the argmin
        _, actions = scatter_min(target_decision, id_graphs, dim=0)
    else:
        # we take the argmax
        _, actions = scatter_max(target_decision, id_graphs, dim=0)
    values = targets[actions[:,0]]

    return actions.view(-1).tolist(), targets, values.view(-1).tolist()


def sample_action_batch(neural_net, player, next_player, rewards, next_afterstates, id_graphs,
                        eps_end, eps_decay, eps_start, count_steps, **kwargs):
    """Sample an action given the possible afterstates.
    The action is the greedy one with a certain probability and sampled at random
    among all possible ones with the complementary probability."""

    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * np.exp(-1. * count_steps / eps_decay)
    if sample > eps_threshold:
        return take_action_deterministic_batch(neural_net, player, next_player, rewards,
                                               next_afterstates, id_graphs, **kwargs)
    else:
        targets = rewards
        actions = []
        values = []
        n_batches = int(id_graphs[-1]) + 1
        all_actions = torch.arange(len(id_graphs))
        for i in range(n_batches):
            mask_i = id_graphs.eq(i)
            actions_possible = all_actions[mask_i]
            random_id = int(np.random.randint(0, len(actions_possible)))
            random_action = int(actions_possible[random_id])
            actions.append(random_action)
            values.append(float(targets[random_action]))

        return actions, targets, values


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


def save_models(date_str, dict_args, value_net, optimizer, count, targets_experts=None):

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
    path = os.path.join("models",date_str)
    if not os.path.exists(path):
        os.mkdir(path)
    # inside this directory, saves the hyperparameters
    # used in a txt file if it doesn't already exist
    if not os.path.exists(os.path.join(path,"hyperparameters.txt")):
        f = open(os.path.join(path,"hyperparameters.txt"), "w")
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
        os.path.join(path,"value_net.tar"),
    )
    print("\nSaved models in " + path, '\n')

    if targets_experts is not None:
        # create a directory for the experts
        path = os.path.join(path,"experts")
        if not os.path.exists(path):
            os.mkdir(path)
        # saves every expert in the list of experts
        count = 0
        for target_net in targets_experts.list_target_nets:
            if target_net is not None:
                name = os.path.join(path,"expert_" + str(count) + ".pt")
                torch.save({"model_state_dict": target_net.state_dict()}, name)
                count += 1


def load_saved_experts(path, **kwargs):
    """Load all the saved experts models from a directory and
    and returns them as a list of pytorch modules

    Parameters:
    ----------
    path: str,
          path to the directory containing the saved experts"""

    # Initialize the list of experts
    list_experts = []
    if os.path.isdir(path):
        # for everything in the directory
        list_files = os.listdir(path)
        max_budget = max([int(re.findall(r'([0-9]+)\.pt', name)[0]) for name in list_files])
        list_experts = [None] * (max_budget + 1)
        for f in list_files:
            if '.pt' in f:
                expert = ValueNet(**kwargs).to(device)
                expert_path = os.path.join(path, f)
                # load the model
                checkpoint = torch.load(expert_path)
                expert.load_state_dict(checkpoint['model_state_dict'])
                #expert = torch.load(expert_path)
                expert.eval()
                # get the expert's budget
                budget = int(re.findall(r'([0-9]+)\.pt', f)[0])
                # append the model to the list
                list_experts[budget] = expert
    else:
        raise ValueError("no directory found at given path")

    return list_experts


def load_training_param(value_net, optimizer, path):

    """Load the training parameters of a previous training session
    and resume training where things where left with the value_net
    and the optimizer"""

    checkpoint = torch.load(path)
    value_net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    value_net.train()

    return(value_net, optimizer)


def count_param_NN(torch_module):

    """Count the number of parameters in a Neural Network and returns it"""

    n_param = 0
    for parameter in torch_module.parameters():

        k_param = 1
        for element in parameter.size():
            k_param *= element
        n_param += k_param

    return n_param


def compute_loss_test(test_set_generators, value_net=None, list_experts=None, id_to_test=None):

    """Compute the list of losses of the value_net or the list_of_experts
    over the list of exactly solved datasets that constitutes the test set"""

    list_losses = []
    with torch.no_grad():
        if id_to_test is None:
            iterator = range(len(test_set_generators))
        else:
            iterator = [id_to_test]
        for k in iterator:
            target = []
            val_approx = []
            if list_experts is not None:
                try:
                    target_net = list_experts[k]
                except IndexError:
                    target_net = None
            elif value_net is not None:
                target_net = value_net
            if target_net is None:
                list_losses.append(0)
            else:
                for i_batch, batch_instances in enumerate(test_set_generators[k]):
                    values_approx = target_net(
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
                loss_target_net = float(torch.sqrt(torch.mean((val_approx[:, 0] - target[:, 0]) ** 2)))
                list_losses.append(loss_target_net)

    return list_losses


