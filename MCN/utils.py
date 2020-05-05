import os
import re
import numpy as np
import torch
import networkx as nx
import random
import matplotlib.pyplot as plt
from torch_geometric.data import Batch
from torch_geometric.utils import from_networkx
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
            # set the corresponding max number of connected comp
            max_n_comp = 1 + Omega_max + Lambda_max - Lambda
        # if the target net is learning the attack values
        elif Budget_target <= Phi_max + Lambda_max:
            Omega = 0
            Phi = Budget_target - Lambda_max
            Lambda = np.random.randint(0, Lambda_max + 1)
            remaining_attack_budget = Phi_max - Phi
            Phi_attacked = np.random.randint(0, remaining_attack_budget + 1)
            # set the corresponding max number of connected comp
            max_n_comp = 1 + Omega_max
        # else, the target net is learning the vaccination values
        elif Budget_target <= Omega_max + Phi_max + Lambda_max:
            Omega = Budget_target - (Phi_max + Lambda_max)
            # we oblige that at least one node is attacked
            Phi = np.random.randint(1, Phi_max + 1)
            Lambda = np.random.randint(0, Lambda_max + 1)
            Phi_attacked = 0
            # set the corresponding max number of connected comp
            max_n_comp = 1 + Omega_max - Omega
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
            # set the corresponding max number of connected comp
            max_n_comp = 1 + Omega_max - Omega
        # if attacker
        elif player == 1:
            Omega = 0
            Phi = np.random.randint(1, Phi_max + 1)
            Lambda = np.random.randint(0, Lambda_max + 1)
            # no nodes pre-attacked
            Phi_attacked = 0
            # set the corresponding max number of connected comp
            max_n_comp = 1 + Omega_max
        # if protector
        elif player == 2:
            Omega = 0
            Phi = 0
            Lambda = np.random.randint(1, Lambda_max + 1)
            # some nodes are pre-attacked
            Phi_attacked = np.random.randint(1, Phi_max + 1)
            # set the corresponding max number of connected comp
            max_n_comp = 1 + Omega_max + Lambda_max - Lambda

    # random number of nodes
    n_free = random.randrange(n_free_min, n_free_max)
    n = n_free + Omega + Phi + Lambda + Phi_attacked
    # random number of components
    n_comp = int(1 + np.random.poisson(1, 1))
    n_comp = min(max_n_comp, n_comp)
    partition = list(
        np.sort(np.random.choice(range(1, n + 1), n_comp - 1, replace=False))
    )
    # Generate the graphs
    G = nx.DiGraph()
    partition = [0] + partition + [n]
    for k in range(n_comp):
        n_k = partition[k + 1] - partition[k]
        d_k = d_edge_min + (d_edge_max - d_edge_min)*np.random.random()
        G_k = generate_random_graph(n_k, d_k, directed=directed, draw=False)
        G = nx.union(G, G_k, rename=("G-", "H-"))
    # Generate the attack
    I = list(np.random.choice(range(n), Phi_attacked, replace=False))
    G = nx.convert_node_labels_to_integers(G)
    # Generate the weights
    if weighted:
        for node in G.nodes():
            G.nodes[node]['weight'] = random.randint(1, w_max)
    # Create the instance
    instance = Instance(G, Omega, Phi, Lambda, I, value=0)

    return instance


class Instance:

    """Creates an instance object to store the parameters defining an instance"""

    def __init__(self, G, Omega, Phi, Lambda, J, value, D=None, P=None):
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
        self.P = P


class InstanceTorch:

    """Creates an instance object to store all the tensors necessary to compute
    the approximate values with the ValueNet"""

    def __init__(self, G_torch, n_nodes, Omegas, Phis, Lambdas, Omegas_norm, Phis_norm, Lambdas_norm, J,
                 saved_nodes, infected_nodes, size_connected, target):

        self.G_torch = G_torch
        self.n_nodes = n_nodes
        self.Omegas = Omegas
        self.Phis = Phis
        self.Lambdas = Lambdas
        self.Omegas_norm = Omegas_norm
        self.Phis_norm = Phis_norm
        self.Lambdas_norm = Lambdas_norm
        self.J = J
        self.saved_nodes = saved_nodes
        self.infected_nodes = infected_nodes
        self.size_connected = size_connected
        self.target = target


def instance_to_torch(instance):

    """Transform an Instance object to an InstanceTorch one"""

    # Transform the graph
    G_torch = graph_torch(instance.G)
    # Compute the features from the connected components
    (
        _,
        J,
        saved_nodes,
        infected_nodes,
        size_connected,
    ) = features_connected_comp(instance.G, instance.J)
    # Put the number of nodes into a tensor
    n = len(instance.G)
    n_nodes = torch.tensor([n], dtype=torch.float).view([1, 1]).to(device)
    # Put the normalized budgets into tensors
    Omega_norm = torch.tensor([instance.Omega / n], dtype=torch.float).view([1, 1]).to(device)
    Lambda_norm = torch.tensor([instance.Lambda / n], dtype=torch.float).view([1, 1]).to(device)
    Phi_norm = torch.tensor([instance.Phi / n], dtype=torch.float).view([1, 1]).to(device)
    # Put the budgets into tensors
    Omega_tensor = torch.tensor([instance.Omega], dtype=torch.float).view([1, 1]).to(device)
    Lambda_tensor = torch.tensor([instance.Lambda], dtype=torch.float).view([1, 1]).to(device)
    Phi_tensor = torch.tensor([instance.Phi], dtype=torch.float).view([1, 1]).to(device)
    # Put the value into a tensor
    target = torch.tensor([instance.value], dtype=torch.float).view([1, 1]).to(device)
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
        saved_nodes,
        infected_nodes,
        size_connected,
        target,
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
    value: int,
           the value of the saved nodes
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
    # Initialize the variables
    value = 0
    connected_infected = set()
    size_connected = [1] * n
    is_weighted = len(nx.get_node_attributes(G, 'weight').values()) != 0
    is_directed = False in [(v, u) in G.edges() for (u, v) in G.edges()]
    # Gather the weights
    if is_weighted:
        weights = np.array([G.nodes[node]['weight'] for node in G.nodes()])
        sum_weights = np.sum(weights)
    else:
        weights = np.ones(n)
        sum_weights = n
    # Compute the features in the directed case
    if is_directed:
        for node in G.nodes():
            connected = set(u for u in nx.dfs_preorder_nodes(G, node))
            size_connected[node] = np.sum(weights[list(connected)]) / sum_weights
            if node in I:
                connected_infected = connected_infected.union(connected)
        value = np.sum(weights[list(set(G.nodes()) - connected_infected)])
    else:
        # insure G is undirected
        G1 = G.to_undirected()
        for c in nx.connected_components(G1):
            size_c = np.sum(weights[list(c)])
            # update the vector of size of the connected comp
            for node in c:
                # we normalize the size by the total size of the graph
                size_connected[node] = size_c / sum_weights
            # a connected component is saved if
            # there is not any attacked node inside it
            if set(I).intersection(c) == set():
                value += size_c
            else:
                connected_infected = connected_infected.union(c)

    # transform the variables to tensor
    # init the tensors
    J_tensor = np.zeros(n)
    indic_saved = np.ones(n)
    indic_infected = np.zeros(n)
    # compute the one hot encoding
    J_tensor[I] = 1
    indic_infected[list(connected_infected)] = 1
    indic_saved -= indic_infected
    J_tensor = torch.tensor(J_tensor, dtype=torch.float).view([n, 1]).to(device)
    indic_saved = torch.tensor(indic_saved, dtype=torch.float).view([n, 1]).to(device)
    indic_infected = torch.tensor(indic_infected, dtype=torch.float).view([n, 1]).to(device)
    size_connected_tensor = torch.tensor(size_connected, dtype=torch.float).view([n, 1]).to(device)

    return (value, J_tensor, indic_saved, indic_infected, size_connected_tensor)


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


def sample_action(neural_net, player, next_player, rewards, next_afterstates,
                  eps_end, eps_decay, eps_start, count_steps, **kwargs):
    """Sample an action given the possible afterstates.
    The action is the greedy one with a certain probability and sampled at random
    among all possible ones with the complementary probability."""

    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * np.exp(-1. * count_steps / eps_decay)
    if sample > eps_threshold:
        return take_action_deterministic(neural_net, player, next_player, rewards, next_afterstates, **kwargs)
    else:
        targets = rewards
        n = targets.size()[0]
        action = np.random.randint(0, n)
        value = float(targets[action])

        return action, targets, value


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
                torch.save(target_net, name)
                count += 1


def load_saved_experts(path):
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
        max_budget = int(re.findall(r'\d+', list_files[-1])[0])
        list_experts = [None] * (max_budget + 1)
        for f in list_files:
            expert_path = os.path.join(path, f)
            # load the model
            expert = torch.load(expert_path)
            expert.eval()
            # get the expert's budget
            budget = int(re.findall(r'\d+', expert_path)[0])
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


def compute_loss_test(test_set_generators, value_net=None, list_experts=None):

    """Compute the list of losses of the value_net or the list_of_experts
    over the list of exactly solved datasets that constitutes the test set"""

    list_losses = []
    with torch.no_grad():
        for k in range(len(test_set_generators)):
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
                        batch_instances.saved_nodes,
                        batch_instances.infected_nodes,
                        batch_instances.size_connected,
                    )
                    val_approx.append(values_approx)
                    target.append(batch_instances.target)
                # Compute the loss
                target = torch.cat(target)
                val_approx = torch.cat(val_approx)
                loss_target_net = float(torch.sqrt(torch.mean((val_approx[:, 0] - target[:, 0]) ** 2)))
                list_losses.append(loss_target_net)

    return list_losses


