import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GlobalAttention, APPNP, global_add_pool, GATConv, BatchNorm
from torch_scatter import scatter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AttentionLayer(nn.Module):

    """Implementation of the AttentionLayer described in the
    'Attention learn to solve routing problem' paper https://arxiv.org/abs/1803.08475"""

    def __init__(self, n_heads, dim_embedding, dim_values, dim_hidden):
        super(AttentionLayer, self).__init__()

        self.n_heads = n_heads
        self.dim_values = dim_values

        self.GAT = GATConv(dim_embedding, dim_values, heads=n_heads, concat=True, bias=False)
        self.lin1 = nn.Linear(dim_values, dim_embedding, bias=False)
        self.BN1 = BatchNorm(dim_embedding)
        self.lin2 = nn.Linear(dim_embedding, dim_hidden)
        self.lin3 = nn.Linear(dim_hidden, dim_embedding)
        self.BN2 = BatchNorm(dim_embedding)

    def forward(self, x, edge_index):
        # Message passing using attention
        h = self.GAT(x, edge_index)
        # undo the concatenation of the results of the M heads
        # at the output of the GAT layer
        h = h.view(-1, self.n_heads, self.dim_values)
        # project back to embedding space
        h = self.lin1(h)
        # sum the results of the M heads
        h = torch.sum(h, 1)
        # apply Batch Norm and skip connection
        h = self.BN1(x + h)
        # apply the feedforward  layer
        h2 = self.lin2(h)
        h2 = F.relu(h2)
        h2 = self.lin3(h2)
        # apply Batch Norm and skip connection
        h2 = self.BN2(h2 + h)

        return h2


class NodeEncoder(nn.Module):

    """Create the node embeddings"""

    def __init__(self, dim_input, n_heads, n_att_layers, dim_embedding, dim_values, dim_hidden, K, alpha, weighted):
        super(NodeEncoder, self).__init__()

        # if the graph is weighted
        self.weighted = weighted
        if weighted:
            # there are 4 features that are added to the input data
            first_dim = dim_input + 2
        else:
            # else, only 2 features are added
            first_dim = dim_input + 1
        self.n_att_layers = n_att_layers
        self.Lin1 = nn.Linear(first_dim, dim_embedding)
        self.attention_layers = nn.ModuleList(
            [
                AttentionLayer(n_heads, dim_embedding, dim_values, dim_hidden) for k in range(n_att_layers)
            ]
        )
        self.power = APPNP(K, alpha, bias=False).to(device)

    def forward(self, G_torch, J):

        # retrieve the data
        x, edge_index, batch = G_torch.x, G_torch.edge_index, G_torch.batch
        # gather together the node features with J and size_connected
        h = torch.cat([x, J], 1)
        # if we are considering weighted graphs
        if self.weighted:
            # add the normalized weights to the features to consider
            weights = G_torch.weight.view([-1,1]).type(dtype=torch.float)
            weights_sum = global_add_pool(weights, batch)
            weights_norm = weights / weights_sum[batch]
            h = torch.cat([h, weights_norm], 1)
        # project the features into a dim_embedding vector space
        h = self.Lin1(h)
        # apply the attention layers
        for k in range(self.n_att_layers):
            h = self.attention_layers[k](h, edge_index)
        # apply the power layer
        h = self.power(h, edge_index)
        # re-add the information about the node's state
        h = torch.cat([h, J], 1)
        # if we are considering weighted graphs
        if self.weighted:
            h = torch.cat([h, weights_norm], 1)
        G_torch.x = h

        return G_torch


class ContextEncoder(nn.Module):

    """Create the graph embedding, which is then concatenated with other
    variables describing the current context"""

    def __init__(self, n_pool, dim_embedding, dim_hidden, weighted):
        super(ContextEncoder, self).__init__()

        self.weighted = weighted
        if weighted:
            # there are 8 features that are added to the input data
            first_dim = dim_embedding + 2
        else:
            # else, only 4 features are added
            first_dim = dim_embedding + 1

        self.n_pool = n_pool
        self.graph_pool = nn.ModuleList(
            [
                GlobalAttention(
                    nn.Sequential(nn.Linear(first_dim, dim_hidden),
                                  nn.ReLU(),
                                  nn.Linear(dim_hidden, 1)),
                    nn.Sequential(nn.Linear(first_dim, dim_hidden),
                                  nn.ReLU(),
                                  nn.Linear(dim_hidden, dim_embedding)))
                for k in range(n_pool)
            ]
        )

    def forward(self, G_torch, n_nodes, Omegas, Phis, Lambdas, Omegas_norm, Phis_norm, Lambdas_norm):

        # retrieve the data
        x, edge_index, batch = G_torch.x, G_torch.edge_index, G_torch.batch
        # concatenate the n_pool graph pool
        context_embedding = []
        for k in range(self.n_pool):
            context_embedding.append(self.graph_pool[k](x, batch))
        context_embedding = torch.cat(context_embedding, 1)
        # create the final context tensor
        context = torch.cat(
        [
            context_embedding,
            n_nodes,
            Omegas,
            Phis,
            Lambdas,
            Omegas_norm,
            Phis_norm,
            Lambdas_norm,
        ], 1)
        if self.weighted:
            weights = G_torch.weight.view([-1, 1]).type(dtype=torch.float)
            weights_sum = global_add_pool(weights, batch)
            context = torch.cat([context, weights_sum], 1)

        return context


class ValueNet(nn.Module):
    r"""The Value Network used in order to estimate the number of saved
        nodes in the end given the current situation.

        This is done in 4 steps:

            - STEP 1 : compute a node embedding for each node
            - STEP 2 : use the nodes embedding and some global
                       information about the graphs to compute
                       a graph embedding
            - STEP 3 : use the graph embeddings, node embeddings
                       and information about the nodes in order
                       to compute a score for each node \in [0,1]
                       (can be thought of as the probability of
                       the node being saved in the end)
            - STEP 4 : sum the scores of the nodes to obtain
                       the value of the graph.
                       if the graph is weighted, before summing we multiply the
                       scores \in [0,1] with the node weights"""

    def __init__(self, dim_input, dim_embedding, dim_values, dim_hidden,
                 n_heads, n_att_layers, n_pool, K, alpha, p, weighted=False):
        super(ValueNet, self).__init__()

        self.weighted = weighted
        self.node_encoder = NodeEncoder(dim_input, n_heads, n_att_layers, dim_embedding,
                                        dim_values, dim_hidden, K, alpha, weighted)
        self.context_encoder = ContextEncoder(n_pool, dim_embedding, dim_hidden, weighted)
        # Score for each node
        if weighted:
            dim_context = dim_embedding * n_pool + 8
            first_dim = dim_context + dim_embedding + 2
        else:
            dim_context = dim_embedding * n_pool + 7
            first_dim = dim_context + dim_embedding + 1
        self.lin1 = nn.Linear(first_dim, dim_hidden)
        self.BN1 = BatchNorm(dim_hidden)
        self.lin2 = nn.Linear(dim_hidden, dim_embedding)
        self.BN2 = BatchNorm(dim_embedding)
        self.lin3 = nn.Linear(dim_embedding, 1)
        # dropout
        self.dropout = nn.Dropout(p=p)


    def forward(self, G_torch, n_nodes, Omegas, Phis, Lambdas, Omegas_norm, Phis_norm, Lambdas_norm, J):
        """ Take a batch of states as input and returns a the values of each state.

                Parameters:
                ----------
                G_torch: Pytorch Geometric batch data,
                         Pytorch Geometric representation of the graphs G along
                         with their node features
                n_nodes: float tensor (size = Batch x 1)
                         the number of nodes of each graph in the batch
                Omegas: float tensor (size = Batch x 1),
                        the budget omega for each graph in the batch
                Phis: float tensor (size = Batch x 1),
                      the budget phi for each graph in the batch
                Lambdas: float tensor (size = Batch x 1),
                         the budget Lambda for each graph in the batch
                Omegas_norm: float tensor (size = Batch x 1),
                             the normalized budget omega for each graph in the batch
                Phis_norm: float tensor (size = Batch x 1),
                           the normalized budget phi for each graph in the batch
                Lambdas_norm: float tensor (size = Batch x 1),
                              the normalized budget Lambda for each graph in the batch
                J: float tensor (size = nb tot of nodes x 1),
                   indicator 1_{node infected} for each node
                   in each graph in the batch
                saved_nodes: float tensor (size = nb tot of nodes x 1),
                             indicator 1_{node currently saved} for each node
                             in each graph in the batch
                infected_nodes: float tensor (size = nb tot of nodes x 1),
                                indicator 1_{node currently infected}
                                for each node in each graph in the batch
                size_connected: float tensor (size = nb tot of nodes x 1),
                                size of the component component each node
                                belongs to in each graph in the batch
                                (normalized by the size of the graph)

                Returns:
                -------
                score_state: float tensor (size = Batch x 1),
                             score of each possible afterstate"""

        G = self.node_encoder(G_torch, J)
        context = self.context_encoder(G, n_nodes, Omegas, Phis, Lambdas, Omegas_norm, Phis_norm, Lambdas_norm)
        # retrieve the data from G
        x, edge_index, batch = G.x, G.edge_index, G.batch
        x_score = torch.cat([x, context[batch]], 1)
        score = self.lin1(x_score)
        score = self.dropout(score)
        score = F.leaky_relu(score, 0.2)
        score = self.BN1(score)
        score = self.lin2(score)
        score = self.dropout(score)
        score = F.leaky_relu(score, 0.2)
        score = self.BN2(score)
        score = self.lin3(score)
        # put the score in [0,1]
        score = torch.sigmoid(score)
        if self.weighted:
            score = score * G_torch.weight.view([-1,1])
        # sum the scores for each afterstates
        score_state = global_add_pool(score, batch).to(device)

        return score_state


class DQN(nn.Module):
    r"""Instead of computing a scrore for each state, compute the state-action values.
    The only change compared to ValueNet is in the end"""

    def __init__(self, dim_input, dim_embedding, dim_values, dim_hidden,
                 n_heads, n_att_layers, n_pool, K, alpha, p, weighted=False):
        super(DQN, self).__init__()

        self.weighted = weighted
        self.node_encoder = NodeEncoder(dim_input, n_heads, n_att_layers, dim_embedding,
                                        dim_values, dim_hidden, K, alpha, weighted)
        self.context_encoder = ContextEncoder(n_pool, dim_embedding, dim_hidden, weighted)
        # Score for each node
        if weighted:
            dim_context = dim_embedding * n_pool + 8
            first_dim = dim_context + dim_embedding + 2
        else:
            dim_context = dim_embedding * n_pool + 7
            first_dim = dim_context + dim_embedding + 1
        self.lin1 = nn.Linear(first_dim, dim_hidden)
        self.BN1 = BatchNorm(dim_hidden)
        self.lin2 = nn.Linear(dim_hidden, dim_embedding)
        self.BN2 = BatchNorm(dim_embedding)
        self.lin3 = nn.Linear(dim_embedding, 1)
        # dropout
        self.dropout = nn.Dropout(p=p)


    def forward(self, G_torch, n_nodes, Omegas, Phis, Lambdas, Omegas_norm, Phis_norm, Lambdas_norm, J, player,
                return_actions=False):
        """ Take a batch of states as input and returns the values of each state-action values.

                Returns:
                -------
                score: float tensor (size = nb tot of nodes in Batch x 1),
                       score of each possible state-action values"""

        G = self.node_encoder(G_torch, J)
        context = self.context_encoder(G, n_nodes, Omegas, Phis, Lambdas, Omegas_norm, Phis_norm, Lambdas_norm)
        # retrieve the data from G
        x, edge_index, batch = G.x, G.edge_index, G.batch
        x_score = torch.cat([x, context[batch]], 1)
        score = self.lin1(x_score)
        score = self.dropout(score)
        score = F.leaky_relu(score, 0.2)
        score = self.BN1(score)
        score = self.lin2(score)
        score = self.dropout(score)
        score = F.leaky_relu(score, 0.2)
        score = self.BN2(score)
        score = self.lin3(score)
        score = F.relu(score)
        #score_state = scatter(score, batch, dim=0, reduce='max')

        return score
