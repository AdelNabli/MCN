import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GlobalAttention, APPNP, global_add_pool, GATConv, BatchNorm

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

    def __init__(self, dim_input, n_heads, n_att_layers, dim_embedding, dim_values, dim_hidden, K, alpha, weighted):
        super(NodeEncoder, self).__init__()

        # if the graph is weighted
        self.weighted = weighted
        if weighted:
            # there are 4 features that are added to the input data
            first_dim = dim_input + 4
        else:
            # else, only 2 features are added
            first_dim = dim_input + 2
        self.n_att_layers = n_att_layers
        self.Lin1 = nn.Linear(first_dim, dim_embedding)
        self.attention_layers = nn.ModuleList(
            [
                AttentionLayer(n_heads, dim_embedding, dim_values, dim_hidden) for k in range(n_att_layers)
            ]
        )
        self.power = APPNP(K, alpha, bias=False).to(device)

    def forward(self, G_torch, J, saved_nodes, infected_nodes, size_connected):

        # retrieve the data
        x, edge_index, batch = G_torch.x, G_torch.edge_index, G_torch.batch
        # gather together the node features with J and size_connected
        h = torch.cat([x, J, size_connected], 1)
        # if we are considering weighted graphs
        if self.weighted:
            # add the weights and the normalized weights to the features to consider
            weights = G_torch.weight.view([-1,1])
            weights_sum = global_add_pool(weights, batch).to_device()
            weights_norm = weights / weights_sum[batch]
            h = torch.cat([h, weights, weights_norm], 1)
        # project the features into a dim_embedding vector space
        h = self.Lin1(h)
        # apply the attention layers
        for k in range(self.n_att_layers):
            h = self.attention_layers[k](h, edge_index)
        # apply the power layer
        h = self.power(h, edge_index)
        # re-add the information about the node's state
        h = torch.cat([h, size_connected, J, saved_nodes, infected_nodes], 1)
        # if we are considering weighted graphs
        if self.weighted:
            h = torch.cat([h, weights, weights_norm], 1)
        G_torch.x = h

        return G_torch


class ContextEncoder(nn.Module):

    def __init__(self, n_pool, dim_embedding, dim_hidden, weighted):
        super(ContextEncoder, self).__init__()

        if weighted:
            # there are 6 features that are added to the input data
            first_dim = dim_embedding + 6
        else:
            # else, only 4 features are added
            first_dim = dim_embedding + 4

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
                       the value of the graph"""

    def __init__(self, dim_input, dim_embedding, dim_values, dim_hidden,
                 n_heads, n_att_layers, n_pool, K, alpha, p, weighted=False):
        super(ValueNet, self).__init__()

        dim_context = dim_embedding * n_pool + 7
        self.node_encoder = NodeEncoder(dim_input, n_heads, n_att_layers, dim_embedding,
                                        dim_values, dim_hidden, K, alpha, weighted)
        self.context_encoder = ContextEncoder(n_pool, dim_embedding, dim_hidden, weighted)
        # Score for each node
        if weighted:
            first_dim = dim_context + dim_embedding + 6
        else:
            first_dim = dim_context + dim_embedding + 4
        self.lin1 = nn.Linear(first_dim, dim_hidden)
        self.BN1 = BatchNorm(dim_hidden)
        self.lin2 = nn.Linear(dim_hidden, dim_embedding)
        self.BN2 = BatchNorm(dim_embedding)
        self.lin3 = nn.Linear(dim_embedding, 1)
        # dropout
        self.dropout = nn.Dropout(p=p)


    def forward(self, G_torch, n_nodes, Omegas, Phis, Lambdas, Omegas_norm, Phis_norm, Lambdas_norm,
                J, saved_nodes, infected_nodes, size_connected):
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

        G = self.node_encoder(G_torch, J, saved_nodes, infected_nodes, size_connected)
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
        # sum the scores for each afterstates
        score_state = global_add_pool(score, batch).to(device)

        return score_state



class ValueNetOld(torch.nn.Module):
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
                   the value of the graph"""

    def __init__(self, input_dim, hidden_dim1, hidden_dim2, n_heads, K, alpha, p):
        super(ValueNetOld, self).__init__()
        r"""Initialize the Value Network

        Parameters:
        ----------
        input_dim: int,
                   the number of features in the Pytorch Geometric data
        hidden_dim1: int,
                     size of the first hidden layer in all of the
                     subparts of the neural net
        hidden_dim2: int,
                     size of the second hidden layer in all of the
                     subparts of the neural net
        n_heads: int,
                 number of heads in the Graph Attention Networks
                 used to create the nodes embedding
        K: int,
           power used in the APPNP part
        alpha: float (\in [0,1]),
               teleport factor of the APPNP part"""

        # Structural node embeddings
        self.lin1 = nn.Linear(input_dim + 2, hidden_dim2)
        self.Attention1 = AttentionLayer(n_heads, hidden_dim2, hidden_dim1, hidden_dim1)
        self.Attention2 = AttentionLayer(n_heads, hidden_dim2, hidden_dim1, hidden_dim1)
        self.Attention3 = AttentionLayer(n_heads, hidden_dim2, hidden_dim1, hidden_dim1)
        self.power = APPNP(K, alpha, bias=False).to(device)
        # Graph embedding
        self.pool1 = GlobalAttention(
            nn.Sequential(
                nn.Linear(hidden_dim2 + 3, hidden_dim1),
                nn.ReLU(),
                nn.Linear(hidden_dim1, 1),
            ),
            nn.Sequential(
                nn.Linear(hidden_dim2 + 3, hidden_dim1),
                nn.ReLU(),
                nn.Linear(hidden_dim1, hidden_dim2),
            ),
        )
        self.pool2 = GlobalAttention(
            nn.Sequential(
                nn.Linear(hidden_dim2 + 3, hidden_dim1),
                nn.ReLU(),
                nn.Linear(hidden_dim1, 1),
            ),
            nn.Sequential(
                nn.Linear(hidden_dim2 + 3, hidden_dim1),
                nn.ReLU(),
                nn.Linear(hidden_dim1, hidden_dim2),
            ),
        )
        self.pool3 = GlobalAttention(
            nn.Sequential(
                nn.Linear(hidden_dim2 + 3, hidden_dim1),
                nn.ReLU(),
                nn.Linear(hidden_dim1, 1),
            ),
            nn.Sequential(
                nn.Linear(hidden_dim2 + 3, hidden_dim1),
                nn.ReLU(),
                nn.Linear(hidden_dim1, hidden_dim2),
            ),
        )
        # Score for each node
        self.lin3 = nn.Linear(hidden_dim2 * 4 + 11, hidden_dim1)
        self.BN1 = BatchNorm(hidden_dim1)
        self.lin4 = nn.Linear(hidden_dim1, hidden_dim2)
        self.BN2 = BatchNorm(hidden_dim2)
        self.lin5 = nn.Linear(hidden_dim2, 1)
        # dropout
        self.dropout = nn.Dropout(p=p)

    def forward(self, G_torch, n_nodes, Omegas, Phis, Lambdas, Omegas_norm, Phis_norm, Lambdas_norm,
                J, saved_nodes, infected_nodes, size_connected):

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

        # retrieve the data
        x, edge_index, batch = G_torch.x, G_torch.edge_index, G_torch.batch

        # Structural node embedding:
        #    Concatenate the 5 features in x
        #    with the indicator of the nodes infected J
        #    and the size of the connected component
        #    each node belongs to
        x_struc_node = torch.cat([x, J, size_connected], 1)
        x_struc_node = self.lin1(x_struc_node)
        x_struc_node = self.Attention1(x_struc_node, edge_index)
        x_struc_node = self.Attention2(x_struc_node, edge_index)
        x_struc_node = self.Attention3(x_struc_node, edge_index)
        x_struc_node = self.power(x_struc_node, edge_index)

        # Graph embedding:
        #    Concatenate the structural node embedding
        #    with J and the one hot encodings
        #    of the nodes saved/infected currently
        x_g_embedding = torch.cat([x_struc_node, J, saved_nodes, infected_nodes], 1)
        g_1 = self.pool1(x_g_embedding, batch)
        g_2 = self.pool2(x_g_embedding, batch)
        g_3 = self.pool3(x_g_embedding, batch)
        g_embedding = torch.cat([g_1, g_2, g_3], 1)

        # Node score:
        #     Concatenate the structural node embedding
        #     with the graph embedding and the descriptors
        #     of the situation, i.e n_nodes, J, saved_nodes,
        #     infected_nodes, size_connected and the budgets
        x_score = torch.cat(
            [
                x_struc_node,
                g_embedding[batch],
                n_nodes[batch],
                J,
                saved_nodes,
                infected_nodes,
                size_connected,
                Omegas[batch],
                Phis[batch],
                Lambdas[batch],
                Omegas_norm[batch],
                Phis_norm[batch],
                Lambdas_norm[batch],
            ],
            1,
        )
        score = self.lin3(x_score)
        score = self.dropout(score)
        score = F.leaky_relu(score, 0.2)
        score = self.BN1(score)
        score = self.lin4(score)
        score = self.dropout(score)
        score = F.leaky_relu(score, 0.2)
        score = self.BN2(score)
        score = self.lin5(score)
        # put the score in [0,1]
        score = torch.sigmoid(score)
        # sum the scores for each afterstates
        score_state = global_add_pool(score, batch).to(device)

        return score_state
