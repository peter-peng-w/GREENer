import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATNET(torch.nn.Module):
    def __init__(self, in_dim, out_dim, head_num, dropout_rate):
        super(GATNET, self).__init__()

        self.gat_layer_1 = GATConv(in_dim, out_dim, heads=head_num, dropout=dropout_rate)
        # default, concat all attention head
        self.gat_layer_2 = GATConv(head_num*out_dim, out_dim, heads=1, concat=False, dropout=dropout_rate)

    def forward(self, x, edge_index, attention_flag=False):
        x_0 = F.dropout(x, p=0.6, training=self.training)

        if attention_flag:
            # e_1, e_2: edge attention weights (edge_index, attention_weights)
            x_1, e_1 = self.gat_layer_1(x_0, edge_index, return_attention_weights=True)
            x_1 = F.elu(x_1)
            x_1 = F.dropout(x_1, p=0.6, training=self.training)
            x_2, e_2 = self.gat_layer_2(x_1, edge_index, return_attention_weights=True)

            return x_2, e_1, e_2
        else:
            x_1 = self.gat_layer_1(x_0, edge_index)
            x_1 = F.elu(x_1)
            x_1 = F.dropout(x_1, p=0.6, training=self.training)
            x_2 = self.gat_layer_2(x_1, edge_index)

            return x_2
