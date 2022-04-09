import torch
from torch_geometric.nn import GCN, GAT, GAE

from .discriminator import Discriminator


class Contrastive_Net(torch.nn.Module):
    def __init__(self, topo_in_channels, feat_in_channels, out_channels):
        super(Contrastive_Net, self).__init__()
        self.model_topo = GAE(GCN(topo_in_channels, out_channels, 2))
        self.model_feat = GAT(feat_in_channels, out_channels)
        self.disc = Discriminator(out_channels)

    def reset_parameters(self):
        self.model_feat.reset_parameters()
        self.model_topo.reset_parameters()

    def forward(self, topo_data, feat_data):
        topo_z = self.model_topo.encode(topo_data.x, topo_data.edge_index)
        feat_z = self.model_feat(feat_data.x, feat_data.edge_index)

        res = self.disc(topo_z, feat_z)
        # return topo_z, feat_z
        return res
