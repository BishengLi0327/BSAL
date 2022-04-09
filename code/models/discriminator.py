import torch


class Discriminator(torch.nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.lin = torch.nn.Linear(n_h, n_h)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, torch.nn.Bilinear):
            torch.nn.init.xavier_normal_(m.weight.data)

    def forward(self, h_pl, h_mi):
        assert h_pl.shape == h_mi.shape

        logits = torch.mm(self.lin(h_pl), torch.transpose(h_mi, dim0=0, dim1=1))

        return logits
