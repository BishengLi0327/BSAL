import torch
from torch_geometric.nn import Node2Vec
from torch_geometric.transforms import KNNGraph
# from models.constrastive import Contrastive_Net


def add_flags_from_config(parser, config_dict):
    """
    Adds a flag (and default value) to an ArgumentParser for each parameter in a config
    """

    for param in config_dict:
        default, type, description = config_dict[param]
        parser.add_argument(f"--{param}", default=default, type=type, help=description)

    return parser


def construct_knn_graph(data):
    if (not data.pos) and (data.x is not None):
        data.pos = data.x
    else:
        raise ValueError('No data pos and data features!')
    k = int(data.num_edges / data.num_nodes) + 1
    trans = KNNGraph(k, loop=False, force_undirected=True)
    knn_graph = trans(data.clone())
    data.pos, knn_graph.pos, knn_graph.x = None, None, None
    return knn_graph


def train_node2vec_emb(data):
    print('=' * 50)
    print('Start train node2vec model on the knn graph.')
    model = Node2Vec(data.edge_index, embedding_dim=32, walk_length=10, context_size=5, walks_per_node=10,
                     num_negative_samples=1, p=1, q=1, sparse=False, num_nodes=data.num_nodes)
    loader = model.loader(batch_size=128, shuffle=True, num_workers=4)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=0.001)
    minimal_loss = 1e9
    patience = 0
    patience_threshold = 20
    for epoch in range(1, 201):
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw, neg_rw)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        loss = total_loss / len(loader)
        if loss < minimal_loss:
            minimal_loss = loss
            patience = 0
        else:
            patience += 1
        if patience >= patience_threshold:
            print('Early Stop.')
            break
        print("Epoch: {:02d}, loss: {:.4f}".format(epoch, loss))
    print('Finished training.')
    print('=' * 50)
    return model().detach()

