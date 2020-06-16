import torch
from torch_geometric.nn import Node2Vec


def main_node2vec(data):
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_nodes = data.x.size(0)
    model = Node2Vec(num_nodes, embedding_dim=64,
                     walk_length=10, context_size=10, walks_per_node=10)
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    for i in range(100):
        model.train()
        optimizer.zero_grad()
        loss = model.loss(data.edge_index)
        loss.backward()
        optimizer.step()
    node_index = torch.tensor([i for i in range(num_nodes)]).to(device)
    return model.forward(node_index).cpu()
