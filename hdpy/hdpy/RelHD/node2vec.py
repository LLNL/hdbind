import os.path as osp

import torch
from sklearn.manifold import TSNE

from torch_geometric.datasets import Planetoid, Reddit
from torch_geometric.nn import Node2Vec

# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/node2vec.py

def main():
    # dataset = Planetoid(root='./data/Cora', name='Cora')
    dataset = Planetoid(root='./data/CiteSeer', name='CiteSeer')
    # dataset = Planetoid(root='./data/Pubmed', name='Pubmed')
    #dataset = Reddit("./data/Reddit")

    print(dataset)
    data = dataset[0]
    #device = 'cpu'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Node2Vec(data.edge_index, embedding_dim=128, walk_length=20,
                     context_size=10, walks_per_node=10,
                     num_negative_samples=1, p=1, q=1, sparse=True).to(device)
                     #num_negative_samples=1, p=4, q=0.25, sparse=True).to(device)  # Reddit uses this one
    loader = model.loader(batch_size=128, shuffle=True, num_workers=4)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    def train():
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def test():
        model.eval()
        z = model()
        acc = model.test(z[data.train_mask], data.y[data.train_mask],
                         z[data.test_mask], data.y[data.test_mask],
                         max_iter=200)
        return acc

    from timeit import default_timer as timer
    start=timer()
    for epoch in range(1, 101):
        loss = train()
        print(epoch)
    end=timer()
    print("[STATS] Train time (100 epochs): {}".format(end - start))
    start=timer()
    acc=test()
    end=timer()
    print("[STATS] Test time: {}".format(end - start))
    print(f'Loss: {loss:.4f}, Acc: {acc:.4f}')
if __name__ == "__main__":
    main()
