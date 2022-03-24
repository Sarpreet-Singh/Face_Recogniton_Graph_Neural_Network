import dgl
from dgl.data import DGLDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt



#Create DGL dataset
class FaceDataset(DGLDataset):

    def __init__(self):
        super().__init__(name = "DGL_Graphs_Faces")

    def process(self):

        graphs, labels = dgl.load_graphs('dgl_graphs-1.dgl')

        self.graphs = graphs
        self.labels = labels['labels']

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)

        
dataset = FaceDataset()

from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler

num_examples = len(dataset)
num_train = int(num_examples * 0.8)

print(f'The dataset is made of {num_examples} graphs ')
print(f'{num_train} grahs are used for training')


train_sampler = SubsetRandomSampler(torch.arange(num_train))
test_sampler = SubsetRandomSampler(torch.arange(num_train, num_examples))

train_dataloader = GraphDataLoader(dataset, sampler=train_sampler, batch_size=25, drop_last=False)
test_dataloader = GraphDataLoader(dataset, sampler=test_sampler, batch_size=4, drop_last=False)

from dgl.nn import GraphConv

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super().__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        g.ndata['node_features'] = h
        hg = dgl.mean_nodes(g, 'node_features')
        return self.classify(hg)





# Create the model with given dimensions
model = GCN(8,12,2)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

epoch_losses = []
for epoch in range(20):
    epoch_loss = 0

    for iter, (batched_graph, labels) in enumerate(train_dataloader):
        pred = model(batched_graph, batched_graph.ndata['node_features'].float())
        loss = F.cross_entropy(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
    epoch_loss /= (iter + 1)
    print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
    epoch_losses.append(epoch_loss)

num_correct = 0
num_tests = 0
for batched_graph, labels in test_dataloader:
    pred = model(batched_graph, batched_graph.ndata['node_features'].float())
    num_correct += (pred.argmax(1) == labels).sum().item()
    num_tests += len(labels)

print('Test accuracy:', num_correct / num_tests)

#net = GCN(10,16,2)
#print(net)

plt.title('cross entropy averaged over minibatches')
plt.plot(epoch_losses)
plt.show()

# model.eval()
# # Convert a list of tuples to two lists
# test_X, test_Y = map(list, zip(*test_sampler))
# test_bg = dgl.batch(test_X)
# test_Y = torch.tensor(test_Y).float().view(-1, 1)
# probs_Y = torch.softmax(model(test_bg), 1)
# sampled_Y = torch.multinomial(probs_Y, 1)
# argmax_Y = torch.max(probs_Y, 1)[1].view(-1, 1)
# print('Accuracy of sampled predictions on the test set: {:.4f}%'.format((test_Y == sampled_Y.float()).sum().item() / len(test_Y) * 100))
# print('Accuracy of argmax predictions on the test set: {:4f}%'.format((test_Y == argmax_Y.float()).sum().item() / len(test_Y) * 100))