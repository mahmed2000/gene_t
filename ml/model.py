####################################################################################
#
#   Trains models from .pt files
#   
#   Requires:
#       Packages: pytorch, scikit-learn
#
####################################################################################

import glob, sys
import torch, torch.nn, torch.utils.data, torch.optim
import sklearn.metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

TRAIN_PROP = 0.8    # Proportion of datasets to use for training
BATCH_SIZE = 64
LR = 0.01   # Learning rate
EPOCHS = 100
POOLED_FEAT = 10
LOSS_FUNCT = torch.nn.CrossEntropyLoss()

# for data loading
class cust_dataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        super().__init__()
        self.data = data
        self.labels = labels

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index, :], self.labels[index]

# model generation
class cust_model(torch.nn.Module):
    def __init__(self, layer_nodes):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, POOLED_FEAT // 2, (805,4))
        self.maxpool1 = torch.nn.MaxPool2d((806,5), stride=1)
        self.conv2 = torch.nn.Conv2d(POOLED_FEAT // 2, POOLED_FEAT, (806, 5))
        self.maxpool2 = torch.nn.MaxPool2d((806,5), stride=1)
        # creates a net of linear layers using a given list of number of nodes per layer
        layers = [torch.nn.Linear(layer_nodes[i], layer_nodes[i+1]) for i in range(len(layer_nodes) - 1)]
        self.lins = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        output = self.lins(x.squeeze())
        return output

def train_model(m):
    m.train()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = BATCH_SIZE, shuffle = True)
    optim = torch.optim.SGD(m.parameters(), lr = LR)
    for epoch in range(EPOCHS):
        running_loss = 0.0
        total = 0
        correct = 0

        for _, (data, labels) in enumerate(train_loader):
            optim.zero_grad()

            outputs = m(data.type(torch.float))
            loss = LOSS_FUNCT(outputs, labels)
            loss.backward()
            optim.step()

            running_loss += loss.item()
            total += labels.size(0)
            predicted = torch.topk(outputs, 1).indices
            correct += (predicted == labels.view(*predicted.size())).sum().item()

        if epoch % 10 == 9: # every ten epochs
            print(f"Epoch: {epoch}\tAcc: {round(correct / total, 4)}\tLoss: {round(running_loss,4)}")


def test_model(m):
    m.eval()
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = BATCH_SIZE, shuffle = False)
    test_labels = torch.tensor([], dtype=torch.long)
    test_preds = torch.tensor([], dtype=torch.long)
    for _, (data, labels) in enumerate(test_loader):
        outputs = m(data.type(torch.float))
        predicted = torch.topk(outputs, 1).indices

        test_labels = torch.cat((test_labels, labels))
        test_preds = torch.cat((test_preds, predicted))

    # prints confusion matrix stats.
    print(sklearn.metrics.classification_report(test_labels, test_preds, zero_division=0))


if __name__ == '__main__':
    # To train multiple genes
    models = []
    # get all gene files
    data_files = glob.glob('./data/*.pt')
    for i, file in enumerate(data_files):
        gene = file.split('/')[-1].split('.')[0]
        print(f"Training for {gene}")
        
        # load file for given gene, calc and get split datasets
        tmp = torch.load(file)
        tmp['data'] = tmp['data'].unsqueeze(1)
        train_size = int(TRAIN_PROP * len(tmp['labels']))
        train_set, test_set = torch.utils.data.random_split(cust_dataset(tmp['data'], tmp['labels']), [train_size, len(tmp['labels']) - train_size])
        models.append(cust_model([POOLED_FEAT, 256, 64, 2]))
        train_model(models[i])
        
        # seperator for training and metrics
        print('\n')

        test_model(models[i])


