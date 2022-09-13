####################################################################################
#
#   Trains models from .pt files
#   
#   Requires:
#       Packages: pytorch, scikit-learn
#
####################################################################################

import glob, sys, os, time, json, copy
import torch, torch.nn, torch.utils.data, torch.optim
import sklearn.metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

TRAIN_PROP = 0.8    # Proportion of datasets to use for training
BATCH_SIZE = 64
LR = 0.01   # Learning rate
EPOCHS = 100
LOSS_FUNCT = torch.nn.BCELoss()
LOG_FILE = 'model_out.txt'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(LOG_FILE, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


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
    def __init__(self, in_feat, arch):
        super().__init__()
        self.config = arch
        if self.config.get('conv', None):
            self.conv = torch.nn.Conv1d(1, self.config['conv']['n_channels'], self.config['conv']['kernel'])
            conv_out = in_feat - self.config['conv']['kernel'] + 1

            if self.config['conv']['lin_conn'] == 'max':
                self.maxpool = torch.nn.MaxPool1d(conv_out)
                self.config['linear'] = [self.config['conv']['n_channels']] + self.config['linear']
            else:
                self.config['linear'] = [self.config['conv']['n_channels'] * conv_out] + self.config['linear']
        else:
            self.config['linear'] = [in_feat] + self.config['linear']
        # creates a net of linear layers using a given list of number of nodes per layer
        layers = [torch.nn.Linear(self.config['linear'][i], self.config['linear'][i+1]) for i in range(len(self.config['linear']) - 1)]
        
        if self.config.get('drop_prob', None):
            drop_layers = [torch.nn.Dropout(self.config['drop_prob']) for _ in range(len(layers) - 1)]
            fc_layers = [None] * (len(layers) + len(drop_layers))
            fc_layers[::2] = layers
            fc_layers[1::2] = drop_layers
        else:
            fc_layers = layers

        print(fc_layers)
        self.model = torch.nn.Sequential(*fc_layers)
        self.activation = torch.nn.Sigmoid()

    def forward(self, x):
        if self.config.get('conv', None):
            x = torch.unsqueeze(x, 1)
            x = self.conv(x)
            if self.config['conv']['lin_conn'] == 'max':
                x = self.maxpool(x)
            x = x.flatten(1)
        x = self.model(x)
        output = self.activation(x)
        return torch.squeeze(output)

def train_model(m):
    m.train()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = BATCH_SIZE, shuffle = True)
    optim = torch.optim.SGD(m.parameters(), lr = m.config.get('lr', LR))
    for epoch in range(m.config.get('epochs', EPOCHS)):
        running_loss = 0.0
        total = 0
        correct = 0

        for _, (data, labels) in enumerate(train_loader):
            data, labels = data.to(DEV), labels.to(DEV)
            optim.zero_grad()

            outputs = m(data.type(torch.float))
            loss = LOSS_FUNCT(outputs, labels.type(torch.float))
            loss.backward()
            optim.step()

            running_loss += loss.item()
            total += labels.size(0)
            predicted = torch.round(outputs).type(torch.int)
            correct += (predicted == labels.view(*predicted.size())).sum().item()

        if epoch % 10 == 9: # every ten epochs
            print(f"Epoch: {epoch + 1}\tAcc: {round(correct / total, 4)}\tLoss: {round(running_loss,4)}")


def test_model(m):
    m.eval()
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = BATCH_SIZE, shuffle = False)
    test_labels = torch.tensor([], dtype=torch.long)
    test_preds = torch.tensor([], dtype=torch.long)
    for _, (data, labels) in enumerate(test_loader):
        outputs = m(data.type(torch.float))
        predicted = torch.round(outputs).type(torch.int)

        test_labels = torch.cat((test_labels, labels))
        test_preds = torch.cat((test_preds, predicted))

    # prints confusion matrix stats.
    print(sklearn.metrics.classification_report(test_labels, test_preds, zero_division=0))


if __name__ == '__main__':
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    sys.stdout = Logger()
    os.makedirs('models', exist_ok = True)
    # To train multiple genes
    with open('model_arch.json', 'r') as f:
        model_archs = json.loads(f.read())
    models = []
    # get all gene files
    data_files = glob.glob(f".{os.sep}data{os.sep}*train.pt")
    for i, file in enumerate(data_files):
        gene = file.split(os.sep)[-1].split('.')[0]
        gene = '_'.join(gene.split('_')[:-1])
        print(f"Training for {gene}")
        
        # load file for given gene, calc and get split datasets
        tmp = torch.load(file)
        train_size = int(TRAIN_PROP * len(tmp['labels']))
        n_features = tmp['data'].size(1)
        
        models.append([])

        for j, arch in enumerate(model_archs):
            print(f"Model {j + 1}")
            train_set, test_set = torch.utils.data.random_split(cust_dataset(tmp['data'], tmp['labels']), [train_size, len(tmp['labels']) - train_size])
            
            models[i].append(cust_model(n_features, copy.deepcopy(arch)))
            print(models[i][j])

            models[i][j].to(DEV)
            train_model(models[i][j])
        
            # seperator for training and metrics
            print('\n')

            models[i][j].to('cpu')
            test_model(models[i][j])
            
            torch.save(models[i][j].state_dict(), f"models/{gene} Model_{j + 1}.pt")
            models[i][j] = None


