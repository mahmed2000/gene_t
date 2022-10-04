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
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy

TRAIN_PROP = 0.8    # Proportion of datasets to use for training
BATCH_SIZE = 64
LR = 0.01   # Learning rate
EPOCHS = 100
LOSS_FUNCT = torch.nn.MSELoss()
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

def bce_l1(x, y):
    loss = -1.0 * (y * torch.log(x, 1) + (1.0 - y) * torch.log(1.0 - x, 1))


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

        self.model = torch.nn.Sequential(*fc_layers)
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

        self.activation = torch.nn.Sigmoid()

    def forward(self, x):
        if self.config.get('conv', None):
            x = torch.unsqueeze(x, 1)
            x = self.conv(x)
            if self.config['conv']['lin_conn'] == 'max':
                x = self.maxpool(x)
            x = x.flatten(1)
        output = self.activation(self.model(x))
        return torch.squeeze(output)

def train_model(m):
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = m.config.get('batch', BATCH_SIZE), shuffle = True)
    val_loader = torch.utils.data.DataLoader(test_set, batch_size = m.config.get('batch', BATCH_SIZE), shuffle = False)
    optim = torch.optim.SGD(m.parameters(), lr = m.config.get('lr', LR))

    acc_log = []
    loss_log = []

    acc_log_val = []
    loss_log_val = []

    for epoch in range(m.config.get('epochs', EPOCHS)):
        running_loss = 0.0
        total = 0
        correct = 0
        
        m.train()
        for _, (data, labels) in enumerate(train_loader):
            data, labels = data.to(DEV), labels.to(DEV)
            optim.zero_grad()

            outputs = m(data.type(torch.float))
            loss = LOSS_FUNCT(outputs, labels.type(torch.float))
            loss.backward()
            optim.step()

            running_loss += loss.item() * data.size(0)
            total += labels.size(0)
            predicted = torch.round(outputs).type(torch.int)
            correct += (predicted == labels.view(*predicted.size())).sum().item()
        epoch_loss = running_loss / total
        acc_log.append(correct / total)
        loss_log.append(epoch_loss)

        if epoch % 10 == 9: # every ten epochs
            print(f"Epoch: {epoch + 1}\tAcc: {round(correct / total, 4)}\tLoss: {round(epoch_loss,4)}")

        m.eval()
        running_loss = 0.0
        total = 0
        correct = 0
        for _, (data, labels) in enumerate(val_loader):
            data, labels = data.to(DEV), labels.to(DEV)
            outputs = m(data.type(torch.float))
            loss = LOSS_FUNCT(outputs, labels.type(torch.float))

            running_loss += loss.item() * data.size(0)
            total += labels.size(0)
            predicted = torch.round(outputs).type(torch.int)
            correct += (predicted == labels.view(*predicted.size())).sum().item()
        epoch_loss = running_loss / total
        acc_log_val.append(correct / total)
        loss_log_val.append(epoch_loss)
        
        if epoch % 10 == 9:
            print(f"\t\tVal Acc: {round(correct / total, 4)}\tVal Loss: {round(epoch_loss,4)}")

    return acc_log, loss_log, acc_log_val, loss_log_val

def test_model(m):
    m.eval()
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = m.config.get('batch', BATCH_SIZE), shuffle = False)
    test_labels = torch.tensor([], dtype=torch.long)
    test_preds = torch.tensor([], dtype=torch.long)
    raw_out = torch.tensor([], dtype=torch.float)
    for _, (data, labels) in enumerate(test_loader):
        outputs = m(data.type(torch.float))
        predicted = torch.round(outputs).type(torch.int)

        test_labels = torch.cat((test_labels, labels))
        test_preds = torch.cat((test_preds, predicted))
        raw_out = torch.cat((raw_out, outputs))
    raw_out = raw_out.detach()
    # prints confusion matrix stats.
    print(sklearn.metrics.classification_report(test_labels, test_preds, zero_division=0))
    print(f"AUC-ROC: {sklearn.metrics.roc_auc_score(test_labels, raw_out)}")
    fpr, tpr, _ = sklearn.metrics.roc_curve(test_labels, raw_out)
    return fpr, tpr


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
        n_features = tmp['data'].size(1)
        
        models.append([])

        for j, arch in enumerate(model_archs):
            print(f"Model {j + 1}")
            if arch.get('kmer'):
                kmers = numpy.zeros((tmp['data'].size(0), 2 ** (2 * int(gene.split('_')[1]))), dtype=numpy.int32)
                data_numpy = tmp['data'].detach().numpy()
                for numpy_index, value in numpy.ndenumerate(data_numpy):
                    kmers[numpy_index[0], value] += 1
                tmp['data'] = torch.tensor(data_numpy)
            data_train, data_val, labels_train, labels_val = train_test_split(tmp['data'], tmp['labels'], test_size = 0.2, shuffle = True)
            scaler = MinMaxScaler().fit(data_train)
            if arch.get('pca'):
                pca = PCA().fit(scaler.transform(data_train))
                sum_var = 0
                for var_i, var_ratio in enumerate(pca.explained_variance_ratio_):
                    sum_var += var_ratio
                    if sum_var >= 0.95:
                        n_features = var_i + 1
                        break
                data_train, data_val = torch.tensor(pca.transform(scaler.transform(data_train))[:, :n_features]), torch.tensor(pca.transform(scaler.transform(data_val))[:, :n_features])
            else:
                pca = None
                n_features = tmp['data'].size(1)
                data_train, data_val = torch.tensor(scaler.transform(data_train)), torch.tensor(scaler.transform(data_val))
            train_set, test_set = cust_dataset(data_train, labels_train), cust_dataset(data_val, labels_val)
            
            models[i].append(cust_model(n_features, copy.deepcopy(arch)))
            print(models[i][j])

            models[i][j].to(DEV)
            acc_log, loss_log, acc_log_val, loss_log_val = train_model(models[i][j])
            fig, ax = plt.subplots(2)
            ax[0].plot(range(len(acc_log)), acc_log)
            ax[0].plot(range(len(acc_log_val)), acc_log_val)
            ax[1].plot(range(len(loss_log)), loss_log)
            ax[1].plot(range(len(loss_log_val)), loss_log_val)
            plt.savefig(f"models/{gene} Model_{j + 1} Train.png")
            plt.close(fig)
        
            # seperator for training and metrics
            print('\n')

            models[i][j].to('cpu')
            fpr, tpr = test_model(models[i][j])
            fig, ax = plt.subplots(1)
            ax.plot(fpr, tpr)
            plt.savefig(f"models/{gene} Model_{j + 1} ROC.png")
            plt.close(fig)
            
            torch.save({'model': models[i][j].state_dict(), 'scaler': scaler, 'pca': pca}, f"models/{gene} Model_{j + 1}.pt")
            models[i][j] = None


