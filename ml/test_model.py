import sys, torch, os, json, re
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
from model import cust_model, cust_dataset
import matplotlib.pyplot as plt

if __name__ == '__main__':
    model_file_path = sys.argv[1]
    res = re.search('(.*\\/)*(.*) Model_(\\d*).pt', model_file_path)
    gene, model_i = res.groups()[1:]
    
    with open('model_arch.json', 'r') as f:
        arch = json.loads(f.read())[int(model_i) - 1]
    
    test_data_path = f"data{os.sep}{gene}_test.pt"
    tmp = torch.load(test_data_path)
    model = cust_model(tmp['data'].size(1), arch)
    model_data = torch.load(model_file_path)
    model.load_state_dict(model_data['model'])
    model.eval()
    
    test_set = cust_dataset(torch.tensor(model_data['scaler'].transform(tmp['data'])), tmp['labels'])
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = 64, shuffle = False)
    test_labels = torch.tensor([], dtype=torch.long)
    test_preds = torch.tensor([], dtype=torch.long)
    raw_out = torch.tensor([], dtype=torch.float)

    for _, (data, labels) in enumerate(test_loader):
        outputs = model(data.type(torch.float))
        predicted = torch.round(outputs).type(torch.int)

        test_labels = torch.cat((test_labels, labels))
        test_preds = torch.cat((test_preds, predicted))
        raw_out = torch.cat((raw_out, outputs))
    raw_out = raw_out.detach()

    print(classification_report(test_labels, test_preds, zero_division=0))
    print(f"AUC-ROC: {roc_auc_score(test_labels, raw_out)}")
    fpr, tpr, _ = roc_curve(test_labels, raw_out)

    fig, ax = plt.subplots(1)
    ax.plot(fpr, tpr)
    plt.show()
