import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import time
import sys
import getopt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
from sklearn import metrics
import numpy as np
import glob
import pickle
from components.encoders import MassEncoder, PeakEncoder, PositionalEncoder

pairmaxlength=500
threshold=0.9
def LabelToDict(fp):
    sample = fp.read().strip().split('\n')
    label_dic = dict()
    for scan in sample:
        s = scan.strip().split('\t')
        idx = s[1]
        qvalue = float(s[2])
        if s[0] == 'True':
            label = 1
            label_dic[idx] = [1 - qvalue, label]
        else:
            label = 0
            label_dic[idx] = [(1-qvalue)/2, label]

    fp.close()
    return label_dic


def pad_control(data):
    data = sorted(data, key=lambda x: x[1], reverse=True)
    if len(data) > pairmaxlength:
        data = data[:pairmaxlength]
    else:
        while (len(data) < pairmaxlength):
            data.append([0, 0])
    data = sorted(data, key=lambda x: x[0])
    return np.asarray(data,dtype=float)


def readData(psms, features):
    L = []
    Yweight = []
    positive=0
    negative=0

    for i in range(len(psms)):
        with open(psms[i]) as f:
            D_Label=LabelToDict(f)
        with open(features[i],'rb') as f:
            D_features=pickle.load(f)

        for j in D_Label.keys():
            if D_Label[j][1]==1:
                if D_Label[j][0]>threshold:
                    L.append(D_features[j])
                    Y = D_Label[j][1]
                    weight = D_Label[j][0]
                    positive+=1
                    Yweight.append([Y, weight])
            else:
                L.append(D_features[j])
                Y = D_Label[j][1]
                weight = D_Label[j][0]
                negative+=1
                Yweight.append([Y, weight])

        del D_features
    print(positive)
    print(negative)
    return L, Yweight


class DefineDataset(Data.Dataset):
    def __init__(self, X, yweight):
        self.X = X
        self.yweight = yweight

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        xspectra1 = pad_control(self.X[idx][0])
        xspectra2 = pad_control(self.X[idx][1])
        y = self.yweight[idx][0]
        weight = self.yweight[idx][1]
        xspectra1 = torch.FloatTensor(xspectra1)
        xspectra2 = torch.FloatTensor(xspectra2)

        return xspectra1, xspectra2, y, weight



class my_loss(torch.nn.Module):
    def __init__(self):
        super(my_loss, self).__init__()

    def forward(self, outputs, targets, weight_label):
        weight_label = weight_label.float()
        entropy = -F.log_softmax(outputs, dim=1)
        w_entropy = weight_label * entropy[:, 1] + (1 - weight_label) * entropy[:, 0]
        losssum = torch.sum(w_entropy)
        return losssum


class MS2Encoder(nn.Module):
    def __init__(
        self,
        dim_model: int,
        dim_intensity: int,
        n_heads: int,
        dim_feedforward: int,
        n_layers: int,
        dropout: float = 0.1,
        max_len: int = 150
    ):
        super().__init__()
        self.peak_encoder = PeakEncoder(
            dim_model=dim_model,
            dim_intensity=dim_intensity,
            min_wavelength=0.001,
            max_wavelength=7000,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim_model))
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len+1, dim_model))
        layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, spectra: torch.Tensor, mask: torch.Tensor=None):
        B, P, _ = spectra.shape
        x = self.peak_encoder(spectra) 
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_emb[:, : P+1, :]
        if mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=mask.device)
            attn_mask = torch.cat([cls_mask, mask == 0], dim=1)  # True=>ignore
        else:
            attn_mask = None
        out = self.transformer(x, src_key_padding_mask=attn_mask)
        spec_rep = out[:, 0, :]
        return spec_rep

class DualPeakClassifier(nn.Module):
    def __init__(
        self,
        dim_model: int = 256,
        dim_intensity: int = 128,
        n_heads: int = 4,
        dim_feedforward: int = 512,
        n_layers: int = 4,
        num_classes: int = 2,
        dropout: float = 0.2,
        max_len: int = 500,
    ):
        super().__init__()
        self.encoder1 = MS2Encoder(
            dim_model, dim_intensity, n_heads, dim_feedforward, n_layers, dropout, max_len
        )
        self.encoder2 = MS2Encoder(
            dim_model, dim_intensity, n_heads, dim_feedforward, n_layers, dropout, max_len
        )
        self.classifier = nn.Linear(2 * dim_model, num_classes)

    def forward(
        self,
        spectra1: torch.Tensor,
        spectra2: torch.Tensor,
        mask1: torch.Tensor = None,
        mask2: torch.Tensor = None,
    ):
        rep1 = self.encoder1(spectra1, mask1)
        rep2 = self.encoder2(spectra2, mask2)
        joint = torch.cat([rep1, rep2], dim=-1)
        outputs = self.classifier(joint)
        return outputs

def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def evaluate(data, model, loss, device):

    model.eval()
    data_loader = Data.DataLoader(data,batch_size=8)

    data_len = len(data)
    total_loss = 0.0
    y_true, y_pred = [], []

    for input1, input2, label, weight in data_loader:
        input1, input2, label, weight = Variable(input1), Variable(input2),Variable(label), Variable(weight)
        input1, input2, label, weight = input1.to(device),input2.to(device),label.to(device), weight.to(device)

        output = model(input1,input2)
        losses = loss(output, label, weight)

        total_loss += losses.data.item()
        pred = torch.max(output.data, dim=1)[1].cpu().numpy().tolist()
        y_pred.extend(pred)
        y_true.extend(label.data.cpu().numpy().tolist())

    acc = (np.array(y_true) == np.array(y_pred)).sum()
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    Pos_prec = 0
    Neg_prec = 0

    for idx in range(len(y_pred)):
        if y_pred[idx] == 1:
            if y_true[idx] == 1:
                TP += 1
            else:
                FP += 1
        else:
            if y_true[idx] == 1:
                TN += 1
            else:
                FN += 1

    if y_pred.count(1) == 0:
        Pos_prec = 0
        Neg_prec = FN / (TN + FN)
    elif y_pred.count(0) == 0:
        Pos_prec = TP / (TP + FP)
        Neg_prec = 0
    else:
        Pos_prec = TP / (TP + FP)
        Neg_prec = FN / (TN + FN)

    return acc / data_len, total_loss / data_len, Pos_prec, Neg_prec



def test_model(model, test_data, device,model_str):
    print("Testing...")
    model.eval()
    start_time = time.time()
    test_loader = Data.DataLoader(test_data,batch_size=8)

    model.load_state_dict(torch.load(model_str, map_location=lambda storage, loc: storage))

    y_true, y_pred, y_pred_prob = [], [], []
    for data1,addfeat,label, weight in test_loader:
        y_true.extend(label.data)
        data1,addfeat,label, weight = Variable(data1), Variable(addfeat),Variable(label), Variable(weight)
        data1,addfeat,label, weight = data1.to(device),Variable(addfeat),label.to(device), weight.to(device)

        output = model(data1,addfeat)
        pred = torch.max(output.data, dim=1)[1].cpu().numpy().tolist()
        pred_prob = torch.softmax(output.data, dim=1).cpu()
        pred_prob = np.asarray(pred_prob, dtype=float)
        y_pred.extend(pred)
        y_pred_prob.extend(pred_prob[:, 1].tolist())

    test_acc = metrics.accuracy_score(y_true, y_pred)
    test_f1 = metrics.f1_score(y_true, y_pred, average='macro')
    print(
        "Test accuracy: {0:>7.2%}, F1-Score: {1:>7.2%}".format(test_acc, test_f1))

    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(
        y_true, y_pred, target_names=['T', 'D']))

    print('Confusion Matrix...')
    cm = metrics.confusion_matrix(y_true, y_pred)
    print(cm)

    print("Time usage:", get_time_dif(start_time))


def train_model(X_train, X_val, X_test, yweight_train, yweight_val, yweight_test,model_name, pretrained_model):
    LR = 1e-3
    train_data = DefineDataset(X_train, yweight_train)
    val_data = DefineDataset(X_val, yweight_val)
    test_data = DefineDataset(X_test, yweight_test)
    device = torch.device("cuda")
    model = DualPeakClassifier(dim_model=256,n_heads=4,dim_feedforward=512,n_layers=4,dim_intensity=64,num_classes=2,dropout=0.2,max_len=500)
    model.cuda()
    model = nn.DataParallel(model)
    model.to(device)
    if len(pretrained_model)>0:
        print("loading pretrained_model")
        model.load_state_dict(torch.load(pretrained_model, map_location=lambda storage, loc: storage))
    criterion = my_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    #model.load_state_dict(torch.load('cnn_pytorch.pt', map_location=lambda storage, loc: storage))
    #test_model(model, test_data, device)
    best_loss = 10000
    for epoch in range(0, 50):
        start_time = time.time()
        best_epoch_loss = 10000
        # load the training data in batch
        batch_count = 0
        model.train()
        train_loader = Data.DataLoader(train_data, batch_size=32, num_workers=8, shuffle=True, pin_memory=True)
        for input1, input2, y_batch, weight in train_loader:
            batch_count = batch_count + 1
            input1, input2,targets, weight = Variable(input1),Variable(input2),Variable(y_batch), Variable(weight)
            input1, input2,targets, weight = input1.to(device), input2.to(device),targets.to(device), weight.to(device)
            optimizer.zero_grad()
            outputs = model(input1,input2)  # forward computation
            loss = criterion(outputs, targets, weight)
            # backward propagation and update parameters
            loss.backward()
            optimizer.step()

            # evaluate on both training and test dataset

            # train_acc, train_loss, train_Posprec, train_Negprec = evaluate(train_data, model, criterion, device)
            val_acc, val_loss, val_PosPrec, val_Negprec = evaluate(val_data, model, criterion, device)
            # print("val acc: "+str(val_acc)+" val loss: "+str(val_loss)+" val negprecision: "+str(val_Negprec)+" val posprec: "+str(val_PosPrec))
            if val_loss < best_epoch_loss:
                # store the best result
                best_epoch_loss = val_loss
                torch.save(model.state_dict(), 'checkpoints/epoch' + str(epoch) + '.pt')

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), model_name)

        model.load_state_dict(
            torch.load('checkpoints/epoch' + str(epoch) + '.pt', map_location=lambda storage, loc: storage))
        train_acc, train_loss, train_Posprec, train_Negprec = evaluate(train_data, model, criterion, device)
        val_acc, val_loss, val_PosPrec, val_Negprec = evaluate(val_data, model, criterion, device)
        time_dif = get_time_dif(start_time)
        msg = "Epoch {0:3}, Train_loss: {1:>7.2}, Train_acc {2:>6.2%}, Train_Posprec {3:>6.2%}, Train_Negprec {" \
              "4:>6.2%}, " + "Val_loss: {5:>6.2}, Val_acc {6:>6.2%},Val_Posprec {7:6.2%}, Val_Negprec {8:6.2%} " \
                             "Time: {9} "
        print(msg.format(epoch + 1, train_loss, train_acc, train_Posprec, train_Negprec, val_loss, val_acc,
                         val_PosPrec, val_Negprec, time_dif))

    model.load_state_dict(
        torch.load(model_name, map_location=lambda storage, loc: storage))
    for i in range(0,50):
        test_model(model, test_data, device,'checkpoints/epoch'+str(i)+'.pt')


if __name__ == "__main__":
    argv=sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv, "hi:m:p:t:")
    except:
        print("Error Option, using -h for help information.")
        sys.exit(1)
    if len(opts)==0:
        print("\n\nUsage:\n")
        print("-i\t Directories for spectra features and Label\n")
        print("-m\t Pre-trained model name\n")
        print("-p\t Output trained model name\n")
        sys.exit(1)
        start_time=time.time()
    input_directory=""
    model_name=""
    pretrained_model=""
    for opt, arg in opts:
        if opt in ("-h"):
            print("\n\nUsage:\n")
            print("-i\t Directories for spectra features\n")
            print("-m\t ms2 format spectrum information\n")
            print("-p\t Output trained model name\n")
            sys.exit(1)
        elif opt in ("-i"):
            input_directory=arg
        elif opt in ("-m"):
            model_name=arg
        elif opt in ("-p"):
            pretrained_model=arg
    psms = glob.glob(input_directory+'/*tsv')
    features = glob.glob(input_directory+'/*pkl')
    start = time.time()
    L, Yweight = readData(psms,features)
    X_train, X_test, yweight_train, yweight_test= train_test_split(L, Yweight, test_size=0.1,random_state=10)
    X_train, X_val, yweight_train, yweight_val = train_test_split(X_train, yweight_train, test_size=0.1,random_state=10)
    end = time.time()
    print('loading data: ' + str(end - start))
    print("length of training data: " + str(len(X_train)))
    print("length of validation data: " + str(len(X_val)))
    print("length of test data: " + str(len(X_test)))
    train_model(X_train, X_val, X_test, yweight_train, yweight_val, yweight_test, model_name, pretrained_model)
    print('done')
