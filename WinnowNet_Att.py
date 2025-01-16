import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
from sklearn import metrics
import numpy as np
import glob
import pickle
from SpectraFeatures import expToDict, theoryToDict

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
    return data

def pad_control_3d(data):
    data = sorted(data, key=lambda x: x[1], reverse=True)
    if len(data) > pairmaxlength:
        data = data[:pairmaxlength]
    else:
        while (len(data) < pairmaxlength):
            data.append([0, 0, 0])
    data = sorted(data, key=lambda x: x[0])
    return data


def readData(psms,exps,theoreticals):
    L = []
    Yweight = []

    for i in range(len(psms)):
        with open(psms[i]) as f:
            D_Label=LabelToDict(f)
        with open(exps[i],'r') as f:
            D_exp=expToDict(f)
        with open(theoreticals[i],'r') as f:
            D_theory=theoryToDict(f)

        for j in D_Label.keys():
            if D_Label[j][1]==1:
                if D_Label[j][0]>0.9:
                    scan=j.split('_')[-3]
                    L.append(D_exp[scan][1])
                    L.append(D_theory[j])
                    Y = D_Label[j][1]
                    weight = D_Label[j][0]
                    Yweight.append([Y, weight])
            else:
                if D_Label[j][0]<0.25:
                    scan=j.split('_')[-3]
                    L.append(D_exp[scan][1])
                    L.append(D_theory[j])
                    Y = D_Label[j][1]
                    weight = 0
                    Yweight.append([Y, weight])

        del D_exp
        del D_theory
        del D_Label

    return L, Yweight


class DefineDataset(Data.Dataset):
    def __init__(self, X, yweight):
        self.X = X
        self.yweight = yweight

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        xFeatures = self.X[idx]
        y = self.yweight[idx][0]
        weight = self.yweight[idx][1]
        xFeatures = torch.FloatTensor(xFeatures)
        addfeat=torch.FloatTensor([0,0,0,0,0,0,0,0,0,0,0])

        return xFeatures, addfeat,y, weight



class my_loss(torch.nn.Module):
    def __init__(self):
        super(my_loss, self).__init__()

    def forward(self, outputs, targets, weight_label):
        weight_label = weight_label.float()
        entropy = -F.log_softmax(outputs, dim=1)
        w_entropy = weight_label * entropy[:, 1] + (1 - weight_label) * entropy[:, 0]
        losssum = torch.sum(w_entropy)
        return losssum

class T_Net(nn.Module):
    def __init__(self, k):
        super(T_Net, self).__init__()
        self.k = k
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 256, 1)
        self.fc1 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 256)

        x = self.fc1(x)

        identity_matrix = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k * self.k).repeat(batchsize, 1)
        if x.is_cuda:
            identity_matrix = identity_matrix.cuda()
        x = x + identity_matrix
        x = x.view(-1, self.k, self.k)
        return x


class Transform(nn.Module):
    def __init__(self):
        super().__init__()
        self.stn = T_Net(k=3)

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 256, 1)
        self.conv3 = nn.Conv1d(256, 512, 1)
        self.conv4 = nn.Conv1d(512,2048,1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(2048)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 2048)

        return x



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.transform = Transform()
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(1024, 256)
        self.layer1=nn.Linear(11,256)
        self.layer2 = nn.Linear(512, 2)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x,addfeat):
        x = self.transform(x)
        x = F.relu(self.bn1(self.dropout(self.fc1(x))))
        #x = F.relu(self.bn1(self.fc1(x)))
        #x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        #xadd=F.relu(self.layer1(addfeat))
        #x=torch.cat((x,xadd),dim=1)
        output = self.layer2(x)
        return output

    #def forward(self, input1, input2):
    #    output1 = self.forward_once(input1)
    #    output2 = self.forward_once(input2)
    #    return self.fc(torch.cat((output1, output2), dim=1))

def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def evaluate(data, model, loss, device):
    # Evaluation, return accuracy and loss

    model.eval()  # set mode to evaluation to disable dropout
    data_loader = Data.DataLoader(data,batch_size=8)

    data_len = len(data)
    total_loss = 0.0
    y_true, y_pred = [], []

    for data1, addfeat,label, weight in data_loader:
        data1, addfeat,label, weight = Variable(data1), Variable(addfeat),Variable(label), Variable(weight)
        data1, addfeat,label, weight = data1.to(device),addfeat.to(device),label.to(device), weight.to(device)

        output = model(data1,addfeat)
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

    if y_pred.count(1) == 0:
        Pos_prec = 0
    elif y_pred.count(0) == 0:
        Neg_prec = 0
    else:
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


def train_model(X_train, X_val, X_test, yweight_train, yweight_val, yweight_test):
    LR = 1e-3
    train_data = DefineDataset(X_train, yweight_train)
    val_data = DefineDataset(X_val, yweight_val)
    test_data = DefineDataset(X_test, yweight_test)
    device = torch.device("cuda")
    model = Net()
    model.cuda()
    model = nn.DataParallel(model)
    model.to(device)
    #model.load_state_dict(torch.load('./models_80easy_20difficult/epoch49.pt', map_location=lambda storage, loc: storage))
    criterion = my_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    #model.load_state_dict(torch.load('cnn_pytorch.pt', map_location=lambda storage, loc: storage))
    #test_model(model, test_data, device)
    best_loss = 10000
    for epoch in range(50, 100):
        start_time = time.time()
        best_epoch_loss = 10000
        # load the training data in batch
        batch_count = 0
        model.train()
        train_loader = Data.DataLoader(train_data, batch_size=8, num_workers=8, shuffle=True, pin_memory=True)
        for x1_batch, addfeat_batch,y_batch, weight in train_loader:
            batch_count = batch_count + 1
            inputs, addfeat,targets, weight = Variable(x1_batch),Variable(addfeat_batch),Variable(y_batch), Variable(
                weight)
            inputs, addfeat,targets, weight = inputs.to(device), addfeat.to(device),targets.to(device), weight.to(
                device)
            optimizer.zero_grad()
            outputs = model(inputs,addfeat)  # forward computation
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
                torch.save(model.state_dict(), './models_80easy_20difficult/epoch' + str(epoch) + '.pt')

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), 'cnn_pytorch_att.pt')

        model.load_state_dict(
            torch.load('./models_80easy_20difficult/epoch' + str(epoch) + '.pt', map_location=lambda storage, loc: storage))
        train_acc, train_loss, train_Posprec, train_Negprec = evaluate(train_data, model, criterion, device)
        val_acc, val_loss, val_PosPrec, val_Negprec = evaluate(val_data, model, criterion, device)
        time_dif = get_time_dif(start_time)
        msg = "Epoch {0:3}, Train_loss: {1:>7.2}, Train_acc {2:>6.2%}, Train_Posprec {3:>6.2%}, Train_Negprec {" \
              "4:>6.2%}, " + "Val_loss: {5:>6.2}, Val_acc {6:>6.2%},Val_Posprec {7:6.2%}, Val_Negprec {8:6.2%} " \
                             "Time: {9} "
        print(msg.format(epoch + 1, train_loss, train_acc, train_Posprec, train_Negprec, val_loss, val_acc,
                         val_PosPrec, val_Negprec, time_dif))

    model.load_state_dict(
        torch.load('cnn_pytorch_att.pt', map_location=lambda storage, loc: storage))
    for i in range(0,50):
        test_model(model, test_data, device,'./models_80easy_20difficult/epoch'+str(i)+'.pt')


if __name__ == "__main__":
    psms = glob.glob('/media/fs0199/easystore/Protein/DeepFilterV2_local/240k_PSMs_new/*tsv')
    exps=[]
    theoreticals=[]
    for name in psms:
        exps.append(name.replace('240k_PSMs_new/', '240k_ms2/').replace('.tsv', '.ms2'))
    for name in psms:
        theoreticals.append(name.replace('240k_PSMs_new/', '240k_theoretical/').replace('.tsv', '.txt'))
    start = time.time()
    L, Yweight = readData(psms,exps,theoreticals)
    eight_easy_X_train, twenty_easy_X_train, eight_easy_yweight_train, twenty_easy_yweight_train= train_test_split(L[:5000], Yweight[:5000], test_size=0.2,random_state=10)

    features = glob.glob('/home/UNT/fs0199/WinnowNet_training/marine_data/spectra_features/*pkl')
    psm = []
    for name in features:
        psm.append(name.replace('spectra_features/', 'Label_PSMs/').replace('.pkl', '.tsv'))
    L, Yweight = readData(features, psm)
    X_train, X_test, yweight_train, yweight_test= train_test_split(L[:1215], Yweight[:1215], test_size=0.1,random_state=10)
    eight_difficult_X_train, twenty_difficult_X_train, eight_difficult_yweight_train, twenty_difficult_yweight_train= train_test_split(X_train, yweight_train, test_size=0.2,random_state=10)
    X_train=twenty_easy_X_train+eight_difficult_X_train
    yweight_train=twenty_easy_yweight_train+eight_difficult_yweight_train
    X_train, X_val, yweight_train, yweight_val = train_test_split(X_train, yweight_train, test_size=0.1,random_state=10)
    end = time.time()
    print('loading data: ' + str(end - start))
    print("length of training data: " + str(len(X_train)))
    print("length of validation data: " + str(len(X_val)))
    print("length of test data: " + str(len(X_test)))
    train_model(X_train, X_val, X_test, yweight_train, yweight_val, yweight_test)
    print('done')
