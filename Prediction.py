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
import sys

pairmaxlength = 500
diffDa=1


def expToDict(f):
    msdict = dict()
    header = ['S', 'I', 'Z']
    scan = []
    charge0 = []
    charge1 = []
    charge2 = []
    charge3 = []
    for line_id, line in enumerate(f):
        if line_id == 0:
            scanid = str(int(line.strip().split('\t')[1]))
            scan = []
        else:
            if line[0] == 'S':
                scan.append(charge0)
                scan.append(charge1)
                scan.append(charge2)
                scan.append(charge3)
                msdict[scanid] = scan
                scan = []
                charge0 = []
                charge1 = []
                charge2 = []
                charge3 = []
                scanid = str(int(line.strip().split('\t')[1]))
            else:
                if line[0] not in header:
                    attrs=line.strip().split(' ')
                    pair = [float(attrs[0]), float(attrs[1])]
                    if attrs[2]=='1':
                        charge1.append(pair)
                    elif attrs[2]=='2':
                        charge2.append(pair)
                    elif attrs[2]=='3':
                        charge3.append(pair)
                    else:
                        charge0.append(pair)

    scan.append(charge0)
    scan.append(charge1)
    scan.append(charge2)
    scan.append(charge3)
    msdict[scanid] = scan
    f.close()
    return msdict


def theoryToDict(f):
    theory_dict = dict()
    scan=[]
    charge1=[]
    charge2=[]
    charge3=[]
    for line_id, line in enumerate(f):
        if line_id % 7 == 0:
            if len(charge1) != 0:
                scan.append(sorted(charge1))
                scan.append(sorted(charge2))
                scan.append(sorted(charge3))
                theory_dict[key] = scan
            key = line.strip()
            scan = []
            charge1=[]
            charge2=[]
            charge3=[]
        else:
            elements = line.strip().split(' ')
            if len(elements)==1:
                continue
            status=line_id%7
            if status in [1,4]:
                for eleid, ele in enumerate(elements):
                    if eleid == 0:
                        pair = [float(ele)]
                    else:
                        if eleid % 2 == 0:
                            pair = [float(ele)]
                        else:
                            pair.append(float(ele))
                            charge1.append(pair)
            elif status in [2,5]:
                for eleid, ele in enumerate(elements):
                    if eleid == 0:
                        pair = [float(ele)]
                    else:
                        if eleid % 2 == 0:
                            pair = [float(ele)]
                        else:
                            pair.append(float(ele))
                            charge2.append(pair)
            else:
                for eleid, ele in enumerate(elements):
                    if eleid == 0:
                        pair = [float(ele)]
                    else:
                        if eleid % 2 == 0:
                            pair = [float(ele)]
                        else:
                            pair.append(float(ele))
                            charge3.append(pair)
    scan.append(sorted(charge1))
    scan.append(sorted(charge2))
    scan.append(sorted(charge3))
    theory_dict[key] = scan

    f.close()
    return theory_dict


def LabelToDict(fp):
    sample = fp.read().strip().split('\n')
    label_dic = dict()
    for scan in sample:
        s = scan.strip().split('\t')
        idx = s[1]
        qvalue = float(s[2])
        if s[0] == 'True':
            label = 1

        else:
            label = 0
        label_dic[idx] = [1-qvalue, label]

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


def readData(exp_train, theoretical_train, psm_train):
    L = []
    Yweight = []

    for i in range(len(theoretical_train)):
        f = open(theoretical_train[i])
        D_theory = theoryToDict(f)
        f = open(exp_train[i])
        D_exp = expToDict(f)
        f = open(psm_train[i])
        D_Label = LabelToDict(f)

        for j in D_Label.keys():
            scanid = j.split('_')[-3]
            l = []
            l.append(D_exp[scanid])
            l.append(D_theory[j])
            L.append(l)
            Y = D_Label[j][1]
            weight = D_Label[j][0]
            Yweight.append([Y, weight])
        D_theory = dict()
        D_exp = dict()
        D_Label = dict()

    return L, Yweight


class DefineDataset(Data.Dataset):
    def __init__(self, X, yweight):
        self.X = X
        self.yweight = yweight

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        Xexp = self.X[idx][0]
        Xtheory = self.X[idx][1]
        y = self.yweight[idx][0]
        weight = self.yweight[idx][1]
        newXexp=[]
        newXtheory=Xtheory[0]
        for cid, chargearray in enumerate(Xexp):
            if cid==0:
                for peak in chargearray:
                    newXexp.append(peak)
            else:
                for peak in chargearray:
                    mass=peak[0]*cid
                    newXexp.append([mass,peak[1]])


        Xexp=sorted(newXexp)
        Xtheory=sorted(newXtheory)
        Xexp = np.asarray(Xexp, dtype=float)
        Xtheory = np.asarray(Xtheory, dtype=float)

        xFeatures=[]
        for mz in Xexp:
            for tmz in Xtheory:
                if abs(mz[0]-tmz[0])<diffDa:
                    xFeatures.append([mz[0]-tmz[0],mz[1],tmz[1]])

        xFeatures=np.asarray(pad_control_3d(xFeatures),dtype=float)

        transformer = MinMaxScaler()
        Norm = transformer.fit_transform(xFeatures)
        xFeatures[:, 1] = Norm[:, 1]
        xFeatures = xFeatures.transpose()
        xFeatures = torch.FloatTensor(xFeatures)

        return xFeatures, y, weight




class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, outputs1, outputs2, targets, weight_label):
        euclidean_distance = F.pairwise_distance(outputs1, outputs2, keepdim=True)
        loss_contrastive = torch.mean((1 - weight_label) * torch.pow(euclidean_distance, 2) + weight_label * torch.pow(
            torch.clamp(self.margin - euclidean_distance, min=0), 2))

        return loss_contrastive

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
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

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
        self.fstn = T_Net(k=64)

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        trans_feat = self.fstn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans_feat)
        x = x.transpose(2, 1)

        pointfeat=x

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        return x, trans, trans_feat



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.transform = Transform()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        x, matrix3x3, matrix64x64 = self.transform(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        output = self.fc3(x)
        return output

    #def forward(self, input1, input2):
    #    output1 = self.forward_once(input1)
    #    output2 = self.forward_once(input2)
    #    return self.fc(torch.cat((output1, output2), dim=1))

class SegNet(nn.Module):
    def __init__(self,k=2):
        super(SegNet,self).__init__()
        self.k=k
        self.transform = Transform()
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 32, 1)
        #self.conv4 = torch.nn.Conv1d(32, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc1 = nn.Linear(16000, 4000)
        self.fc2 = nn.Linear(4000,1000)
        self.fc3 = nn.Linear(1000, 250)
        self.fc = nn.Linear(500,2)

    def forward_once(self,x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.transform(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        #x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x=self.fc1(x)
        x=self.fc2(x)
        x=self.fc3(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return self.fc(torch.cat((output1, output2), dim=1))

class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()
        self.transform = Transform()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)

    def forward_once(self, x):
        x, matrix3x3, matrix64x64 = self.transform(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        output = self.fc3(x)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def evaluate(data, model, loss, device):
    # Evaluation, return accuracy and loss

    model.eval()  # set mode to evaluation to disable dropout
    data_loader = Data.DataLoader(data,batch_size=32)

    data_len = len(data)
    total_loss = 0.0
    y_true, y_pred = [], []

    for data1, label, weight in data_loader:
        data1, label, weight = Variable(data1), Variable(label), Variable(weight)
        data1, label, weight = data1.to(device),label.to(device), weight.to(device)

        output = model(data1)
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



def test_model(model, test_data, device):
    print("Testing...")
    model.eval()
    start_time = time.time()
    test_loader = Data.DataLoader(test_data,batch_size=32)

    y_true, y_pred, y_pred_prob = [], [], []
    for data1,label, weight in test_loader:
        y_true.extend(label.data)
        data1,label, weight = Variable(data1), Variable(label), Variable(weight)
        data1,label, weight = data1.to(device),label.to(device), weight.to(device)

        output = model(data1)
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
    return y_pred_prob




if __name__ == "__main__":
    exp_test = glob.glob('ms2/*ms2')
    psm_test = []
    for name in exp_test:
        psm_test.append(name.replace('ms2/', 'PSMs/').replace('.ms2', '.tsv'))
    theoretical_test = []
    for name in exp_test:
        theoretical_test.append(name.replace('ms2/', 'theoreticals_original/').replace('.ms2', '.txt'))
    start = time.time()
    X_test, yweight_test = readData(exp_test, theoretical_test, psm_test)
    X_test = X_test
    yweight_test = yweight_test
    end = time.time()
    print('loading data: ' + str(end - start))
    print("length of test data: " + str(len(X_test)))
    print('result of '+sys.argv[2])

    test_data = DefineDataset(X_test,yweight_test)
    device = torch.device("cuda")
    model = Net()
    model.cuda()
    model = nn.DataParallel(model)
    model.to(device)
    model.load_state_dict(torch.load(sys.argv[2], map_location=lambda storage, loc: storage))
    y_pred_prob=test_model(model, test_data, device)
    with open(sys.argv[1],'w') as f:
        for line in y_pred_prob:
            f.write(str(line))
            f.write('\n')

    print('done')
