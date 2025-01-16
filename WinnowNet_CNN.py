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
import sys
import getopt

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

def readData(features_train, psm_train):
    L = []
    Yweight = []

    for i in range(len(features_train)):
        with open(features_train[i],'rb') as f:
            D_features=pickle.load(f)
        with open(psm_train[i]) as f:
            D_Label = LabelToDict(f)

        for j in D_Label.keys():
            L.append(D_features[j])
            Y = D_Label[j][1]
            weight = D_Label[j][0]
            Yweight.append([Y, weight])
        del D_features
        del D_Label

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

    model.load_state_dict(torch.load(
        'cnn_pytorch.pt', map_location=lambda storage, loc: storage))

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


def train_model(X_train, X_val, X_test, yweight_train, yweight_val, yweight_test,model_name):
    LR = 1e-3
    train_data = DefineDataset(X_train, yweight_train)
    val_data = DefineDataset(X_val, yweight_val)
    test_data = DefineDataset(X_test, yweight_test)
    device = torch.device("cuda")
    model = Net()
    model.cuda()
    model = nn.DataParallel(model)
    model.to(device)
    #model.load_state_dict(torch.load('./models_original/epoch21.pt', map_location=lambda storage, loc: storage))
    criterion = my_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    #model.load_state_dict(
    #    torch.load('cnn_pytorch.pt', map_location=lambda storage, loc: storage))
    #test_model(model, test_data, device)
    best_loss = 10000
    for epoch in range(0, 50):
        start_time = time.time()
        best_epoch_loss = 10000
        # load the training data in batch
        batch_count = 0
        model.train()
        train_loader = Data.DataLoader(train_data, batch_size=32, num_workers=8, shuffle=False, pin_memory=True)
        for x1_batch, y_batch, weight in train_loader:
            batch_count = batch_count + 1
            inputs, targets, weight = Variable(x1_batch),Variable(y_batch), Variable(
                weight)
            inputs, targets, weight = inputs.to(device), targets.to(device), weight.to(
                device)
            optimizer.zero_grad()
            outputs = model(inputs)  # forward computation
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
                torch.save(model.state_dict(), 'epoch' + str(epoch) + '.pt')

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), model_name)

        model.load_state_dict(
            torch.load('epoch' + str(epoch) + '.pt', map_location=lambda storage, loc: storage))
        train_acc, train_loss, train_Posprec, train_Negprec = evaluate(train_data, model, criterion, device)
        val_acc, val_loss, val_PosPrec, val_Negprec = evaluate(val_data, model, criterion, device)
        time_dif = get_time_dif(start_time)
        msg = "Epoch {0:3}, Train_loss: {1:>7.2}, Train_acc {2:>6.2%}, Train_Posprec {3:>6.2%}, Train_Negprec {" \
              "4:>6.2%}, " + "Val_loss: {5:>6.2}, Val_acc {6:>6.2%},Val_Posprec {7:6.2%}, Val_Negprec {8:6.2%} " \
                             "Time: {9} "
        print(msg.format(epoch + 1, train_loss, train_acc, train_Posprec, train_Negprec, val_loss, val_acc,
                         val_PosPrec, val_Negprec, time_dif))

    model.load_state_dict(torch.load('cnn_pytorch.pt', map_location=lambda storage, loc: storage))
    test_model(model, test_data, device)


if __name__ == "__main__":
    argv=sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv, "hi:s:o:t:")
    except:
        print("Error Option, using -h for help information.")
        sys.exit(1)
    if len(opts)==0:
        print("\n\nUsage:\n")
        print("-i\t Directories for spectra features\n")
        print("-m\t Output trained model name\n")
        print("-t\t Number of threads\n")
        sys.exit(1)
    start_time=time.time()
    exp_file=""
    tsv_file=""
    theoretical_file=""
    output_file=""
    for opt, arg in opts:
        if opt in ("-h"):
            print("\n\nUsage:\n")
            print("-i\t Directories for spectra features\n")
            print("-m\t ms2 format spectrum information\n")
            print("-t\t Number of threads\n")
            sys.exit(1)
        elif opt in ("-i"):
            input_directory=arg
        elif opt in ("-m"):
            model_name=arg
        elif opt in ("-t"):
            num_cpus=arg
    features_train = glob.glob(input_directory+'/*pkl')
    psm_train = []
    for name in features_train:
        psm_train.append(name.replace('spectra_features/', 'feature_PSMs/').replace('.pkl', '.tsv'))
    features_test = glob.glob('spectra_features/test/*pkl')
    psm_test = []
    for name in features_test:
        features_test.append(name.replace('ms2/', 'PSMs/').replace('.pkl', '.tsv'))
    start = time.time()
    L_train, Yweight_train = readData(features_train, psm_train)
    X_test, yweight_test = readData(features_test, psm_test)
    L_train = L_train
    Yweight_train = Yweight_train
    X_test = X_test
    yweight_test = yweight_test
    end = time.time()
    print('loading data: ' + str(end - start))
    X_train, X_val, yweight_train, yweight_val = train_test_split(L_train, Yweight_train, test_size=0.1,
                                                                  random_state=10)
    print("length of training data: " + str(len(X_train)))
    print("length of validation data: " + str(len(X_val)))
    print("length of test data: " + str(len(X_test)))
    train_model(X_train, X_val, X_test, yweight_train, yweight_val, yweight_test,model_name)
    print('done')
