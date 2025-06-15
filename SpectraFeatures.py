import numpy as np
import sys
import pickle
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import time
import multiprocessing as mp
from multiprocessing import Pool
from multiprocessing import Manager
import os
import subprocess
import sys
import getopt

pairmaxlength = 500
diffDa=1

AA_dict = {
    "G": 57.021464,
    "R": 156.101111,
    "V": 99.068414,
    "P": 97.052764,
    "S": 87.032028,
    "U": 150.95363,
    "L": 113.084064,
    "M": 131.040485,
    "Q": 128.058578,
    "N": 114.042927,
    "Y": 163.063329,
    "E": 129.042593,
    "C": 103.009185 + 57.0214637236,   #fixed modification
    "F": 147.068414,
    "I": 113.084064,
    "A": 71.037114,
    "T": 101.047679,
    "W": 186.079313,
    "H": 137.058912,
    "D": 115.026943,
    "K": 128.094963,
    "~": 15.99491
}

PROTON_MASS = 1.00727646688
H = 1.007825
O = 15.9949
N_TERMINUS = H
C_TERMINUS = O + H

class peptide:

    def __init__(self):
        self.identified_pep = ""
        self.qvalue = 0
        self.qn=0
        self.qnl=0
        self.num_missed_cleavages = 0
        self.theory_mass=0
        self.delta_mass=0
        self.peplen=0
        self.PSMId=''

class scan:

    def __init__(self):
        self.fidx = 0
        self.charge = 0
        self.scan_number = 0
        self.exp_mass=0
        self.pep_list = []

    def add_pep(self, pep):
        self.pep_list.append(pep)


    def get_features(self):
        pep_sorted_list = sorted(self.pep_list, key=lambda pep: pep.qvalue)
        if len(pep_sorted_list)==1:
            pep_sorted_list[0].qn=1
            pep_sorted_list[0].qnl=1
            pep_sorted_list[0].delta_mass=pep_sorted_list[0].theory_mass-self.exp_mass
        else:
            lth_qvalue = pep_sorted_list[-1].qvalue
            for i in range(len(pep_sorted_list) - 1):
                pep_sorted_list[i].qn = (pep_sorted_list[i+1].qvalue - pep_sorted_list[i].qvalue)/pep_sorted_list[i+1].qvalue
                pep_sorted_list[i].qnl = (lth_qvalue - pep_sorted_list[i].qvalue)/lth_qvalue
                pep_sorted_list[i].delta_mass=pep_sorted_list[i].theory_mass-self.exp_mass
            pep_sorted_list[i+1].qn = 1
            pep_sorted_list[i+1].qnl = 1
            pep_sorted_list[i+1].delta_mass=pep_sorted_list[i+1].theory_mass-self.exp_mass


def expToDict(f):
    msdict = dict()
    flag = 0
    mzs = []
    ms_scan = 0
    count = 0
    charge=0
    for ms in f:
        ms = ms.strip()
        if ms[0].isalpha() is True:
            if ms[0] == 'S':
                if flag > 0:
                    msdict[ms_scan].append(mzs)
                    count += 1
                    mzs = []
                    charge=0
                ms_scan = str(int(ms.split()[1]))
                mass = str(float(ms.split()[-1]))
                msdict[ms_scan] = [mass+'_'+str(charge)]
                flag += 1
            elif ms[0]=='Z':
                charge=int(ms.split()[1])
                mass=float(ms.split()[2])
                msdict[ms_scan][0] = str((mass-PROTON_MASS))+'_'+str(charge)
            else:
                continue
        else:
            mzs.append(ms.split(' ')[:2])
    msdict[ms_scan].append(mzs)
    f.close()
    return msdict


def theoryToDict(f):
    theory_dic = dict()
    scan = []
    for line_id, line in enumerate(f):
        line = line.strip()
        if line_id % 7 == 0:
            if len(scan) != 0:
                theory_dic[key] = sorted(scan)
            scan = []
            key = line
        else:
            junk = line.split(' ')

            if len(junk) > 1:
                for eachid,each in enumerate(junk):
                    if eachid%2==0:
                        x=[float(each)]
                    else:
                        x.append(float(each))
                        scan.append(x)
            else:
                continue
    theory_dic[key] = sorted(scan)  
    return theory_dic

def read_tsv(f,psm_dict,Xexp):
    f1=open('idx.txt','w')
    f2=open('charge.txt','w')
    f3=open('peptide.txt','w')
    idxarray=[]
    chargearray=[]
    peptidearray=[]
    for line_id, line in enumerate(f):
        s=line.strip().split('\t')
        idx=s[0]
        PSMId=idx.strip().split('_')
        fileidx='_'.join(PSMId[:-3])
        scannum=PSMId[-3]
        charge=PSMId[-2]
        qvalue=1
        peptidestr=s[1].split('.')[1]
        pep = peptide()
        pep.PSMId='_'.join(PSMId)
        pep.qvalue=qvalue
        pep.identified_pep=peptidestr
        pep.num_missed_cleavages=peptidestr[:-1].count('K')+peptidestr[:-1].count('R')
        pep.theory_mass=sum([AA_dict[aa] for aa in peptidestr])+N_TERMINUS+C_TERMINUS
        pep.peplen=len(peptidestr)
        uniqueID=fileidx+'_'+scannum+'_'+charge
        if uniqueID in psm_dict.keys():
            psm_dict[uniqueID].add_pep(pep)
        else:
            one_scan=scan()
            one_scan.fidx=fileidx
            one_scan.scan_number=scannum
            one_scan.charge=charge
            one_scan.add_pep(pep)
            if Xexp[scannum][0][-1]=='0':
                mass=(float(Xexp[scannum][0].split('_')[0])-PROTON_MASS)*int(charge)
                one_scan.exp_mass=mass
            else:
                one_scan.exp_mass=float(Xexp[scannum][0].split('_')[0])
            psm_dict[uniqueID]=one_scan
        idxarray.append(idx)
        chargearray.append(charge)
        peptidearray.append(peptidestr)
    f1.write('\n'.join(idxarray))
    f1.close()
    f2.write('\n'.join(chargearray))
    f2.close()
    f3.write('\n'.join(peptidearray))
    f3.close()

def feature_dict(f_dict):
    D_feature=dict()
    for psm in f_dict:
        for pep in f_dict[psm].pep_list:
            D_feature[pep.PSMId]=[pep.qvalue,pep.qn,pep.qnl,pep.theory_mass+H,pep.delta_mass,abs(pep.delta_mass),pep.peplen,pep.num_missed_cleavages]
            if f_dict[psm].charge=='1':
                D_feature[pep.PSMId].extend([1,0,0])
            elif f_dict[psm].charge=='2':
                D_feature[pep.PSMId].extend([0,1,0])
            else:
                D_feature[pep.PSMId].extend([0,0,1])

    return D_feature


def pad_control_3d(data):
    data = sorted(data, key=lambda x: x[1], reverse=True)
    if len(data) > pairmaxlength:
        data = data[:pairmaxlength]
    else:
        while (len(data) < pairmaxlength):
            data.append([0, 0, 0])
    data = sorted(data, key=lambda x: x[0])
    return data

def IonExtract(Xexp,Xtheory,X_add_feature,key,return_dict):
    Xexp = np.asarray(Xexp[1:][0], dtype=float)
    Xtheory = np.asarray(Xtheory, dtype=float)

    xFeatures = []
    for mz in Xtheory:
        index=np.where(np.logical_and(Xexp[:,0]>mz[0]-diffDa,Xexp[:,0]<mz[0]+diffDa))[0]
        if len(index)>0:
            for idx in index:
                xFeatures.append([Xexp[idx][0]-mz[0], Xexp[idx][1], mz[1]])

    xFeatures = np.asarray(pad_control_3d(xFeatures), dtype=float)

    transformer = StandardScaler()
    Norm = transformer.fit_transform(xFeatures)
    xFeatures[:, 1] = Norm[:, 1]
    xFeatures[:, 2] = Norm[:, 2]
    xFeatures = xFeatures.transpose()
    return_dict[key]=[xFeatures,X_add_feature]

def IonExtract_Att(Xexp,Xtheory,X_add_feature,key,return_dict):
    Xexp = np.asarray(Xexp[1:][0], dtype=float)
    Xtheory = np.asarray(Xtheory, dtype=float)

    transformer = StandardScaler()
    Norm = transformer.fit_transform(Xexp)
    Xexp[:, 1] = Norm[:, 1]
    Norm = transformer.fit_transform(Xtheory)
    Xtheory[:, 1] = Norm[:, 1]
    return_dict[key]=[Xexp,Xtheory]



if __name__ == "__main__":
    argv=sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv, "hi:s:o:t:f:")
    except:
        print("Error Option, using -h for help information.")
        sys.exit(1)
    if len(opts)==0:
        print("\n\nUsage:\n")
        print("-i\t tab delimited file which contains the PSM condidates\n")
        print("-s\t ms2 format spectrum information\n")
        print("-o\t Spectrum features output file\n")
        print("-t\t Number of threads\n")
        print("-f\t Attention mode or CNN mode")
        sys.exit(1)
    start_time=time.time()
    exp_file=""
    tsv_file=""
    theoretical_file=""
    output_file=""
    mode=""
    for opt, arg in opts:
        if opt in ("-h"):
            print("\n\nUsage:\n")
            print("-i\t tab delimited file which contains the PSM condidates\n")
            print("-s\t ms2 format spectrum information\n")
            print("-o\t Spectrum features output file\n")
            print("-t\t Number of threads\n")
            print("-f\t Attention mode or CNN mode")
            sys.exit(1)
        elif opt in ("-i"):
            tsv_file=arg
        elif opt in ("-s"):
            exp_file=arg
        elif opt in ("-o"):
            output_file=arg
        elif opt in ("-t"):
            num_cpus=arg
        elif opt in ("-f"):
            mode=arg

    theoretical_file=exp_file.split('/')[-1].replace('.ms2','.txt')
    f = open(exp_file)
    D_exp = expToDict(f)
    f.close()
    print('Experimental spectra loaded!')
    f = open(tsv_file)
    psm_dict=dict()
    read_tsv(f,psm_dict,D_exp)
    f.close()
    for psm in psm_dict:
        psm_dict[psm].get_features()
    D_feature=feature_dict(psm_dict)
    print('Additional features loaded!')
    subprocess.run('./Sipros_OpenMP -i1 idx.txt -i2 charge.txt -i3 peptide.txt -i4 '+theoretical_file, shell=True, executable="/bin/bash")
    subprocess.run('rm idx.txt charge.txt peptide.txt test.SE.Spe2Pep.txt', shell=True, executable="/bin/bash")
    f = open(theoretical_file)
    D_theory = theoryToDict(f)
    f.close()
    print('Theoretical features loaded!')
    if mode=='cnn':
        manager = Manager()
        return_dict = manager.dict()
        processors = os.cpu_count()
        pool = Pool(processes=int(num_cpus))
        for key in D_theory:
            pool.apply_async(IonExtract, args=(D_exp[key.split('_')[-3]],D_theory[key],D_feature[key],key,return_dict))
        pool.close()
        pool.join()
    else:
        manager = Manager()
        return_dict = manager.dict()
        processors = os.cpu_count()
        pool = Pool(processes=int(num_cpus))
        for key in D_theory:
            pool.apply_async(IonExtract_Att, args=(D_exp[key.split('_')[-3]],D_theory[key],D_feature[key],key,return_dict))
        pool.close()
        pool.join()
    print('Features generated!')
    subprocess.run('rm '+theoretical_file, shell=True, executable="/bin/bash")
    return_dict=dict(return_dict)
    return_dict = { k: return_dict[k] for k in D_feature.keys() }
    with open(output_file,'wb') as f:
        pickle.dump(return_dict,f)
    print('time:'+str(time.time()-start_time))
