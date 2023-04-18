#! /bin/bash
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 128
#SBATCH -t 2:00:00

python /work/08315/fs0199/ls6/PrositTraining/300k_training/SpectraFeatures.py ${1} ${2} ${3}
(base) login1.ls6(1008)$ cat SpectraFeatures.py 
import numpy as np
import sys
import pickle
from sklearn.preprocessing import MinMaxScaler
import time
import multiprocessing as mp
from multiprocessing import Pool
from multiprocessing import Manager
import os

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

def pad_control_3d(data):
    data = sorted(data, key=lambda x: x[1], reverse=True)
    if len(data) > pairmaxlength:
        data = data[:pairmaxlength]
    else:
        while (len(data) < pairmaxlength):
            data.append([0, 0, 0])
    data = sorted(data, key=lambda x: x[0])
    return data

def IonExtract(Xexp,Xtheory,key,return_dict):
    newXexp = []
    newXtheory = Xtheory[0]
    for cid, chargearray in enumerate(Xexp):
        if cid == 0:
            for peak in chargearray:
                newXexp.append(peak)
        else:
            for peak in chargearray:
                mass = peak[0] * cid
                newXexp.append([mass, peak[1]])

    Xexp = sorted(newXexp)
    Xtheory = sorted(newXtheory)
    Xexp = np.asarray(Xexp, dtype=float)
    Xtheory = np.asarray(Xtheory, dtype=float)

    xFeatures = []
    for mz in Xexp:
        for tmz in Xtheory:
            if abs(mz[0] - tmz[0]) < diffDa:
                xFeatures.append([mz[0] - tmz[0], mz[1], tmz[1]])

    xFeatures = np.asarray(pad_control_3d(xFeatures), dtype=float)

    transformer = MinMaxScaler()
    Norm = transformer.fit_transform(xFeatures)
    xFeatures[:, 1] = Norm[:, 1]
    xFeatures = xFeatures.transpose()
    return_dict[key]=xFeatures





if __name__ == "__main__":
    exp_file=sys.argv[1]
    theoretical_file=sys.argv[2]
    output_file=sys.argv[3]
    start_time=time.time()
    #exp_file='ms2/train/01625b_GE3-TUM_first_pool_21_01_01-2xIT_2xHCD-1h-R1.ms2'
    #theoretical_file='theoreticals_original/train/01625b_GE3-TUM_first_pool_21_01_01-2xIT_2xHCD-1h-R1.txt'
    #output_file='01625b_GE3-TUM_first_pool_21_01_01-2xIT_2xHCD-1h-R1.pkl'
    f = open(theoretical_file)
    D_theory = theoryToDict(f)
    f.close()
    f = open(exp_file)
    D_exp = expToDict(f)
    f.close()
    print('exp and theoretical loaded!')

    manager = Manager()
    return_dict = manager.dict()
    processors = os.cpu_count()
    pool = Pool(processes=6)
    for key in D_theory:
        pool.apply_async(IonExtract, args=(D_exp[key.split('_')[-3]],D_theory[key],key,return_dict))
    pool.close()
    pool.join()

    return_dict=dict(return_dict)
    with open(output_file,'wb') as f:
        pickle.dump(return_dict,f)
    print('time:'+str(time.time()-start_time))
