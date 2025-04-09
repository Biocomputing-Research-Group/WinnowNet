import pandas as pd
import shutil
import os
import json
import math
import time
import numpy as np
import sys

Neutron = 1.008665
Hydrogen = 1.007825
max_charge = 3
max_neutron = 2


def theory():
    # call functions from inference_process.py
    return


def merge_apl(dir):
    files = os.listdir(dir)
    scanmzs = []
    scanlist = []
    for file in files:
        with open(dir+file) as f:
            for line in f:
                s = line.strip()
                if s is '':
                    continue
                if s[0].isalpha() is True:
                    if 'header' in s:
                        scan = int(s.split()[3])
                        scanlist.append(scan)
    scanlist = list(set(scanlist))
    scanlist.sort()
    maxscan = max(scanlist)
    for i in range(0, maxscan + 1):
        scanmzs.append([])

    for file in files:
        with open(dir+file) as f:
            for line in f:
                mz = []
                s = line.strip()
                if s is '':
                    continue
                if s[0].isalpha() is True:
                    if 'header' in s:
                        scan = int(s.split()[3])
                else:
                    m = line.strip().split('\t')[0]
                    it = line.strip().split('\t')[1]
                    mz.append(float(m))
                    mz.append(float(it))
                    scanmzs[scan].append(mz)

    f = open('merge_all.txt', 'w')
    sortlist = []
    for scan in scanlist:
        f.write('scan=' + str(scan) + '\n')
        mz = sorted(scanmzs[scan], key=lambda mz: mz[0])
        for line in mz:
            f.write(str(line[0]) + ' ' + str(line[1]) + '\n')
        f.write('\n')
    f.close()


def MzDLoad(msfile):
    apldict = dict()
    msdict = dict()

    with open('merge_all.txt') as f1:
        aplArray = f1.read().strip().split('\n\n')
        for scanBlock in aplArray:
            mz = []
            scan = scanBlock.strip().split('\n')
            scannum = int(scan[0].replace('scan=', ''))
            for i in range(1, len(scan)):
                mz.append(scan[i].strip().split(' '))
            apldict[scannum] = mz

    f2 = open(msfile)
    flag = 0
    mzs = []
    ms_scan = 0
    count = 0
    for ms in f2:
        ms = ms.strip()
        if ms[0].isalpha() is True:
            if ms[0] == 'S':
                if flag > 0:
                    msdict[ms_scan] = mzs
                    count += 1
                    mzs = []
                ms_scan = int(ms.split()[1])
                flag += 1
            else:
                continue
        else:
            mzs.append(ms.split(' '))
    msdict[ms_scan] = mzs
    f2.close()
    return apldict, msdict


def compare(ms_mz, apl_mz, e):
    exist = False
    ms_mz = round(float(ms_mz), 3)
    apl_mz = round(float(apl_mz), 3)
    control = abs(ms_mz - apl_mz)
    if control < ms_mz * e:
        exist = True
    return exist


def BinarySearch(aplscan, l, r, x, e):
    while l <= r:
        mid = int(l + (r - l) / 2)
        if aplscan[mid][0] == x:
            return True
        elif aplscan[mid][0] < x:
            l = mid + 1
        else:
            r = mid - 1
    if l>len(aplscan)-1:
        exist = compare(x, aplscan[r][0], e)
    elif r<0:
        exist = compare(x, aplscan[l][0], e)
    else:
        exist = compare(x, aplscan[l][0], e) | compare(x, aplscan[r][0], e)
    return exist


def generate(apldict, msdict, outfile,fileid):
    f = open(outfile, 'w')
    for line in apldict:
        writeArray = []
        for x in range(max_charge + 1):
            writeArray.append([])
        if line in msdict:
            ms_scan = msdict[line]
        aplscan = apldict[line]
        aplscan = np.asarray(aplscan, dtype=float)
        for pair in ms_scan:
            exist = False
            mz = float(pair[0])
            for i in range(1, max_charge + 1):
                detectMz = mz * i - (i - 1) * Hydrogen
                exist= BinarySearch(aplscan, 0, len(aplscan) - 1, detectMz, 0.0001)
                if exist:
                    writeArray[i - 1].append(str(pair[0]) + ' ' + str(pair[1]))
                    break
                else:
                    for j in range(-max_neutron, max_neutron + 1):
                        detectMz = mz * i - (i - 1) * Hydrogen + j * Neutron
                        exist= BinarySearch(aplscan, 0, len(aplscan) - 1, detectMz, 0.0001)
                        if exist:
                            writeArray[i - 1].append(str(pair[0]) + ' ' + str(pair[1]))
                            break
                    if exist:
                        break

            if exist is False:
                writeArray[max_charge].append(str(pair[0]) + ' ' + str(pair[1]))

        f.write('scan='+str(fileid)+'_'+str(line) + '\n')
        for s in writeArray:
            for scan in s:
                f.write(scan + ' ')
            f.write('\n')
    f.close()


def exp(msfile,apldir,outfile):
    # 1. merge different apl files and remain unique scans
    #merge_apl()
    # 2. Load ms2 file and apl (after merge) file
    namearray=msfile.split('.')
    name=namearray[0]
    fileinfo=name.split('_')
    idx=int(fileinfo[-1])
    print('process: file '+msfile)
    merge_apl(apldir)
    apldict, msdict = MzDLoad(msfile)
    # 3. detect and generate
    generate(apldict, msdict,outfile,idx)



if __name__ == "__main__":
    msfile=sys.argv[1]
    apldir=sys.argv[2]
    outfile=sys.argv[3]
    start = time.time()
    exp(msfile,apldir,outfile)
    end = time.time()
    print(end - start)



