import sys
import sys
def read_file(filename):
    peptideID={}
    proteinID={}
    pep_prob_dict={}
    peptide2protein_dict={}
    protein2peptide_dict={}
    with open(filename) as f:
        ID = 0
        proID = 0
        for line in f:
            line=line.strip().split('\t')
            peptide=line[0]
            protein=line[1]
            probability=line[2]
            if peptide not in peptideID:
                peptideID[peptide]=ID
                ID+=1
            if protein not in proteinID:
                proteinID[protein]=proID
                proID+=1
            if peptideID[peptide] not in peptide2protein_dict:
                peptide2protein_dict[peptideID[peptide]]=[proteinID[protein]]
            else:
                peptide2protein_dict[peptideID[peptide]].append(proteinID[protein])
            if proteinID[protein] not in protein2peptide_dict:
                protein2peptide_dict[proteinID[protein]]=[peptideID[peptide]]
            else:
                protein2peptide_dict[proteinID[protein]].append(peptideID[peptide])
            pep_prob_dict[peptideID[peptide]]=probability

    proteinGroup={}
    for key, value in protein2peptide_dict.items():
        if tuple(value) not in proteinGroup:
            proteinGroup[tuple(value)] = [key]
        else:
            proteinGroup[tuple(value)].append(key)
    proteinGroupIDs={}
    proteinGroup2peptide = {}
    groupIDs=0
    for key, value in proteinGroup.items():
        proteinGroupIDs[groupIDs]=value
        proteinGroup2peptide[groupIDs]=list(key)
        groupIDs+=1

    print(len(proteinID))
    print(len(proteinGroup2peptide))

if __name__=="__main__":
    data=dict()
    prefix=sys.argv[1]
    with open(prefix+'.pep.txt') as f:
        for line_id,line in enumerate(f):
            if line_id<53:
                continue
            query=line.strip().split('\t')
            peptide=query[0].replace('[','')
            peptide=peptide.replace(']','')
            proteins=query[3]
            proteins=proteins.replace('{','')
            proteins=proteins.replace('}','')
            proteins=proteins.split(',')
            score=query[7]
            if peptide in data:
                if score > data[peptide][1]:
                    data[peptide][1]=score
                else:
                    continue
            else:
                data[peptide+'_'+query[1]]=[proteins,score]

    with open(prefix+'_sipros.identification_100%.txt','w') as f:
        for pep in data:
            for p in data[pep][0]:
                f.write(pep+'\t'+p+'\t'+data[pep][1]+'\n')
    print('Done')
