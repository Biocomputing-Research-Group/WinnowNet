import xml.etree.ElementTree as ET
import sys
import math
W_proton=1.00724
W_hydrogen=1.00784

class peptide:

    def __init__(self):
        self.identified_pep = ""
        self.charge = None
        self.calculated_mass = None
        self.measured_mass = None
        self.EValue = None
        self.protein_list = []
        self.mz_diff = 0
        self.local_rank = 0


class scan:

    def __init__(self):
        self.fidx=0
        self.scan_number = 0
        self.pep_list = []

    def add_pep(self, pep):
        self.pep_list.append(pep)

    def get_percolator_output(self):

        self.find_diff()

        # 'SpecId\tLabel\tScanNr\tExpMass\tCalcMass\tEValue\tMass\tdeltamz\tPepLen\tdM\tabsdM\tCharge1\tCharge2\tCharge3\tCharge4\tCharge5\tenzInt\tPeptide\tProtein'


        results = []

        for pep in self.pep_list:
            l = []
            l.append(str(self.fidx)+'_'+str(self.scan_number)+'_'+str(pep.charge)+'_'+str(pep.local_rank))
            proteinstring = '\t'.join(pep.protein_list)
            Label='-1'
            for p in pep.protein_list:
                if 'Rev_' not in p:
                    Label='1'
                    break
            l.append(Label)
            l.append(self.scan_number)
            l.append(str(pep.measured_mass))
            l.append(str(pep.calculated_mass))
            l.append(str(pep.EValue))
            l.append(str(float(pep.measured_mass)+W_hydrogen))
            l.append(str(pep.mz_diff))
            l.append(str(len(pep.identified_pep.split('.')[1])))
            dM, absdM = float(pep.calculated_mass) - float(pep.measured_mass), abs((float(pep.calculated_mass) - float(pep.measured_mass)))
            l.append(str(dM))
            l.append(str(absdM))
            c = int(pep.charge)
            for i in range(1, 5):  # Charge 1-4
                if c == i:
                    l.append("1")
                else:
                    l.append("0")
            if c > 5:  # Charge5
                l.append("1")
            else:
                l.append("0")
            l.append(str(get_num_missed_cleavages(pep.identified_pep.replace('+16','').strip().split('.')[1])))
            l.append(pep.identified_pep)
            l.append('\t'.join(pep.protein_list))
            results.append(l)

        return results

    def find_diff(self):
        pep_sorted_list = sorted(self.pep_list, key=lambda pep: -float(pep.EValue))
        for i in range(len(pep_sorted_list) - 1):
            pep_sorted_list[i].mz_diff = float(pep_sorted_list[i].EValue) - float(pep_sorted_list[i + 1].EValue)


def get_num_missed_cleavages(s):
    return s.count('K')+s.count('R')-s.count('KP')


def pep2proMap(input_file_str,Map):
    tree = ET.parse(input_file_str)
    root = tree.getroot()
    SequenceCollection=root.find('{http://psidev.info/psi/pi/mzIdentML/1.1}SequenceCollection')
    for DBSequence in SequenceCollection.iter('{http://psidev.info/psi/pi/mzIdentML/1.1}DBSequence'):
        Map[DBSequence.attrib['id']]=DBSequence.attrib['accession']

def prepostDBref(input_file_str,prepostMap):
    tree = ET.parse(input_file_str)
    root = tree.getroot()
    SequenceCollection=root.find('{http://psidev.info/psi/pi/mzIdentML/1.1}SequenceCollection')
    for PeptideEvidence in SequenceCollection.iter('{http://psidev.info/psi/pi/mzIdentML/1.1}PeptideEvidence'):
        prepostMap[PeptideEvidence.attrib['id']]=[PeptideEvidence.attrib['pre'],PeptideEvidence.attrib['post'],PeptideEvidence.attrib['dBSequence_ref']]

def read_one_pep_xml_msgf(input_file_str, psm_list,fidx):
    Map=dict()
    prepostMap=dict()
    pep2proMap(input_file_str,Map)
    prepostDBref(input_file_str,prepostMap)
    tree = ET.parse(input_file_str)
    root = tree.getroot()
    DataCollection=root.find('{http://psidev.info/psi/pi/mzIdentML/1.1}DataCollection')
    AnalysisData=DataCollection.find('{http://psidev.info/psi/pi/mzIdentML/1.1}AnalysisData')
    SpectrumIdentificationList=AnalysisData.find('{http://psidev.info/psi/pi/mzIdentML/1.1}SpectrumIdentificationList')
    for SpectrumIdentificationResult in SpectrumIdentificationList.iter('{http://psidev.info/psi/pi/mzIdentML/1.1}SpectrumIdentificationResult'):
        one_scan=scan()
        for cvParam in SpectrumIdentificationResult.iter('{http://psidev.info/psi/pi/mzIdentML/1.1}cvParam'):
            one_scan.scan_number=cvParam.attrib['value']
            one_scan.fidx=fidx
        for SpectrumIdentificationItem in SpectrumIdentificationResult.iter('{http://psidev.info/psi/pi/mzIdentML/1.1}SpectrumIdentificationItem'):
            pep=peptide()
            pep.charge=SpectrumIdentificationItem.attrib['chargeState']
            pep.measured_mass=(float(SpectrumIdentificationItem.attrib['experimentalMassToCharge'])-W_proton)*int(pep.charge)
            pep.calculated_mass=(float(SpectrumIdentificationItem.attrib['calculatedMassToCharge'])-W_proton)*int(pep.charge)
            pep.local_rank=SpectrumIdentificationItem.attrib['rank']
            p='.'+SpectrumIdentificationItem.attrib['peptide_ref'].replace('Pep_','')+'.'
            for PeptideEvidenceRef in SpectrumIdentificationItem.iter('{http://psidev.info/psi/pi/mzIdentML/1.1}PeptideEvidenceRef'):
                pep.identified_pep=prepostMap[PeptideEvidenceRef.attrib['peptideEvidence_ref']][0]+p+prepostMap[PeptideEvidenceRef.attrib['peptideEvidence_ref']][1]
                pep.protein_list.append(Map[prepostMap[PeptideEvidenceRef.attrib['peptideEvidence_ref']][2]])
            for cvP in SpectrumIdentificationItem.iter('{http://psidev.info/psi/pi/mzIdentML/1.1}cvParam'):
                if cvP.attrib['name']=="MS-GF:EValue":
                    pep.EValue=-math.log(float(cvP.attrib['value']))
            one_scan.add_pep(pep)
        for psmcid,psmc in enumerate(one_scan.pep_list):
            psmc.local_rank=psmcid+1
        if len(one_scan.pep_list)>0:
            psm_list.append(one_scan)


def write_output(psm_list, output_file_str):
    with open(output_file_str, 'w') as fw:
        header_str = 'SpecId\tLabel\tScanNr\tExpMass\tCalcMass\tEValue\tMass\tdeltamz\tPepLen\tdM\tabsdM\tCharge1\tCharge2\tCharge3\tCharge4\tCharge5\tenzInt\tPeptide\tProtein'
        fw.write(header_str)
        fw.write('\n')
        for v in psm_list:
            if v.pep_list:
                ll = v.get_percolator_output()
                for l in ll:
                    fw.write('\t'.join(l))
                    fw.write('\n')

    print('write_output is done.')

def main():
    psm_list = []

    # read myrimatch results
    '''
    for i in range(1,12):
        read_one_pep_xml_msgf('marine1_'+str(i)+'.mzid',psm_list,i)
    '''
    inputname=sys.argv[1]
    #inputname='01625b_GA1-TUM_first_pool_1_01_01-2xIT_2xHCD-1h-R1.mzid'
    prefix=inputname.replace('.mzid','')
    read_one_pep_xml_msgf(inputname,psm_list,prefix)
    output_file_str = prefix+'.pin'
    write_output(psm_list, output_file_str)

    print('All done.')

if __name__ == '__main__':
    sys.exit(main())
