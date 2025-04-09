import xml.etree.ElementTree as ET
import sys
W_hydrogen=1.00784

class peptide:

    def __init__(self):
        self.identified_pep = ""
        self.original_pep = ""
        self.calculated_mass = None
        self.mvh_score = None
        self.mz_score = None
        self.protein_list = []
        self.mvh_diff = 0
        self.mz_diff = 0
        self.local_rank = 0
        num_missed_cleavages = 0

class scan:

    def __init__(self):
        self.fidx = 0
        self.measured_mass = None
        self.charge = None
        self.scan_number = 0
        self.pep_list = []
        self.mvh_max = 0

    def add_pep(self, pep):
        self.pep_list.append(pep)

    def get_percolator_output(self):

        self.find_diff()

        # SpecId\tLabel\tScanNr\tExpMass\tCalcMass\tMass\tmvh\tmz\tdeltamvh\tdeltamz\tPepLen\tdM\tabsdM\tCharge1\tCharge2\tCharge3\tCharge4\tCharge5\tenzInt\tPeptide\tProtein

        common_list = [str(self.fidx) + '_' + str(self.scan_number) + '_' +
                       str(self.charge) + '_']  # scan number
        common_list.append('1')
        common_list.append(str(self.scan_number))  # scan number
        common_list.append(self.measured_mass)  # measured mass

        results = []

        rank=0
        for pep in self.pep_list:
            rank+=1
            l = []
            common_list[0]=common_list[0]+str(rank)
            l.extend(common_list)
            common_list[0]=common_list[0][0:len(common_list[0])-1]
            proteinstring = '\t'.join(pep.protein_list)
            Label='-1'
            for p in pep.protein_list:
                if 'Rev_' not in p:
                    Label='1'
                    break
            l[1]=Label
            l.append(pep.calculated_mass)
            l.append(str(float(self.measured_mass)+W_hydrogen))
            l.append(str(pep.mvh_score))
            l.append(str(pep.mvh_diff))
            l.append(str(pep.mz_score))
            l.append(str(pep.mz_diff))
            l.append(str(len(pep.original_pep)))
            dM, absdM = float(pep.calculated_mass) - float(self.measured_mass), abs((float(pep.calculated_mass) - float(self.measured_mass)))
            l.append(str(dM))
            l.append(str(absdM))
            c = int(self.charge)
            for i in range(1, 5):  # Charge 1-4
                if c == i:
                    l.append("1")
                else:
                    l.append("0")
            if c > 5:  # Charge5
                l.append("1")
            else:
                l.append("0")
            l.append(str(pep.num_missed_cleavages))
            l.append(pep.identified_pep)
            l.append('\t'.join(pep.protein_list))
            results.append(l)

        return results

    def find_diff(self):
        pep_sorted_list = sorted(self.pep_list, key=lambda pep: -pep.mvh_score)
        self.mvh_max = pep_sorted_list[0].mvh_score
        for i in range(len(pep_sorted_list) - 1):
            pep_sorted_list[i].mvh_diff = pep_sorted_list[i].mvh_score - \
                pep_sorted_list[i + 1].mvh_score

        pep_sorted_list = sorted(self.pep_list, key=lambda pep: -pep.mz_score)
        for i in range(len(pep_sorted_list) - 1):
            pep_sorted_list[i].mz_diff = pep_sorted_list[i].mz_score - \
                pep_sorted_list[i + 1].mz_score


def read_one_pep_xml_myrimatch(input_file_str, psm_list, fidx):
    global mvh_minimum
    tree = ET.parse(input_file_str)
    root = tree.getroot()
    msms_run_summary = root.find(
        '{http://regis-web.systemsbiology.net/pepXML}msms_run_summary')
    for spectrum_query in msms_run_summary.iter('{http://regis-web.systemsbiology.net/pepXML}spectrum_query'):
        # scan information
        split_list = spectrum_query.attrib['spectrum'].split('.')
        one_scan = scan()
        one_scan.fidx = fidx
        one_scan.scan_number = int(split_list[-2])
        one_scan.measured_mass = spectrum_query.attrib['precursor_neutral_mass']
        one_scan.charge = spectrum_query.attrib['assumed_charge']
        for search_result in spectrum_query.iter('{http://regis-web.systemsbiology.net/pepXML}search_result'):
            for search_hit in search_result.iter('{http://regis-web.systemsbiology.net/pepXML}search_hit'):
                pep = peptide()
                # set the original peptide
                pep.original_pep = search_hit.attrib['peptide']
                # get the calculated mass
                pep.calculated_mass = search_hit.attrib['calc_neutral_pep_mass']
                pep.num_missed_cleavages = search_hit.attrib['num_missed_cleavages']
                pep.local_rank=search_hit.attrib['hit_rank']
                # get the identified peptide
                pos_list = []
                for modification_info in search_hit.findall('{http://regis-web.systemsbiology.net/pepXML}modification_info'):
                    for mod_aminoacid_mass in modification_info.iter('{http://regis-web.systemsbiology.net/pepXML}mod_aminoacid_mass'):
                        pos_int = int(mod_aminoacid_mass.attrib['position'])
                        if pep.original_pep[pos_int - 1] == 'M':
                            pos_list.append(pos_int)
                pep.identified_pep = pep.original_pep
                if len(pos_list) > 0:
                    pos_list = sorted(pos_list)
                    for idx, v in enumerate(pos_list):
                        pep.identified_pep = pep.identified_pep[:(
                            v + idx)] + '~' + pep.identified_pep[(v + idx):]
                pep.identified_pep = search_hit.attrib['peptide_prev_aa']+'.'+pep.identified_pep+'.'+search_hit.attrib['peptide_next_aa']
                # get the negative log e valeu score
                for search_score in search_hit.iter('{http://regis-web.systemsbiology.net/pepXML}search_score'):
                    if search_score.attrib['name'] == 'mvh':
                        pep.mvh_score = float(search_score.attrib['value'])
                    if search_score.attrib['name'] == 'mzFidelity':
                        pep.mz_score = float(search_score.attrib['value'])
                # get proteins
                pep.protein_list.append(search_hit.attrib['protein'])
                for alternative_protein in search_hit.iter('{http://regis-web.systemsbiology.net/pepXML}alternative_protein'):
                    pep.protein_list.append(
                        alternative_protein.attrib['protein'])
                # add to this scan
                one_scan.add_pep(pep)
        if len(one_scan.pep_list) > 0:
            psm_list.append(one_scan)

    print(input_file_str + ' done.')


def write_output(psm_list, output_file_str):
    with open(output_file_str, 'w') as fw:
        header_str = 'SpecId\tLabel\tScanNr\tExpMass\tCalcMass\tMass\tmvh\tmz\tdeltamvh\tdeltamz\tPepLen\tdM\tabsdM\tCharge1\tCharge2\tCharge3\tCharge4\tCharge5\tenzInt\tPeptide\tProtein'
        fw.write(header_str)
        fw.write('\n')
        for v in psm_list:
            if v.pep_list:
                ll = v.get_percolator_output()
                for l in ll:
                    fw.write('\t'.join(l))
                    fw.write('\n')

    print('write_output is done.')


if __name__ == '__main__':
    psm_list = []
    filename=sys.argv[1]
    prefix=filename.replace('.pepXML','')
    # read myrimatch results
    read_one_pep_xml_myrimatch(filename, psm_list, prefix)
    # output results
    output_file_str = prefix+'.pin'
    write_output(psm_list, output_file_str)

    print('All done.')
