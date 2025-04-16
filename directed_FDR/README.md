# Comparison between combined FDR and paired FDR control
## 1. Protein Database Generation
Generate a set of entrapment proteins. This script creates a shuffled (or decoy) version of the protein entries, which will be used later in FDR estimation.
```bash
# Generating entrapment proteins
python Shuffled_proteins.py
```
## 2. Paired protein database generation
Create a paired protein database that includes both target and entrapment proteins, as well as a corresponding peptide list obtained by in silico digestion using basic rules.
### a. Generate Entrapment Protein Database
```bash
# Generate entrapped protein sequences using I2L strategy,
# with additional flags to fix N- and C-term modifications, check sequence quality, and disable small peptides.
java -jar fdrbench-0.0.1.jar -level protein -db ~/entrapment/foreign_species.fasta -o ~/entrapment/foreign_species_I2L_entrapment.fasta -I2L -fix_nc c -check -ns
```
### b. Generate Entrapment Digested Peptides
```bash
#Generate entrapment digested peptide with basic rules
java -jar fdrbench-0.0.1.jar -level peptide -db ~/entrapment/foreign_species.fasta -o ~/entrapment/foreign_species_entrapment_peptide.txt -I2L -minLength 7 -maxLength 60 -fix_nc c -check -ns
```
## 3. Combined FDP and Paired FDR estimation
Once you have the peptide identifications and the paired peptide database, run the FDR estimation using the following command. Here, the tool takes your primary peptide results (`pep_compliment.txt`) and the entrapped peptide set to estimate both the combined FDP and paired FDR for each peptide entry.
```bash
java -jar fdrbench-0.0.1.jar -i ../pep_compliment.txt -fold 1 -pep ../foreign_species_pep_entrapment.txt -level peptide -o ../pep_compliment_out.csv -score 'score:0'
```
  * `-i ../pep_compliment.txt`: Input file containing primary peptide identifications.
  * `-pep ../foreign_species_pep_entrapment.txt`: Entrapped peptide list for paired FDR analysis.
  * `-level peptide`: Sets the level for FDR estimation (peptide in this case).
  * `-o ../pep_compliment_out.csv`: Output CSV file for the results.
  
### Sample Output Results
The output file (`pep_compliment_out.csv`) will contain several columns, such as peptide sequence, modified peptide (if applicable), charge state, q-value, score, estimated combined FDP and paired FDP, among other statistics. Below is an example snippet of the result at peptide level:
```css
peptide	mod_peptide	charge	q_value	score	combined_fdp	n_t	n_p	paired_fdp	n_p_t_s	n_p_s_t	vt	lower_bound_fdp

EALVTGENTSDAYTAATKALDK	EALVTGENTSDAYTAATKALDK	2	0.00197172	19	0.0	176	0	0.0	0	0	0	0.0
QQSSANNGDLVVALLGDEATCK	QQSSANNGDLVVALLGDEATC[57.0215]K	2	0.00197172	21	0.0	176	0	0.0	0	0	0	0.0
HEGDTGSPEVQVALLTAR	HEGDTGSPEVQVALLTAR	2	0.00197172	27	0.0	176	0	0.0	0	0	0	0.0
YMGASAANLSASSSSVQK	YM[15.9949]GASAANLSASSSSVQK	2	0.00197172	38	0.0	176	0	0.0	0	0	0	0.0
KATEELQQSFYDLSSK	KATEELQQSFYDLSSK	2	0.00197172	39	0.0	176	0	0.0	0	0	0	0.0
ATTYYNDPQLLAEVSEELGTAMDSLDVR	ATTYYNDPQLLAEVSEELGTAM[15.9949]DSLDVR	3	0.00197172	55	0.0	176	0	0.0	0	0	0	0.0
AAAEALGLGLYQYLGGVNAK	AAAEALGLGLYQYLGGVNAK	2	0.00197172	95	0.0	176	0	0.0	0	0	0	0.0
VLKGSSNTESAQLNNK	VLKGSSNTESAQLNNK	2	0.00197172	116	0.0	176	0	0.0	0	0	0	0.0
NAKPAAVAPAPAASPAEDAGVLDFEDFQK	NAKPAAVAPAPAASPAEDAGVLDFEDFQK	3	0.00197172	126	0.0	176	0	0.0	0	0	0	0.0
LLVDTGEMQPLVSEDR	LLVDTGEMQPLVSEDR	2	0.00197172	130	0.0	176	0	0.0	0	0	0	0.0
VNVAGGGLSGQAEAVR	VNVAGGGLSGQAEAVR	2	0.00197172	134	0.0	176	0	0.0	0	0	0	0.0
```
Below is an example snippet of the result at protein level:
```css
Protein.Group  PG.Q.Value  q_value     protein  score  combined_fdp  n_t   n_p  paired_fdp  n_p_t_s  n_p_s_t  vt  lower_bound_fdp

sp|B8I601|GATA_RUMCH        0.00174821  0.00174821 B8I601   1      0.0           229  0    0.0         0        0        0   0.0
sp|B8I6T0|PROA_RUMCH        0.00174821  0.00174821 B8I6T0   2      0.0           229  0    0.0         0        0        0   0.0
sp|Q726J4|CARA_DESVH        0.00174821  0.00174821 Q726J4   3      0.0           229  0    0.0         0        0        0   0.0
sp|Q72AQ6|PHNC_DESVH        0.00174821  0.00174821 Q72AQ6|  4      0.0           229  0    0.0         0        0        0   0.0
sp|Q72EU7|TRPA_DESVH        0.00174821  0.00174821 Q72EU7   5      0.0           229  0    0.0         0        0        0   0.0
```
  * Columns **combined_fdp** and **paired_fdp** are used to estimated combined FDR and paired FDR by selecting identifications whose FDP are less than 0.01
