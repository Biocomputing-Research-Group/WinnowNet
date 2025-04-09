# Comparison between combined FDR and paired FDR control
## Protein Database generation
```
# Generating entrapment proteins
python Shuffled_proteins.py
```
## Paired protein database generation
```
#Generate entrapment proteins
java -jar fdrbench-0.0.1.jar -level protein -db ~/entrapment/foreign_species.fasta -o ~/entrapment/foreign_species_I2L_entrapment.fasta -I2L -fix_nc c -check -ns
#Generate entrapment digested peptide with basic rules
java -jar fdrbench-0.0.1.jar -level peptide -db ~/entrapment/foreign_species.fasta -o ~/entrapment/foreign_species_entrapment_peptide.txt -I2L -minLength 7 -maxLength 60 -fix_nc c -check -ns
```
## Combined FDP and Paired FDR estimation
```
java -jar fdrbench-0.0.1.jar -i ../pep_compliment.txt -fold 1 -pep ../foreign_species_pep_entrapment.txt -level peptide -o ../pep_compliment_out.csv -score 'score:0'
```
## Result is shown as:
```
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