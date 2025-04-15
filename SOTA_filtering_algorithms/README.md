# Benchmark filtering algorithmsi
This repository provides tools and scripts to convert database search output files and benchmark various filtering algorithms. The workflow below shows how to:
  * Convert the output files from MS-GF+ and MyriMatch into a unified PIN format.
  * Run several filtering algorithms to process these files.
  * Applying FDR control to obtain reliable outputs.
## Converting Database Search Output Files
MS-GF+ Output Conversion
```bash
python msgf2pin.py -i inputfile.mzid -o outputfile.pin # Convert MS-GF+ mzIdentML file to PIN format
```
MyriMatch Output Conversion
```bash
python myrimatch.py -i inputfile.PepXML -o outputfile.pin # Convert MyriMatch pepXML file to PIN format
```
## Running Filtering Algorithms
Execute the following filtering algorithms:
### MS2Rescore
```bash
ms2rescore -p outputfile.pin -s *.ms2 -c configrationfile -f proteinfastafile -n 8
```
Rescores MS2 spectra with 8 threads, using the specified configuration and FASTA file.

### Percolator
```bash 
Percolator outputfile.pin -m output_target.tsv -M output_decoy.tsv
```
Processes the PIN file to separate target and decoy matches, outputting TSV files.
### Crux Q-Ranker
```bash
crux q-ranker --inpout-file input.xml --output-dir output-dir
```
Executes Crux Q-Ranker, reading from the given input XML file and writing outputs to the specified directory.
### Additional Parsing and Aggregation Tools
Use the following parsing tools to further process the output files from various search engines for PeptideProphet/iProphet:
### InteractParser
```bash
InteractParser input.pep.xml *.pep.xml -Dfastafile -Tfasta -Estricttrypsin -a/ms2_dir/
```
Aggregates multiple pepXML files, using the provided FASTA database and MS2 directory.
### PeptideProphetParser
```bash
PeptideProphetParser input.pep.xml ZERO DECOY=Rev_ DECOYPROBS
```
Processes the output from PeptideProphet with decoy configuration settings.
### iProphet
```bash
InterProphetParser THREADS=8 DECOY=Rev_ NONSI NONSM NONSP NONRS input.pep.xml iProphet.pep.xml
```
Runs iProphet with 8 threads and specific decoy and normalization options.

## Applying FDR Control
After filtering, reading the outputs and applying an FDR (False Discovery Rate) control to finalize the results.

```bash
python filtering_combineFDR.py -i inputfile -f 0.01 -m tsv 
```
This script reads output files (in TSV format for ms2rescore, Percolator, and Crux; or prophetxml/iprophetxml for PeptideProphet/iProphet) and applies an FDR threshold of 1%.
* `-i inputfile` specifies the input file generated from filtering algorithms
* `-f 0.01` specifies the FDR
* `-m tsv` indicates the format of input files
