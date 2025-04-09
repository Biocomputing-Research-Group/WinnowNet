# Benchmark filtering algorithms
## Transfer DB search output files as benchmarking filtering algorithms' input
```
python msgf2pin.py -i inputfile.mzid -o outputfile.pin# to transfer MS-GF+ output files as input files for MS2Rescore or Percolator
python myrimatch.py -i inputfile.PepXML -o outputfile.pin
```
## Filtering algorithms
```
ms2rescore -p outputfile.pin -s *.ms2 -c configrationfile -f proteinfastafile -n 8
Percolator outputfile.pin -m output_target.tsv -M output_decoy.tsv
crux q-ranker --inpout-file input.xml --output-dir output-dir

InteractParser input.pep.xml *.pep.xml -Dfastafile -Tfasta -Estricttrypsin -a/ms2_dir/
PeptideProphetParser input.pep.xml ZERO DECOY=Rev_ DECOYPROBS
InterProphetParser THREADS=8 DECOY=Rev_ NONSI NONSM NONSP NONRS input.pep.xml iProphet.pep.xml
```
## Readoutput and applying fdr control
```
python filtering_combineFDR.py -f 0.01 -m tsv # -m for output mode options, including tsv for ms2rescore, percolator and crux; prophetxml for PeptideProphet; and iprophetxml for iProphet
```