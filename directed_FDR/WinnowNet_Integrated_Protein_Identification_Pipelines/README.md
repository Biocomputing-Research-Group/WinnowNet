# Original command lines of pipelines and the step-step instruction to integrate WinnowNet to the pipeline
Note: the marine2 dataset is used as an example
## Original Sipros-Ensemble 
### Execution of Sipros-Ensemble search engine
```
./Sipros_OpenMP -w marine3_WorkDirectory -c marine3_WorkDirectory/SiprosConfig_Marine.cfg -o marine3_WorkDirectory/ #The working direcotory include the MS2 files
```
### Execution of Sipros-Ensemble's filtering algorithm and FDR controlling at PSM/Peptide/Protein levels
```
./runSiprosFiltering.sh -in marine3_WorkDirectory -c marine3_WorkDirectory/SiprosConfig_Marine.cfg -o marine3_WorkDirectory/ 
```

## Integrating WinnowNet with Sipros-Ensemble
### Execution of Sipros-Ensemble search engine and data format convertion
``` 
./Sipros_OpenMP -w marine3_WorkDirectory -c marine3_WorkDirectory/SiprosConfig_Marine.cfg -o marine3_WorkDirectory/ #The working direcotory include the MS2 fil
es
python sipros2win.py -in marine3_WorkDirectory -o marine3_WorkDirectory/marine3_spectra.pkl -t 48
```
### Execution of WinnowNet for rescoring
```
python Prediction.py -i marine3_WorkDirectory/marine3_spectra.pkl -o marine3_WorkDirectory/marine3.rescore.txt -m att_pytorch.pt
```
### FDR controlling at PSM/Peptide level and protein assembling
```
python filtering_combineFDR.py -i marine3_WorkDirectory/marine3.rescore.txt -f 0.01
python sipros_peptides_assembling.py -w marine3_WorkDirectory
```
## Original FragPipe
### Execution of FragPipe in headless mode
```
fragpipe --headless --workflow marine3_WorkDirectory/fragpipe.workflow --manifest marine3_WorkDirectory/fragpipe-files.fp-manifest --workdir marine3_WorkDirectory/
```
## Integrating WinnowNet with FragPipe
### Execution of FragPipe search engine and data format convertion
```
java -jar -Dfile.encoding=UTF-8 -Xmx16G FragPipe-22.0/fragpipe/tools/MSFragger-4.1/MSFragger-4.1.jar fragger.params marine3_WorkDirectory/OSU_D7_FASP_Elite_03172014_01.raw
python xml2win.py -in marine3_WorkDirectory/ -o marine3_WorkDirectory/marine3_spectra.pkl -t 48
```
### Execution of WinnowNet for rescoring
```
python Prediction.py -i marine3_WorkDirectory/marine3_spectra.pkl -o marine3_WorkDirectory/marine3.rescore.txt -m att_pytorch.pt
```
### Data format convertion for FragPipe's philosopher tool
```
python win2prophet.py -i marine3_WorkDirectory/marine3.rescore.txt -w marine3_WorkDirectory/
```
### Protein inference and fdr controlling by philosopher
```
./FragPipe-22.0/fragpipe/tools/Philosopher/philosopher-v5.1.1 proteinprophet --maxppmdiff 2000000 --output combined marine3_WorkingDirectory/filelist_proteinprophet.txt
./FragPipe-22.0/fragpipe/tools/Philosopher/philosopher-v5.1.1 database --annotate marine3_WorkingDirectory/Marine_shuffled.fasta --prefix shuffled_
./FragPipe-22.0/fragpipe/tools/Philosopher/philosopher-v5.1.1 filter --sequential --prot 0.01 --tag rev_ --pepxml marine3_WorkingDirectory --protxml marine3_WorkingDirectory/combined.prot.xml --razor
```
