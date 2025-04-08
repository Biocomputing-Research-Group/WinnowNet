#Original command lines of pipelines and the step-step instruction to integrate WinnowNet to the pipeline
Note: the marine2 dataset is used as an example
##Original Sipros-Ensemble 
###Execution of Sipros-Ensemble search engine
```
./Sipros_OpenMP -w marine3_WorkDirectory -c marine3_WorkDirectory/SiprosConfig_Marine.cfg -o marine3_WorkDirectory/ #The working direcotory include the MS2 files
```
###Execution of Sipros-Ensemble's filtering algorithm and FDR controlling at PSM/Peptide/Protein levels
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
###Execution of WinnowNet for rescoring
```
python Prediction.py -i marine3_WorkDirectory/marine3_spectra.pkl -o marine3.rescore.txt -m att_pytorch.pt
```
###FDR controlling at PSM/Peptide and Protein level
```
python filtering_combineFDR.py -i marine3.rescore.txt -f 0.01
python sipros_peptides_assembling.py -w marine3_WorkDirectory
```

