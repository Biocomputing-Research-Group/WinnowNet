# Pipeline Integration Instructions for WinnowNet
Note: This document describes both the original command lines for the Sipros-Ensemble and FragPipe pipelines (using the marine3 dataset as an example) and the step-by-step instructions to integrate WinnowNet for rescoring.
## 1. Sipros-Ensemble Workflow
### A. Original Sipros-Ensemble Pipeline
#### Step 1: Execute the Sipros-Ensemble Search Engine
Run the search engine by specifying the working directory (which must contain the MS2 files) and the configuration file.
```bash
./Sipros_OpenMP -w marine3_WorkDirectory -c marine3_WorkDirectory/SiprosConfig_Marine.cfg -o marine3_WorkDirectory/
```
#### Step 2: Apply Filtering and FDR Control at PSM/Peptide/Protein Levels
```
./runSiprosFiltering.sh -in marine3_WorkDirectory -c marine3_WorkDirectory/SiprosConfig_Marine.cfg -o marine3_WorkDirectory/ 
```

### B. Integrating WinnowNet with Sipros-Ensemble
#### Step 1: Execute Search Engine and Convert Data Format
Run the Sipros-Ensemble search engine and then convert the output for WinnowNet. The conversion script produces a pickle file for downstream rescoring.
```bash
./Sipros_OpenMP -w marine3_WorkDirectory -c marine3_WorkDirectory/SiprosConfig_Marine.cfg -o marine3_WorkDirectory/ # Execute search engine (MS2 files required)
python sipros2win.py -in marine3_WorkDirectory -o marine3_WorkDirectory/marine3_spectra.pkl -t 8 # Convert output to pickle format; use 8 threads
```
#### Step 2: Rescore with WinnowNet
Rescore the converted data using the WinnowNet model.
```bash
python Prediction.py -i marine3_WorkDirectory/marine3_spectra.pkl -o marine3_WorkDirectory/marine3.rescore.txt -m att_pytorch.pt
```
Note: The model file att_pytorch.pt is used by WinnowNet for rescoring.

#### Step 3: Apply FDR Control and Assemble Proteins
Combine the rescored output, apply a 1% FDR threshold (at the PSM/Peptide level), and assemble peptides to infer proteins.
```bash
python filtering_combineFDR.py -i marine3_WorkDirectory/marine3.rescore.txt -f 0.01
python sipros_peptides_assembling.py -w marine3_WorkDirectory
```
Note: The first command applies FDR control using a threshold of 0.01, and the second script performs protein assembly.

## 2. FragPipe Workflow
### A. Original FragPipe Pipeline
#### Step 1. Run FragPipe in Headless Mode
Execute FragPipe using a predefined workflow and manifest file.
```
fragpipe --headless --workflow marine3_WorkDirectory/fragpipe.workflow --manifest marine3_WorkDirectory/fragpipe-files.fp-manifest --workdir marine3_WorkDirectory/
```
### B. Integrating WinnowNet with FragPipe
#### Step 1: Execute MSFragger via FragPipe and Convert Data Format
Use MSFragger for the search and then convert the resulting XML output for WinnowNet processing.
```bash
java -jar -Dfile.encoding=UTF-8 -Xmx16G FragPipe-22.0/fragpipe/tools/MSFragger-4.1/MSFragger-4.1.jar fragger.params marine3_WorkDirectory/OSU_D7_FASP_Elite_03172014_01.raw
python xml2win.py -in marine3_WorkDirectory/ -o marine3_WorkDirectory/marine3_spectra.pkl -t 8
```
Note: Adjust fragger.params and the raw file name as required. The conversion script xml2win.py prepares data for WinnowNet using 8 threads.
#### Step 2: Rescore with WinnowNet
Apply WinnowNet rescoring to the converted data file.
```bash
python Prediction.py -i marine3_WorkDirectory/marine3_spectra.pkl -o marine3_WorkDirectory/marine3.rescore.txt -m att_pytorch.pt
```
#### Step 3: Convert Data for Philosopher Tool
Convert the rescored output into a format compatible with FragPipe's Philosopher tool.
```bash
python win2prophet.py -i marine3_WorkDirectory/marine3.rescore.txt -w marine3_WorkDirectory/
```
#### Step 4: Protein Inference and FDR Control Using Philosopher
```bash
./FragPipe-22.0/fragpipe/tools/Philosopher/philosopher-v5.1.1 proteinprophet --maxppmdiff 2000000 --output combined marine3_WorkingDirectory/filelist_proteinprophet.txt

./FragPipe-22.0/fragpipe/tools/Philosopher/philosopher-v5.1.1 database --annotate marine3_WorkingDirectory/Marine_shuffled.fasta --prefix shuffled_

./FragPipe-22.0/fragpipe/tools/Philosopher/philosopher-v5.1.1 filter --sequential --prot 0.01 --tag rev_ --pepxml marine3_WorkingDirectory --protxml marine3_WorkingDirectory/combined.prot.xml --razor
```
Note: These commands handle protein inference (`proteinprophet`), FASTA database annotation, and sequential FDR filtering (`filter`). Ensure that input file paths and parameters match your environment.

## 3. AlphaPept Workflow
### A. Original AlphaPept Pipeline
#### Step 1. Run AlphaPept in Workflow Mode
Execute  AlphaPept using a predefined workflow and setting up parameters in the `.yaml` configuration file.
```bash
alphapept workflow 2025_03_12_marine3.yaml
```
### B. Integrating WinnowNet with AlphaPept
#### Step 1. Execute AlphaPept Shown as Above and Covert Data Format
```bash
python hdf2win.py -in marine3_WorkDirectory/ -o marine3_WorkDirectory/marine3_spectra.pkl -t 8
```
#### Step 2. Rescore with WinnowNet
```bash
python Prediction.py -i marine3_WorkDirectory/marine3_spectra.pkl -o marine3_WorkDirectory/marine3.rescore.txt -m att_pytorch.pt
```
#### Step 3. Using score from WinnowNet and Follow the Steps to Control FDR by AlphaPept
[**FDR Control**](https://github.com/MannLabs/alphapept/blob/master/nbs/06_score.ipynb)

## 4. PEAKS Workflow
### A. Original PEAKS Pipeline
#### Step 1. Runs DDA database search identification mode PEAKS Studio 12.5 GUI

### D. Integrating WinnowNet with PEAKS
#### Step 1: Start a workflow as shown above in PEAKS Studio 12.5 and convert data file
```bash
python  xml2win.py -in marine3_WorkDirectory/ -o marine3_WorkDirectory/marine3_spectra.pkl
```
#### Step 2: Rescore with WinnowNet
Apply WinnowNet rescoring to the converted data file.
```bash
python Prediction.py -i marine3_WorkDirectory/marine3_spectra.pkl -o marine3_WorkDirectory/marine3.rescore.txt -m att_pytorch.pt
```
#### Step 3: Convert Data for Philosopher Tool
Convert the rescored output into a format compatible with FragPipe's Philosopher tool.
```bash
python win2prophet.py -i marine3_WorkDirectory/marine3.rescore.txt -w marine3_WorkDirectory/
```
#### Step 4: Protein Inference and FDR Control Using Philosopher
```bash
./FragPipe-22.0/fragpipe/tools/Philosopher/philosopher-v5.1.1 proteinprophet --maxppmdiff 2000000 --output combined marine3_WorkingDirectory/filelist_proteinprophet.txt

./FragPipe-22.0/fragpipe/tools/Philosopher/philosopher-v5.1.1 database --annotate marine3_WorkingDirectory/Marine_shuffled.fasta --prefix shuffled_

./FragPipe-22.0/fragpipe/tools/Philosopher/philosopher-v5.1.1 filter --sequential --prot 0.01 --tag rev_ --pepxml marine3_WorkingDirectory --protxml marine3_WorkingDirectory/combined.prot.xml --razor
```
Note: These commands handle protein inference (`proteinprophet`), FASTA database annotation, and sequential FDR filtering (`filter`). Ensure that input file paths and parameters match your environment.
