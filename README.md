# WinnowNet
Note: This algorithm was implemented and tested on Ubuntu 20.04.6 LTS (GNU/Linux 5.15.0-84-generic, x86_64).
## Overview
WinnowNet is designed for advanced processing of mass spectrometry data with two core methods: a CNN-based approach and a self-attention-based approach. The repository includes scripts for feature extraction, model training, prediction (inference), and evaluation. A toy example is included to help users get started.

## Table of Contents
- [Setup and Installation](#setup-and-installation)
- [Requirements](#Requirements)
- [Downloading Required Files](#download-required-files)
- [Feature Extraction](#feature-extraction)
- [Training](#training)
  - [CNN-based WinnowNet](#cnn-based-winnownet)
  - [Self-Attention-based WinnowNet](#self-attention-based-winnownet)
- [Inference](#inference)
  - [PSM Re-scoring](#psm-rescoring)
- [Evaluation](#evaluation)
- [Contact and Support](#contact-and-support)

## Setup and installation
### 1. Create a new conda environment and activate it.
It is recommended to use Conda for dependency management. Run the following commands in your terminal:
```bash
conda create --name WinnowNet python=3.8
conda activate WinnowNet
```
### 2. Install dependencies:
CUDA version 11.8
Pytorch GPU version is compatible with corresponding cuda version
```bash
pip install -r ./requirements.txt
```
## Requirements
* **Operation system**: Linux
* **GPU Memory**
  * **Inference Mode**: At least 8 GB (adjust batch size if necessary)
  * **Training Mode**: At least 20 GB

## Download Required Files
* Pre-trained model can be downloaded via:
  * **CNN-based WinnowNet**: [cnn_pytorch.pt](https://figshare.com/articles/dataset/Models/25513531)
  * **Self-Attention-based WinnowNet**: [marine_att.pt](https://figshare.com/articles/dataset/Models/25513531)
* A toy example is provided in this repository.
* **Sample Input Datasets**
[Mass spectra data for Benchmark datasets](https://figshare.com/articles/dataset/Datasets/25511770)
Other raw files benchmark datasets can be downloaded via:
[PXD007587](https://www.ebi.ac.uk/pride/archive/projects/PXD007587), [PXD006118](https://www.ebi.ac.uk/pride/archive/projects/PXD006118), [PXD013386](https://www.ebi.ac.uk/pride/archive/projects/PXD006118), [PXD023217](https://www.ebi.ac.uk/pride/archive/projects/PXD023217), [PXD035759](https://www.ebi.ac.uk/pride/archive/projects/PXD035759)

## Feature extraction

Extract fragment ion matching features along with 11 additional features derived from both theoretical and experimental spectra. The PSM (peptide-spectrum match) candidate information should be provided in a tab-delimited file (e.g., a TSV file output from Percolator).
```bash
python SpectraFeatures.py -i <tsv_file> -s <ms2_file> -o spectra.pkl -t 48 -f cnn
```
* Replace `<tsv_file>` with the path to your PSM candidates file.
* Replace `<ms2_file>` with the path to your experimental spectra file.
* The `-t 48` option sets the number of threads (adjust this value as needed).
* Use `-f cnn` when preparing input for the CNN-based architecture or -f att for the self-attention-based model.

## Training
### CNN-based WinnowNet
Train the CNN-based model using the extracted spectrum features:
```bash
python WinnowNet_CNN.py -i spectra.pkl -m cnn_pytorch.pt
```
* `-i spectra.pkl` specifies the input file containing extracted features.
* `-m cnn_pytorch.pt` indicates the file path to save (or load) the trained CNN model.

### Self-Attention-based WinnowNet
Train the self-attention model similarly:
```bash
python WinnowNet_Att.py -i spectra.pkl -m att_pytorch.pt
```
* `-i spectra.pkl` specifies the input file containing extracted features.
* `-m att_pytorch.pt` indicates the file path to save (or load) the trained self-attention model.

## Inference
### PSM Rescoring
To generate input representations for PSM candidates and perform re-scoring using the self-attention model, run:
```bash
python SpectraFeatures.py -i tsv_file -s ms2_file -o spectra.pkl -t 48 -f att 
python Prediction.py -i spectra.pkl -o rescore.out.txt -m att_pytorch.pt  
```
* `rescore.out.txt` will contain the predicted scores for each PSM candidate.

## Evaluation
### FDR Control at the PSM/Peptide Levels
Filter the re-scored PSM candidates to control the false discovery rate (FDR) at both the PSM and peptide levels (targeted at 1% FDR). You will need both the original PSM file and the re-scoring results.
```bash
python filtering.py -i rescore.out.txt -p tsv_file -o filtered
```
* The filtered output files include updated PSM information (new predicted scores, spectrum IDs, identified peptides, and corresponding proteins).
* Assembling filtered identified peptides into proteins
```bash
python sipros_peptide_assembling.py
```
When assembling filtered, identified peptides into proteins, the overall protein-level FDR depends on the quality of the filtered peptide list. An initial peptide-level FDR (for example, 1%) may lead to a protein-level FDR that is higher than desired. In such cases, you need to re-filter the peptides using a stricter (i.e., lower) FDR threshold until you achieve a 1% protein-level FDR. 

## Contact and Support
For further assistance, please consult the GitHub repository or reach out to the project maintainers.