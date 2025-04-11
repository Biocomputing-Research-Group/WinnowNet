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
conda create --name WinnowNet python=3.7
conda activate WinnowNet
```
### 2. Install dependencies:
CUDA version 11.6
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
  * **Self-Attention-based WinnowNet**: [att_pytorch.pt](https://figshare.com/articles/dataset/Models/25513531)
* A toy example is provided in this repository.
* **Sample Input Datasets**
[Mass spectra data for Benchmark datasets](https://figshare.com/articles/dataset/Datasets/25511770)
Other raw files benchmark datasets can be downloaded via:
[PXD007587](https://www.ebi.ac.uk/pride/archive/projects/PXD007587),[PXD006118](https://www.ebi.ac.uk/pride/archive/projects/PXD006118),[PXD013386](https://www.ebi.ac.uk/pride/archive/projects/PXD006118)[PXD023217](https://www.ebi.ac.uk/pride/archive/projects/PXD023217),[PXD035759](https://www.ebi.ac.uk/pride/archive/projects/PXD035759)

## Feature extraction

Extract fragment ion matching features along with 11 additional features derived from both theoretical and experimental spectra. The PSM (peptide-spectrum match) candidate information should be provided in a tab-delimited file (e.g., a TSV file output from Percolator).
```bash
python SpectraFeatures.py -i <tsv_file> -s <ms2_file> -o spectra.pkl -t 48 -f cnn
```
* Replace `<tsv_file>` with the path to your PSM candidates file.
* Replace `<ms2_file>` with the path to your experimental spectra file.
* The `-t 48` option sets the number of threads (adjust this value as needed).
* Use `-f cnn` when preparing input for the CNN-based architecture or -f att for the self-attention-based model.

## WinnowNet training mode:
The training model for CNN-WinnowNet requires two parameters which include a input file for spectrum features from last step and a specified physical address for the trained model 
```
python WinnowNet_CNN.py -i spectra.pkl -m cnn_pytorch.pt
```
The training model for self-attention-based WinnowNet requires two parameters which include a input file for spectrum features from last step and a specified physical address for the trained model
```
python WinnowNet_Att.py -i spectra.pkl -m att_pytorch.pt
```
## WinnwoNet inference mode:
### PSM re-scoring
Generate input representations for PSM candidates then rescore them. Example:
```
python SpectraFeatures.py -i tsv_file -s ms2_file -o spectra.pkl -t 48 -f att 
python Prediction.py -i spectra.pkl -o rescore.out.txt -m att_pytorch.pt # rescore.out.txt contains the predicted PSMs' scores 
```

## Evaluation
### FDR control in psm/peptide levels
Filtering with re-score psm candidates, input files include original plain file for PSM candidates and rescoring results, option "o" indicates the prefix for output files. Output files include filtering results after controlling FDR at PSM and peptide levels within 1%
```
python filtering.py -i rescore.out.txt -p tsv_file -o filtered #filtered output files contains PSMs' information including new predicted score, spectrum ID, identified peptides and corresponding proteins.
```
Assembled filtered identified peptides into proteins
```
python sipros_peptide_assembling.py
```
