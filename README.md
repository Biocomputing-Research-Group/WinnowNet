# WinnowNet
This algorithm was implemented and tested on Ubuntu 20.04.6 LTS (GNU/Linux 5.15.0-84-generic, x86_64).
## Note: 
This repository contains the development version of WinnowNet. For the code used to reproduce the experiments in the paper, please refer to the following repository: https://github.com/Biocomputing-Research-Group/WinnowNet4Review

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

## Input pre-processing

Extract fragment ion matching features along with 11 additional features derived from both theoretical and experimental spectra. The PSM (peptide-spectrum match) candidate information should be provided in a tab-delimited file (e.g., a TSV file output from Percolator).
```bash
python SpectraFeatures.py -i <tsv_file> -s <ms2_file> -o spectra.pkl -t 48 -f cnn
```
* Replace `<tsv_file>` with the path to your PSM candidates file.
* Replace `<ms2_file>` with the path to your experimental spectra file.
* The `-t 48` option sets the number of threads (adjust this value as needed).
* Use `-f cnn` when preparing input for the CNN-based architecture or `-f att` for the self-attention-based model.

## Training WinnowNet Models

This folder contains scripts, datasets, and instructions for training two variants of the WinnowNet deep learning model: a self-attention-based model and a CNN-based model. Training is carried out in two phases to enable curriculum learning from synthetic (easy) to real-world metaproteomic (difficult) datasets.

## Requirements

- Python 3.7+
- PyTorch
- NumPy, Pandas, scikit-learn

### Datasets

- **Prosit_train.zip** (Phase 1 training set):   https://figshare.com/articles/dataset/Datasets/25511770?file=55257041
- **marine1_train.zip** (Phase 2 training set): https://figshare.com/articles/dataset/Datasets/25511770?file=55257035

---

### Self-Attention-Based WinnowNet

#### Phase 1: Training on Easy Tasks (Synthetic Data)

```bash
python SpectraFeatures_training.py -i filename.tsv -s filename.ms2 -o spectra_feature.pkl -t 20 -f att
python WinnowNet_Att.py -i spectra_feature_directory -m prosit_att.pt
```

**Explanation of options:**
- `-i`: Input tab-delimited file with PSMs, including labels and weights.
- `-s`: Corresponding MS2 file (filename should match TSV).
- `-o`: Output file to store extracted features as a `pkl` file.
- `-t`: Number of threads for parallel processing.
- `-f`: Feature type (`att` for self-attention model).
- `-m`: Filename to save the trained model.
- A for-loop is needed to convert all `tsv` files to `pkl` files.

#### Phase 2: Training on Difficult Tasks (Real Data)

```bash
python SpectraFeatures_training.py -i filename.tsv -s filename.ms2 -o spectra_feature.pkl -t 20 -f att
python WinnowNet_Att.py -i spectra_feature_directory -m marine_att.pt -p prosit_att.pt
```

- `-p`: Pre-trained model from Phase 1.
- A for-loop is needed to convert all `tsv` files to `pkl` files.

**Pre-trained model:** marine_att.pt,  https://figshare.com/articles/dataset/Models/25513531

---

### CNN-Based WinnowNet

#### Phase 1: Training on Easy Tasks (Synthetic Data)

```bash
python SpectraFeatures_training.py -i filename.tsv -s filename.ms2 -o spectra_feature.pkl -t 20 -f cnn
python WinnowNet_CNN.py -i spectra_feature_directory -m prosit_cnn.pt
```

#### Phase 2: Training on Difficult Tasks (Real Data)

```bash
python SpectraFeatures_training.py -i filename.tsv -s filename.ms2 -o spectra_feature.pkl -t 20 -f cnn
python WinnowNet_CNN.py -i spectra_feature_directory -m cnn_pytorch.pt -p prosit_cnn.pt
```

**Pre-trained model:** cnn_pytorch.pt, https://figshare.com/articles/dataset/Models/25513531

---

### Notes

- All input MS2/TSV files must be preprocessed properly.
- Models trained in Phase 1 are reused to initialize weights in Phase 2.
- Training with GPU is recommended for performance.

## Inference
### PSM Rescoring
#### Self-Attention-Based WinnowNet
To generate input representations for PSM candidates and perform re-scoring using the self-attention model, run:
```bash
python SpectraFeatures.py -i tsv_file -s ms2_file -o spectra.pkl -t 48 -f att 
python Prediction.py -i spectra.pkl -o rescore.out.txt -m att_pytorch.pt  

```
#### CNN-Based WinnowNet
To generate input representations for PSM candidates and perform re-scoring using the CNN model, run:
```bash
python SpectraFeatures.py -i filename.tsv -s filename.ms2 -o spectra.pkl -t 48 -f cnn
python Prediction_CNN.py -i spectra.pkl -o rescore.out.txt -m cnn_pytorch.pt 

```
**Explanation of options:**
- `-i`: Input tab-delimited file with PSMs
- `-s`: Corresponding MS2 file (filename should match TSV).
- `-o`: Output file to store extracted features as a `pkl` file.
- `-t`: Number of threads for parallel processing.
- `-f`: Feature type (`att` for self-attention model, `cnn`for CNN model).
- `-m`: Filename to save the trained model.
- A for-loop is needed to convert all `tsv` files to `pkl` files.

## Evaluation
### FDR Control at the PSM/Peptide Levels
Filter the re-scored PSM candidates to control the false discovery rate (FDR) at both the PSM and peptide levels (targeted at 1% FDR). You will need both the original PSM file and the re-scoring results.
```bash
python filtering.py -i rescore.out.txt -p tsv_file -o filtered -d Rev_ -f 0.01
```
**Explanation of options:**
- `-i`: Rescoring file from WinnowNet
- `-p`: Input tab-delimited file with PSMs
- `-o`: filtered results' prefix
- `-d`: Decoy prefix used for target-decoy strategy. Default: Rev_
- `-f`: False Discovery Rate. Default: 0.01
- A for-loop is needed to convert all `tsv` files to `pkl` files.

* The filtered output files include updated PSM information (new predicted scores, spectrum IDs, identified peptides, and corresponding proteins).
* Assembling filtered identified peptides into proteins
* This script is needed to run at the working directory inlucding filtered results at PSM and Peptide levels.
```bash
python sipros_peptide_assembling.py
```
When assembling filtered, identified peptides into proteins, the overall protein-level FDR depends on the quality of the filtered peptide list. An initial peptide-level FDR (for example, 1%) may lead to a protein-level FDR that is higher than desired. In such cases, you need to re-filter the peptides using a stricter (i.e., lower) FDR threshold until you achieve a 1% protein-level FDR. 

## Contact and Support
For further assistance, please consult the GitHub repository or reach out to the project maintainers.