# WinnowNet
Note: The algorithm is implementated and tested in Ubuntu 20.04.6 LTS GNU/Linux 5.15.0-84-generic x86_64
## Setup and installation
### Create a new conda environment and activate it.
```
conda create --name WinnowNet python=3.7
conda activate WinnowNet
```
### Install dependencies:
CUDA version 11.6
Pytorch GPU version is compatible with corresponding cuda version
```
pip install -r ./requirements.txt
```
### Requirement
* Linux operation system
* GPU memory should be more than 8 Gb for inference mode otherwise the batchsize should be adjusted
* GPU memory should be more than 20 Gb for training mode
### Download Required Files
Pre-trained model can be downloaded via:
[cnn_pytorch.pt](https://figshare.com/articles/dataset/Models/25513531) for CNN-based WinnowNet
[att_pytorch.pt](https://figshare.com/articles/dataset/Models/25513531) for self-attention-based WinnowNet
Toy example was provided in the repo
Other raw files benchmark datasets can be downloaded via:
[Mass spectra data for Benchmark datasets](https://figshare.com/articles/dataset/Datasets/25511770)

## Feature extraction

Generate fragment ion matching features and 11 additional features from theoretical, experimental spectrum. The PSM condidates' information is from a telimited file (e.g. a tsv file output from Percolator):
```
python SpectraFeatures.py -i tsv_file -s ms2_file -o spectra.pkl -t 48 -f cnn
```
This function contains two functions of feature extraction methods which are controlled by option "f", when the spectrum features are used as input for CNN-based WinnowNet, please pass "cnn" to the option "f", otherwise, please pass "att" to the option "f"
## WinnowNet training mode:
The training model for CNN-WinnowNet requires two parameters which include a input file for spectrum features from last step and a specified physical address for the trained model 
```
python WinnowNet_CNN.py -i spectra.pkl -m cnn_pytorch.pt
```
The training model for self-attention-based WinnowNet requires two parameters which include a input file for spectrum features from last step and a specified physical address for the trained model
```
python WinnowNet_CNN.py -i spectra.pkl -m cnn_pytorch.pt
```
##WinnwoNet inference mode:
### PSM re-scoring
Generate input representations for PSM candidates then rescore them. Example:
```
python SpectraFeatures.py -i tsv_file -s ms2_file -o spectra.pkl -t 48 -f att 
python Prediction.py -o rescore.out.txt att_pytorch.pt
```
##Evaluation
### FDR control in psm/peptide levels
Filtering with re-score psm candidates, input files include original plain file for PSM candidates and rescoring results, option "o" indicates the prefix for output files. Output files include filtering results after controlling FDR at PSM and peptide levels within 1%
```
python filtering.py -i rescore.out.txt -p tsv_file -o filtered
```
Assembled filtered identified peptides into proteins
```
python sipros_peptide_assembling.py
```
