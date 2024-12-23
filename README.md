# WinnowNet
Note: The algorithm is implementated and tested in Ubuntu 20.04.6 LTS GNU/Linux 5.15.0-84-generic x86_64
## Setup and installation
### Create a new conda environment and activate it.
```
conda create --name WinnowNet python=3.7
conda activate WinnowNet
```
### Install dependencies:

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

### Feature extraction

Generate fragment ion matching features and 11 additional features from theoretical and experimental data:
```
python SpectraFeatures.py -i tsv_file -s ms2_file -o spectra.pkl -t 48
```
### WinnowNet training mode:
The training model for CNN-WinnowNet requires two parameters which include a input file for spectrum features from last step and a specified physical address for the trained model 
'''
python WinnowNet_CNN.py -i spectra.pkl -m cnn_pytorch.pt
'''
### PSM re-scoring
```
python Prediction.py rescore.out.txt cnn_pytorch.pt
```
### Post-Processing
Filtering with re-score psm candidates
```
python filtering.py
```
