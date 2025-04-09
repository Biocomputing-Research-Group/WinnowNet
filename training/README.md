# Download pre-trained model from: https://figshare.com/articles/dataset/Models/25513531
## Generate dataset
```
python train_process.py -i tsv -m msfile -o training_input_file
```
## Training
```
python WinnowNet_CNN.py -i spectra.pkl -m cnn_pytorch.pt
```