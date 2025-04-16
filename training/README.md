## Download pre-trained model from: https://figshare.com/articles/dataset/Models/25513531
## Generate dataset
```bash
python train_process.py -i tsv -m msfile -o training_input_file
```
## Training
```bash
python WinnowNet_CNN.py -i spectra.pkl -m cnn_pytorch.pt
```
  * `-m` indicates the model with lowest loss after the model converging, if the training process is beginning from scratch (e.g., easy learning phase in learning), it will be generated automatically
  * In curriculum learning phase (e.g, difficult learning phase), `-m` indicates the pre-trained model from the easy learning phase

## Sample Output in the Screen
The output in the screen includding the size and loading time of the dataset; training time for each epoch, performance of training and test datasets for each epoch

```bash
loading data: 4.147444248199463 mins
length of training data: 1707478
length of validation data: 189720
length of test data: 210800
......
Epoch  51, Train_loss:    0.12, Train_acc 95.82%, Train_Posprec 96.48%, Train_Negprec 94.97%, Val_loss:   0.13, Val_acc 95.41%,Val_Posprec 96.18%, Val_Negprec 94.43% Time: 7:54:49 
Epoch  52, Train_loss:    0.12, Train_acc 95.97%, Train_Posprec 96.92%, Train_Negprec 94.77%, Val_loss:   0.13, Val_acc 95.59%,Val_Posprec 96.81%, Val_Negprec 94.06% Time: 7:42:17 
Epoch  53, Train_loss:    0.12, Train_acc 95.91%, Train_Posprec 97.12%, Train_Negprec 94.37%, Val_loss:   0.13, Val_acc 95.50%,Val_Posprec 97.03%, Val_Negprec 93.62% Time: 7:43:06 
Epoch  54, Train_loss:    0.12, Train_acc 96.03%, Train_Posprec 96.76%, Train_Negprec 95.08%, Val_loss:   0.13, Val_acc 95.54%,Val_Posprec 96.44%, Val_Negprec 94.40% Time: 7:42:45 
Epoch  55, Train_loss:    0.11, Train_acc 96.12%, Train_Posprec 97.01%, Train_Negprec 94.98%, Val_loss:   0.13, Val_acc 95.61%,Val_Posprec 96.78%, Val_Negprec 94.16% Time: 7:43:12
......
```