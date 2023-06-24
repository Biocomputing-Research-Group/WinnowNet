# WinnowNet

## Setup and installation
### Dependency
* python == 3.7
* numpy == 1.17.2
* scikit-learn >= 0.23.1
* pytorch(gpu version) >= 1.19.5
* CUDA Version 10.2
### Requirement
* Linux operation system
* GPU memory should be more than 8 Gb for inference mode otherwise the batchsize should be adjusted
* GPU memory should be more than 20 Gb for training mode

### Processing

Generate fragment ion matching features from theoretical and experimental data:
```
./Sipros_OpenMP -i1 tempidx -i2 tempcharge -i3 temppeptide -i4 theoryEncode -c SiprosConfig.cfg
python SpectraFeatures.py experimental_ms.ms2 theoryEncode  features.out.pkl
python Prediction.py rescore.out.txt model.pt
```

### Post-Processing
Filtering with re-score psm candidates
```
python filtering.py
```
