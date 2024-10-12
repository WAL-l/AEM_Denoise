# TEMrwkv
this is the code implementation of "[Semi-Airborne Transient Electromagnetic Signal Denoising With Disentangled Representation Learning And RWKV](https://github.com/WAL-l/AEM_Denoise)"

# Train

## Installation
run:
```
pip install -r requirements.txt
```
## Training models

### Preparing Data

The training code reads ‘npy’ or ‘txt’ data from a directory of data files. 

For creating your own dataset, simply dump all of your datas into a directory, and clean data name 'data.npy/txt', noise data name 'data_noise.npy/txt'. 

Specify 'train_dir' and 'val_dir' inside main DataModule.

## training
```
python main.py
```
You may also want to train in a distributed manner. In this case, Specify 'accelerator' and 'devices' in the 'Trainer'.

The logs and saved models will be written to a logging directory determined by the 'dirpath' in main.

## Denoiseing
```bash
python test.py
```
Note that you should specify 'checkpoint' and 'noise_data' in the 'test'.


