# # Extrinsic Intrinsic Representation Learning
## Usage
We provide classes in three modules:
* `train.py`: core EIR modules 
* `EIR.dataset`: datasets used in EIR
* `EIR.py`: implementations of modules of EIR

The core modules in `EIR` are meant to be as general as possible, but you will likely have to modify `EIR.data` and `EIR.models` for your specific application, with the existing classes serving as examples.

## Dataset
First unzip the dataset in `dataset` directory with the following command.
```
unzip -o ./dataset/dataset.zip -d dataset
```

Then, you can use the `EIR.dataset` module to load the dataset.



## Training / testing
To train a model, simply run `python train.py` with the following options.

```
$ python trian.py 
usage: train.py[--dataset-name NAME] [--setting SETTING] [--clu-thre N] [--maxlen N] [--attention_depth N] [--attention_hidden N] [--attention_dropout ] [--batch_size N] [--name NAME] 

optional arguments:
  measure          options for dataset. choices=['IC50', 'KIKD']
  setting   options for experimental settings. choices=['pid', 'pit', 'ppi']
  clu_thre     clustering threshold. choices=['0.3', '0.4', '0.5', '0.6']
  --maxlen N       max number of protein sequence length, default=3072
  --attention_depth N          attention_depth, default=2
  --attention_hidden N    hidden dimension for attention layer, default=256
  --attention_dropout  attention_dropout, default=0.1
  --batch_size N         batch size, default=16
  --name             name for the model

```

For example:
```
# train a model on KIKD dataset (cluster = 0.3) under PID setting.
python train.py KIKD pid 0.3 
```
