# Colonoscopy tissue screen and segmentation. [[DigestPath 2019]](https://digestpath2019.grand-challenge.org/Home/) [[2nd place method]](http://www.digestpath-challenge.org/#/) 

This respository is PyTorch implementation of our method in **Task 2: Colonoscopy tissue segmentation and classification** of DigestPath 2019.
We won 2nd place in the competition, 
## Requirements
* Python3 (Python3.7 is recommended)
* PyTorch >= 1.1
PyTorch1.1, Python3.7

## Installation:
* create conda environment: `conda create -n torch1.1 python=3.7`
* activate conda environment: `source activate torch1.1`
* install PyTorch1.1 following [https://pytorch.org/get-started/locally/]()
* install other package `pip install -r requirement.txt`

## Usage
### Training model
##### Quick start
`CUDA_VISIBLE_DEVICES=0 python train_net.py --config-file ./configs/unet/layer4_dicelossv1_dsv2_down4.yaml`

This will train our final submitted model on fold 0 (total 4 fold). It will take about 5GB GPU memory for this configuration.


Model, training and validation result will be saved in tensorboard in `cfg.OUTPUT_DIR`, use tensorboard to have a look.
##### configure your own model
To use other model, se `configs/defaults.py` for default configuration. You can create your own `yaml` 
file to overwrite default configuration. Eg. set `cfg.MODEL.MODEL = 'unet16layer4'` to use our proposed model

### Test
Use the same yaml file to test the trained model.

eg. `CUDA_VISIBLE_DEVICES=0 python test_net.py --config-file ./configs/unet/layer4_dicelossv1_dsv2_down4.yaml`
This will give dice score and auc of the trained model on test dataset.

set `cfg.SOLVER.DRAW` to True to save visualization result as image in `cfg.OUTPUT_DIR` for a better view.
