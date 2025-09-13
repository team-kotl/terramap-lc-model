# TerraMap Land Cover Model

This repository contains the training script of the TerraMap U-Net Land Cover Classification model.

### Create your conda environment

```shell
conda create --prefix ./tmlc
conda activate ./tmlc
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 # Please install torch with CUDA first
conda install --file requirements.txt -c conda-forge
```

### Files

1. [Truth Labels](https://drive.google.com/drive/folders/1bQEWO6k64sb9HzSqChxYBSdGdfs10M7m?usp=drive_link)
2. [Raw Imagery](https://drive.google.com/drive/folders/1pF6Jvt8Jh0RtbVSauz5szt5dMOT4kjU_?usp=sharing)
