# Automated Knowledge Component Generation and Knowledge Tracing for Coding Problems
This repository contains the code for the paper <a href="https://arxiv.org/abs/2502.18632">Automated Knowledge Component Generation and Knowledge Tracing for Coding Problems</a>. The primary contributions here include 1. A completely automatic KC generation pipeline using solution AST. 2. Utilizing KC mastery level for knowledge tracing task 

## Setup

### Data
We use [CSEDM](https://sites.google.com/ncsu.edu/csedm-dc-2021/) dataset. The dataset can be downloaded as follow:
```
pip install gdown
cd data
bash data.sh
```

### Environment
We use Python 3.8 in the development of this work. Run the following to set up a Conda environment and install the packages required. After activate the conda environment, we install pytorch with cuda version 11.8, you may want to install according to your own situation. More info can be found on the PyTorch website: https://pytorch.org/get-started/locally/ :
```
conda create --name <env_name> python=3.8
conda activate <env_name>
pip3 install torch --index-url https://download.pytorch.org/whl/cu118 
conda env update --name <env_name> --file environment.yml
```


