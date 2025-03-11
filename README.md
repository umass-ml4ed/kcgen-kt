# Automated Knowledge Component Generation and Knowledge Tracing for Coding Problems
This repository contains the code for the paper <a href="https://arxiv.org/abs/2410.10829">Automated Knowledge Component Generation and Knowledge Tracing for Coding Problems</a>. The primary contributions here include 1. A completely automatic KC generation pipeline using solution AST. 2. Utilizing KC mastery level for knowledge tracing task 

## Setup

### Data
We use [CSEDM](https://sites.google.com/ncsu.edu/csedm-dc-2021/) dataset. The dataset can be downloaded as follow:
```
pip install gdown
cd data
bash data.sh
```

### Environment
We used Python 3.8.18 in the development of this work. Run the following to set up a Conda environment:
```
conda create --name <env> --file requirements.txt
```

