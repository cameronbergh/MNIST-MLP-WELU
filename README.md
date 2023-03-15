# MNIST-MLP-WELU
A simple pytorch script to train an MLP using the WELU (Weighted Exponential Linear Units) activation function on the MNIST dataset and record results to wandb.com

I really dont know if i am using this correctly.

SELU and WELU is from the authors in this paper. i cant find the paper for WELU. https://arxiv.org/abs/1706.02515v5

here are my results: 
https://wandb.ai/impudentstrumpet/mnist-welu/runs/zxnmuh4q
https://wandb.ai/impudentstrumpet/mnist-selu

By cameron bergh 2023


installation:

git clone this repo
cd to repo directory

pipenv install

pip3 install torch torchvision wandb

(login to wandb by running wandb.login(), look it up on their website)

python3 welu_mlp.py

