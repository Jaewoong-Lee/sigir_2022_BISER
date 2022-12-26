This code is the implementation of the paper "Bilateral Self-unbiased Learning from Biased Implicit Feedback" in SIGIR2022
(https://dl.acm.org/doi/abs/10.1145/3477495.3531946)

# Main contributors for this code

Jae-woong Lee and Seongmin Park (https://github.com/psm1206)


# About this code and datasets
We modified the code below.

https://github.com/usaito/unbiased-implicit-rec-real

We referred to the code below.

RelMF: https://github.com/usaito/unbiased-implicit-rec-real

CJMF: https://github.com/Zziwei/Unbiased-Propensity-and-Recommendation/blob/master/CJMF.py

MACR: https://github.com/weitianxin/MACR/tree/main/macr_lightgcn

# Preliminaries

Set up your user Python environment as follows:

python==3.7.7
tqdm
joblib
bottleneck
numpy==1.16.2
pandas==0.24.2
scikit-learn==0.20.3
tensorflow==1.15.0 (cpu)
mlflow==1.4.0
pyyaml==5.1


# How to run
1. Set a model's hyperparameters you want in ```main.py```. The adjustable hyperparameters are as follows.
    - model name {mf, relmf, uae, iae, proposed}
    - dataset {coat, yahoo}
    - learning rate
    - regularization term
    - hidden dimension
    - batch size
    - etc.
    - **In case of our proposed method, the best hyperparmeters are fixed in the code.**

2. Run ```main.py``` with the hyperparameters you set.
    - e.g., 
    - MF: ```main.py --model_name mf --dataset coat --lr 0.005 --reg 1e-9 --hidden 128 --batch_size 1024```
    - Proposed: ```main.py --model_name proposed --dataset coat```

3. You can see the results in the './log' folder.


# Example to run in Coat dataset
## MF
python main.py -m mf --dataset coat -lr 0.005 -reg 1e-9 -hidden 128 --batch_size 1024 -ran 10

## RelMF
python main.py -m relmf --dataset coat -lr 0.005 -reg 1e-5 -hidden 128 --batch_size 1024 -ran 10

## UAE
python main.py -m uae --dataset coat -lr 0.1 -reg 1e-6 -hidden 50 --batch_size 1 -ran 10

## IAE
python main.py -m iae --dataset coat -lr 0.2 -reg 1e-7 -hidden 50 --batch_size 1 -ran 10

## CJMF
python main.py -m cjmf --dataset coat -lr 0.005 -reg 1e-9 -hidden 128 --batch_size 1024 -ran 10

## MACR
python main.py -m macr --dataset coat -lr 0.01 -reg 1e-5 -hidden 64 --batch_size 2048 -ran 10

## BISER
python main.py -m proposed --dataset coat -ran 10


# Example to run in Yahoo! R3 dataset
## MF
python main.py -m mf --dataset yahoo -lr 0.001 -reg 1e-7 -hidden 64 --batch_size 1024 -ran 5

## RelMF
python main.py -m relmf --dataset yahoo -lr 0.001 -reg 1e-7 -hidden 64 --batch_size 1024 -ran 5

## UAE
python main.py -m uae --dataset yahoo -lr 0.01 -reg 0.0 -hidden 200 --batch_size 1 -ran 5

## IAE
python main.py -m iae --dataset yahoo -lr 0.05 -reg 0.0 -hidden 200 --batch_size 1 -ran 5

## CJMF
python main.py -m cjmf --dataset yahoo -lr 0.001 -reg 1e-6 -hidden 64 --batch_size 1024 -ran 5

## MACR
python main.py -m macr --dataset yahoo -lr 0.001 -reg 1e-7 -hidden 64 --batch_size 8192 -ran 5

## BISER
python main.py -m proposed --dataset yahoo -ran 5

