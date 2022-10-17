# Sequence Modeling Benchmarks and Temporal Convolutional Networks (TCN)

This repository is a fork of [Temporal Convolutional Networks](https://github.com/locuslab/TCN/tree/master/TCN/adding_problem), which implements the methods/experiments of [An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://arxiv.org/abs/1803.01271) by Shaojie Bai, J. Zico Kolter and Vladlen Koltun:

```
@article{BaiTCN2018,
	author    = {Shaojie Bai and J. Zico Kolter and Vladlen Koltun},
	title     = {An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling},
	journal   = {arXiv:1803.01271},
	year      = {2018},
}
```

## Quadrant Problem Experiment

To help my understanding, I've created this fork to add my own additional experiment that puts forward the following sequence prediction problem:

Suppose there are coordinates $(x_1,y_1)$, $(x_2, y_2)$, ... $(x_n, y_n)$ that correspond to $n$ characters depending on which plane the coordinate resides in:

```
A | C    A | C         A | C
-----  , ----- , ... , ----
B | ?    B | ?         B | ?
```

To prevent the possibility of perfect prediction, the 4th quadrant is denoted by "?". If the coordinate resides there, there is an equal probability of being one of A, B, or C.

For example:

${(x1,y1), (x2,y2), (x3,y3)} = {(-1,-2), (-4,5), (3,3)} =$ BAC

${(x1,y1), (x2,y2), (x3,y3)} = {(-1,-2), (-4,5), (3,-3)} =$ BA[one of {A,B,C}]


## Setup

Create and activate a Python 3.8 virtual environment using [pyenv](https://github.com/pyenv/pyenv-virtualenv):
```
pyenv install -v 3.8.14
pyenv virtualenv 3.8.14 tcn-3.8.14
pyenv activate tcn-3.8.14
```

Install requirements via [Poetry](https://python-poetry.org/):
```
poetry install
```

## Usage

The TCN model can and does learn to improve predictions on the sequences:

```
poetry run python quadrant_test.py
```

### Sample Output:

```
Train Epoch:  1 [   198/   800 (25%)]   Learning rate: 0.0040   Loss: 1.022202
Train Epoch:  1 [   398/   800 (50%)]   Learning rate: 0.0040   Loss: 0.608966
Train Epoch:  1 [   598/   800 (75%)]   Learning rate: 0.0040   Loss: 0.510911
Train Epoch:  1 [   798/   800 (100%)]  Learning rate: 0.0040   Loss: 0.386826

Test set: Average loss: 0.434732

Train Epoch:  2 [   198/   800 (25%)]   Learning rate: 0.0040   Loss: 0.367880
Train Epoch:  2 [   398/   800 (50%)]   Learning rate: 0.0040   Loss: 0.316921
Train Epoch:  2 [   598/   800 (75%)]   Learning rate: 0.0040   Loss: 0.322155
Train Epoch:  2 [   798/   800 (100%)]  Learning rate: 0.0040   Loss: 0.286899

Test set: Average loss: 0.320079

...

Train Epoch: 10 [   198/   800 (25%)]   Learning rate: 0.0040   Loss: 0.232142
Train Epoch: 10 [   398/   800 (50%)]   Learning rate: 0.0040   Loss: 0.212543
Train Epoch: 10 [   598/   800 (75%)]   Learning rate: 0.0040   Loss: 0.218244
Train Epoch: 10 [   798/   800 (100%)]  Learning rate: 0.0040   Loss: 0.224373

Test set: Average loss: 0.249233
```