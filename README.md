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

## Quadrant Sequence Experiment

To increase my own understanding, I've created this fork to add my own experiment that puts forward the following sequence prediction problem:

Suppose there are coordinates $(x_1,y_1)$, $(x_2, y_2)$, ... $(x_n, y_n)$ that correspond to $n$ characters depending on which plane the coordinate resides in:

```
A | C    A | C         A | C
-----  , ----- , ... , ----
B | ?    B | ?         B | ?
```

So that this can't be a perfect prediction (0 loss), for the 4th quadrant denoted by "?", there is an equal probability of being one of A, B, or C of the coordinate is there.

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
TODO
```