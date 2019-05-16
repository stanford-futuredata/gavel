# Recoder
[![Pypi version](https://img.shields.io/badge/dynamic/json.svg?label=pypi&url=https%3A%2F%2Fpypi.org%2Fpypi%2Frecsys-recoder%2Fjson&query=%24.info.version&colorB=blue)](https://pypi.org/project/recsys-recoder/)
[![Docs status](https://readthedocs.org/projects/recoder/badge/?version=latest)](https://recoder.readthedocs.io/en/latest/)
[![Build Status](https://travis-ci.org/amoussawi/recoder.svg?branch=master)](https://travis-ci.org/amoussawi/recoder)

### Introduction

Recoder is a fast implementation for training collaborative filtering latent
factor models with mini-batch based negative sampling following recent work:
- [Towards Large Scale Training Of Autoencoders For Collaborative Filtering](https://arxiv.org/abs/1809.00999).

Recoder contains two implementations of factorization models: Autoencoder and Matrix Factorization.

Check out the [Documentation](https://recoder.readthedocs.io/en/latest/) and
the [Tutorial](https://recoder.readthedocs.io/en/latest/tutorial.html).

### Installation
Recommended to use python 3.6. Python 2 is not supported.

```bash
pip install -U recsys-recoder
```

### Examples
Check out the `scripts/` directory for some good examples on different datasets.
You can get MovieLens-20M dataset fully trained with mean squared error in less
than a minute on a Nvidia Tesla K80 GPU.

### Further Readings
- [Collaborative Filtering for Implicit Feedback Datasets](http://yifanhu.net/PUB/cf.pdf)
- [Variational Autoencoders for Collaborative Filtering](https://arxiv.org/abs/1802.05814)

### Citing
Please cite this paper in your publications if it helps your research:
```
@inproceedings{recoder,
  author = {Moussawi, Abdallah},
  title = {Towards Large Scale Training Of Autoencoders For Collaborative Filtering},
  booktitle = {Proceedings of Late-Breaking Results track part of the Twelfth ACM Conference on Recommender Systems},
  series = {RecSys'18},
  year = {2018},
  address = {Vancouver, BC, Canada}
}
```

### Acknowledgements
- I would like to thank [Anghami](https://www.anghami.com) for supporting this work,
and my colleagues, [Helmi Rifai](https://twitter.com/RifaiHelmi) and
[Ramzi Karam](https://twitter.com/ramzikaram), for great discussions on Collaborative Filtering at scale.

- This project started as a fork of [NVIDIA/DeepRecommender](https://github.com/NVIDIA/DeepRecommender),
and although it went in a slightly different direction and was entirely refactored,
the work in [NVIDIA/DeepRecommender](https://github.com/NVIDIA/DeepRecommender) was a
great contribution to the work here.
