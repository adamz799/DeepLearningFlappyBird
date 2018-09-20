# Overview
This project is a pytorch implemention of ["Playing Flappy Bird with DQN"](https://github.com/yenchenlin1994/DeepLearningFlappyBird.git). The network architecture is built following the ["Readme"](https://github.com/yenchenlin/DeepLearningFlappyBird/blob/master/README.md) file, which is a bit different to the [tensorflow](https://github.com/yenchenlin/DeepLearningFlappyBird/blob/master/deep_q_network.py) version.

## Installation Dependencies:
* Python 3.5
* Pytorch 0.4.1
* pygame
* OpenCV-Python
* Numpy

## How to Run?
```
git clone https://github.com/adamz799/DeepLearningFlappyBird.git
cd DeepLearningFlappyBird
python deep_q_network-pytorch.py
```

#### How to reproduce?
1. Comment out [this line](https://github.com/adamz799/DeepLearningFlappyBird/blob/master/deep_q_network-pytorch.py#L225) and comment [this line](https://github.com/adamz799/DeepLearningFlappyBird/blob/master/deep_q_network-pytorch.py#L226)

2. Modify `deep_q_network-pytorch.py`'s parameter as follow:
```python
OBSERVE = 10000
EXPLORE = 3000000
FINAL_EPSILON = 0.0001
INITIAL_EPSILON = 0.1
```

## References

[1] Mnih Volodymyr, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, Stig Petersen, Charles Beattie, Amir Sadik, Ioannis Antonoglou, Helen King, Dharshan Kumaran, Daan Wierstra, Shane Legg, and Demis Hassabis. **Human-level Control through Deep Reinforcement Learning**. Nature, 529-33, 2015.

[2] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin Riedmiller. **Playing Atari with Deep Reinforcement Learning**. NIPS, Deep Learning workshop

[3] Kevin Chen. **Deep Reinforcement Learning for Flappy Bird** [Report](http://cs229.stanford.edu/proj2015/362_report.pdf) | [Youtube result](https://youtu.be/9WKBzTUsPKc)

## Disclaimer
This work is highly based on the following repos:

1. [sourabhv/FlapPyBird] (https://github.com/sourabhv/FlapPyBird)
2. [asrivat1/DeepLearningVideoGames](https://github.com/asrivat1/DeepLearningVideoGames)
3. [yenchenlin/DeepLearningFlappyBird](https://github.com/yenchenlin/DeepLearningFlappyBird)

